"""
Training Script — BiGRU + Self-Attention Sign Recogniser
=========================================================

Usage
-----
    # Train from scratch
    python -m ml.training.train

    # Resume from checkpoint
    python -m ml.training.train --resume ml/models/checkpoints/epoch_20.pt

    # Train then export to ONNX
    python -m ml.training.train --export

Outputs
-------
    ml/models/checkpoints/epoch_{N}.pt   — per-epoch checkpoints
    ml/models/best_model.pt              — best val-accuracy checkpoint
    ml/models/sign_recognizer.onnx       — ONNX model (if --export)
    ml/models/training_log.csv           — loss & accuracy per epoch

Tips for 10-class, 300-sample-per-class dataset
------------------------------------------------
- Use label smoothing (0.1) to prevent overconfidence on small data
- Early stopping at patience=15 prevents overfitting
- Cosine LR schedule with warm-up helps generalisation
- Target val accuracy > 90% before exporting to ONNX
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import SignDataset, build_references
from .model import SignRecognizer

# ─── Hyper-parameters ────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 150
LR            = 3e-4          # Transformer는 낮은 LR 선호
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
LABEL_SMOOTH  = 0.1
PATIENCE      = 20
WINDOW_T      = 60
INPUT_DIM     = 272   # 136 raw + 136 velocity
# Transformer 하이퍼파라미터
D_MODEL       = 256
NHEAD         = 4
NUM_LAYERS    = 3
DIM_FF        = 512


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    train_ds = SignDataset(split="train", augment=True,  window_T=WINDOW_T, data_dir=data_dir)
    val_ds   = SignDataset(split="val",   augment=False, window_T=WINDOW_T, data_dir=data_dir)

    if len(train_ds) == 0:
        print(
            f"\n[ERROR] No training data found in {data_dir}\n"
            "Run the extractor first:\n"
            "  python -m ml.preprocessing.extract_aihub --keypoint ... --morpheme ... --output <data_dir>\n"
        )
        return

    num_classes = train_ds.num_classes
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {num_classes} | Data: {data_dir}")

    # Colab/멀티프로세스 환경에 따라 num_workers 자동 설정
    nw = min(2, args.num_workers)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=nw, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw, pin_memory=(device.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────
    model = SignRecognizer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from: {args.resume}")

    # ── Optimiser & scheduler ─────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine LR with linear warm-up (5 epochs)
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )

    # ── Logging ───────────────────────────────────────────────────
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    no_improve = 0

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += len(y)

        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(y)

        tr_acc  = tr_correct  / tr_total  if tr_total  else 0.0
        val_acc = val_correct / val_total if val_total else 0.0
        tr_loss_avg  = tr_loss  / tr_total  if tr_total  else 0.0
        val_loss_avg = val_loss / val_total if val_total else 0.0
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{EPOCHS}  "
            f"train_loss={tr_loss_avg:.4f}  train_acc={tr_acc:.3f}  "
            f"val_loss={val_loss_avg:.4f}  val_acc={val_acc:.3f}  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, tr_loss_avg, tr_acc, val_loss_avg, val_acc, current_lr,
            ])

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            no_improve = 0
            print(f"  ★ New best val accuracy: {best_val_acc:.3f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")

    if args.export:
        model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location="cpu"))
        model.cpu()
        model.export_onnx(str(model_dir / "sign_recognizer.onnx"))
        build_references(data_dir=data_dir, model_dir=model_dir)

        # labels.txt: class_idx 순서대로 한 줄씩 (engine.py에서 읽음)
        from .dataset import _load_vocab as _lv
        vocab = _lv(data_dir)
        labels_by_idx = sorted(vocab.items(), key=lambda kv: kv[1])
        labels_path = model_dir / "labels.txt"
        with open(labels_path, "w", encoding="utf-8") as f:
            for lbl, _ in labels_by_idx:
                f.write(lbl + "\n")
        print(f"labels.txt → {labels_path} ({len(labels_by_idx)} classes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer sign recogniser")
    parser.add_argument("--data_dir",    default="ml/data",   help="학습 데이터 경로")
    parser.add_argument("--model_dir",   default="ml/models", help="모델 저장 경로")
    parser.add_argument("--resume",      default=None,        help="이어서 학습할 체크포인트 .pt")
    parser.add_argument("--export",      action="store_true", help="학습 후 ONNX 내보내기")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 워커 수")
    args = parser.parse_args()
    train(args)
