"""
BiGRU Sign Language Classifier — Training Script
=================================================

Usage:
    python ml/train.py

Output:
    ml/models/sign_recognizer.onnx
    ml/models/labels.txt
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ─── Config ──────────────────────────────────────────────────────
DATA_DIR   = "ml/data"
MODEL_DIR  = "ml/models"
ONNX_PATH  = os.path.join(MODEL_DIR, "sign_recognizer.onnx")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

INPUT_DIM  = 63   # vision landmarks only (77차원 중 앞 63개)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.3
SEQ_LEN    = 60   # 패딩/자르기 기준 프레임 수

EPOCHS     = 100
LR         = 1e-3
BATCH_SIZE = 8


# ─── Dataset ─────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, data_dir: str, labels: list[str]):
        self.samples: list[tuple[torch.Tensor, int]] = []
        for idx, label in enumerate(labels):
            label_dir = os.path.join(data_dir, label)
            for fname in sorted(os.listdir(label_dir)):
                if not fname.endswith(".npy"):
                    continue
                arr = np.load(os.path.join(label_dir, fname))  # (T, 77)
                arr = arr[:, :INPUT_DIM].astype(np.float32)    # (T, 63)
                arr = self._pad_or_trim(arr)
                self.samples.append((torch.tensor(arr), idx))

    def _pad_or_trim(self, arr: np.ndarray) -> np.ndarray:
        T = arr.shape[0]
        if T >= SEQ_LEN:
            return arr[:SEQ_LEN]
        pad = np.zeros((SEQ_LEN - T, INPUT_DIM), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ─── Model ───────────────────────────────────────────────────────

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):                         # x: (B, T, D)
        out, _ = self.gru(x)                      # (B, T, H*2)
        w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        ctx = (out * w).sum(dim=1)                # (B, H*2)
        return self.fc(self.drop(ctx))            # (B, C)


# ─── Training ────────────────────────────────────────────────────

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    labels = sorted(os.listdir(DATA_DIR))
    labels = [l for l in labels if os.path.isdir(os.path.join(DATA_DIR, l))]
    print(f"레이블: {labels}")

    dataset = SignDataset(DATA_DIR, labels)
    print(f"총 샘플: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BiGRUClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, len(labels), DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()

        avg_loss = total_loss / total
        acc = correct / total * 100
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  loss={avg_loss:.4f}  acc={acc:.1f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best.pt"))

    # ─── ONNX export ─────────────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best.pt"), weights_only=True))
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, INPUT_DIM)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"\nONNX 저장 완료: {ONNX_PATH}")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))
    print(f"레이블 저장 완료: {LABELS_PATH}")
    print(f"레이블 순서: {labels}")


if __name__ == "__main__":
    train()
