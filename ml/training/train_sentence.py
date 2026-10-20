"""
문장(연속) 수어 인식 — Transformer encoder + CTC
================================================

단어 모델(train.py)은 클립 하나를 단어 하나로 분류한다. 문장은 단어가 이어지므로
프레임별로 글로스 확률을 내고 CTC 로 순서만 맞추도록 학습한다 (구간 라벨 불필요).

Usage
-----
    python -m ml.training.train_sentence \\
        --data_dir  E:/handtalk_data/sentence \\
        --model_dir E:/handtalk_data/models_sentence \\
        --init_word E:/handtalk_data/models/best_model.pt     # 단어 모델 가중치로 시작 (선택)

데이터: ml.preprocessing.extract_aihub_sentence 출력 (npy/{clip}.npy + labels_*.jsonl)
Split : 수어자 단위. --val_signers / --test_signers (기본 REAL05,REAL12 / REAL17 = AI Hub Validation)
출력  : best_model.pt, last.pt(이어 학습), glosses.txt, sentence_recognizer.onnx(--export)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset import RAW_DIM, _add_noise, _mirror_hands, _scale_jitter, _spatial_jitter, webcam_augment

STRIDE   = 2       # 30fps → 15fps
MAX_T    = 240     # 15fps 기준 16초
D_MODEL  = 256
NHEAD    = 4
LAYERS   = 4
DIM_FF   = 512
DROPOUT  = 0.2
BATCH    = 32
LR       = 3e-4
PATIENCE = 8


# ── 데이터 ────────────────────────────────────────────────────────────────────

def load_labels(data_dir: Path) -> List[dict]:
    rows = []
    for p in sorted(data_dir.glob("labels_*.jsonl")):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    seen, out = set(), []
    for r in rows:
        if r["clip"] not in seen and (data_dir / "npy" / f"{r['clip']}.npy").exists():
            seen.add(r["clip"]); out.append(r)
    return out


def build_vocab(rows: List[dict]) -> List[str]:
    return sorted({g for r in rows for g in r["glosses"]})   # index 0 = CTC blank, 글로스는 1부터


class SentenceDataset(Dataset):
    """float16 packed 캐시(memmap)에서 문장 클립을 읽음 — 수십만 클립도 RAM 부담 없음."""

    def __init__(self, data_dir: Path, rows: List[dict], gloss2id: Dict[str, int],
                 split: str, augment: bool, webcam_aug: float = 0.0) -> None:
        self.augment = augment
        self.webcam_aug = webcam_aug
        self.rows = rows
        self.targets = [[gloss2id[g] for g in r["glosses"]] for r in rows]
        key = hashlib.md5("\n".join(r["clip"] for r in rows).encode()).hexdigest()[:12]
        cache = data_dir / "_cache"; cache.mkdir(exist_ok=True)
        self.data_path = cache / f"sen_{split}_{key}.f16"
        idx_path = cache / f"sen_{split}_{key}.idx.npy"
        if not (self.data_path.exists() and idx_path.exists()):
            print(f"[{split}] packed 캐시 생성 ({len(rows)} clips)", flush=True)
            lens = [np.load(str(data_dir / "npy" / f"{r['clip']}.npy"), mmap_mode="r").shape[0] for r in rows]
            offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
            tmp = self.data_path.with_suffix(".tmp")
            mm = np.memmap(str(tmp), np.float16, "w+", shape=(max(1, int(sum(lens))), RAW_DIM))
            for i, r in enumerate(rows):
                mm[offs[i]:offs[i] + lens[i]] = np.load(str(data_dir / "npy" / f"{r['clip']}.npy"))[:, :RAW_DIM]
            mm.flush(); del mm
            tmp.replace(self.data_path)
            np.save(str(idx_path), np.stack([offs, np.asarray(lens, np.int64)], 1))
        self.index = np.load(str(idx_path))
        self._mm = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        s = self.__dict__.copy(); s["_mm"] = None; return s

    def __getitem__(self, i: int):
        if self._mm is None:
            self._mm = np.memmap(str(self.data_path), np.float16, "r").reshape(-1, RAW_DIM)
        off, n = self.index[i]
        seq = np.asarray(self._mm[off:off + n], np.float32)
        if self.augment:
            seq = _add_noise(seq)
            if random.random() < 0.5: seq = _mirror_hands(seq)
            if random.random() < 0.7: seq = _spatial_jitter(seq)
            if random.random() < 0.5: seq = _scale_jitter(seq)
            # 속도 변화: 프레임 간격을 1~3 중 무작위 (기본 2)
            stride = random.choice([1, 2, 2, 3])
        else:
            stride = STRIDE
        if self.webcam_aug and random.random() < self.webcam_aug:
            seq = webcam_augment(seq)
        seq = seq[::stride][:MAX_T]
        vel = np.zeros_like(seq); vel[1:] = seq[1:] - seq[:-1]
        x = np.concatenate([seq, vel], 1).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.targets[i], dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    T = max(x.shape[0] for x in xs)
    X = torch.zeros(len(xs), T, xs[0].shape[1])
    mask = torch.ones(len(xs), T, dtype=torch.bool)   # True = 패딩
    for i, x in enumerate(xs):
        X[i, :x.shape[0]] = x; mask[i, :x.shape[0]] = False
    lens = torch.tensor([x.shape[0] for x in xs])
    return X, mask, lens, torch.cat(ys), torch.tensor([len(y) for y in ys])


# ── 모델 ──────────────────────────────────────────────────────────────────────

class SentenceRecognizer(nn.Module):
    def __init__(self, num_glosses: int, input_dim: int = RAW_DIM * 2) -> None:
        super().__init__()
        # 이름을 단어 모델(SignRecognizer)과 맞춰서 가중치를 그대로 옮겨 올 수 있게 함
        self.input_proj = nn.Sequential(nn.Linear(input_dim, D_MODEL), nn.LayerNorm(D_MODEL),
                                        nn.ReLU(), nn.Dropout(DROPOUT))
        self.pos = nn.Embedding(MAX_T + 8, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, DIM_FF, DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, num_glosses + 1))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x) + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(h)   # (B, T, V+1) — 0 = blank

    def load_word_encoder(self, path: str) -> int:
        """단어 모델 체크포인트에서 모양이 같은 가중치만 복사 (input_proj, encoder 앞쪽 층)."""
        src = torch.load(path, map_location="cpu")
        if isinstance(src, dict) and "model" in src:
            src = src["model"]
        own = self.state_dict(); n = 0
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; n += 1
        self.load_state_dict(own)
        return n


def greedy_decode(logits: torch.Tensor, lens: torch.Tensor) -> List[List[int]]:
    best = logits.argmax(-1).cpu().numpy()
    out = []
    for b, n in zip(best, lens.tolist()):
        seq, prev = [], 0
        for t in b[:n]:
            if t != prev and t != 0:
                seq.append(int(t))
            prev = t
        out.append(seq)
    return out


def edit_distance(a: List[int], b: List[int]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


def evaluate(model, dl, device) -> Tuple[float, float, float]:
    model.eval(); ctc = nn.CTCLoss(zero_infinity=True)
    loss_sum = n = errs = toks = exact = 0
    with torch.no_grad():
        for X, mask, lens, Y, ylens in dl:
            X, mask = X.to(device), mask.to(device)
            logits = model(X, mask)
            lp = logits.log_softmax(-1).transpose(0, 1)
            loss_sum += ctc(lp, Y, lens, ylens).item() * len(lens); n += len(lens)
            hyp = greedy_decode(logits, lens)
            refs = torch.split(Y, ylens.tolist())
            for h, r in zip(hyp, refs):
                r = r.tolist(); e = edit_distance(h, r)
                errs += e; toks += len(r); exact += int(e == 0)
    return loss_sum / max(n, 1), errs / max(toks, 1), exact / max(n, 1)


# ── 학습 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="문장 수어 인식 (CTC) 학습")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--init_word", default=None, help="단어 모델 best_model.pt (encoder 초기화)")
    ap.add_argument("--val_signers", default="REAL05,REAL12")
    ap.add_argument("--test_signers", default="REAL17")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=LR, help="최대 학습률 (미세조정은 낮게, 예: 5e-5)")
    ap.add_argument("--init_from", default=None, help="문장 모델 best_model.pt 에서 시작 (미세조정)")
    ap.add_argument("--webcam_aug", type=float, default=0.0, help="웹캠 조건 흉내 증강 확률, 예: 0.7")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="model_dir/last.pt 에서 이어서")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()

    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = load_labels(data_dir)
    vocab = build_vocab(rows)
    (model_dir / "glosses.txt").write_text("<blank>\n" + "\n".join(vocab) + "\n", encoding="utf-8")
    g2i = {g: i + 1 for i, g in enumerate(vocab)}
    val_s, test_s = set(args.val_signers.split(",")), set(args.test_signers.split(","))
    split = lambda r: "test" if r["signer"] in test_s else "val" if r["signer"] in val_s else "train"
    parts = {k: [r for r in rows if split(r) == k] for k in ("train", "val", "test")}
    print({k: len(v) for k, v in parts.items()}, f"glosses={len(vocab)}", flush=True)

    tr = SentenceDataset(data_dir, parts["train"], g2i, "train", augment=True, webcam_aug=args.webcam_aug)
    va = SentenceDataset(data_dir, parts["val"], g2i, "val", augment=False)
    nw = min(2, args.num_workers)
    tr_dl = DataLoader(tr, BATCH, shuffle=True, num_workers=nw, collate_fn=collate, persistent_workers=nw > 0)
    va_dl = DataLoader(va, BATCH, shuffle=False, num_workers=nw, collate_fn=collate, persistent_workers=nw > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceRecognizer(len(vocab)).to(device)
    if args.init_from and not args.resume:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"문장 모델 가중치에서 시작: {args.init_from}", flush=True)
    elif args.init_word and not args.resume:
        print(f"단어 모델에서 가중치 {model.load_word_encoder(args.init_word)}개 복사", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(tr_dl)),
                                                pct_start=0.05)
    ctc = nn.CTCLoss(zero_infinity=True)
    start, best, bad = 1, math.inf, 0
    last = model_dir / "last.pt"
    if args.resume and last.exists():
        s = torch.load(last, map_location=device)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, best, bad = s["epoch"] + 1, s["best"], s["bad"]
        print(f"resume from epoch {start}", flush=True)
    log = model_dir / "training_log.csv"
    if start == 1:
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_wer", "val_sentence_acc"])

    for ep in range(start, args.epochs + 1):
        model.train(); t0 = time.time(); tot = cnt = 0
        for step, (X, mask, lens, Y, ylens) in enumerate(tr_dl, 1):
            X, mask = X.to(device), mask.to(device)
            lp = model(X, mask).log_softmax(-1).transpose(0, 1)
            loss = ctc(lp, Y, lens, ylens)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(lens); cnt += len(lens)
            if args.log_every and step % args.log_every == 0:
                print(f"  [epoch {ep}] {cnt}/{len(tr)} loss={tot / cnt:.3f} ({(time.time() - t0) / 60:.1f}분)", flush=True)
        vl, wer, acc = evaluate(model, va_dl, device)
        print(f"Epoch {ep:03d} train_loss={tot / max(cnt, 1):.3f} val_loss={vl:.3f} "
              f"val_WER={wer:.3f} val_sentence_acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep, tot / max(cnt, 1), vl, wer, acc])
        if wer < best:
            best, bad = wer, 0
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            print(f"  ★ best val WER {wer:.3f}", flush=True)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "bad": bad}, last)
        if bad >= PATIENCE:
            print("early stop", flush=True); break

    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    if parts["test"]:
        te = SentenceDataset(data_dir, parts["test"], g2i, "test", augment=False)
        tl, twer, tacc = evaluate(model, DataLoader(te, BATCH, collate_fn=collate), device)
        print(f"TEST (signers {args.test_signers}): WER={twer:.3f} sentence_acc={tacc:.3f}", flush=True)
    if args.export:
        model.cpu().eval()
        torch.onnx.export(model, (torch.zeros(1, 120, RAW_DIM * 2),), str(model_dir / "sentence_recognizer.onnx"),
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {1: "time"}, "logits": {1: "time"}}, opset_version=17, dynamo=False)
        print("exported sentence_recognizer.onnx", flush=True)


if __name__ == "__main__":
    main()
