"""
별도 테스트셋 평가 (예: AI Hub 2.Validation 의 REAL17·18 수어자)
=================================================================

Usage
-----
    python -m ml.training.evaluate \\
        --data_dir  E:/handtalk_data/npy_train \\
        --test_dir  E:/handtalk_data/npy_test \\
        --model_dir E:/handtalk_data/models

학습 데이터의 vocab.json 기준으로 라벨을 맞추고, 학습에 없던 단어 폴더는 건너뜀.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import SignDataset, _load_vocab
from .model import SignRecognizer
from .train import D_MODEL, DIM_FF, DROPOUT, INPUT_DIM, NHEAD, NUM_LAYERS, WINDOW_T


def evaluate(args) -> None:
    vocab = _load_vocab(Path(args.data_dir))
    ds = SignDataset(split="all", augment=False, window_T=WINDOW_T, vocab=vocab,
                     data_dir=Path(args.test_dir), packed=args.packed)
    print(f"Test: {len(ds)} clips | Classes: {len(vocab)} | Data: {args.test_dir}")

    model = SignRecognizer(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
                           num_layers=NUM_LAYERS, dim_ff=DIM_FF,
                           num_classes=len(vocab), dropout=DROPOUT)
    model.load_state_dict(torch.load(Path(args.model_dir) / "best_model.pt", map_location="cpu"))
    model.eval()

    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=min(2, args.num_workers))
    top1 = top5 = total = 0
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            k = min(5, logits.size(1))
            pred = logits.topk(k, dim=1).indices
            top1 += (pred[:, 0] == y).sum().item()
            top5 += (pred == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)

    print(f"Top-1 accuracy: {top1 / max(total, 1):.4f}  ({top1}/{total})")
    print(f"Top-5 accuracy: {top5 / max(total, 1):.4f}  ({top5}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="별도 테스트셋으로 수어 인식 모델 평가")
    parser.add_argument("--data_dir",  required=True, help="학습 데이터 경로 (vocab.json 기준)")
    parser.add_argument("--test_dir",  required=True, help="테스트 데이터 경로")
    parser.add_argument("--model_dir", required=True, help="best_model.pt 가 있는 경로")
    parser.add_argument("--packed",    action="store_true", help="float16 packed 캐시 사용")
    parser.add_argument("--num_workers", type=int, default=2)
    evaluate(parser.parse_args())
