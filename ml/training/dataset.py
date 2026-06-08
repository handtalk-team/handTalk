"""
HandTalk Dataset
================

Directory layout
----------------
ml/data/{label}/{recording_id}.npy   — (T, 136) float32 raw features
ml/data/vocab.json                   — {"label": class_idx}

Feature vector per frame (136D raw → 272D after velocity)
----------------------------------------------------------
Raw 136D (stored in .npy):
  [0:63]    오른손 world landmarks (21×3, wrist-centered + scale-normalised)
  [63:126]  왼손   world landmarks (21×3)
  [126:131] 오른손 flex (5)
  [131:136] 왼손   flex (5)

Model input 272D (computed on-the-fly):
  [0:136]   위 raw 피처 (위치)
  [136:272] 속도 (프레임 간 차이) ← 수어의 움직임 방향/속도 인코딩

Augmentations (training only)
------------------------------
  time_warp      : 시퀀스 길이 ±20% 늘리거나 줄임 (빠른/느린 수어)
  noise          : 미세 가우시안 노이즈 (센서 노이즈 모사)
  mirror_flip    : 좌우 손 교환 (거울 수어)
  spatial_jitter : 전체 손 위치 미세 이동 (몸 위치 변화)
  scale_jitter   : 손 크기 미세 변형 (체형 차이)
  frame_dropout  : 랜덤 프레임 제거 (인식 실패/폐색 모사)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

RAW_DIM     = 136
VISION_DIM  = 126
FLEX_DIM    = 10
FEATURE_DIM = RAW_DIM * 2   # 272 = 위치(136) + 속도(136)

# 환경변수 HANDTALK_DATA_DIR 로 재정의 가능 (Colab/서버 환경)
DATA_DIR = Path(os.environ.get("HANDTALK_DATA_DIR", "ml/data"))


def _load_vocab(data_dir: Path = DATA_DIR) -> Dict[str, int]:
    vocab_path = data_dir / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, encoding="utf-8") as f:
            return json.load(f)
    labels = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    return {lbl: i for i, lbl in enumerate(labels)}


# ── 피처 변환 ──────────────────────────────────────────────────────────────────

def add_velocity(seq: np.ndarray) -> np.ndarray:
    """
    (T, 136) → (T, 272): 원본 피처 + 속도(프레임 간 차이)

    속도 피처가 중요한 이유:
      수어는 '손 모양'뿐 아니라 '움직임 방향과 속도'로 구분됨.
      예) '아프다'와 '건강'이 손 모양은 비슷해도 움직임이 다름.
      첫 프레임 속도는 0으로 패딩.
    """
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)


# ── Augmentation 함수들 ────────────────────────────────────────────────────────

def _time_warp(seq: np.ndarray, factor_range=(0.8, 1.2)) -> np.ndarray:
    """
    시퀀스 속도 변형 (±20%).
    실제 수어: 사람마다 빠르게/느리게 함 → 이 다양성을 학습.
    """
    T = seq.shape[0]
    factor = random.uniform(*factor_range)
    new_T = max(10, int(T * factor))
    idx = np.linspace(0, T - 1, new_T)
    return np.stack([
        np.interp(idx, np.arange(T), seq[:, d])
        for d in range(seq.shape[1])
    ], axis=1).astype(np.float32)


def _add_noise(seq: np.ndarray) -> np.ndarray:
    """
    채널별 가우시안 노이즈.
    랜드마크(촘촘한 좌표)는 σ=0.008, flex(0~1 범위)는 σ=0.02.
    """
    out = seq.copy()
    out[:, :VISION_DIM] += np.random.normal(0, 0.008, (len(seq), VISION_DIM)).astype(np.float32)
    out[:, VISION_DIM:]  += np.random.normal(0, 0.020, (len(seq), FLEX_DIM)).astype(np.float32)
    return np.clip(out, -5.0, 5.0).astype(np.float32)


def _mirror_hands(seq: np.ndarray) -> np.ndarray:
    """
    좌우 손 교환.
    AI Hub 데이터는 오른손 위주 → 왼손잡이 수어 커버.
    """
    out = seq.copy()
    out[:, :63],    out[:, 63:126]  = seq[:, 63:126].copy(),  seq[:, :63].copy()
    out[:, 126:131], out[:, 131:136] = seq[:, 131:136].copy(), seq[:, 126:131].copy()
    return out


def _spatial_jitter(seq: np.ndarray, sigma: float = 0.04) -> np.ndarray:
    """
    전체 손 위치 미세 이동 (손목 기준 좌표에 오프셋 추가).
    실제 수어: 배꼽 앞에서 하기도 하고, 가슴 앞에서 하기도 함.
    정규화 후에도 남은 자세 차이를 커버.
    """
    out = seq.copy()
    r_offset = np.random.normal(0, sigma, 3).astype(np.float32)
    l_offset = np.random.normal(0, sigma, 3).astype(np.float32)
    # 오른손 21포인트 각각에 동일 오프셋
    for i in range(0, 63, 3):
        out[:, i:i+3]    += r_offset
        out[:, 63+i:66+i] += l_offset
    return out


def _scale_jitter(seq: np.ndarray, scale_range=(0.85, 1.15)) -> np.ndarray:
    """
    손 크기 미세 변형.
    체형에 따라 손 크기가 달라도 잘 인식하도록.
    (정규화로 대부분 제거되지만 미세 잔여분 커버)
    """
    scale = np.random.uniform(*scale_range)
    out = seq.copy()
    out[:, :VISION_DIM] *= scale
    return out


def _frame_dropout(seq: np.ndarray, p: float = 0.08) -> np.ndarray:
    """
    랜덤 프레임 제거 (해당 프레임을 직전 프레임으로 대체).
    카메라 프레임 드롭, 손 일시적 미감지 상황 모사.
    """
    out = seq.copy()
    for t in range(1, len(seq)):
        if random.random() < p:
            out[t] = out[t - 1]
    return out


def _pad_or_trim(seq: np.ndarray, target_T: int = 60) -> np.ndarray:
    T = seq.shape[0]
    if T >= target_T:
        start = (T - target_T) // 2
        return seq[start: start + target_T]
    pad = np.repeat(seq[-1:], target_T - T, axis=0)
    return np.concatenate([seq, pad], axis=0)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        augment: bool = True,
        window_T: int = 60,
        seed: int = 42,
        vocab: Optional[Dict[str, int]] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        self.augment  = augment and (split == "train")
        self.window_T = window_T
        self.samples: List[Tuple[np.ndarray, int]] = []

        root = Path(data_dir) if data_dir else DATA_DIR
        sign_vocab = vocab if vocab is not None else _load_vocab(root)
        self.num_classes = len(sign_vocab)
        rng = random.Random(seed)

        for label, class_idx in sign_vocab.items():
            label_dir = root / label
            if not label_dir.exists():
                continue
            files = sorted(label_dir.glob("*.npy"))
            rng.shuffle(files)

            n = len(files)
            n_train = int(n * 0.70)
            n_val   = int(n * 0.15)

            if split == "train":
                chosen = files[:n_train]
            elif split == "val":
                chosen = files[n_train: n_train + n_val]
            else:
                chosen = files[n_train + n_val:]

            for f in chosen:
                arr = np.load(str(f)).astype(np.float32)
                if arr.shape[1] < RAW_DIM:
                    pad = np.zeros((arr.shape[0], RAW_DIM - arr.shape[1]), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=1)
                else:
                    arr = arr[:, :RAW_DIM]
                self.samples.append((arr, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq, label = self.samples[idx]   # (T, 136)

        if self.augment:
            seq = _time_warp(seq)
            seq = _add_noise(seq)
            if random.random() < 0.5:
                seq = _mirror_hands(seq)
            if random.random() < 0.7:
                seq = _spatial_jitter(seq)
            if random.random() < 0.5:
                seq = _scale_jitter(seq)
            if random.random() < 0.5:
                seq = _frame_dropout(seq)

        seq = _pad_or_trim(seq, self.window_T)  # (60, 136)
        seq = add_velocity(seq)                 # (60, 272) ← 속도 추가

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


def build_references(
    vocab: Optional[Dict[str, int]] = None,
    data_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> None:
    root    = Path(data_dir)  if data_dir  else DATA_DIR
    out_dir = Path(model_dir) / "references" if model_dir else Path("ml/models/references")
    out_dir.mkdir(parents=True, exist_ok=True)
    sign_vocab = vocab if vocab is not None else _load_vocab(root)
    for label in sign_vocab:
        label_dir = root / label
        if not label_dir.exists():
            continue
        files = list(label_dir.glob("*.npy"))
        if not files:
            continue
        arrays = [_pad_or_trim(np.load(str(f)).astype(np.float32)) for f in files]
        mean_arr = np.stack(arrays).mean(axis=0)
        np.save(str(out_dir / f"{label}.npy"), mean_arr)
        print(f"Reference saved: {out_dir / label}.npy")


if __name__ == "__main__":
    build_references()
