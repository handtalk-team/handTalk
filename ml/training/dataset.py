"""
HandTalk Dataset
================

Directory layout expected
--------------------------
ml/data/raw/{label}/{recording_id}.npy   — (T, 77) float32 fused feature array
ml/data/references/{label}.npy           — mean reference array for DTW feedback

Each .npy file is one sign attempt:
  axis-0 (T) : number of frames (variable, ~30-90 for 1-3 second gestures)
  axis-1 (77): fused feature vector
                 [0:63]  vision landmarks  (21 × 3)
                 [63:68] flex sensors      (5)
                 [68:71] accelerometer     (3)
                 [71:74] gyroscope         (3)
                 [74:77] euler angles      (3)

Data collection
---------------
Run `python -m ml.training.collect` with a connected camera (and optionally
the real glove) to record samples directly into this structure.

Data augmentation (applied on-the-fly)
---------------------------------------
- Time warp  : randomly stretch/compress the sequence (±20%)
- Additive Gaussian noise : σ = 0.01 on all channels
- Speed jitter: resample to ±10% of original length

MVP target: 10 signs × 300 samples = 3,000 samples
Train / val / test split: 70 / 15 / 15
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

SIGN_VOCAB = {
    "안녕하세요": 0,
    "감사합니다":  1,
    "아프다":     2,
    "약":         3,
    "의사":       4,
    "간호사":     5,
    "도와주세요": 6,
    "화장실":     7,
    "물":         8,
    "괜찮아요":   9,
}

DATA_DIR = Path("ml/data/raw")


def _time_warp(seq: np.ndarray, factor_range=(0.8, 1.2)) -> np.ndarray:
    """Randomly stretch or compress the sequence along the time axis."""
    T = seq.shape[0]
    factor = random.uniform(*factor_range)
    new_T = max(10, int(T * factor))
    old_idx = np.linspace(0, T - 1, new_T)
    new_seq = np.stack([
        np.interp(old_idx, np.arange(T), seq[:, d])
        for d in range(seq.shape[1])
    ], axis=1)
    return new_seq.astype(np.float32)


def _add_noise(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def _pad_or_trim(seq: np.ndarray, target_T: int = 60) -> np.ndarray:
    T = seq.shape[0]
    if T >= target_T:
        # Centre-crop
        start = (T - target_T) // 2
        return seq[start: start + target_T]
    # Pad with last frame
    pad = np.repeat(seq[-1:], target_T - T, axis=0)
    return np.concatenate([seq, pad], axis=0)


class SignDataset(Dataset):
    """
    Loads all .npy files under DATA_DIR and returns fixed-length windows.

    Parameters
    ----------
    split   : "train" | "val" | "test"
    augment : apply time-warp + noise (training only)
    window_T: fixed sequence length fed to the model
    seed    : reproducible split
    """

    def __init__(
        self,
        split: str = "train",
        augment: bool = True,
        window_T: int = 60,
        seed: int = 42,
    ) -> None:
        self.augment  = augment and (split == "train")
        self.window_T = window_T
        self.samples: List[Tuple[np.ndarray, int]] = []

        rng = random.Random(seed)

        for label, class_idx in SIGN_VOCAB.items():
            label_dir = DATA_DIR / label
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
                self.samples.append((arr, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq, label = self.samples[idx]

        if self.augment:
            seq = _time_warp(seq)
            seq = _add_noise(seq)

        seq = _pad_or_trim(seq, self.window_T)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


def build_references() -> None:
    """
    Compute per-class mean sequence and save to ml/data/references/{label}.npy
    Used by the FeedbackEngine for DTW comparison.
    """
    ref_dir = Path("ml/models/references")
    ref_dir.mkdir(parents=True, exist_ok=True)

    for label in SIGN_VOCAB:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            continue
        files = list(label_dir.glob("*.npy"))
        if not files:
            continue
        arrays = [_pad_or_trim(np.load(str(f)).astype(np.float32)) for f in files]
        mean_arr = np.stack(arrays).mean(axis=0)
        out = ref_dir / f"{label}.npy"
        np.save(str(out), mean_arr)
        print(f"Reference saved: {out}")


if __name__ == "__main__":
    build_references()
