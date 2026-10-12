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

import hashlib
import json
import os
import random
import re
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


# ── Split 헬퍼 ─────────────────────────────────────────────────────────────────

_SIGNER_RE = re.compile(r"_(REAL|SYN)(\d+)_")


def _signer_of(path: Path) -> Optional[str]:
    """aihub_NIA_SL_WORD0001_REAL05_F.npy → 'REAL05' (syn_word 는 'SYN01')"""
    m = _SIGNER_RE.search(path.name)
    return m.group(1) + m.group(2) if m else None


def _signer_split(root: Path, vocab: Dict[str, int], seed: int,
                  holdout_test: bool = True) -> Optional[Dict[str, str]]:
    """
    수어자 → 'train'/'val'/'test'. 모든 파일에 수어자 번호가 있을 때만 (아니면 None).
    REAL 수어자만 val/test 로 떼어내고, syn_word(SYN) 클립은 항상 train.
    holdout_test=False: 별도 테스트셋(예: AI Hub 2.Validation)이 있을 때 test 몫도 train 으로.
    """
    signers = set()
    for label in vocab:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for f in label_dir.glob("*.npy"):
            s = _signer_of(f)
            if s is None:
                return None
            signers.add(s)
    real = sorted(s for s in signers if s.startswith("REAL"))
    if len(real) < (3 if holdout_test else 2):
        return None
    random.Random(seed).shuffle(real)
    n_val = max(1, round(len(real) * 0.15))
    n_test = max(1, round(len(real) * 0.15)) if holdout_test else 0
    n_train = len(real) - n_val - n_test
    split = {s: "train" for s in signers if not s.startswith("REAL")}
    split.update({s: "train" for s in real[:n_train]})
    split.update({s: "val" for s in real[n_train:n_train + n_val]})
    split.update({s: "test" for s in real[n_train + n_val:]})
    return split


# ── 대용량 데이터: float16 packed 캐시 ─────────────────────────────────────────
# 전체 AI Hub(수십만 클립)는 RAM에 다 못 올리므로 split별로 하나의 float16 파일에
# 이어 붙여 두고 memmap 으로 필요한 클립만 읽음.

def _build_pack(root: Path, split: str, items: List[Tuple[Path, int]]) -> Tuple[Path, np.ndarray]:
    """items(파일, 클래스) → (packed 데이터 경로, index[N,3] = offset, length, label)"""
    key = hashlib.md5("\n".join(f"{f.parent.name}/{f.name}:{c}" for f, c in items)
                      .encode("utf-8")).hexdigest()[:12]
    cache_dir = root / "_cache"
    cache_dir.mkdir(exist_ok=True)
    data_path = cache_dir / f"{split}_{key}.f16"
    idx_path  = cache_dir / f"{split}_{key}.idx.npy"
    if data_path.exists() and idx_path.exists():
        return data_path, np.load(str(idx_path))

    print(f"[{split}] packed 캐시 생성 중 ({len(items)}개 클립) → {data_path}", flush=True)
    lengths = np.array([np.load(str(f), mmap_mode="r").shape[0] for f, _ in items], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    total = int(lengths.sum())
    tmp_path = data_path.with_suffix(".tmp")
    mm = np.memmap(str(tmp_path), dtype=np.float16, mode="w+", shape=(max(total, 1), RAW_DIM))
    for i, (f, _) in enumerate(items):
        mm[offsets[i]: offsets[i] + lengths[i]] = _load_raw(f).astype(np.float16)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)
    mm.flush()
    del mm
    os.replace(tmp_path, data_path)
    index = np.stack([offsets, lengths, np.array([c for _, c in items], dtype=np.int64)], axis=1)
    np.save(str(idx_path), index)
    return data_path, index


def _load_raw(f: Path) -> np.ndarray:
    arr = np.load(str(f)).astype(np.float32)
    if arr.shape[1] < RAW_DIM:
        pad = np.zeros((arr.shape[0], RAW_DIM - arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1)
    return arr[:, :RAW_DIM]


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
        packed: bool = False,
        holdout_test: bool = True,
    ) -> None:
        """
        packed=True      : 클립을 RAM에 올리지 않고 float16 packed 캐시(memmap)에서 읽음 (전체 AI Hub용)
        holdout_test=False: 별도 테스트셋이 있을 때 수어자 split 에서 test 몫을 train 으로
        split="all"      : 폴더 안 전체 클립 (별도 테스트셋 평가용)
        """
        self.augment  = augment and (split == "train")
        self.window_T = window_T
        self.samples: List[Tuple[np.ndarray, int]] = []
        self._pack_path: Optional[Path] = None
        self._pack_index: Optional[np.ndarray] = None
        self._pack_mm: Optional[np.memmap] = None

        root = Path(data_dir) if data_dir else DATA_DIR
        sign_vocab = vocab if vocab is not None else _load_vocab(root)
        self.num_classes = len(sign_vocab)
        rng = random.Random(seed)

        # AI Hub 파일명(aihub_NIA_SL_WORDxxxx_REALnn_X.npy)에 수어자 번호가 있으면
        # 수어자 단위로 split → 같은 사람의 다른 카메라 각도가 train/val에 섞이지 않음
        signer_split = None if split == "all" else _signer_split(root, sign_vocab, seed, holdout_test)

        items: List[Tuple[Path, int]] = []
        for label, class_idx in sign_vocab.items():
            label_dir = root / label
            if not label_dir.exists():
                continue
            files = sorted(label_dir.glob("*.npy"))

            if split == "all":
                items.extend((f, class_idx) for f in files)
                continue

            if signer_split is not None:
                items.extend((f, class_idx) for f in files
                             if signer_split.get(_signer_of(f)) == split)
                continue

            rng.shuffle(files)

            n = len(files)
            n_train = int(n * 0.70)
            n_val   = int(n * 0.15)
            # 샘플 데이터처럼 클래스당 클립이 적을 때(배치 1개 = 5방향)도 val 최소 1개 확보
            if n_val == 0 and n >= 3:
                n_val = 1
                n_train = min(n_train, n - 1)

            if split == "train":
                chosen = files[:n_train]
            elif split == "val":
                chosen = files[n_train: n_train + n_val]
            else:
                chosen = files[n_train + n_val:]

            items.extend((f, class_idx) for f in chosen)

        if packed and items:
            self._pack_path, self._pack_index = _build_pack(root, split, items)
        else:
            self.samples = [(_load_raw(f), c) for f, c in items]

    def __len__(self) -> int:
        if self._pack_index is not None:
            return len(self._pack_index)
        return len(self.samples)

    def __getstate__(self):
        # DataLoader 워커로 넘길 때 memmap 은 다시 열도록 (통째로 pickle 되지 않게)
        state = self.__dict__.copy()
        state["_pack_mm"] = None
        return state

    def _get_raw(self, idx: int) -> Tuple[np.ndarray, int]:
        if self._pack_index is None:
            return self.samples[idx]
        if self._pack_mm is None:
            self._pack_mm = np.memmap(str(self._pack_path), dtype=np.float16, mode="r").reshape(-1, RAW_DIM)
        off, length, label = self._pack_index[idx]
        return np.asarray(self._pack_mm[off: off + length], dtype=np.float32), int(label)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq, label = self._get_raw(idx)   # (T, 136)

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
