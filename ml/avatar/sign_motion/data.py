"""
학습 데이터 읽기: AI Hub 문장(npz, 15fps)과 단어(handtalk-motion JSON, 30fps)를 같은 형식으로 맞춘다.

샘플 하나 = 토큰 열 + 토큰별 길이(프레임) + 동작(T, 50, 3) + 수어자 id
토큰 열은  [BOS, g1, SEP, g2, ..., gN, EOS]
  BOS  쉬는 자세에서 첫 단어까지      SEP  단어 사이 이어지는 동작      EOS  마지막 단어에서 쉬는 자세로
단어 구간 시각(start/end)으로 각 토큰 길이를 정하므로, 모델이 단어 길이와 연결 동작을 함께 배운다.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

FPS = 15.0
# 모델이 다루는 관절 (서버 motion_synth.JOINTS 에서 mid_hip 을 뺀 것)
JOINTS = ["neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "nose"] \
    + [f"rh_{i}" for i in range(21)] + [f"lh_{i}" for i in range(21)]
NJ = len(JOINTS)

PAD, BOS, SEP, EOS = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>"]


@dataclass
class Sample:
    glosses: List[str]
    spans: List[tuple]          # 단어마다 (시작 프레임, 끝 프레임)
    motion: np.ndarray          # (T, NJ, 3) float32
    signer: str


def _pick(frames: np.ndarray, names: Sequence[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros((frames.shape[0], NJ, 3), dtype=np.float32)
    for j, n in enumerate(JOINTS):
        if n in idx:
            out[:, j] = frames[:, idx[n]]
    return out


def _resample(x: np.ndarray, src_fps: float, dst_fps: float = FPS) -> np.ndarray:
    if abs(src_fps - dst_fps) < 0.5 or len(x) < 2:
        return x
    n = max(int(round(len(x) * dst_fps / src_fps)), 2)
    t = np.linspace(0, len(x) - 1, n)
    a = np.floor(t).astype(int)
    b = np.minimum(a + 1, len(x) - 1)
    w = (t - a)[:, None, None].astype(np.float32)
    return x[a] * (1 - w) + x[b] * w


def _signer(name: str) -> str:
    m = re.search(r"(REAL\d+)", name)
    return m.group(1) if m else "REAL01"


def load_sentences(folder: str) -> List[Sample]:
    """NIA_SL_SEN####_REALxx_F.npz (frames, joints, fps, glosses)"""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True)):
        z = np.load(p, allow_pickle=False)
        fps = float(z["fps"])
        joints = [str(j) for j in z["joints"]]
        motion = _resample(_pick(z["frames"].astype(np.float32), joints), fps)
        gl = json.loads(str(z["glosses"]))
        spans = [(int(round(g["start"] * FPS)), int(round(g["end"] * FPS))) for g in gl]
        out.append(Sample([g["gloss"] for g in gl], spans, motion, _signer(os.path.basename(p))))
    return out


def _active_span(motion: np.ndarray, thresh: float = 0.35) -> tuple:
    """단어 영상에서 손이 쉬는 자세에서 벗어나 있는 구간 (손목이 첫 프레임 위치에서 멀어진 구간)"""
    w = [JOINTS.index("r_wrist"), JOINTS.index("l_wrist")]
    d = np.max([np.linalg.norm(motion[:, j] - motion[0, j], axis=-1) for j in w], axis=0)
    on = np.where(d > thresh)[0]
    if len(on) == 0:
        return 0, len(motion) - 1
    return int(on[0]), int(on[-1])


def load_words(folder: str, lexicon: Optional[Dict[str, str]] = None) -> List[Sample]:
    """*.motion.json (서버 단어 사전). lexicon: clip → 글로스 (없으면 파일의 label)"""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.motion.json"))):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if not m.get("frames"):
            continue
        clip = m.get("clip") or os.path.basename(p).split(".")[0]
        gloss = (lexicon or {}).get(clip) or m.get("label")
        if not gloss:
            continue
        frames = np.asarray(m["frames"], dtype=np.float32).reshape(len(m["frames"]), -1, 3)
        motion = _resample(_pick(frames, m["joints"]), float(m.get("fps") or 30.0))
        a, b = _active_span(motion)
        out.append(Sample([gloss], [(a, b)], motion, _signer(m.get("source", ""))))
    return out


def tokens_and_durations(s: Sample, vocab: Dict[str, int]) -> tuple:
    """토큰 열과 토큰별 프레임 수 (합 = 동작 길이)"""
    T = len(s.motion)
    toks, durs = [BOS], []
    cur = 0
    for i, (g, (a, b)) in enumerate(zip(s.glosses, s.spans)):
        a, b = max(a, cur), max(min(b, T - 1), max(a, cur))
        if i == 0:
            durs.append(a - cur)                  # BOS
        else:
            toks.append(SEP)
            durs.append(a - cur)                  # SEP
        toks.append(vocab.get(g, PAD))
        durs.append(b - a + 1)
        cur = b + 1
    toks.append(EOS)
    durs.append(T - cur)
    durs = [max(int(d), 0) for d in durs]
    diff = T - sum(durs)
    durs[-1] += diff
    return np.asarray(toks, dtype=np.int64), np.asarray(durs, dtype=np.int64)


def build_vocab(samples: Sequence[Sample]) -> Dict[str, int]:
    words = sorted({g for s in samples for g in s.glosses})
    return {w: i + len(SPECIAL) for i, w in enumerate(words)}
