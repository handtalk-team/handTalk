"""
손 모양 보정: 단어 연습 시범에서 사전 설명의 손 모양(엄지-검지 동그라미, 새끼손가락만 펴기, 엄지 세우기 등)이
보이도록, 모델이 만든 동작의 손가락만 그 모양으로 맞춘다. 팔·손목·손바닥 방향은 모델 그대로 둔다.

동작 생성 모델은 손가락 모양을 평균 내 뭉개는 경향이 있어 (예: '죄송'의 동그라미가 사라짐) 이 보정이 필요하다.
손 모양 본(template)은 AI Hub 단어 영상에서 그 모양이 가장 뚜렷한 한 프레임의 손가락 방향만 뽑은 것이다
(server/ml/models/handshapes.json, tools/extract_handshapes.py). 녹화 동작을 재생하는 게 아니라 손가락 방향만 쓴다.

규칙: 단어마다 (모양, 적용 구간). 구간은
  face  손이 얼굴 높이 근처일 때 (예: 죄송은 이마에서만 동그라미, 손등에 내려올 때는 편 손)
  up    손을 들고 있는 동안 내내 (쉬는 자세 제외)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

from app.core.config import local_or

PATH = os.environ.get("HANDTALK_HANDSHAPES") or local_or("handshapes.json", r"C:\handtalk\models\handshapes.json")

RULES: Dict[str, tuple] = {
    "죄송": ("ok", "face"),
    "감기": ("ok", "face"),
    "슬프다": ("ok", "face"),
    "괜찮다": ("pinky", "up"),
    "좋다": ("thumb", "up"),
    "화나다": ("thumb", "up"),
    "싫다": ("l", "up"),
    "병원": ("v", "up"),
    "의사": ("v", "up"),
    "입원": ("v", "up"),
}

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


def palm_frame(h: np.ndarray):
    """손 21점 → (원점, 3x3 축). y: 손목→중지 MCP, x: 새끼→검지 MCP, z: 손바닥 법선"""
    o = h[0]
    y = h[9] - o
    y = y / (np.linalg.norm(y) + 1e-9)
    x = h[5] - h[17]
    x = x - y * np.dot(x, y)
    x = x / (np.linalg.norm(x) + 1e-9)
    z = np.cross(x, y)
    return o, np.stack([x, y, z], axis=1)


def finger_dirs(h: np.ndarray) -> np.ndarray:
    """손가락 마디 방향(손바닥 좌표) (5, 4, 3): 각 손가락의 [부모→1, 1→2, 2→3, 3→4]"""
    _, R = palm_frame(h)
    out = np.zeros((5, 4, 3))
    for f, chain in enumerate(FINGERS):
        prev = 0
        for k, j in enumerate(chain):
            v = h[j] - h[prev]
            out[f, k] = R.T @ (v / (np.linalg.norm(v) + 1e-9))
            prev = j
    return out


class HandShaper:
    def __init__(self, path: str = PATH):
        self.shapes: Dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.shapes = {k: np.asarray(v, dtype=np.float64) for k, v in json.load(f).items()}

    def available(self, gloss: Sequence[str]) -> Optional[tuple]:
        if len(gloss) != 1:
            return None   # 문장 동작은 그대로 (단어 하나 시범에만)
        rule = RULES.get(gloss[0])
        return rule if rule and rule[0] in self.shapes else None

    def apply(self, kp: np.ndarray, joints: Sequence[str], gloss: Sequence[str]) -> np.ndarray:
        rule = self.available(gloss)
        if rule is None:
            return kp
        shape, phase = rule
        T = self.shapes[shape]
        J = {n: i for i, n in enumerate(joints)}
        rh = [J[f"rh_{i}"] for i in range(21)]
        out = kp.astype(np.float64).copy()
        n = len(out)
        w = np.zeros(n)
        for t in range(n):
            h = out[t, rh]
            if np.abs(h).sum() == 0:
                continue
            hz, nz = h[9, 2], out[t, J["nose"], 2]
            if phase == "face":
                w[t] = np.clip((hz - (nz - 0.45)) / 0.2, 0, 1)      # 코보다 0.45 아래부터 서서히, 0.25 아래면 완전히
            else:
                w[t] = np.clip((hz + 0.75) / 0.25, 0, 1)            # 손을 가슴 아래에서 들어 올리면
        # 모양이 갑자기 바뀌지 않게 부드럽게
        k = np.ones(5) / 5
        w = np.convolve(np.pad(w, 2, mode="edge"), k, mode="valid")
        for t in range(n):
            if w[t] <= 0.01:
                continue
            h = out[t, rh]
            o, R = palm_frame(h)
            cur = finger_dirs(h)
            new = h.copy()
            for f, chain in enumerate(FINGERS):
                prev = 0
                base = h[0]
                for kk, j in enumerate(chain):
                    L = np.linalg.norm(h[j] - h[prev])
                    if f > 0 and kk == 0:      # 검지~새끼 MCP 는 손바닥이라 그대로
                        new[j] = h[j]
                    else:
                        d = (1 - w[t]) * cur[f, kk] + w[t] * T[f, kk]
                        d = d / (np.linalg.norm(d) + 1e-9)
                        new[j] = base + (R @ d) * L
                    base = new[j]
                    prev = j
            out[t, rh] = new
        return out.astype(np.float32)
