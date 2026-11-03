"""
손 모양 보정: 아바타가 만드는 모든 수어에서 손가락 모양이 사전 영상의 손 모양과 같도록 손가락만 맞춘다.
팔·손목·손바닥 방향과 움직임은 동작 생성 모델 그대로 둔다.

동작 생성 모델은 손가락 모양을 평균 내 뭉개는 경향이 있다 (단어 2,936개를 실제 영상과 비교하면 손목 위치는 맞는데
손가락 방향이 평균 33° 어긋남. 예: '죄송'의 동그라미가 사라짐). 그래서 두 가지 손 모양 자료로 손가락을 다시 세운다.

1) 손 모양 사전 (server/ml/models/handshape_lexicon.npz, tools/extract_handshapes.py --lexicon)
   모든 단어. 단어 영상에서 손을 든 구간을 손 모양이 바뀌는 곳에서 최대 3구간으로 나눈 구간별 손가락 방향 하나와
   그 구간의 상대 시간(0~1)만 있다. 생성한 동작에서도 그 단어 동안 손을 든 구간을 같은 비율로 나눠 맞춘다.
   문장에서는 단어마다 모델이 정한 길이(글로스 구간)를 따라 적용한다.
2) 사전 설명과 맞춘 규칙 (RULES, handshapes.json) — 단어 연습 시범에서 우선.
   국립국어원 한국수어사전 설명과 대조해 영상 키포인트가 흐린 단어(예: '좋다'는 주먹인데 영상에선 검지가 펴져 보임)를 바로잡는다.
     face  손이 얼굴 높이 근처일 때 (예: 죄송은 이마에서만 동그라미, 손등에 내려올 때는 편 손)
     up    손을 들고 있는 동안 내내 (쉬는 자세 제외)

녹화 동작을 재생하는 게 아니라 손가락 방향(손 모양)만 쓴다.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.config import local_or

PATH = os.environ.get("HANDTALK_HANDSHAPES") or local_or("handshapes.json", r"C:\handtalk\models\handshapes.json")
LEXICON = os.environ.get("HANDTALK_HANDSHAPE_LEXICON") or local_or(
    "handshape_lexicon.npz", r"C:\handtalk\models\handshape_lexicon.npz")
SENTENCE_STRENGTH = float(os.environ.get("HANDTALK_HANDSHAPE_SENTENCE", "0.8"))

RULES: Dict[str, tuple] = {
    "죄송": ("ok", "face"),
    "슬프다": ("ok", "face"),
    "괜찮다": ("pinky", "up"),
    "좋다": ("fist", "up"),       # 국립국어원 사전: 오른 주먹을 코에 댄다
    "화나다": ("thumb", "up"),
    "싫다": ("l", "up"),
    "병원": ("v", "up"),
    "의사": ("v", "up"),
    "입원": ("v", "up"),
}

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
HANDS = ("rh", "lh")
FADE = 0.08   # 사전 구간 경계에서 앞뒤 손 모양을 섞는 폭 (손 든 시간 비율)


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


def _smooth(w: np.ndarray) -> np.ndarray:
    k = np.ones(5) / 5
    return np.convolve(np.pad(w, 2, mode="edge"), k, mode="valid")


def _reshape(h: np.ndarray, T: np.ndarray, w: float) -> np.ndarray:
    """손 21점의 손가락을 템플릿 방향 T (5,4,3) 쪽으로 w 만큼. 마디 길이·손바닥(검지~새끼 MCP)은 그대로."""
    _, R = palm_frame(h)
    cur = finger_dirs(h)
    new = h.copy()
    for f, chain in enumerate(FINGERS):
        prev = 0
        base = h[0]
        for kk, j in enumerate(chain):
            L = np.linalg.norm(h[j] - h[prev])
            if f > 0 and kk == 0:
                new[j] = h[j]
            else:
                d = (1 - w) * cur[f, kk] + w * T[f, kk]
                d = d / (np.linalg.norm(d) + 1e-9)
                new[j] = base + (R @ d) * L
            base = new[j]
            prev = j
    return new


class HandShaper:
    def __init__(self, path: str = PATH, lexicon: str = LEXICON):
        self.shapes: Dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.shapes = {k: np.asarray(v, dtype=np.float64) for k, v in json.load(f).items()}
        # 단어 → 손(0 오른손, 1 왼손) → [(t0, t1, 방향)]
        self.lexicon: Dict[str, Dict[int, List[Tuple[float, float, np.ndarray]]]] = {}
        if os.path.exists(lexicon):
            z = np.load(lexicon)
            names = [str(n) for n in z["glosses"]]
            dirs = z["dirs"].astype(np.float64)
            for g, h, a, b, d in zip(z["gloss"], z["hand"], z["t0"], z["t1"], dirs):
                self.lexicon.setdefault(names[g], {}).setdefault(int(h), []).append((float(a), float(b), d))

    def available(self, gloss: Sequence[str]) -> Optional[tuple]:
        if len(gloss) != 1:
            return None   # 사전 설명 규칙은 단어 하나 시범에만
        rule = RULES.get(gloss[0])
        return rule if rule and rule[0] in self.shapes else None

    def apply(self, kp: np.ndarray, joints: Sequence[str], gloss: Sequence[str],
              spans: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
        """spans: 글로스마다 (시작, 끝) 프레임. 없으면 전체가 단어 하나."""
        J = {n: i for i, n in enumerate(joints)}
        out = kp.astype(np.float64).copy()
        if len(gloss) == 1:
            spans = [(0, len(out))]   # 모델은 시작 토큰 동안 이미 손을 올리므로 단어 하나는 전체
        elif spans is None or len(spans) != len(gloss):
            spans = []
        else:   # 단어 사이 전환 구간은 반씩 나눠 앞뒤 단어에 붙인다 (손 드는 동작이 글로스 구간보다 먼저 시작)
            cuts = [0] + [(spans[i][1] + spans[i + 1][0]) // 2 for i in range(len(spans) - 1)] + [len(out)]
            spans = list(zip(cuts[:-1], cuts[1:]))
        strength = 1.0 if len(gloss) == 1 else SENTENCE_STRENGTH
        for g, (s0, s1) in zip(gloss, spans):
            entry = self.lexicon.get(g)
            if entry:
                self._apply_lexicon(out, J, entry, s0, s1, strength)
        rule = self.available(gloss)
        if rule is not None:
            self._apply_rule(out, J, rule)
        return out.astype(np.float32)

    def _apply_lexicon(self, out, J, entry, s0, s1, strength):
        nz = out[:, J["nose"], 2]
        for hi, segs in entry.items():
            idx = [J[f"{HANDS[hi]}_{i}"] for i in range(21)]
            H = out[:, idx]
            present = np.abs(H).sum((1, 2)) > 0
            up = present & (H[:, 9, 2] - nz > -1.0)
            # 단어 구간 안에서 손을 든 구간 (앞뒤 몇 프레임 여유: 손이 올라오는 동안에도 모양이 잡히게)
            frames = np.where(up[s0:s1])[0] + s0
            if len(frames) < 2:
                continue
            a, b = frames[0], frames[-1] + 1
            raise_w = _smooth(np.clip((H[:, 9, 2] - nz + 1.15) / 0.25, 0, 1) * present)
            lo, hi_ = max(0, a - 4), min(len(out), b + 4)
            for t in range(lo, hi_):
                if not present[t] or raise_w[t] <= 0.01:
                    continue
                u = float(np.clip((t - a + 0.5) / (b - a), 0, 1))
                T = self._template_at(segs, u)
                out[t, idx] = _reshape(out[t, idx], T, raise_w[t] * strength)

    @staticmethod
    def _template_at(segs, u):
        """상대 시간 u 의 손 모양. 구간 경계 근처는 앞뒤 구간을 섞는다."""
        ws, ds = [], []
        for a, b, d in segs:
            dist = max(a - u, u - b, 0.0)
            w = max(0.0, 1.0 - dist / FADE) if dist > 0 else 1.0
            if w > 0:
                ws.append(w)
                ds.append(d)
        if not ws:
            return min(segs, key=lambda s: min(abs(s[0] - u), abs(s[1] - u)))[2]
        T = sum(w * d for w, d in zip(ws, ds)) / sum(ws)
        return T / (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-9)

    def _apply_rule(self, out, J, rule):
        shape, phase = rule
        T = self.shapes[shape]
        rh = [J[f"rh_{i}"] for i in range(21)]
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
        w = _smooth(w)   # 모양이 갑자기 바뀌지 않게
        for t in range(n):
            if w[t] > 0.01:
                out[t, rh] = _reshape(out[t, rh], T, w[t])
