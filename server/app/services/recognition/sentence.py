"""
연속 수어(문장) 인식 — AI Hub 문장 데이터로 학습한 Transformer + CTC 모델 (ml/models/sentence_recognizer.onnx)

단어 인식은 2초 창 하나에 단어 하나를 고르지만, 이 모델은 수어 한 문장을 통째로 받아 글로스 순서를 낸다.
카메라 프레임마다 단어 모델과 같은 136 피처(손 2개 정규화 좌표 + flex)를 모으다가, 손을 내리거나 멈추면
그 구간을 15fps 로 줄이고 속도를 붙여(272) 모델에 넣고 CTC greedy 로 읽는다.

모델 파일은 AI Hub 데이터로 학습한 것이라 저장소에 두지 않는다 (server/ml/models/, .gitignore).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MODEL_DIR = os.path.dirname(settings.ONNX_MODEL_PATH)


class SentenceRecognizer:
    START_MOTION = 0.010      # 손 피처 변화량이 이보다 크면 수어 중
    END_STILL_S = 0.9         # 이만큼 손이 멈추거나 사라지면 문장 끝
    MIN_FRAMES = 20           # 30fps 기준 최소 길이 (짧은 잡음 거름)
    MAX_FRAMES = 470          # 모델 시간축 240 (15fps) 이내

    def __init__(self) -> None:
        self.session = None
        self.glosses: List[str] = []
        path = os.path.join(MODEL_DIR, "sentence_recognizer.onnx")
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
                with open(os.path.join(MODEL_DIR, "glosses.json"), encoding="utf-8") as f:
                    self.glosses = [str(g).strip() for g in json.load(f)]
                logger.info("문장 인식 모델: %d 글로스", len(self.glosses) - 1)
            except Exception as e:
                logger.warning("문장 인식 모델을 못 읽음 (%s)", e)
                self.session = None
        self.vocab = set(g for g in self.glosses[1:] if g)
        self.reset()

    @property
    def available(self) -> bool:
        return self.session is not None

    def reset(self) -> None:
        self.frames: List[np.ndarray] = []
        self.times: List[float] = []
        self.active = False
        self.last_motion = 0.0
        self.prev: Optional[np.ndarray] = None

    def push(self, features: np.ndarray, hands_visible: bool, timestamp: Optional[float] = None,
             allowed: Optional[set] = None) -> Optional[List[str]]:
        """프레임 하나(136 피처). 문장이 끝났다고 보이면 인식한 글로스 목록을 돌려준다."""
        if not self.available:
            return None
        now = time.time()
        ts = timestamp if timestamp is not None else now
        f = np.asarray(features, dtype=np.float32)
        motion = float(np.abs(f - self.prev).mean()) if self.prev is not None else 0.0
        self.prev = f
        if hands_visible and motion > self.START_MOTION:
            self.last_motion = now
            if not self.active:
                self.active = True
                self.frames = []
                self.times = []
        if not self.active:
            return None
        self.frames.append(f)
        self.times.append(ts)
        ended = now - self.last_motion > self.END_STILL_S or len(self.frames) >= self.MAX_FRAMES
        if not ended:
            return None
        frames, times, self.frames, self.active = self.frames, self.times, [], False
        if len(frames) < self.MIN_FRAMES:
            return None
        return self.decode(np.stack(frames), np.asarray(times, dtype=np.float64), allowed)

    def decode(self, frames: np.ndarray, times: Optional[np.ndarray] = None, allowed: Optional[set] = None) -> List[str]:
        """frames: (T, 136) → 글로스 순서. times 가 있으면 실제 프레임 시각으로 15fps 에 맞춘다 (없으면 30fps 로 보고 2개에 1개).
        allowed 가 있으면 그 단어들만 고른다 (상황 문장: 그 상황의 단어로 좁혀 웹캠 오인식을 줄임)."""
        if times is not None and len(times) == len(frames) and times[-1] > times[0]:
            t15 = np.arange(times[0], times[-1], 1 / 15.0)
            idx = np.clip(np.searchsorted(times, t15), 1, len(times) - 1)
            a, b = times[idx - 1], times[idx]
            w = ((t15 - a) / np.maximum(b - a, 1e-6))[:, None].astype(np.float32)
            pos = frames[idx - 1] * (1 - w) + frames[idx] * w
        else:
            pos = frames[::2]                                 # 15fps
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        x = np.concatenate([pos, vel], axis=1)[None].astype(np.float32)
        logits = self.session.run(None, {"input": x})[0][0]
        if allowed:
            keep = np.array([i == 0 or self.glosses[i] in allowed for i in range(logits.shape[-1])])
            logits = np.where(keep[None, :], logits, -1e9)
        best = logits.argmax(-1)
        out, prev = [], -1
        for i in best:
            if i != prev and i != 0:
                g = self.glosses[i] if i < len(self.glosses) else ""
                if g:
                    out.append(g)
            prev = i
        logger.info("sentence: %s%s", " ".join(out) or "(없음)", f"  (후보 {len(allowed)}개로 좁힘)" if allowed else "")
        return out
