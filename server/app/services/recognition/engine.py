"""
Hybrid Sign Language Recognition Engine
========================================

Inference priority
------------------
1. ONNX model  — when ml/models/sign_recognizer.onnx exists (trained)
2. Vision-based heuristic  — landmark geometry (no glove required)
3. Glove+Vision rule-based — flex sensor fallback (when glove is connected)

Sign vocabulary (MVP — hospital scenario)
-----------------------------------------
0  안녕하세요  hello / greeting
1  감사합니다  thank you
2  아프다      it hurts / pain
3  약           medicine
4  의사         doctor
5  간호사      nurse
6  도와주세요  please help me
7  화장실      toilet
8  물           water
9  괜찮아요   I'm okay
10 좋아요     thumbs up
11 브이        V / peace sign
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.models.schemas.ws_messages import ModalityWeights, RecognitionResult
from app.services.sync.fusion import FusedFrame, SensorFusionModule

logger = logging.getLogger(__name__)
settings = get_settings()

def _load_vocab() -> Dict[int, str]:
    labels_path = os.path.join(os.path.dirname(settings.ONNX_MODEL_PATH), "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            labels = [l.strip() for l in f if l.strip()]
        logger.info("SIGN_VOCAB loaded from %s: %s", labels_path, labels)
        return {i: l for i, l in enumerate(labels)}
    return {0: "안녕하세요", 1: "좋아요", 2: "브이"}

SIGN_VOCAB: Dict[int, str] = _load_vocab()
NUM_CLASSES = len(SIGN_VOCAB)


class _GestureState(Enum):
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    COOLDOWN = "cooldown"


# MediaPipe hand landmark indices
_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
_PIPS = [3, 6, 10, 14, 18]   # IP/PIP joints (one below each tip)


class HybridRecognitionEngine:
    """One instance per WebSocket session. Not thread-safe."""

    COOLDOWN_S: float = 2.5              # 인식 후 재인식 방지 대기 시간
    MIN_MOTION_FRAMES: int = 10         # 최소 이만큼 움직임이 있어야 진짜 수어 (노이즈 거름)
    MOTION_THRESHOLD_GYRO: float = 0.02
    MOTION_THRESHOLD_VISION: float = 0.010

    def __init__(self) -> None:
        self._fusion = SensorFusionModule()
        self._onnx_session = self._load_onnx()
        self._state = _GestureState.IDLE
        self._last_motion_t = time.time()
        self._accumulate_start: Optional[float] = None
        self._cooldown_start: Optional[float] = None
        self._prev_vision: Optional[np.ndarray] = None
        self._motion_frame_count: int = 0
        self._last_window: Optional[np.ndarray] = None  # 마지막 분류에 쓰인 윈도우

    # ─────────────────────── public API ─────────────────────────

    async def process_frame(self, frame) -> Optional[RecognitionResult]:
        fused = self._fusion.push_frame(frame)
        return await self._run_state_machine(fused)

    def reset_session(self) -> None:
        self._fusion.reset()
        self._state = _GestureState.IDLE
        self._last_motion_t = time.time()
        self._accumulate_start = None
        self._prev_vision = None
        self._motion_frame_count = 0

    # ─────────────────────── state machine ──────────────────────

    async def _run_state_machine(self, ff: FusedFrame) -> Optional[RecognitionResult]:
        now = time.time()

        if self._state == _GestureState.COOLDOWN:
            if now - self._cooldown_start < self.COOLDOWN_S:
                return None
            self._fusion.reset()
            self._prev_vision = None
            self._motion_frame_count = 0
            self._state = _GestureState.IDLE

        motion_energy = self._compute_motion_energy(ff)
        threshold = (
            self.MOTION_THRESHOLD_GYRO
            if self._glove_present(ff)
            else self.MOTION_THRESHOLD_VISION
        )

        if motion_energy > threshold:
            self._motion_frame_count += 1
            self._last_motion_t = now  # 마지막 움직임 시각 갱신

        if self._state == _GestureState.IDLE:
            if motion_energy > threshold:
                self._fusion.reset()
                self._motion_frame_count = 1
                self._last_motion_t = now
                self._accumulate_start = now
                self._state = _GestureState.ACCUMULATING
            return None

        # ── ACCUMULATING ─────────────────────────────────────────
        # 60프레임(30fps × 2초)이 채워지면 분류. 조기 종료 없음.
        win_len = len(self._fusion._window)
        if win_len >= settings.WINDOW_SIZE:
            if self._motion_frame_count >= self.MIN_MOTION_FRAMES:
                logger.debug("classify(2s-window): frames=%d motion=%d",
                             win_len, self._motion_frame_count)
                return self._finalize(ff, now)
            # 움직임 부족 → 버리고 재시작
            self._fusion.reset()
            self._motion_frame_count = 0
            self._accumulate_start = None
            self._state = _GestureState.IDLE

        return None

    def _finalize(self, ff: FusedFrame, now: float) -> Optional[RecognitionResult]:
        self._last_window = self._fusion.get_window_array()  # 분류 전에 보존
        result = self._infer(partial=False, ff=ff)
        self._state = _GestureState.COOLDOWN
        self._cooldown_start = now
        self._motion_frame_count = 0
        self._accumulate_start = None
        self._fusion.reset()
        return result

    # ─────────────────────── inference ──────────────────────────

    def _infer(self, partial: bool, ff: FusedFrame) -> Optional[RecognitionResult]:
        window = self._fusion.get_window_array()

        result: Optional[Tuple[str, float]] = None
        if window is not None and self._onnx_session is not None:
            result = self._onnx_infer(window)
        # 룰 기반 폴백 비활성화 — ONNX 모델 없으면 추론 안 함

        if result is None:
            return None

        label, confidence = result
        total = ff.vision_weight + ff.glove_weight + 1e-9
        return RecognitionResult(
            text=label,
            confidence=confidence,
            modality_weights=ModalityWeights(
                vision=round(ff.vision_weight / total, 3),
                glove=round(ff.glove_weight / total, 3),
            ),
            is_partial=partial,
        )

    def _vision_infer(self, window: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Gesture recognition from MediaPipe landmark geometry alone.

        Coordinate system (wrist-normalised, MediaPipe image space):
          y increases downward → extended finger has tip_y < pip_y

        Returns None when the gesture doesn't match a known sign.
        """
        lm = window[:, :63].mean(axis=0).reshape(21, 3)

        extended = [lm[t, 1] < lm[p, 1] for t, p in zip(_TIPS, _PIPS)]
        thumb, index, middle, ring, pinky = extended

        # ── 좋아요 (thumbs up) ────────────────────────────────────
        if thumb and not index and not middle and not ring and not pinky:
            if lm[4, 1] < lm[2, 1] - 0.04:
                return "좋아요", 0.78

        # ── 브이 (V / peace sign) ─────────────────────────────────
        if index and middle and not ring and not pinky:
            return "브이", 0.76

        # ── 안녕하세요 (open hand) ────────────────────────────────
        if sum(extended[1:]) >= 3:
            return "안녕하세요", 0.72

        return None  # 인식되지 않은 손 모양 → 결과 없음

    def _glove_rule_infer(self, window: np.ndarray) -> Optional[Tuple[str, float]]:
        """Flex-sensor heuristic when glove is connected but ONNX absent."""
        flex = window[:, 63:68].mean(axis=0)
        total_bend = float(flex.sum())

        if total_bend < 0.3:
            return "안녕하세요", 0.72
        if flex[0] < 0.2 and total_bend > 3.5:
            return "좋아요", 0.70
        if flex[1] < 0.2 and flex[2] < 0.2 and total_bend > 2.5:
            return "브이", 0.68
        return None

    def _onnx_infer(self, window: np.ndarray) -> Tuple[str, float]:
        # 모델은 앞 63차원(vision landmarks)만 사용, SEQ_LEN=60으로 패딩/자르기
        seq = window[:, :63]
        if len(seq) < 60:
            pad = np.zeros((60 - len(seq), 63), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:60]
        inp = seq[np.newaxis].astype(np.float32)
        try:
            logits = self._onnx_session.run(None, {"input": inp})[0][0]
            probs = self._softmax(logits)
            idx = int(np.argmax(probs))
            return SIGN_VOCAB.get(idx, "?"), float(probs[idx])
        except Exception as e:
            logger.warning("ONNX inference failed: %s", e)
            return None

    # ─────────────────────── utilities ──────────────────────────

    def _compute_motion_energy(self, ff: FusedFrame) -> float:
        """
        Returns a motion proxy scalar.
        - Glove present : L2 norm of gyro channels (indices 71-73)
        - Vision only   : frame-to-frame velocity of landmark centroid
        """
        gyro = ff.fused_features[71:74]
        gyro_norm = float(np.linalg.norm(gyro))
        if gyro_norm > 1e-4:
            return gyro_norm

        # Vision velocity: displacement of all landmarks between frames
        vision = ff.fused_features[:63]
        if self._prev_vision is not None:
            delta = float(np.linalg.norm(vision - self._prev_vision))
            self._prev_vision = vision.copy()
            return delta
        self._prev_vision = vision.copy()
        return 0.0

    @staticmethod
    def _glove_present(ff: FusedFrame) -> bool:
        return ff.glove_confidence > 0.01

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _load_onnx():
        path = settings.ONNX_MODEL_PATH
        if not os.path.exists(path):
            logger.info(
                "ONNX model not found at '%s'. Using heuristic fallback.", path
            )
            return None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            sess = ort.InferenceSession(path, sess_options=opts)
            logger.info("ONNX model loaded from '%s'", path)
            return sess
        except Exception as e:
            logger.error("Failed to load ONNX model: %s", e)
            return None
