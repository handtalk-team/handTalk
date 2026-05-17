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

SIGN_VOCAB: Dict[int, str] = {
    0: "안녕하세요",
    1: "감사합니다",
    2: "아프다",
    3: "약",
    4: "의사",
    5: "간호사",
    6: "도와주세요",
    7: "화장실",
    8: "물",
    9: "괜찮아요",
    10: "좋아요",
    11: "브이",
}

NUM_CLASSES = 10  # ONNX model trained on 10 classes (0-9)


class _GestureState(Enum):
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    COOLDOWN = "cooldown"


# MediaPipe hand landmark indices
_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
_PIPS = [3, 6, 10, 14, 18]   # IP/PIP joints (one below each tip)


class HybridRecognitionEngine:
    """One instance per WebSocket session. Not thread-safe."""

    IDLE_TIMEOUT_S: float = 2.0
    COOLDOWN_S: float = 1.5
    # Motion thresholds differ by modality (different physical units)
    MOTION_THRESHOLD_GYRO: float = 0.02    # rad/s  — glove gyro norm
    MOTION_THRESHOLD_VISION: float = 0.006  # normalized units/frame — landmark velocity

    def __init__(self) -> None:
        self._fusion = SensorFusionModule()
        self._onnx_session = self._load_onnx()
        self._state = _GestureState.IDLE
        self._last_motion_t = time.time()
        self._cooldown_start: Optional[float] = None
        self._prev_vision: Optional[np.ndarray] = None  # for velocity-based motion

    # ─────────────────────── public API ─────────────────────────

    async def process_frame(self, frame) -> Optional[RecognitionResult]:
        fused = self._fusion.push_frame(frame)
        return await self._run_state_machine(fused)

    def reset_session(self) -> None:
        self._fusion.reset()
        self._state = _GestureState.IDLE
        self._last_motion_t = time.time()
        self._prev_vision = None

    # ─────────────────────── state machine ──────────────────────

    async def _run_state_machine(self, ff: FusedFrame) -> Optional[RecognitionResult]:
        now = time.time()

        if self._state == _GestureState.COOLDOWN:
            if now - self._cooldown_start < self.COOLDOWN_S:
                return None
            self._state = _GestureState.IDLE

        motion_energy = self._compute_motion_energy(ff)
        threshold = (
            self.MOTION_THRESHOLD_GYRO
            if self._glove_present(ff)
            else self.MOTION_THRESHOLD_VISION
        )

        if self._state == _GestureState.IDLE:
            if motion_energy > threshold:
                self._state = _GestureState.ACCUMULATING
                self._last_motion_t = now
            return None

        # ── ACCUMULATING ─────────────────────────────────────────
        if motion_energy > threshold:
            self._last_motion_t = now

        if not self._fusion.window_full:
            return None

        idle_for = now - self._last_motion_t
        if idle_for < 0.3:
            return self._infer(partial=True, ff=ff)

        # Hand still for > 300 ms → commit gesture
        result = self._infer(partial=False, ff=ff)
        self._state = _GestureState.COOLDOWN
        self._cooldown_start = now
        self._fusion.reset()
        return result

    # ─────────────────────── inference ──────────────────────────

    def _infer(self, partial: bool, ff: FusedFrame) -> RecognitionResult:
        window = self._fusion.get_window_array()

        if window is not None and self._onnx_session is not None:
            label, confidence = self._onnx_infer(window)
        elif window is not None and self._glove_present(ff):
            label, confidence = self._glove_rule_infer(window)
        elif window is not None:
            label, confidence = self._vision_infer(window)
        else:
            label, confidence = "인식 중", 0.0

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

    def _vision_infer(self, window: np.ndarray) -> Tuple[str, float]:
        """
        Gesture recognition from MediaPipe landmark geometry alone.

        Coordinate system (wrist-normalised, MediaPipe image space):
          x  right →  positive
          y  down  →  positive  (so y < 0 means ABOVE wrist)
          z  depth →  negative toward camera

        A finger is EXTENDED when tip_y < pip_y
        (tip is higher in the frame than the proximal joint).
        """
        # Average landmark positions over the window
        lm = window[:, :63].mean(axis=0).reshape(21, 3)  # (21, 3)

        # Per-finger extension flags [thumb, index, middle, ring, pinky]
        extended = [lm[t, 1] < lm[p, 1] for t, p in zip(_TIPS, _PIPS)]
        thumb, index, middle, ring, pinky = extended
        n_fingers = sum(extended[1:])  # non-thumb extended count

        # ── 좋아요 (thumbs up) ────────────────────────────────────
        # Thumb clearly extended upward, other 4 fingers closed
        if thumb and not index and not middle and not ring and not pinky:
            # Extra check: thumb tip is well above its MCP joint
            if lm[4, 1] < lm[2, 1] - 0.04:
                return "좋아요", 0.78

        # ── 브이 (V / peace sign) ─────────────────────────────────
        # Index + middle extended, ring + pinky closed
        if index and middle and not ring and not pinky:
            return "브이", 0.76

        # ── 안녕하세요 (open hand / wave) ────────────────────────
        if n_fingers >= 3:
            return "안녕하세요", 0.72

        # ── 아프다 (fist) ─────────────────────────────────────────
        if not any(extended):
            return "아프다", 0.68

        # ── 도와주세요 (default) ──────────────────────────────────
        return "도와주세요", 0.55

    def _glove_rule_infer(self, window: np.ndarray) -> Tuple[str, float]:
        """Flex-sensor heuristic when glove is connected but ONNX absent."""
        flex = window[:, 63:68].mean(axis=0)
        total_bend = float(flex.sum())

        if total_bend < 0.3:
            return "안녕하세요", 0.72
        if total_bend > 4.0:
            return "아프다", 0.68
        if flex[1] < 0.2 and flex[2] > 0.7:
            return "의사", 0.65
        if flex[0] < 0.2 and total_bend > 3.5:
            return "감사합니다", 0.70
        if total_bend < 1.5:
            return "괜찮아요", 0.66
        return "도와주세요", 0.55

    def _onnx_infer(self, window: np.ndarray) -> Tuple[str, float]:
        inp = window[np.newaxis].astype(np.float32)
        try:
            logits = self._onnx_session.run(None, {"input": inp})[0][0]
            probs = self._softmax(logits)
            idx = int(np.argmax(probs))
            return SIGN_VOCAB.get(idx, "?"), float(probs[idx])
        except Exception as e:
            logger.warning("ONNX inference failed: %s — falling back", e)
            return self._vision_infer(window)

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
