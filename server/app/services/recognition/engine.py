"""
Hybrid Sign Language Recognition Engine
========================================

Input  : sliding window array  (T, 77) float32
Output : (sign_label, confidence, modality_weights)

Inference priority
------------------
1. ONNX model  — when ml/models/sign_recognizer.onnx exists
2. Rule-based  — simple heuristic on flex sensor means (for early testing)
3. Cycle mock  — deterministic, always works; used during development

The engine is stateful: it holds a gesture state-machine that debounces
rapid re-triggers and decides when a gesture is "complete" before passing
the window to the model.

Sign vocabulary (MVP — 10 hospital-scenario words)
--------------------------------------------------
0  안녕하세요  hello / greeting
1  감사합니다  thank you
2  아프다      it hurts / pain
3  약           medicine
4  의사         doctor
5  간호사      nurse
6  도와주세요  please help me
7  화장실      toilet / restroom
8  물           water
9  괜찮아요   I'm okay
"""

from __future__ import annotations

import asyncio
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
}

NUM_CLASSES = len(SIGN_VOCAB)


class _GestureState(Enum):
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    COOLDOWN = "cooldown"


class HybridRecognitionEngine:
    """
    One instance per WebSocket session.
    Not thread-safe — call from a single asyncio task.
    """

    # Seconds of inactivity before the gesture window resets
    IDLE_TIMEOUT_S: float = 2.0
    # Minimum seconds between consecutive recognition events
    COOLDOWN_S: float = 1.0
    # Motion-energy threshold to start accumulating (avoid idle noise)
    MOTION_THRESHOLD: float = 0.02

    def __init__(self) -> None:
        self._fusion = SensorFusionModule()
        self._onnx_session = self._load_onnx()
        self._state = _GestureState.IDLE
        self._last_motion_t = time.time()
        self._cooldown_start: Optional[float] = None
        self._mock_cycle = 0    # for the cycle-mock fallback

    # ─────────────────────── public API ─────────────────────────

    async def process_frame(
        self, frame
    ) -> Optional[RecognitionResult]:
        """
        Feed one SensorFrame into the pipeline.
        Returns a RecognitionResult when a complete gesture is detected,
        None otherwise.
        """
        fused = self._fusion.push_frame(frame)
        return await self._run_state_machine(fused)

    def reset_session(self) -> None:
        self._fusion.reset()
        self._state = _GestureState.IDLE
        self._last_motion_t = time.time()

    # ─────────────────────── state machine ──────────────────────

    async def _run_state_machine(
        self, ff: FusedFrame
    ) -> Optional[RecognitionResult]:
        now = time.time()

        if self._state == _GestureState.COOLDOWN:
            if now - self._cooldown_start < self.COOLDOWN_S:
                return None
            self._state = _GestureState.IDLE

        motion_energy = self._compute_motion_energy(ff)

        if self._state == _GestureState.IDLE:
            if motion_energy > self.MOTION_THRESHOLD:
                self._state = _GestureState.ACCUMULATING
                self._last_motion_t = now
            return None

        # ACCUMULATING
        if motion_energy > self.MOTION_THRESHOLD:
            self._last_motion_t = now

        if not self._fusion.window_full:
            return None

        idle_for = now - self._last_motion_t
        if idle_for < 0.3:
            # Still moving — emit partial result so the UI can show feedback
            return self._infer(partial=True, ff=ff)

        # Hand has been still for > 300 ms → commit the gesture
        result = self._infer(partial=False, ff=ff)
        self._state = _GestureState.COOLDOWN
        self._cooldown_start = now
        self._fusion.reset()
        return result

    # ─────────────────────── inference ──────────────────────────

    def _infer(self, partial: bool, ff: FusedFrame) -> RecognitionResult:
        window = self._fusion.get_window_array()
        conf_summary = self._fusion.get_confidence_summary()

        if window is not None and self._onnx_session is not None:
            label, confidence = self._onnx_infer(window)
        elif window is not None:
            label, confidence = self._rule_based_infer(window)
        else:
            label, confidence = self._cycle_mock()

        total = ff.vision_weight + ff.glove_weight
        vw = ff.vision_weight / (total + 1e-9)
        gw = ff.glove_weight / (total + 1e-9)

        return RecognitionResult(
            text=label,
            confidence=confidence,
            modality_weights=ModalityWeights(vision=round(vw, 3), glove=round(gw, 3)),
            is_partial=partial,
        )

    def _onnx_infer(self, window: np.ndarray) -> Tuple[str, float]:
        """
        Run the BiGRU+Attention ONNX model.

        Input tensor : (1, T, 77)   — batch=1, time, features
        Output tensor: (1, 10)      — class logits (softmax applied here)
        """
        inp = window[np.newaxis].astype(np.float32)        # (1, T, 77)
        try:
            logits = self._onnx_session.run(
                None, {"input": inp}
            )[0][0]                                          # (10,)
            probs = self._softmax(logits)
            idx = int(np.argmax(probs))
            return SIGN_VOCAB.get(idx, "?"), float(probs[idx])
        except Exception as e:
            logger.warning("ONNX inference failed: %s — falling back", e)
            return self._rule_based_infer(window)

    def _rule_based_infer(self, window: np.ndarray) -> Tuple[str, float]:
        """
        Heuristic fallback based on mean flex sensor values.
        Works well enough to validate the pipeline before the model is trained.

        Glove features are at indices 63..67 in the 77-D fused vector
        (positions 0..62 are vision landmarks).
        """
        flex = window[:, 63:68].mean(axis=0)    # (5,) mean flex over time
        total_bend = float(flex.sum())

        # Very rough mapping — replace with trained model ASAP
        if total_bend < 0.3:
            return "안녕하세요", 0.72        # open hand → greeting
        if total_bend > 4.0:
            return "아프다", 0.68            # fist → pain
        if flex[1] < 0.2 and flex[2] > 0.7:
            return "의사", 0.65             # point index → doctor
        if flex[0] < 0.2 and total_bend > 3.5:
            return "감사합니다", 0.70       # thumb up → thank you
        if total_bend < 1.5:
            return "괜찮아요", 0.66         # mostly open → okay
        return "도와주세요", 0.55           # default

    def _cycle_mock(self) -> Tuple[str, float]:
        """Cycles deterministically through vocabulary — dev/demo only."""
        label = SIGN_VOCAB[self._mock_cycle % NUM_CLASSES]
        self._mock_cycle += 1
        return label, 0.80

    # ─────────────────────── utilities ──────────────────────────

    @staticmethod
    def _compute_motion_energy(ff: FusedFrame) -> float:
        """
        Simple proxy for motion: L2 norm of the first 3 IMU gyro values
        (indices 71..73 in the 77-D vector).
        """
        gyro = ff.fused_features[71:74]
        return float(np.linalg.norm(gyro))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _load_onnx():
        path = settings.ONNX_MODEL_PATH
        if not os.path.exists(path):
            logger.info(
                "ONNX model not found at '%s'. "
                "Using rule-based fallback until model is trained.",
                path,
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
