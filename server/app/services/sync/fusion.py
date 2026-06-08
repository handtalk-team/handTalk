"""
Sensor Fusion & Temporal Synchronisation Module
================================================

Problem
-------
Camera runs at ~30 fps (33 ms intervals).
BLE glove runs at ~50 Hz (20 ms intervals).
They are independent clocks — we must align them onto a single time axis
before feeding the recognition model.

Approach
--------
1.  Every SensorFrame carries a Unix timestamp assigned by the client.
2.  The fusion module maintains a sorted deque of VisionSnapshot and
    GloveSnapshot objects indexed by timestamp.
3.  For each new camera frame we find the closest glove packet (within
    MAX_GLOVE_LAG_S seconds) and pair them.  If no glove packet is
    available we use the last known glove state (hold-last-value).
4.  Feature-level fusion: vision → 126-D vector (양손), glove → 14-D vector,
    concatenated → 140-D fused vector.
5.  Dynamic weighting: confidence scores gate each modality's contribution.
    If vision confidence < VISION_MIN we zero-pad vision features so the
    model learns not to rely on them.  Same for glove.
6.  The fused frames are pushed into a fixed-size sliding window.
    Once full, the window numpy array is handed to the recogniser.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.models.schemas.sensor import GloveData, SensorFrame, VisionData

settings = get_settings()

VISION_DIM_PER_HAND = 63   # 21 landmarks × 3 (x, y, z)
VISION_DIM = 126           # 양손: 63 × 2
FLEX_DIM   = 5             # per hand
FUSED_DIM  = VISION_DIM + FLEX_DIM * 2   # 136

# MediaPipe hand landmark finger chains (for flex synthesis)
_FINGER_CHAINS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]


@dataclass
class FusedFrame:
    timestamp: float
    vision_features: np.ndarray      # shape (VISION_DIM,)   = 126
    flex_features: np.ndarray        # shape (FLEX_DIM*2,)   = 10
    fused_features: np.ndarray       # shape (FUSED_DIM,)    = 136
    vision_weight: float             # 0.0 – 1.0
    glove_weight: float              # 0.0 – 1.0
    vision_confidence: float
    glove_confidence: float
    glove_is_mock: bool = False



class SensorFusionModule:
    """
    Thread-safe sensor fusion for one WebSocket session.
    One instance per connected client.
    """

    def __init__(self) -> None:
        self._window: Deque[FusedFrame] = deque(maxlen=settings.WINDOW_SIZE)

    # ─────────────────────── public API ─────────────────────────

    def push_frame(self, frame: SensorFrame) -> FusedFrame:
        """
        Ingest one SensorFrame, fuse it, append to the sliding window,
        and return the resulting FusedFrame.
        """
        vision_feat, v_conf = self._extract_vision(frame.camera, frame.camera_left)
        flex_feat, g_conf, is_mock = self._extract_flex(
            frame.glove, frame.glove_left, frame.camera, frame.camera_left
        )

        v_w, g_w = self._compute_weights(v_conf, g_conf)

        if v_conf < settings.VISION_CONFIDENCE_MIN:
            vision_feat = np.zeros(VISION_DIM, dtype=np.float32)

        fused = np.concatenate([vision_feat, flex_feat]).astype(np.float32)

        ff = FusedFrame(
            timestamp=frame.timestamp,
            vision_features=vision_feat,
            flex_features=flex_feat,
            fused_features=fused,
            vision_weight=v_w,
            glove_weight=g_w,
            vision_confidence=v_conf,
            glove_confidence=g_conf,
            glove_is_mock=is_mock,
        )
        self._window.append(ff)
        return ff

    @property
    def window_full(self) -> bool:
        return len(self._window) >= settings.MIN_WINDOW_FRAMES

    def get_window_array(self) -> Optional[np.ndarray]:
        """
        Return the sliding window as (T, FUSED_DIM*2) with velocity appended.
        Matches the 272D input the model was trained on.
        """
        if not self.window_full:
            return None
        arr = np.array([f.fused_features for f in self._window], dtype=np.float32)
        vel = np.zeros_like(arr)
        vel[1:] = arr[1:] - arr[:-1]
        return np.concatenate([arr, vel], axis=1)

    def get_confidence_summary(self) -> dict:
        """Average modal confidences over the last 10 frames."""
        recent = list(self._window)[-10:] if self._window else []
        if not recent:
            return {"vision": 0.0, "glove": 0.0}
        return {
            "vision": float(np.mean([f.vision_confidence for f in recent])),
            "glove": float(np.mean([f.glove_confidence for f in recent])),
        }

    def reset(self) -> None:
        self._window.clear()

    # ─────────────────────── internals ──────────────────────────

    def _extract_vision(
        self,
        right: Optional[VisionData],
        left: Optional[VisionData],
    ) -> Tuple[np.ndarray, float]:
        right_feat, r_conf = self._extract_one_hand(right)
        left_feat, l_conf  = self._extract_one_hand(left)
        combined = np.concatenate([right_feat, left_feat]).astype(np.float32)
        confidence = max(r_conf, l_conf)
        return combined, confidence

    def _extract_one_hand(
        self, vision: Optional[VisionData]
    ) -> Tuple[np.ndarray, float]:
        if vision is None:
            return np.zeros(VISION_DIM_PER_HAND, dtype=np.float32), 0.0

        # Use world landmarks (metric, hand-centric) for position invariance.
        # Fall back to image landmarks if world not provided.
        source = vision.world_landmarks if vision.world_landmarks else vision.landmarks
        arr = np.array([[lm.x, lm.y, lm.z] for lm in source], dtype=np.float32)

        # Wrist-center + scale normalise (mirrors extract_landmarks.py)
        arr -= arr[0]
        scale = float(np.linalg.norm(arr[9])) + 1e-6
        arr /= scale

        return arr.flatten(), float(vision.confidence)

    def _extract_flex(
        self,
        glove_right: Optional[GloveData],
        glove_left: Optional[GloveData],
        vision_right: Optional[VisionData],
        vision_left: Optional[VisionData],
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Build a (10,) flex feature vector:  [right×5, left×5].

        Priority:
          1. Real glove data (ble_quality > 0.1)
          2. Synthetic flex estimated from world landmarks
          3. Zeros (no hand visible)
        """
        r_flex, r_quality, r_mock = self._flex_one_hand(glove_right, vision_right)
        l_flex, l_quality, l_mock = self._flex_one_hand(glove_left,  vision_left)

        flex_feat = np.concatenate([r_flex, l_flex]).astype(np.float32)
        quality   = max(r_quality, l_quality)
        is_mock   = r_mock and l_mock
        return flex_feat, quality, is_mock

    def _flex_one_hand(
        self,
        glove: Optional[GloveData],
        vision: Optional[VisionData],
    ) -> Tuple[np.ndarray, float, bool]:
        # 1. Real glove
        if glove is not None and glove.ble_quality > 0.1:
            return (
                np.array(glove.flex, dtype=np.float32),
                float(glove.ble_quality),
                False,
            )
        # 2. Synthesise from world landmarks
        if vision is not None:
            source = vision.world_landmarks if vision.world_landmarks else vision.landmarks
            arr = np.array([[lm.x, lm.y, lm.z] for lm in source], dtype=np.float32)
            arr -= arr[0]
            scale = float(np.linalg.norm(arr[9])) + 1e-6
            arr /= scale
            return self._estimate_flex(arr), 0.5, True
        # 3. No data
        return np.zeros(FLEX_DIM, dtype=np.float32), 0.0, True

    @staticmethod
    def _estimate_flex(pts_21x3: np.ndarray) -> np.ndarray:
        """Estimate per-finger flex ∈ [0,1] from normalised world landmarks."""
        flex = []
        for chain in _FINGER_CHAINS:
            angles = []
            for i in range(1, len(chain) - 1):
                a = pts_21x3[chain[i - 1]] - pts_21x3[chain[i]]
                b = pts_21x3[chain[i + 1]] - pts_21x3[chain[i]]
                denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
                cos_a = float(np.dot(a, b) / denom)
                angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            mean_angle = float(np.mean(angles))
            flex.append(float(np.clip(1.0 - mean_angle / np.pi, 0.0, 1.0)))
        return np.array(flex, dtype=np.float32)

    @staticmethod
    def _compute_weights(v_conf: float, g_conf: float) -> Tuple[float, float]:
        """
        Derive normalised fusion weights from raw confidence scores.
        When both modalities are bad the weights are equal (0.5 / 0.5).
        """
        v_min = settings.VISION_CONFIDENCE_MIN
        g_min = settings.GLOVE_QUALITY_MIN

        v_eff = max(0.0, v_conf - v_min) / (1.0 - v_min + 1e-9)
        g_eff = max(0.0, g_conf - g_min) / (1.0 - g_min + 1e-9)

        total = v_eff + g_eff
        if total < 1e-9:
            return 0.5, 0.5

        return v_eff / total, g_eff / total
