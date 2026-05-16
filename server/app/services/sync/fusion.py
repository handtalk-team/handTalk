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
4.  Feature-level fusion: vision → 63-D vector, glove → 14-D vector,
    concatenated → 77-D fused vector.
5.  Dynamic weighting: confidence scores gate each modality's contribution.
    If vision confidence < VISION_MIN we zero-pad vision features so the
    model learns not to rely on them.  Same for glove.
6.  The fused frames are pushed into a fixed-size sliding window.
    Once full, the window numpy array is handed to the recogniser.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.models.schemas.sensor import GloveData, SensorFrame, VisionData

settings = get_settings()

# Maximum time difference (seconds) to consider a glove packet
# as "contemporaneous" with a camera frame.
MAX_GLOVE_LAG_S: float = 0.10  # 100 ms

VISION_DIM = 63   # 21 landmarks × 3 (x, y, z)
GLOVE_DIM = 14    # 5 flex + 3 accel + 3 gyro + 3 euler-from-quat
FUSED_DIM = VISION_DIM + GLOVE_DIM   # 77


@dataclass
class FusedFrame:
    timestamp: float
    vision_features: np.ndarray      # shape (VISION_DIM,)
    glove_features: np.ndarray       # shape (GLOVE_DIM,)
    fused_features: np.ndarray       # shape (FUSED_DIM,)
    vision_weight: float             # 0.0 – 1.0
    glove_weight: float              # 0.0 – 1.0
    vision_confidence: float
    glove_confidence: float
    glove_is_mock: bool = False


@dataclass
class _GloveSnapshot:
    timestamp: float
    data: GloveData


class SensorFusionModule:
    """
    Thread-safe sensor fusion for one WebSocket session.
    One instance per connected client.
    """

    def __init__(self) -> None:
        self._window: Deque[FusedFrame] = deque(
            maxlen=settings.WINDOW_SIZE
        )
        # Short ring of recent glove packets for nearest-neighbour lookup
        self._glove_ring: Deque[_GloveSnapshot] = deque(maxlen=10)
        self._last_glove: Optional[_GloveSnapshot] = None

    # ─────────────────────── public API ─────────────────────────

    def push_frame(self, frame: SensorFrame) -> FusedFrame:
        """
        Ingest one SensorFrame, fuse it, append to the sliding window,
        and return the resulting FusedFrame.
        """
        # Register any glove data that arrived with this frame
        if frame.glove is not None:
            snap = _GloveSnapshot(timestamp=frame.timestamp, data=frame.glove)
            self._glove_ring.append(snap)
            self._last_glove = snap

        vision_feat, v_conf = self._extract_vision(frame.camera)
        glove_feat, g_conf, is_mock = self._extract_glove(frame.timestamp)

        v_w, g_w = self._compute_weights(v_conf, g_conf)

        # Zero-pad the weaker modality when it is below the quality floor
        # (so the model sees explicit zeros, not noisy garbage)
        if v_conf < settings.VISION_CONFIDENCE_MIN:
            vision_feat = np.zeros(VISION_DIM, dtype=np.float32)
        if g_conf < settings.GLOVE_QUALITY_MIN:
            glove_feat = np.zeros(GLOVE_DIM, dtype=np.float32)

        fused = np.concatenate([vision_feat, glove_feat]).astype(np.float32)

        ff = FusedFrame(
            timestamp=frame.timestamp,
            vision_features=vision_feat,
            glove_features=glove_feat,
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
        Return the sliding window as a float32 array of shape (T, FUSED_DIM).
        Returns None if not enough frames have been collected yet.
        """
        if not self.window_full:
            return None
        return np.array(
            [f.fused_features for f in self._window], dtype=np.float32
        )

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
        self._glove_ring.clear()
        self._last_glove = None

    # ─────────────────────── internals ──────────────────────────

    def _extract_vision(
        self, vision: Optional[VisionData]
    ) -> Tuple[np.ndarray, float]:
        if vision is None:
            return np.zeros(VISION_DIM, dtype=np.float32), 0.0

        coords: List[float] = []
        for lm in vision.landmarks:
            coords.extend([lm.x, lm.y, lm.z])

        # Normalise: subtract wrist (landmark 0) so the vector is
        # translation-invariant (hand position in frame doesn't matter)
        arr = np.array(coords, dtype=np.float32)
        wrist = arr[:3]
        arr = arr - np.tile(wrist, VISION_DIM // 3)

        return arr, float(vision.confidence)

    def _extract_glove(
        self, ts: float
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Find the glove snapshot closest in time to `ts`.
        Falls back to the last known snapshot (hold-last-value strategy).
        Returns (features, quality, is_mock).
        """
        snap = self._nearest_glove(ts)
        if snap is None:
            return np.zeros(GLOVE_DIM, dtype=np.float32), 0.0, True

        g = snap.data
        euler = self._quat_to_euler(g.imu.quaternion)
        features = np.array(
            g.flex + g.imu.accel + g.imu.gyro + list(euler),
            dtype=np.float32,
        )
        return features, float(g.ble_quality), g.is_mock

    def _nearest_glove(self, ts: float) -> Optional[_GloveSnapshot]:
        """Return the glove snapshot with the smallest |Δt| to ts."""
        if not self._glove_ring and self._last_glove is None:
            return None

        candidates = list(self._glove_ring)
        if self._last_glove and self._last_glove not in candidates:
            candidates.append(self._last_glove)

        best = min(candidates, key=lambda s: abs(s.timestamp - ts))
        if abs(best.timestamp - ts) > MAX_GLOVE_LAG_S:
            # Too old to be useful, but still return it (hold-last-value)
            # so the model sees a non-zero glove signal
            return best
        return best

    @staticmethod
    def _quat_to_euler(q: List[float]) -> Tuple[float, float, float]:
        """Convert quaternion [w, x, y, z] → Euler [roll, pitch, yaw] in rad."""
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        return roll, pitch, yaw

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
