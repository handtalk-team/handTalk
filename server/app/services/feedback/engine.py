"""
Feedback Engine
================

Analyses a completed sign attempt and produces:
  - InlineError list   : per-body-part errors (part, description, expected, observed)
  - DTW score          : similarity to reference motion (lower = better match)
  - Suggestions        : short Korean coaching tips
  - Session summary    : aggregated stats after the session ends

Reference motions
-----------------
Ground-truth feature windows (T, 77) are stored in ml/models/references/{label}.npy
These are produced by the data-collection pipeline (ml/training/collect.py).
If a reference file is missing, the engine falls back to heuristic checks only.

DTW implementation
------------------
We use a simplified 1-D DTW on the flex-sensor subspace (63:68) to measure
how well the user's time-warp matches the reference trajectory.
Full 77-D DTW is computationally too expensive for real-time inline feedback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models.schemas.ws_messages import FeedbackMessage, InlineError, SessionSummary

logger = logging.getLogger(__name__)

REFERENCE_DIR = Path("ml/models/references")

# Flex sensor indices in the 77-D fused vector
FLEX_SLICE = slice(63, 68)    # [thumb, index, middle, ring, pinky]
# IMU accel indices
ACCEL_SLICE = slice(68, 71)
# IMU gyro indices
GYRO_SLICE = slice(71, 74)

# Thresholds for inline error detection
FLEX_ERROR_THRESHOLD = 0.25   # normalised flex deviation
WRIST_ANGLE_THRESHOLD = 0.30  # rad deviation


class FeedbackEngine:
    """
    One instance per WebSocket session.
    Accumulates per-sign results for the end-of-session report.
    """

    def __init__(self) -> None:
        self._references: Dict[str, np.ndarray] = {}
        self._session_log: List[dict] = []
        self._errors_by_sign: Dict[str, List[dict]] = {}

    def load_references(self) -> None:
        """Load all reference .npy files from REFERENCE_DIR."""
        if not REFERENCE_DIR.exists():
            logger.info("Reference directory not found — heuristic-only feedback active")
            return
        for path in REFERENCE_DIR.glob("*.npy"):
            label = path.stem
            try:
                self._references[label] = np.load(str(path))
                logger.debug("Loaded reference for '%s'", label)
            except Exception as e:
                logger.warning("Failed to load reference %s: %s", path, e)

    def analyse(
        self,
        sign_label: str,
        feature_window: np.ndarray,     # (T, 77)
        is_correct: bool,
        confidence: float,
    ) -> FeedbackMessage:
        """
        Produce inline feedback for one completed sign attempt.
        Called by the WebSocket session handler immediately after recognition.
        """
        errors: List[InlineError] = []
        dtw_score: Optional[float] = None
        suggestions: List[str] = []

        # ── DTW against reference ─────────────────────────────────
        if sign_label in self._references:
            ref = self._references[sign_label]
            dtw_score, path = self._dtw(
                feature_window[:, FLEX_SLICE],
                ref[:, FLEX_SLICE],
            )
            if dtw_score > 15.0:
                suggestions.append("전체적인 동작 속도와 궤적을 다시 확인해보세요.")

        # ── Per-part heuristic checks ─────────────────────────────
        errors += self._check_flex_errors(sign_label, feature_window)
        errors += self._check_wrist_stability(feature_window)

        if not errors:
            suggestions.append("동작이 정확합니다! 잘 하셨어요.")
        else:
            suggestions.append("틀린 부분을 확인하고 천천히 다시 연습해보세요.")

        # ── Log for session summary ───────────────────────────────
        entry = {
            "sign": sign_label,
            "is_correct": is_correct,
            "confidence": confidence,
            "dtw_score": dtw_score,
            "errors": [e.model_dump() for e in errors],
        }
        self._session_log.append(entry)
        if errors:
            self._errors_by_sign.setdefault(sign_label, []).extend(
                e.model_dump() for e in errors
            )

        return FeedbackMessage(
            errors=errors,
            suggestions=suggestions,
            dtw_score=dtw_score,
        )

    def build_session_summary(self, session_id: str) -> SessionSummary:
        total = len(self._session_log)
        correct = sum(1 for e in self._session_log if e["is_correct"])
        accuracy = correct / total if total else 0.0
        return SessionSummary(
            session_id=session_id,
            total_signs=total,
            correct_signs=correct,
            accuracy=round(accuracy, 3),
            errors_by_sign={
                sign: [InlineError(**e) for e in errs]
                for sign, errs in self._errors_by_sign.items()
            },
        )

    # ─────────────────────── heuristics ─────────────────────────

    def _check_flex_errors(
        self, label: str, window: np.ndarray
    ) -> List[InlineError]:
        """Compare mean flex posture to known reference or target posture."""
        errors: List[InlineError] = []

        if label not in self._references:
            return errors

        ref = self._references[label]
        user_flex = window[:, FLEX_SLICE].mean(axis=0)   # (5,)
        ref_flex = ref[:, FLEX_SLICE].mean(axis=0)        # (5,)
        diff = np.abs(user_flex - ref_flex)

        finger_names = ["엄지", "검지", "중지", "약지", "새끼"]
        for i, (name, d, exp, obs) in enumerate(
            zip(finger_names, diff, ref_flex, user_flex)
        ):
            if d > FLEX_ERROR_THRESHOLD:
                direction = "더 구부려야" if obs < exp else "덜 구부려야"
                errors.append(
                    InlineError(
                        part=name,
                        description=f"{name}손가락을 {direction} 합니다.",
                        expected=round(float(exp), 2),
                        observed=round(float(obs), 2),
                    )
                )
        return errors

    def _check_wrist_stability(self, window: np.ndarray) -> List[InlineError]:
        """Flag excessive wrist jitter during the sign."""
        errors: List[InlineError] = []
        gyro = window[:, GYRO_SLICE]    # (T, 3)
        std = float(np.std(gyro))
        if std > WRIST_ANGLE_THRESHOLD:
            errors.append(
                InlineError(
                    part="손목",
                    description="손목이 너무 흔들립니다. 안정적으로 유지하세요.",
                    expected=round(WRIST_ANGLE_THRESHOLD, 2),
                    observed=round(std, 2),
                )
            )
        return errors

    # ─────────────────────── DTW ─────────────────────────────────

    @staticmethod
    def _dtw(
        seq_a: np.ndarray, seq_b: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Simple 1-D-per-channel DTW.
        Returns (normalised_distance, alignment_path).
        """
        n, m = len(seq_a), len(seq_b)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )

        # Backtrack
        path: List[Tuple[int, int]] = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            best = min(
                (dtw_matrix[i - 1, j], (i - 1, j)),
                (dtw_matrix[i, j - 1], (i, j - 1)),
                (dtw_matrix[i - 1, j - 1], (i - 1, j - 1)),
                key=lambda x: x[0],
            )
            i, j = best[1]

        normalised = dtw_matrix[n, m] / max(n, m)
        return float(normalised), list(reversed(path))
