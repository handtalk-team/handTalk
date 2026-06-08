"""
AI Hub Sign Language Video → Landmark Feature Extractor
========================================================

Usage
-----
    python -m ml.preprocessing.extract_landmarks \\
        --input /path/to/aihub_videos \\
        --output ml/data

Supported input layouts
-----------------------
    /input/{label}/{video}.mp4                    # flat per-label folders
    /input/Training/{label}/{video}.mp4           # with train/val split prefix
    /input/Validation/{label}/{video}.mp4

Output
------
    ml/data/{label}/aihub_{N:04d}.npy   shape=(T, 136) float32
    ml/data/vocab.json                  {"label": class_idx, ...}

Feature vector (136-D per frame)
---------------------------------
    [0 : 63]    오른손 world landmarks (21 × 3) — wrist-centered, scale-normalised
    [63:126]    왼손  world landmarks (21 × 3)
    [126:131]   오른손 flex (5) — joint-angle estimation from landmarks
    [131:136]   왼손  flex (5)

Position invariance
-------------------
MediaPipe world_landmarks are already in a hand-centric metric space (wrist ≈
origin, hand size ≈ constant).  We additionally subtract the wrist and divide by
the wrist→middle-MCP distance so the representation is fully scale-invariant even
across different signer body sizes or camera distances.

Glove compatibility
-------------------
The synthetic flex values (estimated from joint angles) occupy the same
[0, 1] range as the real ESP32 flex-sensor readings.  Training on synthetic
flex teaches the model to use finger-bend information; at inference the
synthetic values are replaced by real glove readings when hardware is present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands

FINGER_CHAINS: List[List[int]] = [
    [1, 2, 3, 4],      # thumb  : CMC → MCP → IP  → TIP
    [5, 6, 7, 8],      # index  : MCP → PIP → DIP → TIP
    [9, 10, 11, 12],   # middle
    [13, 14, 15, 16],  # ring
    [17, 18, 19, 20],  # pinky
]

VISION_DIM = 63   # 21 landmarks × 3
FLEX_DIM   = 5
FEATURE_DIM = VISION_DIM * 2 + FLEX_DIM * 2   # 136


# ── Feature engineering ────────────────────────────────────────────────────────

def _normalize_world_lm(wlm_list) -> np.ndarray:
    """
    Convert a list of 21 MediaPipe WorldLandmark objects to a normalised
    (63,) float32 vector.

    Normalisation:
      1. Wrist-center: subtract landmark[0] from every landmark.
      2. Scale: divide by the wrist→middle-finger-MCP distance (landmark 9).

    This makes the representation invariant to:
      - Absolute hand position in 3-D space (height, distance from camera)
      - Signer body size differences
    """
    arr = np.array([[lm.x, lm.y, lm.z] for lm in wlm_list], dtype=np.float32)
    arr -= arr[0]                                   # wrist → origin
    scale = float(np.linalg.norm(arr[9])) + 1e-6   # middle-finger MCP distance
    arr /= scale
    return arr.flatten()                            # (63,)


def _estimate_flex(wlm_arr_63: np.ndarray) -> np.ndarray:
    """
    Estimate per-finger flex ∈ [0, 1] from normalised world landmarks.

    Geometry
    --------
    For each joint in a finger chain, compute the interior angle (0 = fully
    bent, π = fully straight).  Average across joints, then map:

        flex = 1 − mean_angle / π

    So flex ≈ 0 when the finger is open, ≈ 1 when closed.

    This synthetic value intentionally mirrors the ESP32 ADC-derived flex:
        flex_real = clip((raw_ADC − 800) / 500, 0, 1)
    Both go 0 → 1 as the finger closes, so the model can generalise from
    synthetic (training) to real (inference) without explicit calibration.
    """
    pts = wlm_arr_63.reshape(21, 3)
    flex = []
    for chain in FINGER_CHAINS:
        angles = []
        for i in range(1, len(chain) - 1):
            a = pts[chain[i - 1]] - pts[chain[i]]
            b = pts[chain[i + 1]] - pts[chain[i]]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            cos_a = float(np.dot(a, b) / denom)
            angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        mean_angle = float(np.mean(angles))          # 0 .. π
        flex.append(float(np.clip(1.0 - mean_angle / np.pi, 0.0, 1.0)))
    return np.array(flex, dtype=np.float32)          # (5,)


def _build_frame_feature(
    right_wlm,
    left_wlm,
) -> np.ndarray:
    """
    Combine right + left world landmarks into a 136-D feature vector.
    Missing hands are represented as zeros.
    """
    if right_wlm is not None:
        r_vision = _normalize_world_lm(right_wlm)
        r_flex   = _estimate_flex(r_vision)
    else:
        r_vision = np.zeros(VISION_DIM, dtype=np.float32)
        r_flex   = np.zeros(FLEX_DIM,   dtype=np.float32)

    if left_wlm is not None:
        l_vision = _normalize_world_lm(left_wlm)
        l_flex   = _estimate_flex(l_vision)
    else:
        l_vision = np.zeros(VISION_DIM, dtype=np.float32)
        l_flex   = np.zeros(FLEX_DIM,   dtype=np.float32)

    return np.concatenate([r_vision, l_vision, r_flex, l_flex])   # (136,)


# ── Per-video processing ───────────────────────────────────────────────────────

def _process_video(video_path: Path) -> Optional[np.ndarray]:
    """
    Run MediaPipe on every frame of a video and return a (T, 136) float32 array.
    Returns None if the video has fewer than 10 valid frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path.name}")
        return None

    frames: List[np.ndarray] = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as hands:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            right_wlm = left_wlm = None

            if result.multi_hand_world_landmarks:
                for wlm, handedness in zip(
                    result.multi_hand_world_landmarks,
                    result.multi_handedness,
                ):
                    label = handedness.classification[0].label   # "Left" | "Right"
                    if label == "Right":
                        right_wlm = wlm.landmark
                    else:
                        left_wlm = wlm.landmark

            feat = _build_frame_feature(right_wlm, left_wlm)
            frames.append(feat)

    cap.release()

    if len(frames) < 10:
        print(f"  [SKIP] Too short ({len(frames)} frames): {video_path.name}")
        return None

    return np.stack(frames, axis=0).astype(np.float32)   # (T, 136)


# ── Directory scanning ─────────────────────────────────────────────────────────

_SPLIT_DIRS = {"Training", "Validation", "Test", "train", "val", "test"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _collect_label_videos(root: Path) -> Dict[str, List[Path]]:
    """
    Scan root recursively and return {label: [video_paths]}.

    Handles both:
      root/{label}/...         → label = folder directly under root
      root/{split}/{label}/... → split folders are transparent
    """
    result: Dict[str, List[Path]] = {}

    def _scan(directory: Path, label: Optional[str]) -> None:
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                # If this is a split folder (Training/Validation/…), stay labelless
                new_label = label if entry.name in _SPLIT_DIRS else (label or entry.name)
                _scan(entry, new_label)
            elif entry.suffix.lower() in _VIDEO_EXTS and label:
                result.setdefault(label, []).append(entry)

    _scan(root, None)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def extract(input_dir: str, output_dir: str, overwrite: bool = False) -> None:
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_path}")
        sys.exit(1)

    label_videos = _collect_label_videos(input_path)
    if not label_videos:
        print("[ERROR] No video files found. Check --input path and folder layout.")
        sys.exit(1)

    # Assign deterministic class indices (sorted label names)
    vocab = {label: idx for idx, label in enumerate(sorted(label_videos))}
    print(f"\nFound {len(vocab)} labels: {', '.join(vocab)}")
    print(f"Total videos: {sum(len(v) for v in label_videos.values())}\n")

    total_saved = 0

    for label, videos in sorted(label_videos.items()):
        label_out = output_path / label
        label_out.mkdir(parents=True, exist_ok=True)

        # Count existing aihub samples to avoid index collisions on re-runs
        existing = len(list(label_out.glob("aihub_*.npy")))
        saved = 0

        for video_path in videos:
            out_file = label_out / f"aihub_{existing + saved:04d}.npy"
            if out_file.exists() and not overwrite:
                saved += 1
                continue

            print(f"  [{label}] {video_path.name} ... ", end="", flush=True)
            arr = _process_video(video_path)
            if arr is None:
                continue

            np.save(str(out_file), arr)
            print(f"saved  T={arr.shape[0]}")
            saved += 1

        print(f"  → {label}: {saved} samples\n")
        total_saved += saved

    # Write / update vocab.json
    vocab_path = output_path / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            existing_vocab = json.load(f)
        # Merge: new labels get next available idx
        max_idx = max(existing_vocab.values(), default=-1)
        for lbl in sorted(vocab):
            if lbl not in existing_vocab:
                max_idx += 1
                existing_vocab[lbl] = max_idx
        vocab = existing_vocab

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocab saved → {vocab_path}")
    print(f"\nDone. {total_saved} samples extracted to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe world landmarks from AI Hub sign language videos"
    )
    parser.add_argument("--input",     required=True, help="Root directory of AI Hub videos")
    parser.add_argument("--output",    default="ml/data", help="Output directory (default: ml/data)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if .npy already exists")
    args = parser.parse_args()

    extract(args.input, args.output, args.overwrite)
