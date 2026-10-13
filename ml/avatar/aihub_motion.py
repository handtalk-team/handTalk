"""
AI Hub 수어 키포인트 → 아바타용 모션 JSON
==========================================

AI Hub 클립 폴더(프레임별 OpenPose JSON)에서 상체와 양손 키포인트를 뽑아
캐릭터와 무관한 "수어자 좌표계" 모션 파일로 저장한다. 이 파일을 언리얼 에디터에서
client/unreal/HandTalkTutor/Scripts/build_sign_anim.py 가 메타휴먼 AnimSequence 로 바꾼다.

Usage
-----
    python -m ml.avatar.aihub_motion \\
        --clip     "E:/wordmorpheme/keypoint/01/NIA_SL_WORD0001_REAL01_F" \\
        --morpheme "E:/wordmorpheme/morpheme/01/NIA_SL_WORD0001_REAL01_F_morpheme.json" \\
        --clip_name KSL_hello \\
        --out      ml/avatar/out

수어자 좌표계 (출력)
--------------------
    X = 수어자의 오른쪽, Y = 수어자의 앞쪽(보는 사람 쪽), Z = 위
    원점 = 목(neck), 단위 = 어깨 너비 1.0

    OpenPose 이미지 좌표는 x 오른쪽, y 아래, (3D면) z 카메라에서 멀어지는 쪽이다.
    수어자는 카메라를 보고 있으므로 수어자의 오른쪽은 이미지의 왼쪽(-x)이다.

출력 JSON
---------
    {
      "format": "handtalk-motion/1",
      "clip": "KSL_hello", "label": "안녕하세요", "source": "NIA_SL_...",
      "fps": 30.0, "space": "signer", "depth": "3d" | "2d",
      "joints": ["neck", "r_shoulder", ..., "rh_0", ..., "lh_20"],
      "frames": [[x, y, z, ...joints*3], ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# OpenPose BODY_25 에서 쓰는 관절
BODY_JOINTS: List[Tuple[str, int]] = [
    ("neck", 1),
    ("r_shoulder", 2), ("r_elbow", 3), ("r_wrist", 4),
    ("l_shoulder", 5), ("l_elbow", 6), ("l_wrist", 7),
    ("mid_hip", 8),
    ("nose", 0),
]
HAND_POINTS = 21

JOINT_NAMES: List[str] = (
    [n for n, _ in BODY_JOINTS]
    + [f"rh_{i}" for i in range(HAND_POINTS)]
    + [f"lh_{i}" for i in range(HAND_POINTS)]
)


# ── 프레임 읽기 ────────────────────────────────────────────────────────────────

def _points(values: List[float], n: int, dim: int) -> Optional[np.ndarray]:
    """[x, y, (z,) conf] * n → (n, 4) [x, y, z, conf]. 값이 없으면 None."""
    stride = dim + 1
    if len(values) != n * stride:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(n, stride)
    out = np.zeros((n, 4))
    out[:, :dim] = arr[:, :dim]
    out[:, 3] = arr[:, dim]
    return out


def _read_frame(path: str, use_3d: bool) -> Dict[str, Optional[np.ndarray]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    p = d["people"]
    if isinstance(p, list):          # 표준 OpenPose 는 리스트, AI Hub 는 dict
        p = p[0] if p else {}
    dim, sfx = (3, "_3d") if use_3d else (2, "_2d")
    return {
        "body": _points(p.get("pose_keypoints" + sfx, []), 25, dim),
        "rh": _points(p.get("hand_right_keypoints" + sfx, []), HAND_POINTS, dim),
        "lh": _points(p.get("hand_left_keypoints" + sfx, []), HAND_POINTS, dim),
    }


def _has_3d(path: str) -> bool:
    """3D 키가 있고 실제 값(0이 아닌 좌표)이 들어 있는지"""
    with open(path, encoding="utf-8") as f:
        p = json.load(f)["people"]
    if isinstance(p, list):
        p = p[0] if p else {}
    for key, n in (("pose_keypoints_3d", 25), ("hand_right_keypoints_3d", 21)):
        pts = _points(p.get(key, []), n, 3)
        if pts is None or not np.any(pts[:, :3]):
            return False
    return True


def list_frames(clip_dir: str) -> List[Tuple[int, str]]:
    out = []
    for n in os.listdir(clip_dir):
        m = re.search(r"_(\d{12})_keypoints\.json$", n)
        if m:
            out.append((int(m.group(1)), os.path.join(clip_dir, n)))
    return sorted(out)


# ── 좌표 변환 ──────────────────────────────────────────────────────────────────

def to_signer_space(xyz: np.ndarray, depth: str) -> np.ndarray:
    """OpenPose (이미지/카메라) 좌표 → 수어자 좌표 (X 오른쪽, Y 앞, Z 위)"""
    out = np.empty_like(xyz)
    out[..., 0] = -xyz[..., 0]                       # 수어자 오른쪽 = 이미지 왼쪽
    out[..., 1] = -xyz[..., 2] if depth == "3d" else 0.0   # 카메라 쪽 = 앞
    out[..., 2] = -xyz[..., 1]                       # 이미지 y 는 아래
    return out


def _fill_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, J, 3) 에서 valid[t, j] 가 False 인 값을 시간축 선형 보간. 전부 없으면 0 유지."""
    T = x.shape[0]
    t = np.arange(T)
    out = x.copy()
    for j in range(x.shape[1]):
        ok = valid[:, j]
        if ok.all() or not ok.any():
            continue
        for c in range(3):
            out[:, j, c] = np.interp(t, t[ok], x[ok, j, c])
    return out


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """시간축 이동 평균 (가장자리는 edge padding)"""
    if window <= 1 or x.shape[0] < window:
        return x
    k = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, ((pad, window - 1 - pad), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        for c in range(3):
            out[:, j, c] = np.convolve(xp[:, j, c], k, mode="valid")
    return out


def extract_motion(clip_dir: str,
                   start_sec: float = 0.0,
                   end_sec: float = 1e9,
                   duration: Optional[float] = None,
                   conf_threshold: float = 0.1,
                   smooth_window: int = 5,
                   force_2d: bool = False) -> dict:
    frames = list_frames(clip_dir)
    if not frames:
        raise FileNotFoundError(f"프레임 JSON 없음: {clip_dir}")

    fps = len(frames) / duration if duration else 30.0
    lo, hi = int(start_sec * fps), int(end_sec * fps)
    frames = [(n, p) for n, p in frames if lo <= n <= hi]
    if len(frames) < 2:
        raise ValueError("수어 구간 프레임이 2개 미만")

    depth = "2d" if force_2d or not _has_3d(frames[0][1]) else "3d"
    J = len(JOINT_NAMES)
    raw = np.zeros((len(frames), J, 3))
    valid = np.zeros((len(frames), J), dtype=bool)

    nb = len(BODY_JOINTS)
    for t, (_, path) in enumerate(frames):
        fr = _read_frame(path, depth == "3d")
        if fr["body"] is not None:
            for k, (_, idx) in enumerate(BODY_JOINTS):
                raw[t, k] = fr["body"][idx, :3]
                valid[t, k] = fr["body"][idx, 3] > conf_threshold
        for off, key in ((nb, "rh"), (nb + HAND_POINTS, "lh")):
            pts = fr[key]
            if pts is not None:
                raw[t, off:off + HAND_POINTS] = pts[:, :3]
                valid[t, off:off + HAND_POINTS] = pts[:, 3] > conf_threshold

    pos = to_signer_space(raw, depth)
    pos = _fill_gaps(pos, valid)
    pos = _smooth(pos, smooth_window)

    # 목 기준, 어깨 너비 1.0 으로 정규화 (클립 전체 중앙값 사용 — 프레임별 흔들림 방지)
    neck = pos[:, 0:1, :]
    pos = pos - neck
    sw = np.linalg.norm(pos[:, 1] - pos[:, 4], axis=-1)
    sw = float(np.median(sw[sw > 1e-6])) if np.any(sw > 1e-6) else 1.0
    pos = pos / sw

    return {
        "format": "handtalk-motion/1",
        "fps": round(float(fps), 3),
        "space": "signer",
        "depth": depth,
        "joints": JOINT_NAMES,
        "valid_ratio": {
            "body": round(float(valid[:, :nb].mean()), 3),
            "right_hand": round(float(valid[:, nb:nb + HAND_POINTS].mean()), 3),
            "left_hand": round(float(valid[:, nb + HAND_POINTS:].mean()), 3),
        },
        "frames": np.round(pos.reshape(len(frames), -1), 4).tolist(),
    }


def read_morpheme(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    meta = d["metaData"]
    seg = d["data"][0]
    duration = float(meta.get("duration", 0)) or None
    return {
        "source": meta["name"].replace(".mp4", ""),
        "label": seg["attributes"][0]["name"],
        "start": float(seg.get("start", 0)),
        "end": float(seg.get("end", duration or 1e9)),
        "duration": duration,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Hub 키포인트 클립 → handTalk 모션 JSON")
    ap.add_argument("--clip", required=True, help="프레임 JSON 이 든 클립 폴더")
    ap.add_argument("--morpheme", help="대응하는 *_morpheme.json (수어 구간과 라벨)")
    ap.add_argument("--clip_name", help="언리얼 클립 이름 (기본: 라벨)")
    ap.add_argument("--out", default="ml/avatar/out")
    ap.add_argument("--smooth", type=int, default=5, help="이동 평균 창 (프레임)")
    ap.add_argument("--force_2d", action="store_true", help="3D 키가 있어도 2D 사용")
    args = ap.parse_args()

    info: Dict[str, object] = {"source": Path(args.clip).name, "label": None,
                               "start": 0.0, "end": 1e9, "duration": None}
    if args.morpheme:
        info.update(read_morpheme(args.morpheme))

    motion = extract_motion(args.clip, info["start"], info["end"], info["duration"],
                            smooth_window=args.smooth, force_2d=args.force_2d)
    clip_name = args.clip_name or info["label"] or info["source"]
    motion = {"clip": clip_name, "label": info["label"], "source": info["source"], **motion}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{clip_name}.motion.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(motion, f, ensure_ascii=False)
    print(f"{path}  frames={len(motion['frames'])} fps={motion['fps']} "
          f"depth={motion['depth']} valid={motion['valid_ratio']}")


if __name__ == "__main__":
    main()
