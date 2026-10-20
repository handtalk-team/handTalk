"""
AI Hub 수어 문장(real_sen) 키포인트 zip → 문장 인식 학습 데이터 + 아바타용 문장 모션
====================================================================================

zip 을 풀지 않고 바로 읽는다 (문장 키포인트는 압축 해제 시 배치당 ~30GB).

Usage
-----
    python -m ml.preprocessing.extract_aihub_sentence \\
        --zip      "E:/wordmorpheme/_downloads/Training__01_real_sen_keypoint.zip" \\
        --morpheme "E:/wordmorpheme/_downloads/Training__01_real_sen_morpheme.zip" \\
        --out      E:/handtalk_data/sentence --workers 6

출력
----
    {out}/npy/{clip}.npy            (T, 136) float16 — 단어 모델과 같은 손 피처 (30fps, 문장 구간)
    {out}/motion/{clip}.npz         정면(F)만: 상체+양손 수어자 좌표, 15fps float16 (아바타 학습용)
    {out}/labels_{zip이름}.jsonl    클립별 문장 id, 수어자, 방향, 글로스 순서와 구간
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .extract_aihub import _frame_feature, _parse_hand_3d

CLIP_RE = re.compile(r"(NIA_SL_SEN\d+_(?:REAL|SYN)\d+_[A-Z])/[^/]*_(\d{12})_keypoints\.json$")
MARGIN_SEC = 0.3

# 아바타용 관절 (OpenPose BODY_25 인덱스) + 양손 21점
BODY = [("nose", 0), ("neck", 1), ("r_shoulder", 2), ("r_elbow", 3), ("r_wrist", 4),
        ("l_shoulder", 5), ("l_elbow", 6), ("l_wrist", 7), ("r_eye", 15), ("l_eye", 16),
        ("r_ear", 17), ("l_ear", 18)]
JOINTS = [n for n, _ in BODY] + [f"rh_{i}" for i in range(21)] + [f"lh_{i}" for i in range(21)]


# ── morpheme ──────────────────────────────────────────────────────────────────

def load_morphemes(zip_paths: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if not info.filename.endswith("_morpheme.json"):
                    continue
                try:
                    d = json.loads(z.read(info).decode("utf-8"))
                except Exception:
                    continue
                clip = d["metaData"]["name"].replace(".mp4", "")
                segs = sorted(({"gloss": s["attributes"][0]["name"],
                                "start": float(s.get("start", 0)), "end": float(s.get("end", 0))}
                               for s in d.get("data", []) if s.get("attributes")),
                              key=lambda s: s["start"])
                if segs:
                    out[clip] = {"duration": float(d["metaData"].get("duration") or 0), "glosses": segs}
    return out


# ── 프레임 → 피처 ─────────────────────────────────────────────────────────────

def _pts(values, n, dim=3):
    if len(values) != n * (dim + 1):
        return None
    return np.asarray(values, dtype=np.float32).reshape(n, dim + 1)


def _motion_frame(p: dict) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros((len(JOINTS), 3), np.float32)
    ok = np.zeros(len(JOINTS), bool)
    body = _pts(p.get("pose_keypoints_3d", []), 25)
    if body is not None:
        for k, (_, idx) in enumerate(BODY):
            xyz[k] = body[idx, :3]
            ok[k] = body[idx, 3] > 0.1
    nb = len(BODY)
    for off, key in ((nb, "hand_right_keypoints_3d"), (nb + 21, "hand_left_keypoints_3d")):
        h = _pts(p.get(key, []), 21)
        if h is not None:
            xyz[off:off + 21] = h[:, :3]
            ok[off:off + 21] = h[:, 3] > 0.1
    return xyz, ok


def _to_signer(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """카메라 좌표 → 수어자 좌표 (X 오른쪽, Y 앞, Z 위), 목 원점, 어깨 너비 1, 결측 보간."""
    pos = np.stack([-raw[..., 0], -raw[..., 2], -raw[..., 1]], axis=-1)
    T, J, _ = pos.shape
    t = np.arange(T)
    for j in range(J):
        okj = valid[:, j]
        if okj.any() and not okj.all():
            for c in range(3):
                pos[:, j, c] = np.interp(t, t[okj], pos[okj, j, c])
    neck = pos[:, 1:2, :]
    pos = pos - neck
    sw = np.linalg.norm(pos[:, 2] - pos[:, 5], axis=-1)
    sw = float(np.median(sw[sw > 1e-6])) if np.any(sw > 1e-6) else 1.0
    return pos / sw


# ── 워커 ──────────────────────────────────────────────────────────────────────

# 워커는 ZipFile 을 열지 않는다: 문장 zip 하나에 항목이 수백만 개라 워커마다 목차를 올리면
# RAM 이 바닥남. 메인 프로세스가 각 프레임의 (오프셋, 압축 크기, 방식)만 넘겨 주고 직접 읽음.
_F = None


def _init(zip_path: str) -> None:
    global _F
    _F = open(zip_path, "rb")


def _read_member(off: int, csize: int, method: int) -> bytes:
    _F.seek(off)
    hdr = _F.read(30)
    n_len, x_len = struct.unpack("<HH", hdr[26:30])
    _F.seek(off + 30 + n_len + x_len)
    data = _F.read(csize)
    return zlib.decompress(data, -15) if method == 8 else data


def _process(task: Tuple[str, List[Tuple[int, int, int, int]], dict, str]) -> Optional[dict]:
    clip, frames, meta, out = task
    npy_path = os.path.join(out, "npy", clip + ".npy")
    view = clip[-1]
    mot_path = os.path.join(out, "motion", clip + ".npz")
    need_mot = view == "F"
    if os.path.exists(npy_path) and (not need_mot or os.path.exists(mot_path)):
        return None   # 이미 처리됨 (라벨은 다시 기록)
    try:
        frames.sort()
        fps = len(frames) / meta["duration"] if meta["duration"] > 0 else 30.0
        lo = int(max(0.0, meta["glosses"][0]["start"] - MARGIN_SEC) * fps)
        hi = int((meta["glosses"][-1]["end"] + MARGIN_SEC) * fps)
        sel = [fr for fr in frames if lo <= fr[0] <= hi]
        if len(sel) < 10:
            return {"clip": clip, "error": "too few frames"}
        feats, raw, valid = [], [], []
        for _, off, csize, method in sel:
            p = json.loads(_read_member(off, csize, method))["people"]
            if isinstance(p, list):
                p = p[0] if p else {}
            r = _parse_hand_3d(p.get("hand_right_keypoints_3d", []))
            l = _parse_hand_3d(p.get("hand_left_keypoints_3d", []))
            feats.append(_frame_feature(r, l))
            if need_mot:
                xyz, ok = _motion_frame(p)
                raw.append(xyz); valid.append(ok)
        arr = np.stack(feats).astype(np.float16)
        tmp = npy_path + ".tmp.npy"
        np.save(tmp, arr); os.replace(tmp, npy_path)
        if need_mot:
            pos = _to_signer(np.stack(raw), np.stack(valid))
            step = max(1, int(round(fps / 15)))
            pos = pos[::step].astype(np.float16)
            t0 = sel[0][0] / fps
            gl = [{"gloss": g["gloss"], "start": round(g["start"] - t0, 3), "end": round(g["end"] - t0, 3)}
                  for g in meta["glosses"]]
            tmp = mot_path + ".tmp.npz"
            np.savez_compressed(tmp, frames=pos, fps=np.float32(fps / step),
                                joints=np.array(JOINTS), glosses=json.dumps(gl, ensure_ascii=False))
            os.replace(tmp, mot_path)
        return {"clip": clip, "frames": len(sel), "fps": round(fps, 3)}
    except Exception as e:  # noqa: BLE001
        return {"clip": clip, "error": f"{type(e).__name__}: {e}"[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Hub 문장 키포인트 zip → 학습/아바타 데이터")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--morpheme", nargs="+", required=True, help="문장 morpheme zip (여러 개 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--views", default="DFLRU", help="처리할 카메라 방향 (기본 전부)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "npy").mkdir(parents=True, exist_ok=True)
    (out / "motion").mkdir(parents=True, exist_ok=True)
    morph = load_morphemes(args.morpheme)
    print(f"morpheme: {len(morph)} clips", flush=True)

    clips: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    with zipfile.ZipFile(args.zip) as z:
        for info in z.infolist():
            m = CLIP_RE.search(info.filename)
            if m and m.group(1)[-1] in args.views:
                clips[m.group(1)].append((int(m.group(2)), info.header_offset,
                                          info.compress_size, info.compress_type))
    # zip 안에 저장된 순서대로 처리 → 디스크를 거의 순차로 읽음
    tasks = [(c, fr, morph[c], str(out))
             for c, fr in sorted(clips.items(), key=lambda kv: min(x[1] for x in kv[1])) if c in morph]
    clips.clear()
    print(f"clips in zip: {len(clips)}, with labels: {len(tasks)}", flush=True)

    labels_path = out / f"labels_{Path(args.zip).stem}.jsonl"
    n_ok = n_err = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.zip,)) as ex, \
            open(labels_path, "w", encoding="utf-8") as lf:
        for i, (task, res) in enumerate(zip(tasks, ex.map(_process, tasks, chunksize=4)), 1):
            clip, meta = task[0], task[2]
            if res and "error" in res:
                n_err += 1
                continue
            m = re.match(r"NIA_SL_(SEN\d+)_((?:REAL|SYN)\d+)_([A-Z])", clip)
            lf.write(json.dumps({"clip": clip, "sentence_id": m.group(1), "signer": m.group(2),
                                 "view": m.group(3), "glosses": [g["gloss"] for g in meta["glosses"]]},
                                ensure_ascii=False) + "\n")
            n_ok += 1
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)}  ok={n_ok} err={n_err}", flush=True)
    print(f"done: ok={n_ok} err={n_err} → {out}", flush=True)


if __name__ == "__main__":
    main()
