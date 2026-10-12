"""
AI Hub 수어 영상 Keypoint → handTalk 학습 데이터 변환
=======================================================

Usage
-----
    # 기본 (병원 시나리오 단어 자동 선택)
    python -m ml.preprocessing.extract_aihub \\
        --keypoint "/Users/dongseong/Downloads/수어 영상/1.Training/01" \\
        --morpheme "/Users/dongseong/Downloads/수어 영상/1.Training/morpheme" \\
        --output   ml/data

    # 단어 직접 지정
    python -m ml.preprocessing.extract_aihub \\
        --keypoint "..." --morpheme "..." --output ml/data \\
        --words 병원 아프다 괜찮다 의사 간호사

    # 전체 단어 (3000개), 8개 프로세스 병렬
    python -m ml.preprocessing.extract_aihub \\
        --keypoint "..." --morpheme "..." --output E:/handtalk_data/npy_train \\
        --all_words --workers 8

AI Hub 데이터 구조
------------------
keypoint/
  {batch}/
    NIA_SL_WORD{id}_REAL{n}_{angle}/
      *_{frame:012d}_keypoints.json     ← 프레임별 OpenPose JSON

morpheme/
  {batch}/
    NIA_SL_WORD{id}_REAL{n}_{angle}_morpheme.json  ← 라벨 + 구간 정보

syn_word 데이터(NIA_SL_WORD{id}_SYN{n}_{angle})도 같은 구조라 그대로 처리됨.

출력 파일명은 aihub_{clip_name}.npy — 이미 있으면 스킵하므로 중단 후 재실행하면 이어서 진행.

Feature vector (136-D per frame)
---------------------------------
    [0:63]    오른손 3D 랜드마크 (21×3) — 손목 기준 + 스케일 정규화
    [63:126]  왼손  3D 랜드마크 (21×3)
    [126:131] 오른손 flex (5) — 관절 각도 역산
    [131:136] 왼손  flex (5)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FEATURE_DIM = 136
VISION_DIM  = 63   # 21 × 3
FLEX_DIM    = 5

FINGER_CHAINS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]

# handTalk 병원 시나리오 기본 단어 목록
DEFAULT_WORDS = [
    "병원", "아프다", "괜찮다", "치료", "건강",
    "감기", "의사", "간호사", "검사", "입원",
]

# Windows 폴더명에 쓸 수 없는 문자
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# ── Feature engineering ────────────────────────────────────────────────────────

def _normalize(pts_21x3: np.ndarray) -> np.ndarray:
    """손목 중심 + 손 크기 스케일 정규화 → (63,)"""
    arr = pts_21x3.copy()
    arr -= arr[0]                                    # 손목 → 원점
    scale = float(np.linalg.norm(arr[9])) + 1e-6    # 중지 MCP 거리
    arr /= scale
    return arr.flatten()


def _estimate_flex(norm_63: np.ndarray) -> np.ndarray:
    """정규화된 랜드마크에서 flex [0,1] 추정 → (5,)"""
    pts = norm_63.reshape(21, 3)
    flex = []
    for chain in FINGER_CHAINS:
        angles = []
        for i in range(1, len(chain) - 1):
            a = pts[chain[i - 1]] - pts[chain[i]]
            b = pts[chain[i + 1]] - pts[chain[i]]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            cos_a = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
            angles.append(np.arccos(cos_a))
        flex.append(float(np.clip(1.0 - np.mean(angles) / np.pi, 0.0, 1.0)))
    return np.array(flex, dtype=np.float32)


def _parse_hand_3d(values: List[float]) -> Optional[np.ndarray]:
    """
    AI Hub 3D 손 키포인트 파싱.
    포맷: [x, y, z, conf] × 21 = 84 값
    모든 confidence가 0이면 손이 감지 안 된 것 → None 반환
    """
    if len(values) != 84:
        return None
    arr = np.array(values, dtype=np.float32).reshape(21, 4)
    if arr[:, 3].sum() == 0:   # 감지 실패
        return None
    return arr[:, :3]          # (21, 3)


def _frame_feature(right_3d: Optional[np.ndarray],
                   left_3d:  Optional[np.ndarray]) -> np.ndarray:
    """두 손 → 136-D 피처"""
    def hand_feat(pts):
        if pts is None:
            return np.zeros(VISION_DIM + FLEX_DIM, dtype=np.float32)
        n = _normalize(pts)
        return np.concatenate([n, _estimate_flex(n)])

    r = hand_feat(right_3d)
    l = hand_feat(left_3d)
    # [r_vision(63), l_vision(63), r_flex(5), l_flex(5)]
    return np.concatenate([r[:VISION_DIM], l[:VISION_DIM],
                           r[VISION_DIM:], l[VISION_DIM:]])


# ── JSON 읽기 ──────────────────────────────────────────────────────────────────

def _read_keypoint_json(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """프레임 JSON 하나에서 (right_3d, left_3d) 추출"""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        p = d['people']
        right = _parse_hand_3d(p.get('hand_right_keypoints_3d', []))
        left  = _parse_hand_3d(p.get('hand_left_keypoints_3d',  []))
        return right, left
    except Exception:
        return None, None


def _list_frames(clip_dir: str) -> List[str]:
    # os.listdir: glob보다 빠르고 '[라벨]' 같은 폴더명을 패턴으로 해석하지 않음
    return sorted(os.path.join(clip_dir, n) for n in os.listdir(clip_dir)
                  if n.endswith('_keypoints.json'))


def _process_clip(clip_dir: str,
                  start_sec: float = 0.0,
                  end_sec:   float = 9999.0,
                  fps:       float = 30.0,
                  json_files: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """
    클립 폴더의 프레임 JSON들을 순서대로 읽어 (T, 136) 배열 반환.
    start_sec/end_sec: morpheme에서 추출한 실제 수어 구간 (초)
    """
    if json_files is None:
        json_files = _list_frames(clip_dir)
    if not json_files:
        return None

    # 파일명에서 프레임 번호 추출해서 구간 필터링
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)

    frames = []
    for jf in json_files:
        # 파일명: ..._000000000039_keypoints.json
        m = re.search(r'_(\d{12})_keypoints', jf)
        frame_no = int(m.group(1)) if m else len(frames)
        if frame_no < start_frame or frame_no > end_frame:
            continue
        right, left = _read_keypoint_json(jf)
        frames.append(_frame_feature(right, left))

    if len(frames) < 10:
        return None
    return np.stack(frames).astype(np.float32)


# ── morpheme 매핑 ──────────────────────────────────────────────────────────────

def _safe_label(label: str) -> str:
    """라벨을 폴더명으로 쓸 수 있게 정리 (Windows 금지 문자 → '_')"""
    s = _INVALID_FS_CHARS.sub('_', label.strip()).rstrip('. ')
    return s or '_'


def _build_word_map(morpheme_dir: str) -> Dict[str, str]:
    """WORD0391 → '집중' 매핑 딕셔너리"""
    word_map: Dict[str, str] = {}
    for jf in glob.glob(os.path.join(glob.escape(morpheme_dir), '**', '*_morpheme.json'),
                        recursive=True):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
            filename = d['metaData']['name']           # NIA_SL_WORD1119_REAL01_R.mp4
            word_id  = filename.split('_')[2]          # WORD1119
            label    = d['data'][0]['attributes'][0]['name']
            word_map[word_id] = label
        except Exception:
            pass
    return word_map


def _build_clip_info(morpheme_dir: str,
                     target_labels: Optional[List[str]],
                     batch: str = '01') -> Dict[str, List[dict]]:
    """
    대상 단어의 클립 정보 수집.
    target_labels: None 이면 전체 단어
    batch: keypoint 배치 번호와 일치하는 morpheme 하위 폴더 (기본 '01')
    반환: {label: [{word_id, clip_name, start, end, fps}]}
    """
    target_set = set(target_labels) if target_labels is not None else None
    clips: Dict[str, List[dict]] = defaultdict(list)

    # 지정 배치 폴더만 스캔 (전체 스캔 대비 ~16배 빠름)
    scan_dir = os.path.join(morpheme_dir, batch)
    if not os.path.exists(scan_dir):
        scan_dir = morpheme_dir   # fallback: 전체 스캔

    for jf in glob.glob(os.path.join(glob.escape(scan_dir), '**', '*_morpheme.json'),
                        recursive=True):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
            meta     = d['metaData']
            filename = meta['name']                        # NIA_SL_WORD1119_REAL01_R.mp4
            word_id  = filename.split('_')[2]              # WORD1119
            label    = d['data'][0]['attributes'][0]['name']
            if target_set is not None and label not in target_set:
                continue
            duration = float(meta.get('duration', 0))
            start    = float(d['data'][0].get('start', 0))
            end      = float(d['data'][0].get('end',   duration))
            clip_name = filename.replace('.mp4', '')       # NIA_SL_WORD1119_REAL01_R
            clips[label].append({
                'word_id':   word_id,
                'clip_name': clip_name,
                'start':     start,
                'end':       end,
                'duration':  duration,
            })
        except Exception:
            pass
    return clips


# ── 클립 1개 처리 (병렬 워커) ──────────────────────────────────────────────────

def _extract_one(task: Tuple[str, str, float, float, float]) -> str:
    """'saved' | 'skipped' | 'missing' | 'failed'"""
    clip_path, out_file, start, end, duration = task
    if os.path.exists(out_file):
        return 'skipped'
    if not os.path.isdir(clip_path):
        return 'missing'
    try:
        json_files = _list_frames(clip_path)
        # fps 추정: 프레임 수 / 영상 길이
        fps = len(json_files) / duration if duration > 0 else 30.0
        arr = _process_clip(clip_path, start_sec=start, end_sec=end,
                            fps=fps, json_files=json_files)
        if arr is None:
            return 'failed'
        # 임시 파일에 쓰고 rename → 중간에 끊겨도 깨진 npy가 남지 않음
        tmp = out_file + '.tmp.npy'
        np.save(tmp, arr)
        os.replace(tmp, out_file)
        return 'saved'
    except Exception:
        return 'failed'


# ── 메인 ──────────────────────────────────────────────────────────────────────

def extract(keypoint_dir: str,
            morpheme_dir: str,
            output_dir:   str,
            target_words: Optional[List[str]],
            batch:        str = '01',
            overwrite:    bool = False,
            workers:      int = 1) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] morpheme 파일 스캔 중 (batch={batch}) ...")
    clip_info = _build_clip_info(morpheme_dir, target_words, batch=batch)

    if not clip_info:
        print("[ERROR] 대상 단어를 찾지 못했습니다. --words 옵션을 확인하세요.")
        return

    found = list(clip_info.keys())
    print(f"  찾은 단어: {len(found)}개")
    if target_words is not None:
        missing = [w for w in target_words if w not in clip_info]
        if missing:
            print(f"  데이터 없음: {missing}")

    # 클래스 인덱스 (폴더명 = 정리된 라벨)
    vocab = {lbl: idx for idx, lbl in enumerate(sorted({_safe_label(l) for l in found}))}

    tasks = []
    for label, clips in sorted(clip_info.items()):
        label_dir = out / _safe_label(label)
        label_dir.mkdir(exist_ok=True)
        for info in clips:
            # 파일명에 클립명(REALnn/SYNnn = 수어자)을 남겨 재실행 시 중복 방지 + 수어자 단위 split 가능
            out_file = label_dir / f"aihub_{info['clip_name']}.npy"
            if overwrite and out_file.exists():
                out_file.unlink()
            tasks.append((os.path.join(keypoint_dir, info['clip_name']), str(out_file),
                          info['start'], info['end'], info['duration']))

    print(f"\n[2/3] 키포인트 추출 시작 ... ({len(tasks)}개 클립, workers={workers})")
    counts: Dict[str, int] = defaultdict(int)
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(_extract_one, tasks, chunksize=8)
            for i, r in enumerate(results, 1):
                counts[r] += 1
                if i % 1000 == 0:
                    _print_progress(i, len(tasks), counts, t0)
    else:
        for i, task in enumerate(tasks, 1):
            counts[_extract_one(task)] += 1
            if i % 1000 == 0:
                _print_progress(i, len(tasks), counts, t0)
    _print_progress(len(tasks), len(tasks), counts, t0)

    # vocab.json 저장
    vocab_path = out / 'vocab.json'
    if vocab_path.exists():
        with open(vocab_path, encoding='utf-8') as f:
            existing_vocab = json.load(f)
        max_idx = max(existing_vocab.values(), default=-1)
        for lbl in sorted(vocab):
            if lbl not in existing_vocab:
                max_idx += 1
                existing_vocab[lbl] = max_idx
        vocab = existing_vocab

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"\n[3/3] 완료!")
    print(f"  저장 {counts['saved']} / 기존 {counts['skipped']} / "
          f"키포인트 없음 {counts['missing']} / 실패 {counts['failed']} → {out}")
    print(f"  vocab.json → {vocab_path} ({len(vocab)} classes)")


def _print_progress(i: int, n: int, counts: Dict[str, int], t0: float) -> None:
    el = time.time() - t0
    eta = el / i * (n - i) if i else 0
    print(f"    {i}/{n}  saved={counts['saved']} skipped={counts['skipped']} "
          f"missing={counts['missing']} failed={counts['failed']}  "
          f"{el/60:.1f}분 경과, 남은 시간 약 {eta/60:.1f}분", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI Hub 수어 키포인트 → handTalk npy 변환'
    )
    parser.add_argument('--keypoint', required=True,
                        help='keypoint 루트 디렉토리 (배치 폴더들의 부모, 예: .../1.Training)')
    parser.add_argument('--morpheme', required=True,
                        help='morpheme 루트 (예: .../1.Training/morpheme)')
    parser.add_argument('--output',   default='ml/data',
                        help='출력 디렉토리 (기본: ml/data)')
    parser.add_argument('--words',    nargs='+', default=DEFAULT_WORDS,
                        help='학습할 단어 목록 (기본: 병원 시나리오 10단어)')
    parser.add_argument('--all_words', action='store_true',
                        help='morpheme에 있는 전체 단어 추출 (--words 무시)')
    parser.add_argument('--batches',  nargs='+', default=None,
                        help='처리할 배치 번호 목록 (기본: 자동 감지). 예: 01 02 03')
    parser.add_argument('--workers',  type=int, default=1,
                        help='병렬 프로세스 수 (기본 1)')
    parser.add_argument('--overwrite', action='store_true',
                        help='기존 npy 파일 덮어쓰기')
    args = parser.parse_args()

    # 배치 자동 감지 또는 지정
    if args.batches:
        batches = args.batches
    else:
        # keypoint 루트 아래 숫자 폴더 자동 감지
        kp_root = Path(args.keypoint)
        batches = sorted(d.name for d in kp_root.iterdir()
                         if d.is_dir() and d.name.isdigit())
        if not batches:
            batches = ['01']
        print(f'감지된 배치: {batches}')

    for batch in batches:
        kp_dir = str(Path(args.keypoint) / batch)
        if not os.path.exists(kp_dir):
            print(f'[SKIP] 배치 {batch} keypoint 없음: {kp_dir}')
            continue
        print(f'\n{"="*50}')
        print(f'배치 {batch} 처리 중')
        print(f'{"="*50}')
        extract(
            keypoint_dir=kp_dir,
            morpheme_dir=args.morpheme,
            output_dir=args.output,
            target_words=None if args.all_words else args.words,
            batch=batch,
            overwrite=args.overwrite,
            workers=args.workers,
        )
