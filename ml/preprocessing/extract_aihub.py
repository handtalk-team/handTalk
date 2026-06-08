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

AI Hub 데이터 구조
------------------
keypoint/
  {batch}/
    NIA_SL_WORD{id}_REAL{n}_{angle}/
      *_{frame:012d}_keypoints.json     ← 프레임별 OpenPose JSON

morpheme/
  {batch}/
    NIA_SL_WORD{id}_REAL{n}_{angle}_morpheme.json  ← 라벨 + 구간 정보

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
from collections import defaultdict
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


def _process_clip(clip_dir: str,
                  start_sec: float = 0.0,
                  end_sec:   float = 9999.0,
                  fps:       float = 30.0) -> Optional[np.ndarray]:
    """
    클립 폴더의 프레임 JSON들을 순서대로 읽어 (T, 136) 배열 반환.
    start_sec/end_sec: morpheme에서 추출한 실제 수어 구간 (초)
    """
    json_files = sorted(glob.glob(os.path.join(clip_dir, '*_keypoints.json')))
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

def _build_word_map(morpheme_dir: str) -> Dict[str, str]:
    """WORD0391 → '집중' 매핑 딕셔너리"""
    word_map: Dict[str, str] = {}
    for jf in glob.glob(os.path.join(morpheme_dir, '**', '*_morpheme.json'),
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
                     target_labels: List[str],
                     batch: str = '01') -> Dict[str, List[dict]]:
    """
    대상 단어의 클립 정보 수집.
    batch: keypoint 배치 번호와 일치하는 morpheme 하위 폴더 (기본 '01')
    반환: {label: [{word_id, clip_name, start, end, fps}]}
    """
    target_set = set(target_labels)
    clips: Dict[str, List[dict]] = defaultdict(list)

    # 지정 배치 폴더만 스캔 (전체 스캔 대비 ~16배 빠름)
    scan_dir = os.path.join(morpheme_dir, batch)
    if not os.path.exists(scan_dir):
        scan_dir = morpheme_dir   # fallback: 전체 스캔

    for jf in glob.glob(os.path.join(scan_dir, '**', '*_morpheme.json'),
                        recursive=True):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
            meta     = d['metaData']
            filename = meta['name']                        # NIA_SL_WORD1119_REAL01_R.mp4
            word_id  = filename.split('_')[2]              # WORD1119
            label    = d['data'][0]['attributes'][0]['name']
            if label not in target_set:
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


# ── 메인 ──────────────────────────────────────────────────────────────────────

def extract(keypoint_dir: str,
            morpheme_dir: str,
            output_dir:   str,
            target_words: List[str],
            batch:        str = '01',
            overwrite:    bool = False) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] morpheme 파일 스캔 중 (batch={batch}) ...")
    clip_info = _build_clip_info(morpheme_dir, target_words, batch=batch)

    if not clip_info:
        print("[ERROR] 대상 단어를 찾지 못했습니다. --words 옵션을 확인하세요.")
        sys.exit(1)

    found = list(clip_info.keys())
    missing = [w for w in target_words if w not in clip_info]
    print(f"  찾은 단어: {found}")
    if missing:
        print(f"  데이터 없음: {missing}")

    # 클래스 인덱스
    vocab = {label: idx for idx, label in enumerate(sorted(found))}

    print(f"\n[2/3] 키포인트 추출 시작 ...")
    total_saved = 0

    for label, clips in sorted(clip_info.items()):
        label_dir = out / label
        label_dir.mkdir(exist_ok=True)
        existing  = len(list(label_dir.glob("aihub_*.npy")))
        saved = 0

        print(f"\n  [{label}] {len(clips)}개 클립")
        for info in clips:
            out_file = label_dir / f"aihub_{existing + saved:04d}.npy"
            if out_file.exists() and not overwrite:
                saved += 1
                continue

            # 키포인트 폴더 찾기: {keypoint_dir}/{clip_name}
            clip_name = info['clip_name']
            matches   = glob.glob(os.path.join(keypoint_dir, clip_name))
            if not matches:
                continue

            clip_path = matches[0]
            # fps 추정: 프레임 수 / 영상 길이
            n_frames = len(glob.glob(os.path.join(clip_path, '*_keypoints.json')))
            fps = n_frames / info['duration'] if info['duration'] > 0 else 30.0

            arr = _process_clip(clip_path,
                                start_sec=info['start'],
                                end_sec=info['end'],
                                fps=fps)
            if arr is None:
                continue

            np.save(str(out_file), arr)
            saved += 1
            if saved % 50 == 0:
                print(f"    {saved}/{len(clips)} 저장 중 ...")

        print(f"    → {saved}개 저장")
        total_saved += saved

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
    print(f"  총 {total_saved}개 샘플 → {out}")
    print(f"  vocab.json → {vocab_path}")
    print(f"  클래스: {vocab}")


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
    parser.add_argument('--batches',  nargs='+', default=None,
                        help='처리할 배치 번호 목록 (기본: 자동 감지). 예: 01 02 03')
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
            target_words=args.words,
            batch=batch,
            overwrite=args.overwrite,
        )
