"""
수어 동작 생성 → 메타휴먼 실시간 전송
=====================================

글로스(수어 단어) 순서를 받아 문장 전체를 하나로 이어진 동작으로 만들고, 메타휴먼 본 회전으로 바꿔
Unreal(/ws/avatar) 에 {"type": "motion"} 한 편으로 보낸다. Unreal 은 받은 회전을 그대로 뼈대에 입힌다
(HandTalkLiveAnimInstance). 미리 구운 애니메이션 클립은 쓰지 않는다.

동작을 만드는 부분(MotionGenerator)은 교체할 수 있다.
  ConcatGenerator  단어 키포인트에서 실제 수어 구간만 잘라 자연스럽게 이어 붙임 (지금)
  (학습 모델)       AI Hub 문장 데이터로 학습한 글로스 → 키포인트 생성 모델 (3070 에서 학습 후 교체)

키포인트 좌표는 ml/avatar/aihub_motion.py 의 수어자 공간(X 오른쪽, Y 앞, Z 위, 어깨너비 정규화).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.services.tutor import curriculum as cur

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "client", "unreal", "HandTalkTutor", "Scripts"))
import sign_retarget as sr  # noqa: E402  (언리얼 없이도 도는 순수 파이썬 리타깃)

MOTION_DIR = os.environ.get("HANDTALK_MOTIONS", r"C:\handtalk\motions")
REF_POSE = os.environ.get("HANDTALK_REF_POSE", os.path.join(_ROOT, "client", "assets", "avatar", "ref_pose.json"))
REST_CLIP = "KSL_hospital"   # 이 동작의 마지막 프레임이 쉬는 자세 (Unreal RestClip 과 같음)
FPS = 30.0

JOINTS = ["neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "mid_hip", "nose"] \
    + [f"rh_{i}" for i in range(21)] + [f"lh_{i}" for i in range(21)]
_WRISTS = (JOINTS.index("r_wrist"), JOINTS.index("l_wrist"))


@functools.lru_cache(maxsize=512)
def load_keypoints(clip: str) -> Optional[np.ndarray]:
    """클립의 키포인트 (프레임, 관절, 3). JOINTS 순서로 맞춘다."""
    path = os.path.join(MOTION_DIR, f"{clip}.motion.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    src = np.asarray(m["frames"], dtype=np.float32).reshape(len(m["frames"]), -1, 3)
    idx = {n: i for i, n in enumerate(m["joints"])}
    out = np.zeros((src.shape[0], len(JOINTS), 3), dtype=np.float32)
    for j, n in enumerate(JOINTS):
        if n in idx:
            out[:, j] = src[:, idx[n]]
    return out


def _smoothstep(n: int) -> np.ndarray:
    t = (np.arange(1, n + 1) / (n + 1)).astype(np.float32)
    return t * t * (3 - 2 * t)


class ConcatGenerator:
    """단어마다 손이 올라가 있는 구간만 잘라, 단어 사이를 부드럽게 이어 한 문장으로 만든다."""

    ACTIVE = 0.35      # 쉬는 자세 손목에서 이만큼(어깨너비 배) 벗어나면 수어 중
    MARGIN = 3         # 잘라낸 구간 앞뒤 여유 프레임
    GAP = 7            # 단어 사이 이어 주는 프레임
    EDGE = 12          # 쉬는 자세 ↔ 첫/끝 단어

    def __init__(self) -> None:
        rest = load_keypoints(REST_CLIP)
        self.rest = rest[-1] if rest is not None else None

    def _active(self, kp: np.ndarray) -> np.ndarray:
        if self.rest is None:
            return kp
        d = np.max([np.linalg.norm(kp[:, w] - self.rest[w], axis=-1) for w in _WRISTS], axis=0)
        on = np.where(d > self.ACTIVE)[0]
        if len(on) == 0:
            return kp
        a, b = max(on[0] - self.MARGIN, 0), min(on[-1] + self.MARGIN + 1, len(kp))
        return kp[a:b]

    @staticmethod
    def _bridge(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
        w = _smoothstep(n)[:, None, None]
        return a[None] * (1 - w) + b[None] * w

    def generate(self, clips: Sequence[str], rest_edges: bool = True) -> Optional[np.ndarray]:
        parts = [self._active(k) for k in (load_keypoints(c) for c in clips) if k is not None]
        if not parts:
            return None
        seq: List[np.ndarray] = []
        if rest_edges and self.rest is not None:
            seq += [self.rest[None], self._bridge(self.rest, parts[0][0], self.EDGE)]
        for i, p in enumerate(parts):
            if i:
                seq.append(self._bridge(parts[i - 1][-1], p[0], self.GAP))
            seq.append(p)
        if rest_edges and self.rest is not None:
            seq += [self._bridge(parts[-1][-1], self.rest, self.EDGE), self.rest[None]]
        kp = np.concatenate(seq, axis=0)
        # 떨림 줄이기 (3프레임 이동 평균, 양 끝은 그대로)
        if len(kp) > 2:
            kp[1:-1] = (kp[:-2] + kp[1:-1] + kp[2:]) / 3
        return kp


MODEL_DIR = os.environ.get("HANDTALK_SIGN_MODEL", r"C:\handtalk\models\sign_motion")


class ModelGenerator:
    """학습한 GlossToMotion (ml/avatar/sign_motion) 으로 글로스 순서에서 문장 동작을 새로 만든다."""

    SIGNER = os.environ.get("HANDTALK_SIGNER", "REAL02")

    def __init__(self, folder: str = MODEL_DIR) -> None:
        import onnxruntime as ort
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.vocab: Dict[str, int] = meta["vocab"]
        self.fps = float(meta["fps"])
        self.joints: List[str] = meta["joints"]
        self.signer = np.array([meta["signers"].get(self.SIGNER, 0)], dtype=np.int64)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(os.path.join(folder, "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(folder, "decoder.onnx"), opts, providers=["CPUExecutionProvider"])

    def knows(self, gloss: str) -> bool:
        return gloss in self.vocab

    def generate(self, glosses: Sequence[str], speed: float = 1.0) -> Optional[np.ndarray]:
        ids = [self.vocab[g] for g in glosses if g in self.vocab]
        if not ids:
            return None
        toks = [1]
        for i, g in enumerate(ids):
            toks += ([2] if i else []) + [g]
        toks.append(3)
        hidden, log_dur = self.enc.run(None, {"tokens": np.array([toks], dtype=np.int64), "signer": self.signer})
        dur = np.maximum(np.round(np.expm1(log_dur[0]) / speed), 1).astype(int)
        dur[0], dur[-1] = max(dur[0], 3), max(dur[-1], 3)
        idx = np.repeat(np.arange(len(toks)), dur)
        starts = np.cumsum(dur) - dur
        within = np.arange(len(idx)) - starts[idx]
        L = dur[idx].astype(np.float32)
        phase = np.stack([(within + 0.5) / L, L / 30.0], -1)[None].astype(np.float32)
        out = self.dec.run(None, {"frames": hidden[:, idx], "phase": phase, "signer": self.signer})[0][0]
        kp = np.zeros((len(out), len(JOINTS), 3), dtype=np.float32)
        for j, n in enumerate(self.joints):
            kp[:, JOINTS.index(n)] = out[:, j]
        # 15fps → 30fps
        t = np.linspace(0, len(kp) - 1, int(round(len(kp) * FPS / self.fps)))
        a = np.floor(t).astype(int)
        b = np.minimum(a + 1, len(kp) - 1)
        w = (t - a)[:, None, None].astype(np.float32)
        return kp[a] * (1 - w) + kp[b] * w


class MotionSynth:
    def __init__(self) -> None:
        with open(REF_POSE, encoding="utf-8") as f:
            ref = json.load(f)
        self.retargeter = sr.Retargeter(sr.RefPose(
            {k: tuple(v) for k, v in ref["pos"].items()},
            {k: tuple(v) for k, v in ref["rot"].items()},
            ref["parent"]))
        self.bones = [m[0] for m in self.retargeter.map]
        self.generator = ConcatGenerator()
        self.model: Optional[ModelGenerator] = None
        if os.path.exists(os.path.join(MODEL_DIR, "decoder.onnx")):
            try:
                self.model = ModelGenerator()
                logger.info("수어 동작 생성 모델 사용: %d 단어", len(self.model.vocab))
            except Exception as e:   # 모델이 깨졌으면 이어 붙이기로
                logger.warning("동작 생성 모델을 못 읽음 (%s)", e)

    def knows(self, gloss: str) -> bool:
        return bool(cur.clip_for(gloss)) or bool(self.model and self.model.knows(gloss))

    def from_gloss(self, gloss: Sequence[str], loop: bool = False, loop_gap: float = 0.0) -> Optional[dict]:
        """글로스 순서 → 문장 동작. 모델이 있으면 모델이 새로 만들고, 없으면 단어 동작을 이어 붙인다."""
        kp = None
        if self.model is not None:
            kp = self.model.generate(gloss)
            if kp is not None and self.generator.rest is not None:
                g = self.generator
                kp = np.concatenate([g.rest[None], g._bridge(g.rest, kp[0], g.EDGE), kp,
                                     g._bridge(kp[-1], g.rest, g.EDGE), g.rest[None]])
        if kp is None:
            kp = self.generator.generate(self.clips_for(gloss))
        return self.to_message(kp, loop, loop_gap) if kp is not None else None

    def clips_for(self, gloss: Sequence[str]) -> List[str]:
        return [c for c in (cur.clip_for(g) for g in gloss) if c]

    def to_message(self, kp: np.ndarray, loop: bool = False, loop_gap: float = 0.0) -> dict:
        frames: List[List[float]] = []
        prev: Dict[str, sr.Quat] = {}
        for row in kp:
            world = self.retargeter.solve_frame_world({n: row[j] for j, n in enumerate(JOINTS)})
            flat: List[float] = []
            for b in self.bones:
                q = world[b]
                if b in prev:
                    q = sr.q_align_flip(q, prev[b])
                prev[b] = q
                flat += [round(float(x), 5) for x in q]
            frames.append(flat)
        return {"type": "motion", "fps": FPS, "loop": loop, "loop_gap": loop_gap,
                "bones": self.bones, "frames": frames}

    def from_clips(self, clips: Sequence[str], loop: bool = False, loop_gap: float = 0.0) -> Optional[dict]:
        kp = self.generator.generate(clips)
        return self.to_message(kp, loop, loop_gap) if kp is not None else None

    def rest(self) -> Optional[dict]:
        if self.generator.rest is None:
            return None
        return self.to_message(self.generator.rest[None], loop=True)


@functools.lru_cache(maxsize=1)
def get_synth() -> Optional[MotionSynth]:
    try:
        return MotionSynth()
    except (OSError, KeyError, ValueError) as e:
        logger.warning("실시간 동작 생성 꺼짐 (%s). 기존 클립 재생으로 동작합니다.", e)
        return None


def motion_for(data: dict) -> Optional[dict]:
    """아바타로 가는 메시지(llm_response / lesson)를 실시간 동작 메시지로 바꾼다. 해당 없으면 None."""
    synth = get_synth()
    if synth is None:
        return None
    kind = data.get("type")
    if kind == "llm_response":
        gloss = [g for g in data.get("gloss") or [] if synth.knows(g)]
        return synth.from_gloss(gloss) if gloss else None
    if kind == "lesson":
        target = data.get("target") or {}
        gloss = [] if data.get("done") else [g for g in target.get("gloss") or [] if synth.knows(g)]
        # 연습 중인 목표 동작은 쉬었다가 반복. 목표가 없으면 쉬는 자세로
        return synth.from_gloss(gloss, loop=True, loop_gap=1.8) if gloss else synth.rest()
    return None
