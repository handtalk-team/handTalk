"""
수어 동작 생성 → 메타휴먼 실시간 전송
=====================================

글로스(수어 단어) 순서를 받아, 학습한 동작 생성 모델(GlossToMotion, ml/avatar/sign_motion)이 문장 전체를
하나로 이어진 동작으로 새로 만든다. 그 키포인트를 메타휴먼 본 회전으로 바꿔 Unreal(/ws/avatar) 에
{"type": "motion"} 한 편으로 보내면, Unreal 은 받은 회전을 그대로 뼈대에 입힌다 (HandTalkLiveAnimInstance).
녹화된 단어 동작이나 애니메이션 클립은 쓰지 않는다. 쉬는 자세도 모델이 만든 첫 프레임이다.

키포인트 좌표는 수어자 공간(X 오른쪽, Y 앞, Z 위, 목 원점, 어깨너비 1).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.core.config import local_or

from app.services.avatar import hand_limits

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "client", "unreal", "HandTalkTutor", "Scripts"))
import sign_retarget as sr  # noqa: E402  (언리얼 없이도 도는 순수 파이썬 리타깃)

REF_POSE = os.environ.get("HANDTALK_REF_POSE", os.path.join(_ROOT, "client", "assets", "avatar", "ref_pose.json"))
MODEL_DIR = os.environ.get("HANDTALK_SIGN_MODEL") or local_or("sign_motion", r"C:\handtalk\models\sign_motion")
FPS = 30.0
DEMO_SPEED = float(os.environ.get("HANDTALK_DEMO_SPEED", "0.75"))
EDGE = 8   # 쉬는 자세 ↔ 문장 처음/끝을 잇는 프레임

JOINTS = ["neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "mid_hip", "nose"] \
    + [f"rh_{i}" for i in range(21)] + [f"lh_{i}" for i in range(21)]


def _bridge(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = (np.arange(1, n + 1) / (n + 1)).astype(np.float32)
    w = (t * t * (3 - 2 * t))[:, None, None]
    return a[None] * (1 - w) + b[None] * w


class ModelGenerator:
    """학습한 GlossToMotion (ONNX) 으로 글로스 순서에서 문장 동작을 새로 만든다."""

    SIGNER = os.environ.get("HANDTALK_SIGNER", "REAL02")

    def __init__(self, folder: str = MODEL_DIR) -> None:
        import onnxruntime as ort
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.vocab: Dict[str, int] = meta["vocab"]
        self.fps = float(meta["fps"])
        self.joints: List[str] = meta["joints"]
        self.signers: Dict[str, int] = meta["signers"]
        self.word_signer: Dict[str, str] = meta.get("word_signer", {})
        self.signer = np.array([self.signers.get(self.SIGNER, 0)], dtype=np.int64)
        self.rest_kp: Optional[np.ndarray] = None
        if "rest_kp" in meta:
            src = {n: i for i, n in enumerate(meta["rest_kp"]["joints"])}
            kp = np.asarray(meta["rest_kp"]["kp"], dtype=np.float32)
            self.rest_kp = np.stack([kp[src[n]] if n in src else np.zeros(3, np.float32) for n in JOINTS])
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
        toks = [1]   # BOS, g1, SEP, g2, ..., EOS
        for i, g in enumerate(ids):
            toks += ([2] if i else []) + [g]
        toks.append(3)
        signer = self.signer
        if len(ids) == 1:   # 단어 하나(단어 연습)는 그 단어를 배운 수어자 스타일로
            g = next(g for g in glosses if g in self.vocab)
            if self.word_signer.get(g) in self.signers:
                signer = np.array([self.signers[self.word_signer[g]]], dtype=np.int64)
        hidden, log_dur = self.enc.run(None, {"tokens": np.array([toks], dtype=np.int64), "signer": signer})
        raw = np.maximum(np.round(np.expm1(log_dur[0]) / speed), 0).astype(int)
        # 단어·연결 토큰은 최소 1프레임. 시작/끝(BOS/EOS)은 예측값 그대로 둔다: 모델은 양방향으로 보기 때문에
        # 끝 길이를 억지로 늘리면 앞쪽 동작까지 작아진다 (단어 손 높이가 실제의 절반으로 줄던 원인).
        dur = raw.copy()
        dur[1:-1] = np.maximum(dur[1:-1], 1)
        idx = np.repeat(np.arange(len(toks)), dur)
        starts = np.cumsum(dur) - dur
        within = np.arange(len(idx)) - starts[idx]
        L = dur[idx].astype(np.float32)
        phase = np.stack([(within + 0.5) / L, L / 30.0], -1)[None].astype(np.float32)
        out = self.dec.run(None, {"frames": hidden[:, idx], "phase": phase, "signer": signer})[0][0]
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
        self.model = ModelGenerator()
        logger.info("수어 동작 생성 모델: %d 단어", len(self.model.vocab))
        # 쉬는 자세: 학습 때 meta 에 저장한 '손을 내린 자세' (단어 영상 첫 프레임들의 중앙값, 좌우 대칭)
        self.rest_kp = self.model.rest_kp
        if self.rest_kp is None:
            first = self.model.generate([next(iter(self.model.vocab))])
            self.rest_kp = first[0] if first is not None else None

    def knows(self, gloss: str) -> bool:
        return self.model.knows(gloss)

    def from_gloss(self, gloss: Sequence[str], loop: bool = False, loop_gap: float = 0.0,
                   speed: float = 1.0) -> Optional[dict]:
        kp = self.model.generate(gloss, speed=speed)
        if kp is None:
            return None
        if self.rest_kp is not None:
            r = self.rest_kp
            kp = np.concatenate([r[None], _bridge(r, kp[0], EDGE), kp, _bridge(kp[-1], r, EDGE), r[None]])
        return self.to_message(kp, loop, loop_gap)

    def rest(self) -> Optional[dict]:
        return self.to_message(self.rest_kp[None], loop=True) if self.rest_kp is not None else None

    def to_message(self, kp: np.ndarray, loop: bool = False, loop_gap: float = 0.0) -> dict:
        kp = hand_limits.apply(kp, JOINTS)   # 사람 손이 못 하는 꺾임 제거
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


@functools.lru_cache(maxsize=1)
def get_synth() -> Optional[MotionSynth]:
    try:
        return MotionSynth()
    except Exception as e:   # 모델/참조 포즈가 없으면 아바타는 쉬는 자세 그대로
        logger.warning("실시간 수어 동작 생성 꺼짐 (%s)", e)
        return None


WH_WORDS = {"무엇", "어디", "언제", "누구", "왜", "어떻게", "몇", "얼마", "방법", "곳", "몇호"}
NEG_WORDS = {"없다", "아니다", "안되다", "못하다", "불가능", "싫다", "모르다", "안"}


def non_manual(text: str, gloss: Sequence[str]) -> Optional[dict]:
    """한국수어 비수지 신호(표정·머리 움직임): 의문사 질문 / 예·아니오 질문 / 부정.
    문장 뒷부분(의문사·부정어가 나오는 곳)에 얹는다. start, end 는 동작 길이에 대한 비율."""
    if not gloss:
        return None
    if any(g in NEG_WORDS for g in gloss):
        return {"kind": "neg", "start": 0.45, "end": 0.95}
    if gloss[-1] in WH_WORDS or any(g in WH_WORDS for g in gloss[-2:]):
        return {"kind": "wh", "start": 0.4, "end": 0.97}
    if (text or "").strip().endswith("?"):
        return {"kind": "yn", "start": 0.5, "end": 0.97}
    return None


def motion_for(data: dict) -> Optional[dict]:
    """아바타로 가는 메시지(llm_response / lesson)를 실시간 동작 메시지로 바꾼다. 해당 없으면 None."""
    synth = get_synth()
    if synth is None:
        return None
    kind = data.get("type")
    if kind == "llm_response":
        gloss = [g for g in data.get("gloss") or [] if synth.knows(g)]
        msg = synth.from_gloss(gloss) if gloss else None
        if msg:
            msg["nms"] = non_manual(data.get("text", ""), gloss)
        return msg
    if kind == "lesson":
        target = data.get("target") or {}
        gloss = [] if data.get("done") else [g for g in target.get("gloss") or [] if synth.knows(g)]
        # 연습 중인 목표 문장은 쉬었다가 반복. 목표가 없으면 쉬는 자세로
        # 따라 하라고 보여 주는 시범은 조금 천천히 (손이 닿는 짧은 순간도 내장 그래픽에서 보이게)
        msg = synth.from_gloss(gloss, loop=True, loop_gap=1.8, speed=DEMO_SPEED) if gloss else synth.rest()
        if msg and gloss:
            msg["nms"] = non_manual(target.get("text", ""), gloss)
        return msg
    return None
