"""
수어자 좌표계 모션 → 메타휴먼(UE5 마네킹 호환) 본 회전
=======================================================

언리얼 없이도 돌아가는 순수 파이썬 수학 모듈. build_sign_anim.py 가 언리얼 에디터 안에서
이 모듈을 불러 쓰고, 테스트는 일반 파이썬에서 한다.

방법: 방향 맞추기(FK)
  - 참조 포즈(A포즈)에서 각 본의 방향(본 → 자식 본)을 구한다.
  - 키포인트에서 같은 구간의 방향(예: 어깨 → 팔꿈치)을 구한다.
  - 부모에게서 물려받은 회전에 최소 회전을 더해 두 방향을 맞춘다.
  - 손은 손바닥 방향(검지 MCP → 새끼 MCP)으로 비틀림까지 맞춘다.
  척추, 목, 머리, 다리는 참조 포즈를 유지한다.

쿼터니언은 (x, y, z, w) 튜플. 합성 규칙은 UE FQuat 과 같다: world = parent * local.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

# ── 벡터 / 쿼터니언 ────────────────────────────────────────────────────────────

def v_add(a: Vec, b: Vec) -> Vec: return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a: Vec, b: Vec) -> Vec: return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_scale(a: Vec, s: float) -> Vec: return (a[0] * s, a[1] * s, a[2] * s)
def v_dot(a: Vec, b: Vec) -> float: return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_len(a: Vec) -> float: return math.sqrt(v_dot(a, a))


def v_cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_norm(a: Vec) -> Optional[Vec]:
    n = v_len(a)
    return None if n < 1e-8 else (a[0] / n, a[1] / n, a[2] / n)


QI: Quat = (0.0, 0.0, 0.0, 1.0)


def q_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def q_inv(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def q_norm(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def q_rotate(q: Quat, v: Vec) -> Vec:
    u = (q[0], q[1], q[2])
    w = q[3]
    t = v_scale(v_cross(u, v), 2.0)
    return v_add(v_add(v, v_scale(t, w)), v_cross(u, t))


def q_axis_angle(axis: Vec, angle: float) -> Quat:
    s = math.sin(angle / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def q_between(a: Vec, b: Vec) -> Quat:
    """단위 벡터 a 를 b 로 보내는 최소 회전"""
    d = max(-1.0, min(1.0, v_dot(a, b)))
    if d > 1 - 1e-9:
        return QI
    if d < -1 + 1e-9:
        axis = v_cross(a, (1.0, 0.0, 0.0))
        if v_len(axis) < 1e-6:
            axis = v_cross(a, (0.0, 1.0, 0.0))
        return q_axis_angle(v_norm(axis), math.pi)
    axis = v_norm(v_cross(a, b))
    return q_axis_angle(axis, math.acos(d))


def q_align_flip(q: Quat, ref: Quat) -> Quat:
    """프레임 간 보간이 튀지 않게 ref 와 같은 반구로"""
    if sum(a * b for a, b in zip(q, ref)) < 0:
        return (-q[0], -q[1], -q[2], -q[3])
    return q


# ── 본 매핑 ────────────────────────────────────────────────────────────────────
# (본, 가장 가까운 구동 조상, 키포인트 from, 키포인트 to, 참조 방향용 자식 본, 비틀림)
# 참조 방향용 자식 본이 None 이면 (부모 본 → 이 본) 방향을 이어서 쓴다 (손가락 끝마디).
# 비틀림: (키포인트 a, 키포인트 b, 참조 본 a, 참조 본 b) — 손바닥 가로 방향

FINGERS = [("thumb", (1, 2, 3, 4)), ("index", (5, 6, 7, 8)), ("middle", (9, 10, 11, 12)),
           ("ring", (13, 14, 15, 16)), ("pinky", (17, 18, 19, 20))]


def _side_map(s: str) -> List[tuple]:
    h = "rh" if s == "r" else "lh"
    m = [
        (f"upperarm_{s}", None, f"{s}_shoulder", f"{s}_elbow", f"lowerarm_{s}", None),
        (f"lowerarm_{s}", f"upperarm_{s}", f"{s}_elbow", f"{s}_wrist", f"hand_{s}", None),
        (f"hand_{s}", f"lowerarm_{s}", f"{h}_0", f"{h}_9", f"middle_01_{s}",
         (f"{h}_5", f"{h}_17", f"index_01_{s}", f"pinky_01_{s}")),
    ]
    for name, (a, b, c, d) in FINGERS:
        m += [
            (f"{name}_01_{s}", f"hand_{s}", f"{h}_{a}", f"{h}_{b}", f"{name}_02_{s}", None),
            (f"{name}_02_{s}", f"{name}_01_{s}", f"{h}_{b}", f"{h}_{c}", f"{name}_03_{s}", None),
            (f"{name}_03_{s}", f"{name}_02_{s}", f"{h}_{c}", f"{h}_{d}", None, None),
        ]
    return m


BONE_MAP: List[tuple] = _side_map("r") + _side_map("l")
MAX_TWIST = math.radians(100)   # 손바닥 비틀림 한계 (회내·회외 합쳐 사람은 약 ±90°)
DRIVEN_BONES: List[str] = [b[0] for b in BONE_MAP]


class RefPose:
    """참조 포즈의 컴포넌트 공간 위치/회전과 부모 이름"""

    def __init__(self,
                 world_pos: Dict[str, Vec],
                 world_rot: Dict[str, Quat],
                 parent: Dict[str, Optional[str]]):
        self.pos = world_pos
        self.rot = world_rot
        self.parent = parent

    def has(self, bone: str) -> bool:
        return bone in self.pos


def character_basis(ref: RefPose) -> Tuple[Vec, Vec, Vec]:
    """캐릭터의 (오른쪽, 앞, 위) 축 — 컴포넌트 공간"""
    up = (0.0, 0.0, 1.0)
    right = v_sub(ref.pos["upperarm_r"], ref.pos["upperarm_l"])
    right = v_norm((right[0], right[1], 0.0))
    fwd = v_cross(right, up)
    return right, fwd, up


class Retargeter:
    def __init__(self, ref: RefPose):
        self.ref = ref
        self.right, self.fwd, self.up = character_basis(ref)
        self.map = [m for m in BONE_MAP if ref.has(m[0])]
        self.missing = [m[0] for m in BONE_MAP if not ref.has(m[0])]
        self.ref_dir: Dict[str, Vec] = {}
        self.ref_twist: Dict[str, Vec] = {}
        for bone, _anc, _a, _b, child, twist in self.map:
            if child and ref.has(child):
                d = v_sub(ref.pos[child], ref.pos[bone])
            else:
                p = ref.parent.get(bone)
                d = v_sub(ref.pos[bone], ref.pos[p]) if p else (0.0, 0.0, 1.0)
            self.ref_dir[bone] = v_norm(d) or (0.0, 0.0, 1.0)
            if twist and ref.has(twist[2]) and ref.has(twist[3]):
                self.ref_twist[bone] = v_norm(v_sub(ref.pos[twist[3]], ref.pos[twist[2]]))

    def to_component(self, v: Sequence[float]) -> Vec:
        """수어자 좌표 (X 오른쪽, Y 앞, Z 위) → 컴포넌트 공간"""
        return v_add(v_add(v_scale(self.right, v[0]), v_scale(self.fwd, v[1])), v_scale(self.up, v[2]))

    def solve_frame(self, joints: Dict[str, Sequence[float]]) -> Dict[str, Quat]:
        """키포인트(수어자 좌표) → 구동 본의 로컬 회전"""
        world = self.solve_frame_world(joints)
        local: Dict[str, Quat] = {}
        for bone, *_ in self.map:
            parent = self.ref.parent.get(bone)
            pw = self._world_of(parent, world) if parent else QI
            local[bone] = q_norm(q_mul(q_inv(pw), world[bone]))
        return local

    def solve_frame_world(self, joints: Dict[str, Sequence[float]]) -> Dict[str, Quat]:
        """키포인트(수어자 좌표) → 구동 본의 컴포넌트 공간 회전 (실시간 스트리밍용: 메시마다 로컬로 바꾼다)"""
        world: Dict[str, Quat] = {}
        ref = self.ref
        for bone, anc, ka, kb, _child, twist in self.map:
            # 조상에서 물려받은 회전 (조상 변화량을 참조 회전에 적용)
            if anc and anc in world:
                delta = q_mul(world[anc], q_inv(ref.rot[anc]))
            else:
                delta = QI
            base = q_mul(delta, ref.rot[bone])
            cur_dir = q_rotate(delta, self.ref_dir[bone])

            target = None
            if ka in joints and kb in joints:
                target = v_norm(self.to_component(v_sub(joints[kb], joints[ka])))
            q = q_mul(q_between(cur_dir, target), base) if target else base

            if target and bone in self.ref_twist:
                a, b = joints.get(twist[0]), joints.get(twist[1])
                if a is not None and b is not None:
                    want = self.to_component(v_sub(b, a))
                    have = q_rotate(q_mul(q, q_inv(ref.rot[bone])), self.ref_twist[bone])
                    # 본 축(target) 에 수직인 성분끼리 비교해 비틀림 각도 계산
                    want_p = v_norm(v_sub(want, v_scale(target, v_dot(want, target))))
                    have_p = v_norm(v_sub(have, v_scale(target, v_dot(have, target))))
                    if want_p and have_p:
                        ang = math.atan2(v_dot(v_cross(have_p, want_p), target), v_dot(have_p, want_p))
                        # 손 비틀림은 사람 팔뚝이 돌 수 있는 만큼만 (키포인트 잡음으로 손이 뒤집히는 것 방지)
                        ang = max(-MAX_TWIST, min(MAX_TWIST, ang))
                        q = q_mul(q_axis_angle(target, ang), q)

            q = q_norm(q)
            world[bone] = q
        return world

    def _world_of(self, bone: str, world: Dict[str, Quat]) -> Quat:
        """구동 안 되는 본의 새 월드 회전 = 가장 가까운 구동 조상의 변화량 × 참조 회전"""
        if bone in world:
            return world[bone]
        b = self.ref.parent.get(bone)
        while b is not None and b not in world:
            b = self.ref.parent.get(b)
        if b is None:
            return self.ref.rot[bone]
        return q_mul(q_mul(world[b], q_inv(self.ref.rot[b])), self.ref.rot[bone])

    def solve_motion(self, motion: dict) -> Dict[str, List[Quat]]:
        names = motion["joints"]
        tracks: Dict[str, List[Quat]] = {b[0]: [] for b in self.map}
        for row in motion["frames"]:
            joints = {n: (row[3 * i], row[3 * i + 1], row[3 * i + 2]) for i, n in enumerate(names)}
            for bone, q in self.solve_frame(joints).items():
                prev = tracks[bone][-1] if tracks[bone] else None
                tracks[bone].append(q_align_flip(q, prev) if prev else q)
        return tracks
