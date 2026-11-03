"""
몸 비율 차이 보정: 얼굴 근처 수어에서 손이 아바타 얼굴의 같은 자리에 닿게 팔을 다시 푼다 (두 관절 IK).

리타깃은 팔 마디의 '방향'만 옮긴다. 그런데 메타휴먼은 사람(AI Hub 수어자)보다 어깨 관절 사이가 좁고 목이 길어서,
같은 방향으로 팔을 들면 손이 얼굴보다 낮게 온다 ('죄송'에서 이마에 대야 할 손이 턱에 머물던 원인).
손이 수어자 얼굴 가까이 있을 때는 '아바타 코 + (수어자 코→손목)' 을 손목 목표로 삼고,
어깨·팔꿈치·손목 두 관절 IK 로 팔꿈치와 손목 위치를 새로 구한 뒤, 그 방향을 수어자 좌표로 되돌려 리타깃에 넘긴다.
손 모양(손가락 키포인트)은 손목과 함께 평행 이동만 한다.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

NEAR = 0.35      # 손-코 거리(수어자 어깨너비 단위)가 이보다 가까우면 얼굴 기준 100%
FAR = 0.85       # 이보다 멀면 보정 없음 (원래 리타깃 그대로)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


class ArmIK:
    def __init__(self, ref_pos: Dict[str, Sequence[float]], right, fwd, up):
        P = {k: np.asarray(v, dtype=np.float64) for k, v in ref_pos.items()}
        self.B = np.stack([np.asarray(right), np.asarray(fwd), np.asarray(up)], axis=1)   # 수어자 → 컴포넌트
        self.side = {}
        for s in ("r", "l"):
            sh, el, wr = P[f"upperarm_{s}"], P[f"lowerarm_{s}"], P[f"hand_{s}"]
            self.side[s] = (sh, np.linalg.norm(el - sh), np.linalg.norm(wr - el))
        L = self.side["r"][1] + self.side["r"][2]
        self.k = L / 1.55   # 수어자 팔 길이(위팔 0.83 + 아래팔 0.72 어깨너비) → 아바타 cm
        head = P["head"]
        # 메타휴먼 head 본은 두개골 아래쪽이라 코끝은 그보다 앞·위 (대략값)
        self.nose = head + self.B @ np.array([0.0, 10.0, 5.0])

    def to_comp(self, v):
        return self.B @ v

    def to_signer(self, v):
        return self.B.T @ v

    def solve(self, kp: np.ndarray, J: Dict[str, int]) -> None:
        """kp: (관절, 3) 수어자 좌표 한 프레임. 제자리에서 팔·손 키포인트를 고친다."""
        nose_s = kp[J["nose"]]
        for s, h in (("r", "rh"), ("l", "lh")):
            hand_c = kp[J[f"{h}_9"]] if np.abs(kp[J[f"{h}_9"]]).sum() > 0 else kp[J[f"{s}_wrist"]]
            d = np.linalg.norm(hand_c - nose_s)
            a = float(np.clip((FAR - d) / (FAR - NEAR), 0.0, 1.0))
            if a <= 0:
                continue
            sh_s, el_s, wr_s = kp[J[f"{s}_shoulder"]], kp[J[f"{s}_elbow"]], kp[J[f"{s}_wrist"]]
            sh, L1, L2 = self.side[s]
            # 지금 리타깃이 만들 손목 위치 (방향 그대로) vs 얼굴 기준 목표
            e0 = sh + _unit(self.to_comp(el_s - sh_s)) * L1
            w0 = e0 + _unit(self.to_comp(wr_s - el_s)) * L2
            wf = self.nose + self.to_comp(wr_s - nose_s) * self.k
            T = w0 * (1 - a) + wf * a
            # 두 관절 IK (팔꿈치는 원래 굽힘 방향 쪽으로)
            v = T - sh
            dist = float(np.clip(np.linalg.norm(v), abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3))
            n = _unit(v)
            T = sh + n * dist
            cos_a = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)
            ang = np.arccos(np.clip(cos_a, -1, 1))
            pole = e0 - sh
            pole = _unit(pole - n * np.dot(pole, n))
            if np.linalg.norm(pole) < 1e-6:
                pole = _unit(np.cross(n, self.B[:, 1]))
            E = sh + (n * np.cos(ang) + pole * np.sin(ang)) * L1
            # 수어자 좌표로 되돌려 방향만 넘긴다
            new_el = sh_s + self.to_signer(_unit(E - sh)) * np.linalg.norm(el_s - sh_s)
            new_wr = new_el + self.to_signer(_unit(T - E)) * np.linalg.norm(wr_s - el_s)
            delta = new_wr - wr_s
            kp[J[f"{s}_elbow"]] = new_el
            kp[J[f"{s}_wrist"]] = new_wr
            for i in range(21):
                j = J[f"{h}_{i}"]
                if np.abs(kp[j]).sum() > 0:
                    kp[j] = kp[j] + delta


def apply(kp: np.ndarray, joints: Sequence[str], ik: ArmIK) -> np.ndarray:
    J = {n: i for i, n in enumerate(joints)}
    out = kp.astype(np.float64).copy()
    for t in range(len(out)):
        ik.solve(out[t], J)
    return out.astype(np.float32)
