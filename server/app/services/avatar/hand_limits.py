"""
사람 손이 실제로 할 수 있는 범위로 키포인트를 바로잡는다 (리타깃 전에 적용).

AI Hub 키포인트는 영상에서 추정한 값이라 손가락이 손등 쪽으로 꺾이거나(과신전), 옆으로 휘거나,
손목이 반대로 꺾인 프레임이 섞여 있고, 그걸 배운 모델 출력에도 그대로 나온다. 그대로 뼈대에 입히면
메타휴먼 손가락이 사람 관절로는 불가능하게 꺾인다.

  손가락 2·3마디(PIP, DIP)  경첩 관절: 옆으로 휘지 않고, 손등 쪽으로는 거의 못 젖히고(-5°), 최대 110° 굽힘
  손가락 1마디(MCP)        손등 쪽 최대 -20°, 손바닥 쪽 최대 90°, 옆으로 ±20°
  엄지                     각 마디 최대 80° 굽힘, 손등 쪽 최대 -15°
  손목                     팔뚝 방향과 손 방향 사이 최대 75°
마디 길이는 유지한다. 손바닥 방향은 AI Hub 데이터에서 확인: 오른손은 cross(검지MCP-손목, 새끼MCP-손목) 쪽,
왼손은 그 반대쪽이 손바닥.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
THUMB = (1, 2, 3, 4)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _limit_dir(u: np.ndarray, v: np.ndarray, palm: np.ndarray, side: np.ndarray,
               flex_max: float, ext_max: float, lat_max: float) -> np.ndarray:
    """앞 마디 방향 u 기준으로 다음 마디 방향 v 를 굽힘(손바닥 쪽)/젖힘/옆 범위 안으로. 단위벡터 반환."""
    # u 에 수직인 굽힘 축 성분: 손바닥 쪽(palm)과 옆(side)
    p = _unit(palm - u * np.dot(palm, u))
    s = _unit(np.cross(u, p)) if lat_max < 90 else side
    a, b, c = np.dot(v, u), np.dot(v, p), np.dot(v, s)
    flex = np.degrees(np.arctan2(b, a))            # + 손바닥 쪽으로 굽힘, - 손등 쪽으로 젖힘
    lat = np.degrees(np.arctan2(c, np.hypot(a, b)))
    flex = float(np.clip(flex, -ext_max, flex_max))
    lat = float(np.clip(lat, -lat_max, lat_max))
    fr, lr = np.radians(flex), np.radians(lat)
    return _unit(np.cos(lr) * (np.cos(fr) * u + np.sin(fr) * p) + np.sin(lr) * s)


def limit_hand(kp: np.ndarray, J: Dict[str, int], h: str, elbow: str, wrist_joint: str) -> None:
    """kp: (관절, 3) 한 프레임. 제자리에서 고친다. h = 'rh' | 'lh'"""
    idx = lambda i: J[f"{h}_{i}"]
    w, i5, i17 = kp[idx(0)], kp[idx(5)], kp[idx(17)]
    palm = _unit(np.cross(i5 - w, i17 - w)) * (1.0 if h == "rh" else -1.0)
    side = _unit(i5 - i17)

    # 손목: 팔뚝 방향과 손 방향(손목→중지 MCP) 사이 각도 제한. 손 전체를 손목 기준으로 돌린다.
    fore = _unit(kp[J[wrist_joint]] - kp[J[elbow]])
    hand_dir = _unit(kp[idx(9)] - w)
    ang = np.degrees(np.arccos(np.clip(np.dot(fore, hand_dir), -1, 1)))
    if ang > 75:
        axis = _unit(np.cross(fore, hand_dir))
        th = np.radians(ang - 75)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) - np.sin(th) * K + (1 - np.cos(th)) * (K @ K)   # 초과분만큼 되돌림
        for j in range(1, 21):
            kp[idx(j)] = w + R @ (kp[idx(j)] - w)
        i5, i17 = kp[idx(5)], kp[idx(17)]
        palm = _unit(np.cross(i5 - w, i17 - w)) * (1.0 if h == "rh" else -1.0)
        side = _unit(i5 - i17)

    for chain in FINGERS:
        prev = _unit(kp[idx(chain[0])] - w)
        for k in range(1, 4):
            a, b = idx(chain[k - 1]), idx(chain[k])
            seg = kp[b] - kp[a]
            L = np.linalg.norm(seg)
            if L < 1e-6:
                continue
            if k == 1:   # MCP
                d = _limit_dir(prev, _unit(seg), palm, side, flex_max=90, ext_max=20, lat_max=20)
            else:        # PIP, DIP: 경첩
                d = _limit_dir(prev, _unit(seg), palm, side, flex_max=110, ext_max=5, lat_max=0)
            new_b = kp[a] + d * L
            delta = new_b - kp[b]
            for j in chain[k:]:   # 끝쪽 마디는 같이 옮긴다
                kp[idx(j)] += delta
            prev = d

    prev = _unit(kp[idx(THUMB[0])] - w)
    for k in range(1, 4):
        a, b = idx(THUMB[k - 1]), idx(THUMB[k])
        seg = kp[b] - kp[a]
        L = np.linalg.norm(seg)
        if L < 1e-6:
            continue
        d = _limit_dir(prev, _unit(seg), palm, side, flex_max=80, ext_max=15, lat_max=35)
        delta = kp[a] + d * L - kp[b]
        for j in THUMB[k:]:
            kp[idx(j)] += delta
        prev = d


def apply(kp: np.ndarray, joints: Sequence[str]) -> np.ndarray:
    """kp (T, 관절, 3) → 손 관절 범위를 지킨 복사본"""
    J = {n: i for i, n in enumerate(joints)}
    out = kp.copy()
    for t in range(len(out)):
        limit_hand(out[t], J, "rh", "r_elbow", "r_wrist")
        limit_hand(out[t], J, "lh", "l_elbow", "l_wrist")
    return out
