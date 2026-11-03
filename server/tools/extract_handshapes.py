"""
AI Hub 단어 영상에서 손 모양 본(손가락 방향)을 뽑아 server/ml/models/handshapes.json 으로 저장한다.
모양마다 그 모양이 가장 뚜렷한 단어 영상 한 프레임의 오른손 손가락 방향만 저장 (AI Hub 유래라 저장소에 올리지 않음).

    python tools/extract_handshapes.py C:/handtalk/sentdata/all/word_motion_npz
    python tools/extract_handshapes.py --lexicon C:/handtalk/sentdata/all/word_motion_npz

--lexicon: 모든 단어의 손 모양 사전 (server/ml/models/handshape_lexicon.npz).
단어 영상에서 손을 든 구간을 손 모양이 바뀌는 곳에서 최대 3구간으로 나누고, 구간마다 손가락 방향 중앙값 하나와
그 구간이 손을 든 시간 중 어디쯤인지(0~1)만 저장한다. 궤적이나 프레임은 저장하지 않는다.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.avatar.handshape import finger_dirs  # noqa: E402

SOURCES = {          # 모양: (단어, 프레임 고르는 기준)
    "ok": ("죄송", lambda h, s: -np.linalg.norm(h[4] - h[8]) / s),
    "pinky": ("괜찮다", lambda h, s: (np.linalg.norm(h[20] - h[0]) - np.linalg.norm(h[8] - h[0]) - np.linalg.norm(h[12] - h[0])) / s),
    "thumb": ("좋다", lambda h, s: (np.linalg.norm(h[4] - h[17]) * 1.0 - np.linalg.norm(h[8] - h[0]) - np.linalg.norm(h[12] - h[0])) / s),
    "v": ("병원", lambda h, s: (np.linalg.norm(h[8] - h[0]) + np.linalg.norm(h[12] - h[0]) - np.linalg.norm(h[16] - h[0]) - np.linalg.norm(h[20] - h[0])) / s),
    "fist": ("건강", lambda h, s: -sum(np.linalg.norm(h[t] - h[0]) for t in (8, 12, 16, 20)) / s),
    "l": ("싫다", lambda h, s: (np.linalg.norm(h[4] - h[8]) + np.linalg.norm(h[8] - h[0]) - np.linalg.norm(h[12] - h[0]) - np.linalg.norm(h[16] - h[0])) / s),
}


def main(folder: str, out: str) -> None:
    clips = {}
    for p in glob.glob(os.path.join(folder, "*.npz")):
        z = np.load(p)
        g = json.loads(str(z["glosses"]))[0]["gloss"]
        clips[g] = z
    shapes = {}
    for name, (word, score) in SOURCES.items():
        z = clips[word]
        J = [str(j) for j in z["joints"]]
        fr = z["frames"].astype(np.float64)
        rh = fr[:, [J.index(f"rh_{i}") for i in range(21)]]
        best, bi = -1e9, 0
        for t, h in enumerate(rh):
            s = np.linalg.norm(h[9] - h[0]) + 1e-9
            if np.abs(h).sum() == 0:
                continue
            v = score(h, s)
            if v > best:
                best, bi = v, t
        shapes[name] = finger_dirs(rh[bi]).round(4).tolist()
        print(f"{name}: {word} frame {bi}/{len(rh)} score {best:.2f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(shapes, f)
    print("->", out)


HANDS = ("rh", "lh")
MAX_SEG = 3


def _segments(F: np.ndarray, min_len: int):
    """F: (T, 60) 손가락 방향. 구간 내 분산 합이 가장 작은 K(1~3)구간 분할. [(시작, 끝)] 목록 반환."""
    T = len(F)
    c = np.concatenate([np.zeros((1, F.shape[1])), np.cumsum(F, 0)])
    c2 = np.concatenate([[0], np.cumsum((F * F).sum(1))])

    def cost(a, b):   # [a, b)
        n = b - a
        m = c[b] - c[a]
        return c2[b] - c2[a] - (m * m).sum() / n

    best = {1: (cost(0, T), [(0, T)])}
    for K in (2, 3):
        if T < K * min_len:
            break
        if K == 2:
            cands = [(cost(0, i) + cost(i, T), [(0, i), (i, T)]) for i in range(min_len, T - min_len + 1)]
        else:
            cands = [(cost(0, i) + cost(i, j) + cost(j, T), [(0, i), (i, j), (j, T)])
                     for i in range(min_len, T - 2 * min_len + 1) for j in range(i + min_len, T - min_len + 1)]
        best[K] = min(cands, key=lambda x: x[0])
    K = 1
    for k in (2, 3):   # 나눌 때마다 30% 넘게 줄어야 한 구간 더
        if k in best and best[k][0] < best[K][0] * 0.7:
            K = k
    return best[K][1]


def build_lexicon(folder: str, out: str) -> None:
    gl, hand, t0, t1, dirs = [], [], [], [], []
    names = []
    for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        try:
            z = np.load(p)
            g = json.loads(str(z["glosses"]))[0]["gloss"]
            J = [str(j) for j in z["joints"]]
            fr = z["frames"].astype(np.float64)
        except Exception:
            continue
        if g in names:
            continue
        nose = fr[:, J.index("nose")]
        added = False
        for hi, h in enumerate(HANDS):
            H = fr[:, [J.index(f"{h}_{i}") for i in range(21)]]
            up = np.where((np.abs(H).sum((1, 2)) > 0) & (H[:, 9, 2] - nose[:, 2] > -1.0))[0]
            if len(up) < 3:
                continue
            a, b = up[0], up[-1] + 1
            idx = [t for t in range(a, b) if np.abs(H[t]).sum() > 0]
            F = np.stack([finger_dirs(H[t]) for t in idx]).reshape(len(idx), -1)
            if len(F) >= 3:
                F = np.concatenate([F[:1], (F[:-2] + F[1:-1] + F[2:]) / 3, F[-1:]])
            segs = _segments(F, max(3, int(len(F) * 0.2)))
            for s0, s1 in segs:
                d = np.median(F[s0:s1].reshape(-1, 5, 4, 3), 0)
                d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
                if not added:
                    names.append(g)
                    added = True
                gl.append(len(names) - 1); hand.append(hi)
                t0.append(s0 / len(F)); t1.append(s1 / len(F)); dirs.append(d)
    np.savez_compressed(out, glosses=np.array(names), gloss=np.array(gl, np.int32), hand=np.array(hand, np.int8),
                        t0=np.array(t0, np.float32), t1=np.array(t1, np.float32), dirs=np.array(dirs, np.float16))
    print(f"{len(names)} words, {len(gl)} handshape segments -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "--lexicon":
        build_lexicon(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else r"C:\handtalk\models\handshape_lexicon.npz")
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else r"C:\handtalk\models\handshapes.json")
