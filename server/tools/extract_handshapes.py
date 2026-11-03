"""
AI Hub 단어 영상에서 손 모양 본(손가락 방향)을 뽑아 server/ml/models/handshapes.json 으로 저장한다.
모양마다 그 모양이 가장 뚜렷한 단어 영상 한 프레임의 오른손 손가락 방향만 저장 (AI Hub 유래라 저장소에 올리지 않음).

    python tools/extract_handshapes.py C:/handtalk/sentdata/all/word_motion_npz
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


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else r"C:\handtalk\models\handshapes.json")
