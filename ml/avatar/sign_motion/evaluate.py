"""
학습한 모델들을 같은 검증 문장(처음 보는 수어자)으로 비교한다. 정답 길이로 펼쳐 손·팔 위치 오차(어깨너비 단위)를 잰다.

    python -m ml.avatar.sign_motion.evaluate --data C:\\handtalk\\sentdata\\all\\sentence_motion_npz ^
        --signers REAL17,REAL18 --models C:\\handtalk\\models\\sign_motion C:\\handtalk\\models\\sign_motion_v2
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from .data import SPECIAL, load_sentences, tokens_and_durations
from .model import GlossToMotion

HANDS = slice(8, None)
WRISTS = [3, 6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--signers", default="REAL17,REAL18")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    want = set(args.signers.split(","))
    samples = [s for s in load_sentences(args.data) if s.signer in want]
    random.seed(1)
    random.shuffle(samples)
    samples = samples[:args.n]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for folder in args.models:
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        vocab = meta["vocab"]
        model = GlossToMotion(len(vocab) + len(SPECIAL), len(meta["signers"])).to(dev).eval()
        model.load_state_dict(torch.load(os.path.join(folder, "best.pt"), map_location=dev)["model"])
        sid = meta["signers"].get("REAL02", 0)
        err_all, err_wrist, err_hand, skipped = [], [], [], 0
        with torch.no_grad():
            for s in samples:
                if any(g not in vocab for g in s.glosses):
                    skipped += 1
                    continue
                t, d = tokens_and_durations(s, vocab)
                tok = torch.from_numpy(t)[None].to(dev)
                pred, _, _ = model(tok, torch.tensor([sid], device=dev), tok == 0, torch.from_numpy(d)[None].to(dev), len(s.motion))
                e = np.linalg.norm(pred[0].cpu().numpy() - s.motion, axis=-1)
                err_all.append(e.mean())
                err_wrist.append(e[:, WRISTS].mean())
                err_hand.append(e[:, HANDS].mean())
        print(f"{folder}: n={len(err_all)} skipped={skipped}  all={np.mean(err_all):.3f}  "
              f"wrist={np.mean(err_wrist):.3f}  hand_pts={np.mean(err_hand):.3f}")


if __name__ == "__main__":
    main()
