"""
GlossToMotion 학습 (RTX 3070)

    C:\\handtalk\\mlenv\\Scripts\\python.exe -m ml.avatar.sign_motion.train ^
        --sentences C:\\handtalk\\sentdata\\motion --words C:\\handtalk\\motions ^
        --lexicon client\\assets\\lexicon.json --out C:\\handtalk\\models\\sign_motion

결과 (--out):
  encoder.onnx, decoder.onnx   서버 추론용 (server/app/services/avatar/motion_synth.py 의 ModelGenerator)
  meta.json                    단어 → 토큰 id, 수어자 → id, fps, 관절 이름
  last.pt                      이어서 학습할 때
학습 데이터(AI Hub)와 모델은 저장소에 올리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import FPS, JOINTS, PAD, SPECIAL, build_vocab, load_sentences, load_words, tokens_and_durations
from .model import GlossToMotion, sinusoid

# 손목·손가락은 오차를 더 크게 본다 (수어의 뜻은 손에 있다)
W_JOINT = torch.ones(len(JOINTS))
W_JOINT[8:] = 2.0
W_JOINT[[3, 6]] = 2.0


def make_batch(items, device):
    B = len(items)
    N = max(len(t) for t, *_ in items)
    T = max(len(m) for *_, m in items)
    tokens = torch.full((B, N), PAD, dtype=torch.long)
    durs = torch.zeros(B, N, dtype=torch.long)
    signer = torch.zeros(B, dtype=torch.long)
    motion = torch.zeros(B, T, len(JOINTS), 3)
    for b, (t, d, s, m) in enumerate(items):
        tokens[b, :len(t)] = torch.from_numpy(t)
        durs[b, :len(d)] = torch.from_numpy(d)
        signer[b] = s
        motion[b, :len(m)] = torch.from_numpy(m)
    return tokens.to(device), durs.to(device), signer.to(device), motion.to(device), T


def augment(m: np.ndarray) -> np.ndarray:
    """작은 크기·회전 변화 (수어자 체형/카메라 차이)"""
    s = np.random.uniform(0.92, 1.08)
    a = np.random.uniform(-0.08, 0.08)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)
    return (m @ R.T) * s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default=None)
    ap.add_argument("--words", default=None)
    ap.add_argument("--lexicon", default=None, help="client/assets/lexicon.json (clip → 글로스)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val-signers", default="", help="검증용으로 뺄 수어자 (예: REAL17,REAL18)")
    ap.add_argument("--export-only", action="store_true", help="--out 의 best.pt 를 ONNX 로만 내보냄")
    args = ap.parse_args()
    if args.export_only:
        with open(os.path.join(args.out, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        model = GlossToMotion(len(meta["vocab"]) + len(SPECIAL), len(meta["signers"]))
        model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location="cpu")["model"])
        export_onnx(model.eval(), args.out)
        return

    lex = None
    if args.lexicon and os.path.exists(args.lexicon):
        with open(args.lexicon, encoding="utf-8") as f:
            lex = {e["clip"]: e["gloss"] for e in json.load(f)}
    samples = []
    if args.sentences:
        sen = load_sentences(args.sentences)
        print(f"sentences {len(sen)}")
        samples += sen
    if args.words:
        wd = load_words(args.words, lex)
        print(f"words {len(wd)}")
        samples += wd
    samples = [s for s in samples if 8 <= len(s.motion) <= 300]
    vocab = build_vocab(samples)
    signers = {s: i for i, s in enumerate(sorted({x.signer for x in samples}))}
    items, held = [], []
    val_signers = {v for v in args.val_signers.split(",") if v}
    for s in samples:
        t, d = tokens_and_durations(s, vocab)
        if s.signer in val_signers:   # 처음 보는 사람이라 수어자 임베딩은 시연용(REAL02)으로
            held.append((t, d, signers.get("REAL02", 0), s.motion))
        else:
            items.append((t, d, signers[s.signer], s.motion))
    random.seed(0)
    random.shuffle(items)
    if held:
        # 학습에 안 쓴 수어자로 검증 (처음 보는 사람의 동작을 얼마나 잘 맞히나). 너무 많으면 일부만.
        random.shuffle(held)
        val, train = held[:1000], items
    else:
        n_val = max(len(items) // 50, 1)
        val, train = items[:n_val], items[n_val:]
    print(f"vocab {len(vocab)}  signers {len(signers)}  train {len(train)}  val {len(val)}")

    os.makedirs(args.out, exist_ok=True)
    meta = {"vocab": vocab, "signers": signers, "fps": FPS, "joints": JOINTS, "special": SPECIAL}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlossToMotion(len(vocab) + len(SPECIAL), len(signers)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * (len(train) // args.batch + 1))
    start = 0
    ckpt = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"])
        start = st["epoch"] + 1
    wj = W_JOINT.to(dev)[None, None, :]

    def loss_fn(batch, train_mode: bool):
        tokens, durs, signer, motion, T = batch
        pred, log_dur, mask = model(tokens, signer, tokens == PAD, durs, T)
        valid = (~mask).float()[..., None]
        pos = ((pred - motion).abs().mean(-1) * wj * valid).sum() / (valid.sum() * len(JOINTS))
        vel = (((pred[:, 1:] - pred[:, :-1]) - (motion[:, 1:] - motion[:, :-1])).abs().mean(-1)
               * valid[:, 1:]).sum() / (valid[:, 1:].sum() * len(JOINTS))
        tm = (tokens != PAD).float()
        dur = (((log_dur - torch.log1p(durs.float())) ** 2) * tm).sum() / tm.sum()
        return pos + 0.5 * vel + 0.1 * dur, pos.item()

    best = 1e9
    for ep in range(start, args.epochs):
        model.train()
        random.shuffle(train)
        t0, tot, n = time.time(), 0.0, 0
        for i in range(0, len(train), args.batch):
            chunk = [(t, d, s, augment(m)) for t, d, s, m in train[i:i + args.batch]]
            loss, _ = loss_fn(make_batch(chunk, dev), True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        model.eval()
        with torch.no_grad():
            vp = [loss_fn(make_batch(val[i:i + args.batch], dev), False)[1] for i in range(0, len(val), args.batch)]
        v = float(np.mean(vp))
        print(f"epoch {ep} train {tot / max(n, 1):.4f} val_pos {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "best.pt"))
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev)["model"])
    export_onnx(model.cpu().eval(), args.out)


class _EncWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m.encoder

    def forward(self, tokens, signer):
        h, log_dur = self.m(tokens, signer, tokens == PAD)
        return h, log_dur


class _DecWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m.decoder

    def forward(self, frames, phase, signer):
        mask = torch.zeros(frames.shape[:2], dtype=torch.bool)
        return self.m(frames, phase, signer, mask)


def export_onnx(model: GlossToMotion, out: str) -> None:
    # 빠른 경로(fused MHA)는 길이를 상수로 박아 버려 다른 길이 입력에서 깨진다
    torch.backends.mha.set_fastpath_enabled(False)
    tok = torch.tensor([[1, 4, 2, 5, 3]])
    sg = torch.tensor([0])
    torch.onnx.export(_EncWrap(model), (tok, sg), os.path.join(out, "encoder.onnx"),
                      input_names=["tokens", "signer"], output_names=["hidden", "log_dur"],
                      dynamic_axes={"tokens": {1: "n"}, "hidden": {1: "n"}, "log_dur": {1: "n"}}, opset_version=17,
                      dynamo=True)
    d = model.decoder.d
    torch.onnx.export(_DecWrap(model), (torch.zeros(1, 40, d), torch.zeros(1, 40, 2), sg),
                      os.path.join(out, "decoder.onnx"), input_names=["frames", "phase", "signer"],
                      output_names=["joints"], dynamic_axes={"frames": {1: "t"}, "phase": {1: "t"}, "joints": {1: "t"}},
                      opset_version=17, dynamo=True)
    print(f"exported onnx -> {out}")


if __name__ == "__main__":
    main()
