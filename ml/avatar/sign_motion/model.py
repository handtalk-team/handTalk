"""
GlossToMotion: 글로스 토큰 → 토큰별 길이 예측 → 길이만큼 펼쳐서 → 프레임별 관절 좌표 (FastSpeech 구조)

  인코더   토큰 임베딩 + 수어자 임베딩 → 트랜스포머
  길이     토큰마다 log(프레임 수) 예측 (학습 땐 정답 길이로 펼침)
  디코더   펼친 프레임 + 토큰 안에서의 진행도(0→1) → 트랜스포머 → (관절 50 × 3)

추론은 두 부분으로 ONNX 로 내보내 서버(onnxruntime)에서 돌린다. 펼치는 단계는 numpy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .data import NJ, PAD


def sinusoid(n: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class Encoder(nn.Module):
    def __init__(self, n_vocab: int, n_signer: int, d: int = 256, layers: int = 4, heads: int = 4):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(n_vocab, d, padding_idx=PAD)
        self.signer = nn.Embedding(n_signer, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.dur = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, tokens: torch.Tensor, signer: torch.Tensor, mask: torch.Tensor):
        """tokens (B, N), signer (B,), mask (B, N) True=패딩 → (hidden (B,N,d), log_dur (B,N))"""
        n = tokens.shape[1]
        x = self.tok(tokens) * math.sqrt(self.d) + sinusoid(n, self.d, tokens.device)[None] + self.signer(signer)[:, None]
        h = self.enc(x, src_key_padding_mask=mask)
        return h, self.dur(h).squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, n_signer: int, d: int = 256, layers: int = 6, heads: int = 4):
        super().__init__()
        self.d = d
        self.signer = nn.Embedding(n_signer, d)
        self.phase = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, NJ * 3)

    def forward(self, frames: torch.Tensor, phase: torch.Tensor, signer: torch.Tensor, mask: torch.Tensor):
        """frames (B,T,d) 펼친 토큰 은닉, phase (B,T,2) [진행도, 토큰 길이/30], mask (B,T) → (B,T,NJ,3)"""
        t = frames.shape[1]
        x = frames + self.phase(phase) + sinusoid(t, self.d, frames.device)[None] + self.signer(signer)[:, None]
        y = self.out(self.dec(x, src_key_padding_mask=mask))
        return y.view(y.shape[0], t, NJ, 3)


def expand(h: torch.Tensor, durs: torch.Tensor, t_max: int):
    """토큰 은닉을 길이만큼 펼친다 (학습용, 배치). → frames (B,T,d), phase (B,T,2), mask (B,T)"""
    B, N, d = h.shape
    frames = h.new_zeros(B, t_max, d)
    phase = h.new_zeros(B, t_max, 2)
    mask = torch.ones(B, t_max, dtype=torch.bool, device=h.device)
    for b in range(B):
        idx = torch.repeat_interleave(torch.arange(N, device=h.device), durs[b])[:t_max]
        T = idx.shape[0]
        frames[b, :T] = h[b, idx]
        starts = torch.cumsum(durs[b], 0) - durs[b]
        within = torch.arange(T, device=h.device) - starts[idx]
        L = durs[b][idx].clamp(min=1).float()
        phase[b, :T, 0] = (within.float() + 0.5) / L
        phase[b, :T, 1] = L / 30.0
        mask[b, :T] = False
    return frames, phase, mask


class GlossToMotion(nn.Module):
    def __init__(self, n_vocab: int, n_signer: int, d: int = 256):
        super().__init__()
        self.encoder = Encoder(n_vocab, n_signer, d)
        self.decoder = Decoder(n_signer, d)

    def forward(self, tokens, signer, tok_mask, durs, t_max):
        h, log_dur = self.encoder(tokens, signer, tok_mask)
        frames, phase, mask = expand(h, durs, t_max)
        return self.decoder(frames, phase, signer, mask), log_dur, mask
