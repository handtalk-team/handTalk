"""
Sign Language Recognition — Transformer Encoder
================================================

Architecture
------------
Input projection  : 136 → d_model (256)
Positional encoding: learned (max 120 frames)
Transformer Encoder: N layers × (Multi-head Attention + FFN)
Global avg pool   : T → 1
Classifier        : d_model → num_classes

Why Transformer over BiGRU for 800+ samples
--------------------------------------------
- Self-attention captures simultaneous finger/wrist relationships
  across the entire sequence (BiGRU is sequential)
- Parallel training — faster on GPU
- Scales better as data grows

Input  : (batch, T, 272)   위치(136) + 속도(136)
Output : (batch, num_classes)

Export to ONNX
--------------
    python -m ml.training.model --export
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn as nn


class _LearnedPE(nn.Module):
    """Learned positional encoding (more flexible than sinusoidal for short seqs)."""
    def __init__(self, d_model: int, max_len: int = 120) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return x + self.pe(pos)


class SignRecognizer(nn.Module):
    """
    Transformer Encoder classifier for sign language recognition.

    Parameters
    ----------
    input_dim   : feature dimension per frame (default 136)
    d_model     : transformer hidden size (default 256)
    nhead       : number of attention heads (default 4)
    num_layers  : number of encoder layers (default 3)
    dim_ff      : feedforward hidden size (default 512)
    num_classes : output classes
    dropout     : dropout rate
    """

    def __init__(
        self,
        input_dim:   int = 272,
        d_model:     int = 256,
        nhead:       int = 4,
        num_layers:  int = 3,
        dim_ff:      int = 512,
        num_classes: int = 10,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = _LearnedPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 136)

        Returns
        -------
        logits : (B, num_classes)
        """
        h = self.input_proj(x)        # (B, T, d_model)
        h = self.pos_enc(h)           # add positional info
        h = self.encoder(h)           # (B, T, d_model)
        h = h.mean(dim=1)             # global average pool → (B, d_model)
        return self.classifier(h)     # (B, num_classes)

    def export_onnx(self, path: str = "ml/models/sign_recognizer.onnx") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.eval()
        dummy = torch.zeros(1, 60, 272)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {1: "time"}},
            opset_version=17,
            dynamo=False,
        )
        print(f"Exported ONNX → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    model = SignRecognizer()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded: {args.checkpoint}")

    if args.export:
        model.export_onnx()
