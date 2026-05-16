"""
BiGRU + Self-Attention Sign Language Recognition Model
=======================================================

Why this architecture for 10 words × 300 samples
--------------------------------------------------
- MediaPipe already extracts 63-D spatial features → CNN not needed
- GRU has fewer parameters than LSTM (less overfitting on small data)
- Bidirectional: forward pass captures onset, backward pass captures release
- Self-attention: learns which frames in the 2-second window matter most
  (e.g., the peak of the gesture matters more than the transition frames)
- ~120K parameters total → trains in minutes on a CPU

Input  : (batch, T=60, D=77)   T can vary at inference via pack_padded_sequence
Output : (batch, num_classes)  raw logits

Export to ONNX
--------------
    python -m ml.training.model --export
Produces: ml/models/sign_recognizer.onnx
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head scaled dot-product attention over the time axis."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        q = self.query(x)                       # (B, T, H)
        k = self.key(x)                         # (B, T, H)
        v = self.value(x)                       # (B, T, H)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, T, T)
        weights = F.softmax(scores, dim=-1)     # (B, T, T)
        context = torch.bmm(weights, v)         # (B, T, H)
        return context                          # (B, T, H)


class SignRecognizer(nn.Module):
    """
    Bidirectional GRU + Self-Attention classifier.

    Architecture
    ------------
    Input projection  : 77 → 128
    BiGRU × 2 layers : 128 → 256 hidden (2 × 128)
    Self-attention    : 256 → 256
    Global avg pool   : T → 1
    Classifier        : 256 → num_classes
    """

    def __init__(
        self,
        input_dim: int = 77,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2     # bidirectional doubles hidden size

        self.attn = SelfAttention(gru_out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(gru_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 77)

        Returns
        -------
        logits : (B, num_classes)
        """
        # Project to hidden dim
        h = self.input_proj(x)              # (B, T, 128)

        # BiGRU
        gru_out, _ = self.bigru(h)          # (B, T, 256)

        # Self-attention
        ctx = self.attn(gru_out)            # (B, T, 256)
        ctx = self.norm(gru_out + ctx)      # residual connection

        # Global average pooling over time
        pooled = ctx.mean(dim=1)            # (B, 256)
        pooled = self.drop(pooled)

        return self.classifier(pooled)      # (B, num_classes)

    # ── ONNX export helper ────────────────────────────────────────

    def export_onnx(self, path: str = "ml/models/sign_recognizer.onnx") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.eval()
        dummy = torch.zeros(1, 60, 77)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {1: "time"}},   # variable T
            opset_version=17,
        )
        print(f"Exported ONNX model → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export untrained model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    model = SignRecognizer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.export:
        model.export_onnx()
