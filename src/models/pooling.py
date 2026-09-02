"""MIL attention pooling (spec §5).

    a_t = softmax(u^T tanh(V h_t)) ;  z = Σ_t a_t h_t

Aggregates frame embeddings (B, T, D) into a clip embedding (B, D). The
attention weights ``a`` are returned as well — they double as temporal
localization for analysis (ref [mil-attn]).
"""

from __future__ import annotations

import torch
from torch import nn


class MILAttentionPooling(nn.Module):
    def __init__(self, dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.V = nn.Linear(dim, attn_dim, bias=False)  # V h_t
        self.u = nn.Linear(attn_dim, 1, bias=False)  # u^T (.)

    def forward(
        self, frames: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """frames: (B, T, D), optional bool mask: (B, T) with True = keep.

        Returns (z: (B, D), attention: (B, T)).
        """
        scores = self.u(torch.tanh(self.V(frames))).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=1)  # (B, T)
        if mask is not None:
            # A fully-masked row makes softmax([-inf,...]) NaN; zero it out.
            attention = torch.nan_to_num(attention, nan=0.0)
        z = torch.bmm(attention.unsqueeze(1), frames).squeeze(1)  # (B, D)
        return z, attention
