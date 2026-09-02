"""Classifier and projection heads (spec §5).

    classifier:  z -> FC512 -> GELU -> Dropout(.3) -> FC(C)   (logits)
    projection:  z -> FC256 -> L2-normalize                   (SupCon only)

The classifier outputs raw logits (sigmoid is applied by the loss / at
inference) for numerically stable focal-BCE. The projection head is used only
for the contrastive loss and is dropped at inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(
        self, dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits (B, C)


class ProjectionHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, proj_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(z), dim=-1)  # unit-norm (B, proj_dim)
