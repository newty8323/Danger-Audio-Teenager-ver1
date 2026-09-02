"""Focal binary cross-entropy for multi-label tagging (spec §5, ref focal loss).

    FL = (1 - p_t)^gamma * BCE(logits, targets)

Computed from logits via ``binary_cross_entropy_with_logits`` for numerical
stability. gamma=0 recovers plain BCE. Optional per-class ``alpha`` weighting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """logits, targets: (B, C). Returns scalar (mean/sum) or (B, C) if reduction='none'."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)  # prob of the true class
    focal = (1.0 - p_t).clamp(min=0.0).pow(gamma) * bce

    if alpha is not None:
        a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal = a_t * focal

    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    if reduction == "none":
        return focal
    raise ValueError(f"unknown reduction {reduction!r}")
