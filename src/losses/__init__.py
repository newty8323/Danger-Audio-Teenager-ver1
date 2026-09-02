"""Losses: focal-BCE + multi-label SupCon + combined objective (spec §5)."""

from losses.combined import CombinedLoss, LossConfig
from losses.focal import focal_bce_with_logits
from losses.supcon import jaccard_positive_mask, multilabel_supcon

__all__ = [
    "focal_bce_with_logits",
    "multilabel_supcon",
    "jaccard_positive_mask",
    "CombinedLoss",
    "LossConfig",
]
