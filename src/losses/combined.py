"""Combined training objective (spec §5).

    L = focal-BCE(gamma=2) + mu * SupCon,  mu=0.2 (search 0.1-0.5)

Returns the scalar total plus a detached breakdown for logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from losses.focal import focal_bce_with_logits
from losses.supcon import multilabel_supcon


@dataclass(frozen=True)
class LossConfig:
    gamma: float = 2.0
    mu: float = 0.2
    temperature: float = 0.1
    jaccard_threshold: float = 0.5


class CombinedLoss(nn.Module):
    def __init__(self, cfg: LossConfig | None = None,
                 alpha: torch.Tensor | float | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.alpha = alpha
        # Curriculum toggle: S1 (heads-only warmup) runs focal-BCE only (spec §6).
        self.enable_supcon = True

    def forward(
        self, logits: torch.Tensor, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        focal = focal_bce_with_logits(logits, targets, self.cfg.gamma, self.alpha)
        if self.enable_supcon:
            supcon = multilabel_supcon(
                embeddings, targets, self.cfg.temperature, self.cfg.jaccard_threshold
            )
        else:
            supcon = torch.zeros((), device=logits.device, dtype=logits.dtype)
        total = focal + self.cfg.mu * supcon
        parts = {
            "focal": focal.detach(),
            "supcon": supcon.detach(),
            "total": total.detach(),
        }
        return total, parts
