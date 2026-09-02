"""Multi-label supervised contrastive loss (spec §5, ref [supcon]).

Positives are batch pairs whose label sets have Jaccard overlap >= 0.5 (spec's
multi-label variant); everything else is a negative. With the balanced
harm:confusable sampler this makes confusable pairs (vio_impact vs chair_scrape)
negatives automatically — no manual pair labeling.

Anchors with no positive in the batch are excluded. If the whole batch has no
positive pairs the loss is 0 (no contrastive signal that step).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jaccard_positive_mask(labels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return a (B, B) float mask of Jaccard-overlap positives, self excluded.

    labels: (B, C) multi-hot. Jaccard = |A∩B| / |A∪B|; pairs of empty label sets
    (safe clips) have union 0 and are treated as non-positive.
    """
    binary = (labels > 0).float()
    inter = binary @ binary.t()  # (B, B)
    card = binary.sum(dim=1)
    union = card.unsqueeze(0) + card.unsqueeze(1) - inter
    jacc = torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(inter))
    mask = (jacc >= threshold).float()
    mask = mask - torch.eye(labels.size(0), device=labels.device, dtype=mask.dtype)
    return mask.clamp(min=0.0)


def multilabel_supcon(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    jaccard_threshold: float = 0.5,
) -> torch.Tensor:
    """embeddings: (B, d) (will be L2-normalized). labels: (B, C) multi-hot."""
    z = F.normalize(embeddings, dim=-1)
    batch = z.size(0)
    device = z.device

    sim = (z @ z.t()) / temperature  # (B, B)
    self_mask = torch.eye(batch, device=device, dtype=sim.dtype)
    neg_inf_diag = self_mask * -1e9  # remove self from the denominator

    # log-softmax over all k != i, numerically stabilized.
    sim = sim + neg_inf_diag
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B, B)

    pos_mask = jaccard_positive_mask(labels, jaccard_threshold)  # (B, B)
    pos_count = pos_mask.sum(dim=1)  # (B,)

    valid = pos_count > 0
    if not bool(valid.any()):
        return torch.zeros((), device=device, dtype=z.dtype)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
    loss = -mean_log_prob_pos[valid]
    return loss.mean()
