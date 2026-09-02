"""Hard-negative candidate selection (spec §7 steps 1-3).

From model predictions on an unlabeled pool, pick clips worth human review:
  - false_positive: confident harm prediction on likely-negative audio
    (max harm prob >= fp_prob_threshold)
  - uncertain: max harm prob in [uncertain_low, uncertain_high)

Candidates are ranked (false positives first, then uncertain-by-closeness-to-0.5)
and capped at the per-iteration review budget (top_k). Reviewed false positives
later become confusable-labeled train clips (see mining.hnm).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from datasets.taxonomy import Taxonomy
from mining.config import MiningConfig

FALSE_POSITIVE = "false_positive"
UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class PoolClip:
    """An unlabeled candidate clip (minimal provenance for later promotion)."""

    clip_id: str
    source: str
    source_id: str
    start_sec: float
    duration: float


@dataclass
class ReviewCandidate:
    clip_id: str
    source: str
    source_id: str
    start_sec: float
    duration: float
    top_harm_class: str
    top_harm_prob: float
    reason: str  # FALSE_POSITIVE | UNCERTAIN
    priority: float
    clap_pseudo_label: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> ReviewCandidate:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in obj.items() if k in known})


def select_candidates(
    pool: list[PoolClip],
    probs: np.ndarray,
    taxonomy: Taxonomy,
    config: MiningConfig | None = None,
    clap_labels: dict[str, str] | None = None,
) -> list[ReviewCandidate]:
    """Return up to ``top_k`` review candidates ranked FP-first.

    ``probs`` is (N, num_classes) aligned row-wise with ``pool``.
    """
    config = config or MiningConfig()
    clap_labels = clap_labels or {}
    if probs.ndim != 2 or probs.shape[1] != taxonomy.num_classes:
        raise ValueError(
            f"probs must be (N, {taxonomy.num_classes}); got shape {probs.shape}"
        )
    if len(pool) != probs.shape[0]:
        raise ValueError(f"pool ({len(pool)}) and probs ({probs.shape[0]}) length mismatch")

    harm_idx = list(taxonomy.harm_indices)
    fps: list[ReviewCandidate] = []
    uncertain: list[ReviewCandidate] = []

    for i, clip in enumerate(pool):
        harm_probs = probs[i, harm_idx]
        j = int(np.argmax(harm_probs))
        top_prob = float(harm_probs[j])
        top_class = taxonomy.harm_classes[j]

        if top_prob >= config.fp_prob_threshold:
            reason, priority = FALSE_POSITIVE, top_prob
        elif config.uncertain_low <= top_prob < config.uncertain_high:
            reason, priority = UNCERTAIN, 1.0 - 2.0 * abs(top_prob - 0.5)
        else:
            continue

        cand = ReviewCandidate(
            clip_id=clip.clip_id,
            source=clip.source,
            source_id=clip.source_id,
            start_sec=clip.start_sec,
            duration=clip.duration,
            top_harm_class=top_class,
            top_harm_prob=top_prob,
            reason=reason,
            priority=priority,
            clap_pseudo_label=clap_labels.get(clip.clip_id),
        )
        (fps if reason == FALSE_POSITIVE else uncertain).append(cand)

    fps.sort(key=lambda c: c.priority, reverse=True)
    uncertain.sort(key=lambda c: c.priority, reverse=True)
    return (fps + uncertain)[: config.top_k]


def write_review_queue(candidates: Iterable[ReviewCandidate], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")


def read_review_queue(path: str | Path) -> list[ReviewCandidate]:
    out: list[ReviewCandidate] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ReviewCandidate.from_json(json.loads(line)))
    return out
