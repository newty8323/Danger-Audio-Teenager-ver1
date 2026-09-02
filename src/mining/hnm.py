"""HNM orchestration: promotion, FP distribution, stopping (spec §7 steps 4-5).

After humans review the candidate queue, confirmed false positives are relabeled
as their confusable class and added to the train manifest (spec §7 step 4). The
loop stops after ``max_iterations`` or when FPR@95%recall stops improving
(spec §7 step 5). Per-iteration FP distributions feed the confusion taxonomy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from datasets.manifest import ClipRecord
from datasets.taxonomy import Taxonomy
from mining.candidates import FALSE_POSITIVE, ReviewCandidate
from mining.config import MiningConfig


def fp_distribution(candidates: Iterable[ReviewCandidate]) -> dict[str, int]:
    """Count false-positive candidates by predicted harm class (confusion taxonomy)."""
    return dict(
        Counter(c.top_harm_class for c in candidates if c.reason == FALSE_POSITIVE)
    )


def _promote(
    candidates: list[ReviewCandidate],
    decisions: dict[str, str],
    taxonomy: Taxonomy,
    want_harm: bool,
) -> list[ClipRecord]:
    by_id = {c.clip_id: c for c in candidates}
    records: list[ClipRecord] = []
    for clip_id, label in decisions.items():
        if clip_id not in by_id:
            raise KeyError(f"decision for unknown candidate {clip_id!r}")
        if label not in taxonomy.categories:
            raise ValueError(f"unknown label {label!r}")
        if taxonomy.is_harm(label) != want_harm:
            kind = "harm" if want_harm else "confusable"
            raise ValueError(f"expected a {kind} class label, got {label!r}")
        c = by_id[clip_id]
        records.append(
            ClipRecord(
                clip_id=c.clip_id,
                source=c.source,
                source_id=c.source_id,
                start_sec=c.start_sec,
                duration=c.duration,
                labels=[label],
                label_confidence="verified",  # human-reviewed
                split="train",
            )
        )
    return records


def promote_false_positives(
    candidates: list[ReviewCandidate],
    decisions: dict[str, str],
    taxonomy: Taxonomy,
) -> list[ClipRecord]:
    """Turn reviewed false positives into confusable-labeled train records (spec §7 step 4).

    ``decisions`` maps clip_id -> the confusable class a human assigned. Clips not
    in ``decisions`` are skipped; the label must be a taxonomy confusable class.
    """
    return _promote(candidates, decisions, taxonomy, want_harm=False)


def promote_positives(
    candidates: list[ReviewCandidate],
    decisions: dict[str, str],
    taxonomy: Taxonomy,
) -> list[ClipRecord]:
    """Turn reviewed uncertain clips confirmed harmful into harm-labeled train records.

    Counterpart to :func:`promote_false_positives` for the recall side of
    uncertainty sampling (spec §7 step 2): a reviewer confirming a missed harm
    positive can add it to train. The label must be a taxonomy harm class.
    """
    return _promote(candidates, decisions, taxonomy, want_harm=True)


def should_stop(
    fpr_history: list[float], config: MiningConfig | None = None
) -> bool:
    """Stop HNM when the iteration budget is hit or FPR@95%recall plateaus.

    ``fpr_history`` = FPR@95%recall after each iteration (lower is better).
    """
    config = config or MiningConfig()
    n = len(fpr_history)
    if n >= config.max_iterations:
        return True
    if n >= 2:
        improvement = fpr_history[-2] - fpr_history[-1]  # reduction in FPR
        if improvement < config.min_fpr_improvement:
            return True
    return False
