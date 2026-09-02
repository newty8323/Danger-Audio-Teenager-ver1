"""Source-disjoint train/val/test splitting (spec §3).

Hard constraint: the same ``source_id`` (video/channel) never crosses splits,
so a model can't memorize a source in train and be scored on it in val/test.
Certain sources (in-the-wild broadcast clips) are test-only.

The splitter groups clips by ``source_id`` and greedily assigns whole groups to
whichever split is furthest below its target clip count, which tracks the
requested ratios closely while respecting disjointness. Full label
stratification (keeping per-class balance across splits) is a future refinement;
size-balancing is the honest current behavior.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable

from datasets.manifest import SPLITS, ClipRecord

DEFAULT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
DEFAULT_TEST_ONLY_SOURCES = frozenset({"in_the_wild"})


class SplitError(ValueError):
    pass


def assert_source_disjoint(records: Iterable[ClipRecord]) -> None:
    """Raise if any ``source_id`` appears in more than one split."""
    source_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        source_to_splits[r.source_id].add(r.split)
    leaks = {sid: sorted(s) for sid, s in source_to_splits.items() if len(s) > 1}
    if leaks:
        preview = "; ".join(f"{sid} in {splits}" for sid, splits in list(leaks.items())[:10])
        raise SplitError(f"{len(leaks)} source(s) span multiple splits: {preview}")


def assign_splits(
    records: Iterable[ClipRecord],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    test_only_sources: Iterable[str] = DEFAULT_TEST_ONLY_SOURCES,
) -> dict[str, str]:
    """Return a deterministic ``source_id -> split`` assignment.

    ``test_only_sources`` matches on ``ClipRecord.source`` (the dataset name);
    all clips from such sources go to test.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SplitError(f"ratios must be 3 values summing to 1, got {ratios}")

    records = list(records)
    test_only = set(test_only_sources)

    # Group clip counts by source_id, and remember which sources are test-only.
    counts: dict[str, int] = defaultdict(int)
    forced_test: set[str] = set()
    for r in records:
        counts[r.source_id] += 1
        if r.source in test_only:
            forced_test.add(r.source_id)

    assignment: dict[str, str] = {sid: "test" for sid in forced_test}

    assignable = {sid: n for sid, n in counts.items() if sid not in forced_test}
    total = sum(assignable.values())
    targets = dict(zip(SPLITS, (r * total for r in ratios), strict=True))
    current = {s: 0 for s in SPLITS}
    # test already carries forced clips — count them so ratios stay honest overall.
    current["test"] += sum(counts[sid] for sid in forced_test)

    # Deterministic order: largest groups first (better balance), seeded tie-break.
    rng = random.Random(seed)
    order = sorted(assignable, key=lambda sid: (-assignable[sid], rng.random(), sid))

    for sid in order:
        n = assignable[sid]
        # Assign to the split with the largest remaining deficit.
        split = max(SPLITS, key=lambda s: targets[s] - current[s])
        assignment[sid] = split
        current[split] += n

    return assignment


def apply_split(records: Iterable[ClipRecord], assignment: dict[str, str]) -> list[ClipRecord]:
    """Return copies of ``records`` with ``split`` set from ``assignment``."""
    out: list[ClipRecord] = []
    for r in records:
        if r.source_id not in assignment:
            raise SplitError(f"no split assigned for source_id {r.source_id!r}")
        out.append(ClipRecord(**{**r.to_json(), "split": assignment[r.source_id]}))
    return out
