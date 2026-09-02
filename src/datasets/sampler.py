"""Balanced harm:confusable batch sampler (spec §5).

Forces roughly 1:1 harm vs. non-harm clips per batch so that confusable pairs
(vio_impact vs chair_scrape, etc.) land in the same batch and become SupCon
negatives automatically. A clip counts as *harm* if it carries any harm label.

Use as ``DataLoader(dataset, batch_sampler=BalancedBatchSampler(...))``. The
majority pool is fully covered once per epoch; the minority pool is cycled
(reshuffled on exhaustion). Seeded by ``(seed, epoch)`` for reproducibility.
"""

from __future__ import annotations

import math
import random
import warnings

from torch.utils.data import Sampler

from datasets.manifest import ClipRecord
from datasets.taxonomy import Taxonomy


def _is_harm(record: ClipRecord, taxonomy: Taxonomy) -> bool:
    return any(taxonomy.is_harm(lbl) for lbl in record.labels)


_MASK = 0xFFFFFFFFFFFFFFFF


def _mix(*vals: int) -> int:
    """Fold ints into one 64-bit seed (FNV-1a style).

    Deterministic across processes — uses only int arithmetic, never str hashing
    (which Python randomizes via PYTHONHASHSEED). ``random.Random`` also does not
    accept tuple seeds, so we must reduce to a single int here.
    """
    h = 1469598103934665603
    for v in vals:
        h = ((h ^ (v & _MASK)) * 1099511628211) & _MASK
    return h


class _CyclicShuffler:
    """Yields indices from ``pool`` forever, reshuffling each pass. Deterministic."""

    def __init__(self, pool: list[int], seed: int) -> None:
        self._pool = list(pool)
        self._seed = seed
        self._pass = 0
        self._order: list[int] = []
        self._pos = 0

    def _reshuffle(self) -> None:
        rng = random.Random(_mix(self._seed, self._pass))
        self._order = list(self._pool)
        rng.shuffle(self._order)
        self._pos = 0
        self._pass += 1

    def take(self, n: int) -> list[int]:
        """Return ``n`` indices. Distinct within the call when ``n <= len(pool)``
        (a batch never gets the same clip twice — trivial SupCon positives); only
        a pool smaller than ``n`` forces repeats."""
        if not self._pool:
            return []
        if n <= len(self._pool):
            # Take a contiguous block from a single shuffled pass so all picks are
            # distinct; reshuffle up front (discarding the short tail) if needed.
            if not self._order or self._pos + n > len(self._order):
                self._reshuffle()
            block = self._order[self._pos:self._pos + n]
            self._pos += n
            return block
        out: list[int] = []
        while len(out) < n:
            if self._pos >= len(self._order):
                self._reshuffle()
            out.append(self._order[self._pos])
            self._pos += 1
        return out


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[ClipRecord],
        taxonomy: Taxonomy,
        batch_size: int,
        seed: int = 42,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for a balanced sampler")
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        # Every batch is filled to batch_size by cycling the pools, so there is
        # no partial last batch to drop.

        self.harm_indices = [i for i, r in enumerate(records) if _is_harm(r, taxonomy)]
        # "non-harm" = confusable and/or safe clips; both act as negatives (spec §5).
        self.other_indices = [i for i, r in enumerate(records) if not _is_harm(r, taxonomy)]
        if not self.harm_indices and not self.other_indices:
            raise ValueError("no records to sample")

        self.n_harm_per_batch = batch_size // 2
        self.n_other_per_batch = batch_size - self.n_harm_per_batch

        # If a (non-empty) pool is smaller than its per-batch quota, cycling must
        # repeat clips within a batch -> duplicate SupCon positives. Warn loudly.
        if 0 < len(self.harm_indices) < self.n_harm_per_batch:
            warnings.warn(
                f"only {len(self.harm_indices)} harm clips for a per-batch quota of "
                f"{self.n_harm_per_batch}; batches will repeat harm clips",
                stacklevel=2,
            )
        if 0 < len(self.other_indices) < self.n_other_per_batch:
            warnings.warn(
                f"only {len(self.other_indices)} non-harm clips for a per-batch quota of "
                f"{self.n_other_per_batch}; batches will repeat non-harm clips",
                stacklevel=2,
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _num_batches(self) -> int:
        # Cover the larger pool once per epoch given its per-batch quota.
        candidates = []
        if self.harm_indices:
            candidates.append(len(self.harm_indices) / self.n_harm_per_batch)
        if self.other_indices:
            candidates.append(len(self.other_indices) / self.n_other_per_batch)
        return max(1, math.ceil(max(candidates)))

    def __len__(self) -> int:
        return self._num_batches()

    def __iter__(self):
        harm = _CyclicShuffler(self.harm_indices, _mix(self.seed, self._epoch, 0))
        other = _CyclicShuffler(self.other_indices, _mix(self.seed, self._epoch, 1))

        harm_empty = not self.harm_indices
        other_empty = not self.other_indices

        for _ in range(self._num_batches()):
            if harm_empty:
                batch = other.take(self.batch_size)
            elif other_empty:
                batch = harm.take(self.batch_size)
            else:
                batch = harm.take(self.n_harm_per_batch) + other.take(self.n_other_per_batch)
            yield batch
