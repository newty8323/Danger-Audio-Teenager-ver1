"""Deterministic builder for the violence+gambling combined frozen-BEATs dataset.

This is the *committed, reproducible* replacement for the earlier ad-hoc combined
run (whose split logic lived only in a scratch script). Given a fixed ``split_seed``
it reconstructs the exact train/val/test used to train ``ckpt_beats_combined``:

  - violence: splits are baked into ``violence_v2.jsonl`` (train 2287/val 252/test 705).
  - gambling: re-split per class *by source video* 70/15/15 with ``random.Random(seed)``
    so each gmb class has its own train/val/test (source-disjoint, no leakage).

Both branches read precomputed BEATs frame embeddings from ``data/beats_feats/*.npy``
(shape (T, 768)); the model is PassthroughBackbone + MIL + heads. Keeping the split
seed separate from the training seed lets Phase 1 vary optimization/init noise while
holding the data split fixed — the standard N-seed protocol.
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from datasets.manifest import read_manifest  # noqa: E402
from datasets.taxonomy import load_taxonomy  # noqa: E402

FEAT = "data/beats_feats"
VIOLENCE = "data/manifests/violence_v2.jsonl"
GAMBLING = "data/manifests/gambling.jsonl"
DEFAULT_SPLIT_SEED = 42


def _has_feat(clip_id: str) -> bool:
    return os.path.exists(f"{FEAT}/{clip_id}.npy")


def build_combined_records(split_seed: int = DEFAULT_SPLIT_SEED, exists_fn=_has_feat):
    """Return (train, val, test) record lists for the combined dataset.

    Deterministic in ``split_seed``: same seed -> byte-identical splits, so the
    eval of a checkpoint is reproducible.

    Gambling is split by *physical clip* (source-video-disjoint): clips dual-labeled
    across gmb classes are deduped into one multi-hot record BEFORE splitting, and each
    source video is assigned to exactly one split. This closes a train<->val leak that
    the earlier per-(class,video) split had (a dual-class video landed in two splits).
    Per-class assignment is still used so each gmb class keeps train/val/test coverage.
    """
    # violence: splits already baked in; keep only clips with a cached BEATs feature.
    vio = [r for r in read_manifest(VIOLENCE) if exists_fn(r.clip_id)]

    # gambling: dedup to one multi-hot record per clip_id (union labels).
    gmb_raw = [r for r in read_manifest(GAMBLING) if exists_fn(r.clip_id)]
    by_clip: dict[str, object] = {}
    label_union: dict[str, list] = defaultdict(list)
    for r in gmb_raw:
        by_clip.setdefault(r.clip_id, r)
        for lbl in r.labels:
            if lbl not in label_union[r.clip_id]:
                label_union[r.clip_id].append(lbl)
    for cid, r in by_clip.items():
        r.labels = label_union[cid]
    gmb = list(by_clip.values())

    # assign each source video to exactly one split; per-class 70/15/15 over its videos,
    # skipping videos already placed by an earlier class (keeps videos split-disjoint).
    classes_of_video: dict[str, set] = defaultdict(set)
    for r in gmb:
        classes_of_video[r.source_id].update(r.labels)
    rng = random.Random(split_seed)
    video_split: dict[str, str] = {}
    all_classes = sorted({c for cs in classes_of_video.values() for c in cs})
    for cls in all_classes:
        vids = sorted(v for v, cs in classes_of_video.items() if cls in cs)
        unassigned = [v for v in vids if v not in video_split]
        rng.shuffle(unassigned)
        n = len(unassigned)
        ntr = max(1, int(n * 0.7))
        nva = max(1, int(n * 0.15))
        for j, v in enumerate(unassigned):
            video_split[v] = "train" if j < ntr else ("val" if j < ntr + nva else "test")
    for r in gmb:
        r.split = video_split[r.source_id]

    allr = vio + gmb
    tr = [r for r in allr if r.split == "train"]
    va = [r for r in allr if r.split == "val"]
    te = [r for r in allr if r.split == "test"]

    # guard: splits must be clip-disjoint (no leakage) — cheap and catches regressions.
    tr_ids, va_ids, te_ids = ({r.clip_id for r in s} for s in (tr, va, te))
    assert not (tr_ids & va_ids), f"train/val leak: {tr_ids & va_ids}"
    assert not (tr_ids & te_ids), f"train/test leak: {tr_ids & te_ids}"
    assert not (va_ids & te_ids), f"val/test leak: {va_ids & te_ids}"
    return tr, va, te


class BeatsFeatureDataset(Dataset):
    """Serves (BEATs frame embeddings (T,768), multihot label) from disk."""

    def __init__(self, records, tax=None):
        self.records = records
        self.tax = tax or load_taxonomy()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        x = torch.from_numpy(np.load(f"{FEAT}/{r.clip_id}.npy").astype(np.float32))
        return x, torch.from_numpy(r.multihot(self.tax))
