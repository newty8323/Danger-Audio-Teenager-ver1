"""Feature artifact path conventions (single source of truth).

Kept dependency-free so both the writer (preprocess.precompute) and readers
(datasets.dataset, mining, evaluate) agree on where a clip's ``.npy`` lives.
"""

from __future__ import annotations

from pathlib import Path


def feature_path(clip_id: str, feature_root: str | Path) -> Path:
    return Path(feature_root) / f"{clip_id}.npy"
