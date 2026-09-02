"""Dataset manifest, taxonomy, splits, and training data pipeline (spec §2-§5)."""

from datasets.dataset import AugmentConfig, LogMelDataset
from datasets.manifest import (
    ClipRecord,
    ManifestError,
    load_manifest,
    read_manifest,
    training_records,
    validate_manifest,
    write_manifest,
)
from datasets.sampler import BalancedBatchSampler
from datasets.splits import apply_split, assert_source_disjoint, assign_splits
from datasets.taxonomy import Taxonomy, load_taxonomy

__all__ = [
    "ClipRecord",
    "ManifestError",
    "Taxonomy",
    "load_taxonomy",
    "read_manifest",
    "write_manifest",
    "load_manifest",
    "validate_manifest",
    "training_records",
    "assert_source_disjoint",
    "assign_splits",
    "apply_split",
    "LogMelDataset",
    "AugmentConfig",
    "BalancedBatchSampler",
]
