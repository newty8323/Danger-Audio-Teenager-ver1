"""Batch precompute driver (spec §4, §11).

    manifest + audio -> per-clip log-mel .npy  (unnormalized)
                     -> norm stats fitted on the TRAIN split (versioned .npz)
                     -> output manifest with RMS-gated clips dropped

This is the step whose ``.npy`` output + manifest become a versioned Kaggle
Dataset. Normalization is *not* baked into the features; the saved stats are
applied at load time so they stay swappable.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from datasets.manifest import ClipRecord, read_manifest, training_records, write_manifest
from preprocess.config import PreprocessConfig
from preprocess.logmel import LogMelExtractor
from preprocess.normalize import NormStats, fit_norm_stats
from preprocess.paths import feature_path  # noqa: F401  (re-exported for callers)
from preprocess.pipeline import preprocess_clip

PathResolver = Callable[[ClipRecord, Path], Path]


def default_audio_path(record: ClipRecord, audio_root: Path) -> Path:
    return audio_root / f"{record.clip_id}.wav"


@dataclass
class PrecomputeResult:
    processed_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)  # RMS-gated
    missing_ids: list[str] = field(default_factory=list)  # audio file not found
    stats_path: str | None = None
    output_manifest_path: str | None = None

    @property
    def n_processed(self) -> int:
        return len(self.processed_ids)


def _train_feature_stream(
    kept: list[ClipRecord], feature_root: Path
) -> Iterator[np.ndarray]:
    for r in training_records(kept):
        yield np.load(feature_path(r.clip_id, feature_root))


def precompute_manifest(
    manifest_path: str | Path,
    audio_root: str | Path,
    feature_root: str | Path,
    stats_path: str | Path,
    output_manifest_path: str | Path,
    cfg: PreprocessConfig | None = None,
    path_resolver: PathResolver = default_audio_path,
) -> PrecomputeResult:
    cfg = cfg or PreprocessConfig()
    audio_root = Path(audio_root)
    feature_root = Path(feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)

    records = read_manifest(manifest_path)
    extractor = LogMelExtractor(cfg)
    result = PrecomputeResult()
    kept: list[ClipRecord] = []

    for r in records:
        src = path_resolver(r, audio_root)
        if not src.exists():
            result.missing_ids.append(r.clip_id)
            continue
        logmel = preprocess_clip(str(src), extractor, cfg)
        if logmel is None:  # RMS-gated (too quiet)
            result.dropped_ids.append(r.clip_id)
            continue
        np.save(feature_path(r.clip_id, feature_root), logmel)
        result.processed_ids.append(r.clip_id)
        kept.append(r)

    # Fit normalization stats on the training split only (spec §4).
    stats = _fit_train_stats(kept, feature_root)
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    stats.save(str(stats_path))
    result.stats_path = str(stats_path)

    write_manifest(kept, output_manifest_path)
    result.output_manifest_path = str(output_manifest_path)
    return result


def _fit_train_stats(kept: list[ClipRecord], feature_root: Path) -> NormStats:
    if not any(True for _ in training_records(kept)):
        raise ValueError("no kept training-split clips to fit normalization stats")
    return fit_norm_stats(_train_feature_stream(kept, feature_root))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute log-mel features from a manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--output-manifest", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    result = precompute_manifest(
        manifest_path=args.manifest,
        audio_root=args.audio_root,
        feature_root=args.feature_root,
        stats_path=args.stats_path,
        output_manifest_path=args.output_manifest,
    )
    print(
        f"processed={result.n_processed} dropped(gated)={len(result.dropped_ids)} "
        f"missing={len(result.missing_ids)} stats={result.stats_path}"
    )


if __name__ == "__main__":
    main()
