"""Build an HNM review queue from model predictions on an unlabeled pool (spec §7 steps 1-3).

Closes the pipeline gap before the annotator: scores a pool of unlabeled clips
with a trained model, selects hard-negative / uncertain candidates, and writes
the review queue the annotator (tools/annotator) consumes.

    python -m mining.run --pool-manifest data/manifests/pool.jsonl \
        --feature-root data/features --stats artifacts/norm.npz \
        --ckpt artifacts/checkpoints/best.ckpt --out data/mining/queue.jsonl

The pool manifest lists clips to score (labels/split are ignored — the pool is
unlabeled); clips without a precomputed feature are skipped.
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from datasets.dataset import LogMelDataset
from datasets.manifest import read_manifest
from datasets.taxonomy import load_taxonomy
from evaluate import predict
from mining.candidates import PoolClip, select_candidates, write_review_queue
from mining.config import load_mining_config
from models.harm_model import HarmModel
from preprocess.normalize import NormStats
from preprocess.paths import feature_path
from training.trainer import resolve_device


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build an HNM review queue from pool predictions.")
    p.add_argument("--pool-manifest", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="review-queue jsonl for the annotator")
    p.add_argument("--classes", default=None)
    p.add_argument("--mining-config", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    taxonomy = load_taxonomy(args.classes)
    mining_cfg = load_mining_config(args.mining_config)

    records = [
        r for r in read_manifest(args.pool_manifest)
        if feature_path(r.clip_id, args.feature_root).exists()
    ]
    if not records:
        raise SystemExit(f"no pool clips with precomputed features under {args.feature_root}")
    # The pool is unlabeled by design; drop any labels so a stray/out-of-taxonomy
    # label can't KeyError in the dataset's multi-hot encoding.
    for r in records:
        r.labels = []

    stats = NormStats.load(args.stats)
    ds = LogMelDataset(records, args.feature_root, taxonomy, stats, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device("auto")
    model = HarmModel.from_checkpoint(args.ckpt, taxonomy.num_classes, map_location=device)
    probs, _ = predict(model, loader, device)

    pool = [
        PoolClip(r.clip_id, r.source, r.source_id, r.start_sec, r.duration) for r in records
    ]
    candidates = select_candidates(pool, probs, taxonomy, mining_cfg)
    write_review_queue(candidates, args.out)
    print(f"scored {len(records)} pool clips -> {len(candidates)} review candidates -> {args.out}")


if __name__ == "__main__":
    main()
