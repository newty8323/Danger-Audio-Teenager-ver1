"""Fit the risk scorer on a val split and save its coefficients (spec §1 Task B, §9).

Closes the pipeline gap between training and streaming: `infer_stream` needs a
fitted ``--risk-params`` json, which this produces.

    python -m risk.fit --manifest data/manifests/val.jsonl --feature-root data/features \
        --stats artifacts/norm.npz --ckpt artifacts/checkpoints/best.ckpt \
        --split val --out artifacts/risk.json

Target = 1 if the clip carries any harm label (spec §1 Task B). Also reports the
binary risk AUC (spec §9 target >= .95).
"""

from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset import LogMelDataset
from datasets.manifest import load_manifest
from datasets.taxonomy import load_taxonomy
from evaluate import predict
from models.harm_model import HarmModel
from preprocess.normalize import NormStats
from risk.policy import load_risk_policy
from risk.scorer import RiskScorer
from training.metrics import auroc
from training.trainer import resolve_device


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit the risk scorer on a val split.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--classes", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", required=True, help="write fitted risk params (a,b,c) json")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    taxonomy = load_taxonomy(args.classes)
    policy = load_risk_policy(args.policy)
    records = [r for r in load_manifest(args.manifest, taxonomy) if r.split == args.split]
    if not records:
        raise SystemExit(f"no clips in split {args.split!r} of {args.manifest}")

    stats = NormStats.load(args.stats)
    ds = LogMelDataset(records, args.feature_root, taxonomy, stats, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device("auto")
    model = HarmModel.from_checkpoint(args.ckpt, taxonomy.num_classes, map_location=device)

    probs, labels = predict(model, loader, device)
    harm_idx = list(taxonomy.harm_indices)
    targets = (labels[:, harm_idx].max(axis=1) > 0).astype(np.float64)

    scorer = RiskScorer.from_policy(policy, taxonomy).fit(probs, targets)
    scorer.save_params(args.out)

    risk_auc = auroc(targets, np.asarray(scorer.score(probs)))
    print(f"fit on {len(records)} clips; risk binary AUC={risk_auc:.4f} "
          f"(a={scorer.a:.3f} b={scorer.b:.3f} c={scorer.c:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
