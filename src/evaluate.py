"""Evaluation report against the spec §9 targets.

Given model predictions over a split, reports:
  - per harm class: AP, AUROC, Recall@FPR1%
  - per harm category (sex/vio/gmb): Recall@FPR1% and AUROC (binary, max-pooled)
  - macro mAP (all classes and harm-only)
  - inference latency (ms/clip)
and whether each §9 target is met.

    python -m evaluate --manifest data/manifests/test.jsonl \
        --feature-root data/features --stats artifacts/norm.npz \
        --ckpt artifacts/checkpoints/best.ckpt --split test --out artifacts/eval.json

The scoring core (:func:`harm_report`) is pure numpy and unit-tested; the CLI
wires a checkpoint + data around it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LogMelDataset
from datasets.manifest import load_manifest
from datasets.taxonomy import Taxonomy, load_taxonomy
from models.harm_model import HarmModel
from preprocess.normalize import NormStats
from training.metrics import auroc, average_precision, macro_auroc, macro_map, recall_at_fpr
from training.trainer import resolve_device

# spec §9 targets
TARGET_MACRO_MAP = 0.70
TARGET_HARM_AUROC = 0.90
TARGET_RECALL_AT_FPR = 0.80
TARGET_FPR = 0.01


def _passed(values: list[float], threshold: float) -> bool:
    """True if there is >=1 measurable value and all measurable ones meet threshold."""
    valid = [v for v in values if not math.isnan(v)]
    return bool(valid) and all(v >= threshold for v in valid)


def harm_report(
    probs: np.ndarray,
    labels: np.ndarray,
    taxonomy: Taxonomy,
    target_fpr: float = TARGET_FPR,
) -> dict:
    """Compute the §9 metric report from (N, C) probs and multi-hot labels."""
    per_class: dict[str, dict] = {}
    for name in taxonomy.all_classes:
        c = taxonomy.index_of(name)
        per_class[name] = {
            "ap": average_precision(labels[:, c], probs[:, c]),
            "auroc": auroc(labels[:, c], probs[:, c]),
            "recall_at_fpr": recall_at_fpr(labels[:, c], probs[:, c], target_fpr),
        }

    harm_idx = list(taxonomy.harm_indices)
    macro_all = macro_map(labels, probs)
    macro_harm = macro_map(labels[:, harm_idx], probs[:, harm_idx])
    macro_auroc_harm = macro_auroc(labels[:, harm_idx], probs[:, harm_idx])

    # Per-category rollup: binary "any harm class in this category", score = max prob.
    categories: dict[str, dict] = {}
    for cat in taxonomy.harm_categories:
        idx = [taxonomy.index_of(n) for n in taxonomy.harm_classes
               if taxonomy.category_of(n) == cat]
        cat_label = labels[:, idx].max(axis=1)
        cat_score = probs[:, idx].max(axis=1)
        categories[cat] = {
            "recall_at_fpr": recall_at_fpr(cat_label, cat_score, target_fpr),
            "auroc": auroc(cat_label, cat_score),
        }

    harm_aurocs = [per_class[n]["auroc"] for n in taxonomy.harm_classes]
    cat_recalls = [categories[c]["recall_at_fpr"] for c in categories]

    targets = {
        f"macro_map>={TARGET_MACRO_MAP}": (not math.isnan(macro_all))
        and macro_all >= TARGET_MACRO_MAP,
        f"harm_auroc>={TARGET_HARM_AUROC}": _passed(harm_aurocs, TARGET_HARM_AUROC),
        f"recall@fpr{target_fpr:.0%}>={TARGET_RECALL_AT_FPR}_per_category": _passed(
            cat_recalls, TARGET_RECALL_AT_FPR
        ),
    }

    return {
        "macro_map": macro_all,
        "macro_map_harm": macro_harm,
        "macro_auroc_harm": macro_auroc_harm,
        "per_class": per_class,
        "per_category": categories,
        "targets": targets,
        "target_fpr": target_fpr,
        "n_samples": int(labels.shape[0]),
    }


@torch.no_grad()
def predict(
    model: HarmModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        out = model(x, return_projection=False)
        probs.append(torch.sigmoid(out["logits"]).float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def measure_latency(
    model: HarmModel, example: torch.Tensor, device: torch.device, repeats: int = 20
) -> float:
    """Mean inference latency in ms/clip for a single-clip forward pass."""
    model.eval()
    x = example.to(device)
    for _ in range(3):  # warmup
        model(x, return_projection=False)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x, return_projection=False)
    _sync(device)
    return (time.perf_counter() - start) / repeats * 1000.0


def _sync(device: torch.device) -> None:
    """Block until queued device work finishes so timing is accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _json_safe(obj):
    """Recursively replace non-finite floats with None so json.dump is portable
    (JSON has no NaN/Infinity; strict parsers like jq/wandb reject the bare token)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def format_report(report: dict, taxonomy: Taxonomy) -> str:
    lines = [
        f"samples={report['n_samples']}  "
        f"macro_mAP={report['macro_map']:.3f} (harm {report['macro_map_harm']:.3f})",
        "",
        f"{'harm class':16s} {'AP':>6s} {'AUROC':>6s} {'R@FPR':>6s}",
    ]
    for name in taxonomy.harm_classes:
        m = report["per_class"][name]
        lines.append(f"{name:16s} {m['ap']:6.3f} {m['auroc']:6.3f} {m['recall_at_fpr']:6.3f}")
    lines.append("")
    lines.append(f"{'category':16s} {'AUROC':>6s} {'R@FPR':>6s}")
    for cat, m in report["per_category"].items():
        lines.append(f"{cat:16s} {m['auroc']:6.3f} {m['recall_at_fpr']:6.3f}")
    lines.append("")
    for name, ok in report["targets"].items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint against §9 targets.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--classes", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None, help="write the JSON report here")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    taxonomy = load_taxonomy(args.classes)
    records = [r for r in load_manifest(args.manifest, taxonomy) if r.split == args.split]
    if not records:
        raise SystemExit(f"no clips in split {args.split!r} of {args.manifest}")
    stats = NormStats.load(args.stats)
    ds = LogMelDataset(records, args.feature_root, taxonomy, stats, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device("auto")
    model = HarmModel.from_checkpoint(args.ckpt, taxonomy.num_classes, map_location=device)

    probs, labels = predict(model, loader, device)
    report = harm_report(probs, labels, taxonomy)
    if len(ds) > 0:
        report["latency_ms_per_clip"] = measure_latency(model, ds[0][0].unsqueeze(0), device)

    print(format_report(report, taxonomy))
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(_json_safe(report), f, indent=2, allow_nan=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
