"""Training entrypoint (spec §6, §10, §11).

Wires precomputed features + manifest into the Trainer:

    python -m train --manifest data/manifests/train.jsonl \
        --feature-root data/features --stats artifacts/norm.npz --resume auto

Heavy full training runs on Kaggle GPU (report GPU-hours + get approval first,
CLAUDE.md rule 1); heads-only/dev runs work locally on MPS/CPU.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset import LogMelDataset
from datasets.manifest import load_manifest
from datasets.sampler import BalancedBatchSampler
from datasets.taxonomy import load_taxonomy
from losses.combined import CombinedLoss, LossConfig
from models.harm_model import HarmModel, ModelConfig
from preprocess.normalize import NormStats
from training.config import TrainConfig
from training.trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    manifest_path: str,
    feature_root: str,
    stats_path: str,
    taxonomy,
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, int]:
    records = load_manifest(manifest_path, taxonomy)
    stats = NormStats.load(stats_path)
    train_recs = [r for r in records if r.split == "train" and not r.flagged]
    val_recs = [r for r in records if r.split == "val"]

    train_ds = LogMelDataset(train_recs, feature_root, taxonomy, stats, train=True,
                             seed=cfg.seed)
    val_ds = LogMelDataset(val_recs, feature_root, taxonomy, stats, train=False)

    sampler = BalancedBatchSampler(train_recs, taxonomy, cfg.batch_size, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    return train_loader, val_loader, taxonomy.num_classes


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the harm-detection model.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--classes", default=None, help="taxonomy yaml (default: configs)")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--resume", default="auto", choices=["auto", "none"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = TrainConfig()
    if args.ckpt_dir is not None:
        cfg = TrainConfig(ckpt_dir=args.ckpt_dir)
    seed_everything(cfg.seed)

    taxonomy = load_taxonomy(args.classes)
    train_loader, val_loader, num_classes = build_loaders(
        args.manifest, args.feature_root, args.stats, taxonomy, cfg
    )

    model = HarmModel(num_classes, ModelConfig())
    loss_fn = CombinedLoss(LossConfig())
    trainer = Trainer(model, loss_fn, cfg)

    result = trainer.fit(train_loader, val_loader, resume=args.resume)
    print(
        f"[{result.status}] best val mAP={result.best_metric:.4f} "
        f"at/through epoch {result.last_epoch}"
    )


if __name__ == "__main__":
    main()
