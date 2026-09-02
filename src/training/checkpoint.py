"""Checkpoint save/load + resume-auto (mandatory, CLAUDE.md; spec §11).

A checkpoint carries everything needed to resume a Kaggle 12h session exactly:
model / optimizer / scheduler / AMP-scaler state, the global epoch, best metric,
early-stop counter, and RNG states (python / numpy / torch). ``last.ckpt`` is
overwritten each epoch; ``best.ckpt`` tracks the best val mAP.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAST_CKPT = "last.ckpt"
BEST_CKPT = "best.ckpt"


@dataclass
class TrainState:
    epoch: int  # last completed global epoch
    best_metric: float
    epochs_no_improve: int


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["torch_cuda"]])


def _as_byte_tensor(t: torch.Tensor) -> torch.Tensor:
    # RNG states must be CPU ByteTensors; loading may yield a different dtype/device.
    return t.cpu().to(torch.uint8)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    state: TrainState,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        # Architecture config, so inference can rebuild the exact model (a ckpt
        # trained with a non-default backbone/dims loads into the right shape).
        "model_config": asdict(model.cfg) if is_dataclass(getattr(model, "cfg", None)) else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "epochs_no_improve": state.epochs_no_improve,
        "rng": _rng_state(),
    }
    tmp = Path(path).with_suffix(".ckpt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic-ish: never leave a half-written last.ckpt


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> TrainState:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng") is not None:
        _restore_rng(ckpt["rng"])
    return TrainState(
        epoch=ckpt["epoch"],
        best_metric=ckpt["best_metric"],
        epochs_no_improve=ckpt["epochs_no_improve"],
    )


def find_latest(ckpt_dir: str | Path) -> Path | None:
    """Return the resume checkpoint (``last.ckpt``) if present, else None."""
    p = Path(ckpt_dir) / LAST_CKPT
    return p if p.exists() else None
