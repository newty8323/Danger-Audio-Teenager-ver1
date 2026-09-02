"""Training loop (spec §6, §11).

Grad-accumulation, AMP on CUDA (fp32 on MPS/CPU), curriculum (S1 heads-only ->
S2 full + SupCon), early stopping on val mAP, a Kaggle 11h time-guard (checked
mid-epoch), and mandatory resume-auto checkpointing every epoch.

Note on SupCon batch width: gradient accumulation sums per-micro-batch
gradients, so it widens the *BCE* gradient estimate but does NOT widen the
contrastive set — SupCon only ever sees one physical (sampler) batch. To give
SupCon a wider batch, raise ``batch_size`` (memory permitting); a true
"effective 64 from 32" contrastive batch would need gradient caching (GradCache)
and is a deliberate follow-up (see process.md).

The loop is decoupled from data construction: ``fit`` takes ready DataLoaders,
so it is unit-testable on tiny synthetic data. The clock is injectable so the
time-guard can be exercised deterministically.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses.combined import CombinedLoss
from training.checkpoint import (
    BEST_CKPT,
    LAST_CKPT,
    TrainState,
    find_latest,
    load_checkpoint,
    save_checkpoint,
)
from training.config import CurriculumStage, TrainConfig
from training.metrics import macro_map
from training.optim import build_optimizer, build_scheduler


class _NoScaler:
    """AMP GradScaler no-op for MPS/CPU (fp32)."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


@dataclass
class TrainResult:
    status: str  # "completed" | "early_stop" | "time_guard"
    best_metric: float
    last_epoch: int
    history: list[dict] = field(default_factory=list)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: CombinedLoss,
        cfg: TrainConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_epoch: Callable[[dict], None] | None = None,
    ) -> None:
        self.cfg = cfg or TrainConfig()
        self.device = resolve_device(self.cfg.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.clock = clock
        self.on_epoch = on_epoch  # per-epoch metrics callback (e.g. wandb.log)
        self.amp_enabled = self.cfg.amp and self.device.type == "cuda"
        self._bf16 = self.amp_enabled and self.cfg.amp_dtype == "bf16"
        self._amp_dtype = torch.bfloat16 if self._bf16 else torch.float16
        # bf16 has fp32 range -> no loss scaling needed; only fp16 uses GradScaler.
        self.scaler = (torch.amp.GradScaler("cuda")
                       if (self.amp_enabled and not self._bf16) else _NoScaler())
        self._autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        self._start_time = 0.0
        self._last_train_loss = float("nan")  # avg train loss of the last completed epoch
        self.history: list[dict] = []

    # ---- curriculum ----

    def _apply_stage(self, stage: CurriculumStage) -> None:
        self.loss_fn.enable_supcon = stage.use_supcon
        backbone = getattr(self.model, "backbone", None)
        # a backbone that manages its own selective freezing (e.g. BEATsRawBackbone's
        # top-k strategy-B unfreeze) opts out of the stage's all-or-nothing freeze.
        if backbone is not None and not getattr(backbone, "manages_own_freezing", False):
            for p in backbone.parameters():
                p.requires_grad_(not stage.freeze_backbone)

    # ---- epoch hooks ----

    @staticmethod
    def _set_epoch(loader: DataLoader, epoch: int) -> None:
        for obj in (getattr(loader, "dataset", None),
                    getattr(loader, "batch_sampler", None),
                    getattr(loader, "sampler", None)):
            if hasattr(obj, "set_epoch"):
                obj.set_epoch(epoch)

    def _time_exceeded(self) -> bool:
        return (self.clock() - self._start_time) >= self.cfg.time_guard_hours * 3600.0

    # ---- train / eval ----

    def train_one_epoch(self, loader: DataLoader, stage: CurriculumStage) -> bool:
        """Run one epoch. Returns False if the time-guard interrupted it mid-epoch."""
        self.model.train()
        backbone = getattr(self.model, "backbone", None)
        if stage.freeze_backbone and backbone is not None:
            backbone.eval()  # keep BatchNorm running stats fixed while frozen

        accum = self.cfg.grad_accum_steps
        self.optimizer.zero_grad(set_to_none=True)
        pending = 0
        opt_steps = 0
        running_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.autocast(self._autocast_device, dtype=self._amp_dtype,
                                enabled=self.amp_enabled):
                out = self.model(x)
                loss, _ = self.loss_fn(out["logits"], out["embeddings"], y)
                loss = loss / accum
            self.scaler.scale(loss).backward()
            running_loss += loss.item() * accum  # undo the accum scaling for logging
            n_batches += 1
            pending += 1
            if pending == accum:
                self._optimizer_step()
                pending = 0
                opt_steps += 1
                if opt_steps % self.cfg.time_guard_check_steps == 0 and self._time_exceeded():
                    return False  # save & exit; this epoch is redone on resume
        if pending > 0:  # flush a partial accumulation window
            self._optimizer_step()
        self._last_train_loss = running_loss / max(1, n_batches)
        return True

    def _optimizer_step(self) -> None:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        probs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for x, y in loader:
            x = x.to(self.device)
            out = self.model(x, return_projection=False)
            probs.append(torch.sigmoid(out["logits"]).float().cpu().numpy())
            labels.append(np.asarray(y))
        if not probs:  # empty val split
            return {"map": float("nan")}
        p = np.concatenate(probs, axis=0)
        yv = np.concatenate(labels, axis=0)
        return {"map": macro_map(yv, p)}

    # ---- fit ----

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, resume: str = "auto"
    ) -> TrainResult:
        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.cfg.grad_accum_steps))
        total_steps = self.cfg.total_epochs * steps_per_epoch
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg, total_steps)

        start_epoch = 0
        best = float("-inf")
        no_improve = 0
        if resume == "auto":
            ckpt = find_latest(self.cfg.ckpt_dir)
            if ckpt is not None:
                st = load_checkpoint(
                    ckpt, self.model, self.optimizer, self.scheduler, self.scaler,
                    map_location=self.device,
                )
                start_epoch = st.epoch + 1
                best = st.best_metric
                no_improve = st.epochs_no_improve

        self._start_time = self.clock()
        status = "completed"
        last_epoch = start_epoch - 1

        ckpt_dir = Path(self.cfg.ckpt_dir)
        for epoch in range(start_epoch, self.cfg.total_epochs):
            stage = self.cfg.stage_for_epoch(epoch)
            self._apply_stage(stage)
            self._set_epoch(train_loader, epoch)

            if not self.train_one_epoch(train_loader, stage):
                # Interrupted mid-epoch: checkpoint marking this epoch as NOT done
                # (epoch-1) so resume redoes it from the start; then exit.
                state = TrainState(epoch=epoch - 1, best_metric=best,
                                   epochs_no_improve=no_improve)
                save_checkpoint(ckpt_dir / LAST_CKPT, self.model, self.optimizer,
                                self.scheduler, self.scaler, state)
                status = "time_guard"
                last_epoch = epoch - 1
                break

            metrics = self.evaluate(val_loader)
            val_map = metrics["map"]
            last_epoch = epoch

            improved = (not math.isnan(val_map)) and val_map > best + 1e-6
            if improved:
                best, no_improve = val_map, 0
            else:
                no_improve += 1

            state = TrainState(epoch=epoch, best_metric=best, epochs_no_improve=no_improve)
            save_checkpoint(ckpt_dir / LAST_CKPT, self.model, self.optimizer,
                            self.scheduler, self.scaler, state)
            if improved:
                save_checkpoint(ckpt_dir / BEST_CKPT, self.model, self.optimizer,
                                self.scheduler, self.scaler, state)

            lr = self.optimizer.param_groups[-1]["lr"]  # head LR (last group)
            epoch_info = {"epoch": epoch, "stage": stage.name, "val_map": val_map,
                          "train_loss": self._last_train_loss, "lr": lr, **metrics}
            self.history.append({"epoch": epoch, "stage": stage.name, "val_map": val_map})
            if self.on_epoch is not None:
                self.on_epoch(epoch_info)  # e.g. wandb.log — persistent Kaggle logs (spec §11)

            if self._time_exceeded():
                status = "time_guard"
                break
            if no_improve >= self.cfg.patience:
                status = "early_stop"
                break

        return TrainResult(status, best, last_epoch, self.history)
