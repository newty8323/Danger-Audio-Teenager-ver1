"""Training loop: optimizer, scheduler, checkpointing, metrics, Trainer (spec §6, §11)."""

from training.checkpoint import (
    TrainState,
    find_latest,
    load_checkpoint,
    save_checkpoint,
)
from training.config import CurriculumStage, TrainConfig
from training.metrics import (
    auroc,
    average_precision,
    macro_auroc,
    macro_map,
    recall_at_fpr,
)
from training.optim import build_optimizer, build_scheduler
from training.trainer import Trainer, TrainResult, resolve_device

__all__ = [
    "TrainConfig",
    "CurriculumStage",
    "build_optimizer",
    "build_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest",
    "TrainState",
    "average_precision",
    "macro_map",
    "auroc",
    "macro_auroc",
    "recall_at_fpr",
    "Trainer",
    "TrainResult",
    "resolve_device",
]
