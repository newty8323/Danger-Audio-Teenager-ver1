"""Training configuration (spec §6). Mirrored by configs/train/train.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    epochs: int
    freeze_backbone: bool
    use_supcon: bool


# spec §6: S1 heads-only (5 ep) -> S2 full classes + SupCon, unfreeze (45 ep).
DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("S1-heads", epochs=5, freeze_backbone=True, use_supcon=False),
    CurriculumStage("S2-full", epochs=45, freeze_backbone=False, use_supcon=True),
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    # grad-accum widens the BCE gradient estimate (spec §6 "effective 64"); it does
    # NOT widen the SupCon contrastive set (that sees one physical batch). See trainer.
    grad_accum_steps: int = 2

    # optimizer (spec §6)
    lr_heads: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    layer_decay: float = 0.9

    # schedule
    warmup_pct: float = 0.05  # 5% warmup then cosine

    # early stopping on val mAP
    patience: int = 10

    # precision: AMP only on CUDA, fp32 on MPS/CPU (spec §6).
    # amp_dtype: "fp16" (default; needs GradScaler) or "bf16" (Ampere+, no scaler,
    # fp32-range so no overflow — the safe choice for BEATs, which NaNs in fp16).
    amp: bool = True
    amp_dtype: str = "fp16"

    # Kaggle 12h session guard: save & exit at 11h (spec §11). Checked mid-epoch
    # every time_guard_check_steps optimizer steps so a long epoch can't overshoot.
    time_guard_hours: float = 11.0
    time_guard_check_steps: int = 50

    device: str = "auto"  # auto -> cuda|mps|cpu
    num_workers: int = 4
    ckpt_dir: str = "artifacts/checkpoints"

    curriculum: tuple[CurriculumStage, ...] = field(default=DEFAULT_CURRICULUM)

    @property
    def total_epochs(self) -> int:
        return sum(s.epochs for s in self.curriculum)

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        """Curriculum stage owning a 0-indexed global epoch."""
        cum = 0
        for stage in self.curriculum:
            cum += stage.epochs
            if epoch < cum:
                return stage
        return self.curriculum[-1]
