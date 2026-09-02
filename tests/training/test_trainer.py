import torch
from torch.utils.data import DataLoader, Dataset

from losses.combined import CombinedLoss, LossConfig
from models.harm_model import HarmModel, ModelConfig
from training.config import CurriculumStage, TrainConfig
from training.trainer import Trainer


class ToyDS(Dataset):
    def __init__(self, n=8, num_classes=4, f=32, t=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 1, f, t, generator=g)
        y = (torch.rand(n, num_classes, generator=g) > 0.5).float()
        y[0] = 1.0  # guarantee each class has a positive (val mAP well-defined)
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class FakeClock:
    """Returns successive canned timestamps (last one repeats)."""

    def __init__(self, times):
        self.times = times
        self.i = 0

    def __call__(self):
        v = self.times[min(self.i, len(self.times) - 1)]
        self.i += 1
        return v


def _cfg(tmp_path, curriculum, **kw):
    return TrainConfig(
        device="cpu",
        grad_accum_steps=1,
        ckpt_dir=str(tmp_path / "ckpts"),
        num_workers=0,
        curriculum=curriculum,
        **kw,
    )


def _model_loss(num_classes=4):
    model = HarmModel(num_classes, ModelConfig(backbone_out_dim=32))
    return model, CombinedLoss(LossConfig())


def _loaders(num_classes=4):
    ds = ToyDS(num_classes=num_classes)
    train = DataLoader(ds, batch_size=4, shuffle=True)
    val = DataLoader(ds, batch_size=4, shuffle=False)
    return train, val


def test_smoke_two_stage_curriculum_runs(tmp_path):
    curriculum = (
        CurriculumStage("s1", epochs=1, freeze_backbone=True, use_supcon=False),
        CurriculumStage("s2", epochs=1, freeze_backbone=False, use_supcon=True),
    )
    model, loss = _model_loss()
    trainer = Trainer(model, loss, _cfg(tmp_path, curriculum))
    train, val = _loaders()
    result = trainer.fit(train, val, resume="none")
    assert result.status == "completed"
    assert result.last_epoch == 1
    assert len(result.history) == 2
    assert (tmp_path / "ckpts" / "last.ckpt").exists()


def test_curriculum_toggles_supcon_and_freeze(tmp_path):
    curriculum = (
        CurriculumStage("s1", epochs=1, freeze_backbone=True, use_supcon=False),
        CurriculumStage("s2", epochs=1, freeze_backbone=False, use_supcon=True),
    )
    model, loss = _model_loss()
    trainer = Trainer(model, loss, _cfg(tmp_path, curriculum))
    trainer._apply_stage(curriculum[0])
    assert loss.enable_supcon is False
    assert all(not p.requires_grad for p in model.backbone.parameters())
    trainer._apply_stage(curriculum[1])
    assert loss.enable_supcon is True
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_time_guard_stops_and_checkpoints(tmp_path):
    curriculum = (CurriculumStage("s1", epochs=5, freeze_backbone=False, use_supcon=True),)
    model, loss = _model_loss()
    # start=0, first time-check returns 1e9 -> exceeds 11h guard after epoch 0
    clock = FakeClock([0.0, 1e9])
    trainer = Trainer(model, loss, _cfg(tmp_path, curriculum), clock=clock)
    train, val = _loaders()
    result = trainer.fit(train, val, resume="none")
    assert result.status == "time_guard"
    assert result.last_epoch == 0
    assert (tmp_path / "ckpts" / "last.ckpt").exists()


def test_mid_epoch_time_guard_redoes_epoch_on_resume(tmp_path):
    # ToyDS has 8 samples, batch_size=2 -> 4 optimizer steps/epoch. Check every step.
    curriculum = (CurriculumStage("s1", epochs=2, freeze_backbone=False, use_supcon=True),)
    train, val = _loaders()

    # Run A: guard trips mid-epoch (clock jumps huge on the first in-epoch check).
    model_a, loss_a = _model_loss()
    cfg_a = _cfg(tmp_path, curriculum, time_guard_check_steps=1)
    trainer_a = Trainer(model_a, loss_a, cfg_a, clock=FakeClock([0.0, 1e9]))
    res_a = trainer_a.fit(train, val, resume="none")
    assert res_a.status == "time_guard"
    assert res_a.last_epoch == -1  # epoch 0 was interrupted -> not counted
    assert res_a.history == []  # never reached end-of-epoch eval

    # Run B: resume auto -> redoes epoch 0 from scratch, then epoch 1.
    model_b, loss_b = _model_loss()
    cfg_b = _cfg(tmp_path, curriculum)
    trainer_b = Trainer(model_b, loss_b, cfg_b, clock=FakeClock([0.0]))
    res_b = trainer_b.fit(train, val, resume="auto")
    assert res_b.status == "completed"
    assert [h["epoch"] for h in res_b.history] == [0, 1]


def test_resume_continues_from_checkpoint(tmp_path):
    curriculum = (CurriculumStage("s1", epochs=3, freeze_backbone=False, use_supcon=True),)
    train, val = _loaders()

    # Run A: stop via time-guard after epoch 0.
    model_a, loss_a = _model_loss()
    trainer_a = Trainer(model_a, loss_a, _cfg(tmp_path, curriculum),
                        clock=FakeClock([0.0, 1e9]))
    res_a = trainer_a.fit(train, val, resume="none")
    assert res_a.status == "time_guard" and res_a.last_epoch == 0

    # Run B: resume auto; clock never trips guard -> finishes epochs 1..2.
    model_b, loss_b = _model_loss()
    trainer_b = Trainer(model_b, loss_b, _cfg(tmp_path, curriculum),
                        clock=FakeClock([0.0]))
    res_b = trainer_b.fit(train, val, resume="auto")
    assert res_b.status == "completed"
    assert res_b.last_epoch == 2
    assert [h["epoch"] for h in res_b.history] == [1, 2]  # did not redo epoch 0


def test_early_stop_on_plateau(tmp_path):
    curriculum = (CurriculumStage("s1", epochs=10, freeze_backbone=True, use_supcon=False),)
    model, loss = _model_loss()
    trainer = Trainer(model, loss, _cfg(tmp_path, curriculum, patience=1))
    train, val = _loaders()
    result = trainer.fit(train, val, resume="none")
    # patience=1: stops soon after val mAP stops improving (well before 10 epochs)
    assert result.status in {"early_stop", "completed"}
    assert result.last_epoch < 9
