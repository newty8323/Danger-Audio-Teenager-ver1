import torch

from models.harm_model import HarmModel, ModelConfig
from training.checkpoint import (
    TrainState,
    find_latest,
    load_checkpoint,
    save_checkpoint,
)
from training.config import TrainConfig
from training.optim import build_optimizer, build_scheduler


def _build():
    model = HarmModel(num_classes=4, cfg=ModelConfig(backbone_out_dim=32))
    cfg = TrainConfig()
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=50)
    return model, opt, sched


def test_find_latest(tmp_path):
    assert find_latest(tmp_path) is None
    model, opt, sched = _build()
    save_checkpoint(tmp_path / "last.ckpt", model, opt, sched, None,
                    TrainState(epoch=0, best_metric=0.5, epochs_no_improve=0))
    assert find_latest(tmp_path) is not None


def test_save_load_roundtrip_restores_weights_and_state(tmp_path):
    model, opt, sched = _build()
    path = tmp_path / "last.ckpt"
    save_checkpoint(path, model, opt, sched, None,
                    TrainState(epoch=7, best_metric=0.83, epochs_no_improve=3))

    fresh, opt2, sched2 = _build()
    # perturb fresh weights so we can tell load actually happened
    with torch.no_grad():
        for p in fresh.parameters():
            p.add_(1.0)
    state = load_checkpoint(path, fresh, opt2, sched2, None, restore_rng=False)

    assert state.epoch == 7
    assert abs(state.best_metric - 0.83) < 1e-9
    assert state.epochs_no_improve == 3
    for p_orig, p_loaded in zip(model.parameters(), fresh.parameters(), strict=True):
        torch.testing.assert_close(p_orig, p_loaded)


def test_from_checkpoint_restores_architecture(tmp_path):
    # A non-default architecture must round-trip so inference rebuilds the right shape.
    cfg = ModelConfig(backbone_out_dim=64, attn_dim=16, proj_dim=32)
    model = HarmModel(5, cfg)
    opt = build_optimizer(model, TrainConfig())
    sched = build_scheduler(opt, TrainConfig(), total_steps=10)
    path = tmp_path / "last.ckpt"
    save_checkpoint(path, model, opt, sched, None, TrainState(0, 0.0, 0))

    rebuilt = HarmModel.from_checkpoint(str(path), num_classes=5)
    assert rebuilt.cfg.backbone_out_dim == 64
    assert rebuilt.cfg.attn_dim == 16 and rebuilt.cfg.proj_dim == 32
    for a, b in zip(model.parameters(), rebuilt.parameters(), strict=True):
        torch.testing.assert_close(a.detach().cpu(), b.detach().cpu())


def test_from_checkpoint_restores_mfcc_bilstm_backbone(tmp_path):
    # The non-conv backbone must also round-trip (Major-1 failure class).
    cfg = ModelConfig(backbone="mfcc_bilstm", backbone_out_dim=64)
    model = HarmModel(4, cfg)
    opt = build_optimizer(model, TrainConfig())
    sched = build_scheduler(opt, TrainConfig(), total_steps=10)
    path = tmp_path / "last.ckpt"
    save_checkpoint(path, model, opt, sched, None, TrainState(0, 0.0, 0))

    rebuilt = HarmModel.from_checkpoint(str(path), num_classes=4)
    assert rebuilt.cfg.backbone == "mfcc_bilstm"
    from models.backbones import MFCCBiLSTMBackbone
    assert isinstance(rebuilt.backbone, MFCCBiLSTMBackbone)


def test_rng_state_restored(tmp_path):
    model, opt, sched = _build()
    path = tmp_path / "last.ckpt"
    save_checkpoint(path, model, opt, sched, None,
                    TrainState(epoch=0, best_metric=0.0, epochs_no_improve=0))
    a = torch.rand(5)  # advances the global RNG past the saved state
    load_checkpoint(path, model, restore_rng=True)
    b = torch.rand(5)  # RNG rewound to save time -> should reproduce `a`
    torch.testing.assert_close(a, b)
