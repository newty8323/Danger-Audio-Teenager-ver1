import torch

from models.harm_model import HarmModel, ModelConfig
from training.config import TrainConfig
from training.optim import build_optimizer, build_scheduler


def test_param_groups_have_head_and_backbone_lrs():
    model = HarmModel(num_classes=4, cfg=ModelConfig(backbone_out_dim=32))
    cfg = TrainConfig(lr_heads=1e-4, lr_backbone=1e-5, layer_decay=0.9)
    opt = build_optimizer(model, cfg)
    lrs = {round(g["lr"], 12) for g in opt.param_groups}
    assert round(cfg.lr_heads, 12) in lrs  # heads at full head LR
    assert round(cfg.lr_backbone, 12) in lrs  # last backbone block at full backbone LR
    # layer-wise decay produced a smaller backbone LR too
    assert any(g["lr"] < cfg.lr_backbone - 1e-12 for g in opt.param_groups)


def test_bias_and_norm_params_have_no_weight_decay():
    model = HarmModel(num_classes=4, cfg=ModelConfig(backbone_out_dim=32))
    cfg = TrainConfig()
    opt = build_optimizer(model, cfg)
    for g in opt.param_groups:
        if g["weight_decay"] == 0.0:
            assert all(p.ndim <= 1 for p in g["params"])  # only 1-D params get no wd
        else:
            assert all(p.ndim > 1 for p in g["params"])


def test_scheduler_warmup_then_cosine_decay():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    cfg = TrainConfig()
    total = 100
    sched = build_scheduler(opt, cfg, total_steps=total)
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    warmup_steps = max(1, int(total * cfg.warmup_pct))
    assert lrs[0] < lrs[warmup_steps]  # ramps up during warmup
    assert lrs[warmup_steps] > lrs[-1]  # decays afterwards
    assert lrs[-1] < 0.05  # cosine approaches 0
