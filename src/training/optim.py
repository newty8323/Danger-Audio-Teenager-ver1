"""Optimizer and LR schedule builders (spec §6).

- AdamW with separate LR for heads (1e-4) and backbone (1e-5); optional
  layer-wise LR decay (0.9) across the backbone's ordered child modules (earlier
  layers get a smaller LR). Bias and normalization (1-D) params get no weight
  decay — standard practice.
- Warmup (5% of total steps) then cosine decay to 0.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from training.config import TrainConfig


def _is_no_decay(param: torch.Tensor) -> bool:
    return param.ndim <= 1  # biases and norm weights


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with head/backbone LR split, layer-wise decay, and no-wd on 1-D params."""
    backbone = getattr(model, "backbone", None)
    backbone_params = set(id(p) for p in backbone.parameters()) if backbone is not None else set()

    groups: list[dict] = []

    # Backbone groups with layer-wise decay across ordered layers. A backbone can
    # expose ``layerwise_groups()`` (input->output) for meaningful decay; otherwise
    # it is treated as a single group at lr_backbone.
    if backbone is not None:
        if hasattr(backbone, "layerwise_groups"):
            layers = list(backbone.layerwise_groups())
        else:
            layers = [backbone]
        n = len(layers)
        for depth, layer in enumerate(layers):
            # last layer (closest to head) -> full lr_backbone; earlier -> decayed
            scale = cfg.layer_decay ** (n - 1 - depth)
            lr = cfg.lr_backbone * scale
            _add_param_groups(groups, layer.parameters(), lr, cfg.weight_decay)

    # Head groups: everything not in the backbone.
    head_params = [p for p in model.parameters() if id(p) not in backbone_params]
    _add_param_groups(groups, head_params, cfg.lr_heads, cfg.weight_decay)

    # Drop empty groups (e.g. a frozen module with no params).
    groups = [g for g in groups if len(g["params"]) > 0]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def _add_param_groups(groups: list[dict], params, lr: float, weight_decay: float) -> None:
    params = list(params)
    decay = [p for p in params if not _is_no_decay(p)]
    no_decay = [p for p in params if _is_no_decay(p)]
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup (warmup_pct) then cosine decay to 0, stepped per optimizer step."""
    warmup_steps = max(1, int(total_steps * cfg.warmup_pct))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
