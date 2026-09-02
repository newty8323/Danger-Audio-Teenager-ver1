"""BEATs fine-tuning path (spec §5 strategy B) — raw audio → trainable BEATs → frames.

The frozen-feature path (`PassthroughBackbone`) consumes precomputed BEATs embeddings and
cannot adapt the backbone. Fine-tuning needs the actual BEATs model in the graph, running
on raw 16 kHz waveforms with gradients through the TOP-k transformer blocks (strategy B:
unfreeze top blocks, keep the lower/patch layers frozen — cheap + stable on a T4).

`build_finetune_model()` assembles a ready-to-train `HarmModel` by swapping its passthrough
backbone for `BEATsRawBackbone`, warm-starting MIL+heads from a frozen-head checkpoint so
fine-tuning starts from the strong 0.70M-head solution rather than scratch.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.beats import BEATs, BEATsConfig
from models.beats_extractor import DEFAULT_CKPT
from models.harm_model import HarmModel, ModelConfig


class BEATsRawBackbone(nn.Module):
    """Raw 16 kHz waveform (B, N) -> BEATs frame embeddings (B, T, 768).

    All BEATs params are frozen except the top ``unfreeze_top_k`` encoder layers and the
    encoder's final layer norm — that is strategy B (top-block fine-tuning)."""

    out_dim = 768
    manages_own_freezing = True  # opt out of Trainer's all-or-nothing stage freeze

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT, unfreeze_top_k: int = 4,
                 use_layers: int | None = None) -> None:
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(ckpt["model"])
        self.beats.predictor = None  # encoder frame outputs, not the AudioSet head
        # Depth reduction (lightweighting): keep only the first ``use_layers`` of the 12
        # encoder layers — drops params AND inference compute; uses the layer-k intermediate
        # representation. None = full 12 (unchanged default).
        if use_layers is not None:
            self.beats.encoder.layers = self.beats.encoder.layers[:use_layers]
        self.unfreeze_top_k = unfreeze_top_k
        self._set_trainable()

    def _set_trainable(self) -> None:
        for p in self.beats.parameters():
            p.requires_grad = False
        layers = self.beats.encoder.layers
        for layer in layers[len(layers) - self.unfreeze_top_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        # the encoder's final layer_norm (post-block) also adapts, if present
        ln = getattr(self.beats.encoder, "layer_norm", None)
        if isinstance(ln, nn.Module):
            for p in ln.parameters():
                p.requires_grad = True

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.beats.parameters() if p.requires_grad)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # (B, N) float32 in [-1,1] @16kHz -> (B, T, 768); gradients flow to top-k layers only
        x, _ = self.beats.extract_features(waveform)
        return x.contiguous()


def build_finetune_model(num_classes: int, head_ckpt: str | Path | None = None,
                         beats_ckpt: str | Path = DEFAULT_CKPT, unfreeze_top_k: int = 4,
                         map_location: str = "cpu", use_layers: int | None = None) -> HarmModel:
    """HarmModel with a trainable BEATs backbone; MIL+heads warm-started from ``head_ckpt``
    (a frozen-BEATs/passthrough checkpoint), the backbone from the original BEATs weights.
    ``use_layers`` truncates BEATs to its first k encoder layers (depth-reduction experiment)."""
    model = HarmModel(num_classes, ModelConfig(backbone="passthrough", backbone_out_dim=768))
    if head_ckpt is not None:
        ck = torch.load(head_ckpt, map_location=map_location, weights_only=False)
        # passthrough backbone has no params -> only MIL/classifier/projection load
        model.load_state_dict(ck["model"], strict=True)
    model.backbone = BEATsRawBackbone(beats_ckpt, unfreeze_top_k=unfreeze_top_k,
                                      use_layers=use_layers)
    return model
