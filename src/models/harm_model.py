"""Harm-detection model assembly (spec §5).

    log-mel -> backbone -> frame embeddings -> MIL attention -> z
             -> classifier head (logits)  +  projection head (SupCon embedding)

``forward`` returns logits, attention weights (temporal localization), the
pooled clip embedding, and — unless disabled — the projection embedding used by
the contrastive loss (dropped at inference).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.backbones import build_backbone
from models.heads import ClassifierHead, ProjectionHead
from models.pooling import MILAttentionPooling


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "conv"
    backbone_out_dim: int = 256
    attn_dim: int = 128
    classifier_hidden: int = 512
    dropout: float = 0.3
    proj_dim: int = 256


class HarmModel(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        if self.cfg.backbone in ("conv", "mfcc_bilstm", "passthrough", "beats"):
            self.backbone = build_backbone(self.cfg.backbone, out_dim=self.cfg.backbone_out_dim)
        else:
            self.backbone = build_backbone(self.cfg.backbone)
        dim = self.backbone.out_dim
        self.pool = MILAttentionPooling(dim, self.cfg.attn_dim)
        self.classifier = ClassifierHead(
            dim, num_classes, self.cfg.classifier_hidden, self.cfg.dropout
        )
        self.projection = ProjectionHead(dim, self.cfg.proj_dim)

    def forward(
        self, x: torch.Tensor, return_projection: bool = True
    ) -> dict[str, torch.Tensor]:
        frames = self.backbone(x)  # (B, T', D)
        z, attention = self.pool(frames)  # (B, D), (B, T')
        out = {
            "logits": self.classifier(z),  # (B, C)
            "attention": attention,
            "pooled": z,
        }
        if return_projection:
            out["embeddings"] = self.projection(z)  # (B, proj_dim)
        return out

    @classmethod
    def from_checkpoint(
        cls, path: str, num_classes: int, map_location: str | torch.device = "cpu"
    ) -> HarmModel:
        """Rebuild a model from a checkpoint, using its stored ModelConfig.

        Falls back to default ModelConfig for older checkpoints that predate the
        stored config, so inference doesn't silently mis-shape the architecture.

        weights_only=False unpickles arbitrary objects — only load checkpoints you
        produced/trust.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg_dict = ckpt.get("model_config")
        cfg = ModelConfig(**cfg_dict) if cfg_dict else ModelConfig()
        model = cls(num_classes, cfg)
        model.load_state_dict(ckpt["model"])
        return model.to(map_location)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # Force eval for BN/Dropout, but restore the prior mode so validating
        # mid-epoch doesn't silently leave the model in eval for the rest of training.
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self.forward(x, return_projection=False)["logits"])
        finally:
            self.train(was_training)
