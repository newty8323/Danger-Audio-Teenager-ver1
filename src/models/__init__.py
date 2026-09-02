"""Model: frame backbone + MIL attention pooling + classifier/projection heads (spec §5)."""

from models.backbones import (
    ConvFrameBackbone,
    FrameBackbone,
    MFCCBiLSTMBackbone,
    PassthroughBackbone,
    build_backbone,
)
from models.harm_model import HarmModel, ModelConfig
from models.heads import ClassifierHead, ProjectionHead
from models.pooling import MILAttentionPooling

__all__ = [
    "MILAttentionPooling",
    "ClassifierHead",
    "ProjectionHead",
    "FrameBackbone",
    "ConvFrameBackbone",
    "MFCCBiLSTMBackbone",
    "PassthroughBackbone",
    "build_backbone",
    "HarmModel",
    "ModelConfig",
]
