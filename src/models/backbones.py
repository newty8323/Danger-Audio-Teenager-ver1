"""Frame-level backbones (spec §5).

A backbone maps a log-mel spectrogram (B, 1, F, T) to frame embeddings
(B, T', D) that MIL attention then pools. The spec's primary backbone is BEATs
(with PANNs CNN14 / AST baselines); those require external pretrained weights
and are wired via :func:`build_backbone` once available.

``ConvFrameBackbone`` is a small, self-contained CNN that runs on CPU/MPS so the
full model is trainable and testable today (strategy A, heads-only local runs).
It is a baseline, not CNN14.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torchaudio
from torch import nn


@runtime_checkable
class FrameBackbone(Protocol):
    """Maps (B, 1, F, T) -> (B, T', out_dim)."""

    out_dim: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ConvFrameBackbone(nn.Module):
    def __init__(self, in_ch: int = 1, out_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            _conv_block(in_ch, 64),
            _conv_block(64, 128),
            _conv_block(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)  # (B, out_dim, F', T')
        h = h.mean(dim=2)  # average over frequency -> (B, out_dim, T')
        return h.transpose(1, 2).contiguous()  # (B, T', out_dim)

    def layerwise_groups(self) -> list[nn.Module]:
        """Ordered input->output layer groups for layer-wise LR decay (spec §6)."""
        return list(self.net.children())


class MFCCBiLSTMBackbone(nn.Module):
    """Classic weight-free baseline (spec §5): log-mel -> MFCC -> BiLSTM.

    Applies a DCT to the log-mel to get MFCCs, then a bidirectional LSTM over
    time. Frame count T is preserved. out_dim = 2 * hidden (bidirectional).
    """

    def __init__(self, out_dim: int = 256, n_mels: int = 128, n_mfcc: int = 40) -> None:
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError("out_dim must be even (2 * lstm hidden)")
        self.out_dim = out_dim
        hidden = out_dim // 2
        # Fixed orthonormal DCT-II matrix (n_mels, n_mfcc); not a learned param.
        dct = torchaudio.functional.create_dct(n_mfcc, n_mels, norm="ortho")
        self.register_buffer("dct", dct)
        self.lstm = nn.LSTM(n_mfcc, hidden, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.squeeze(1).transpose(1, 2)  # (B, 1, F, T) -> (B, T, F)
        mfcc = h @ self.dct  # (B, T, n_mfcc)
        frames, _ = self.lstm(mfcc)  # (B, T, 2*hidden)
        return frames.contiguous()


class PassthroughBackbone(nn.Module):
    """Identity backbone for precomputed frame embeddings (spec §5 strategy A).

    When the per-clip feature is ALREADY a frame-embedding sequence (e.g. frozen
    BEATs output, (B, T, D)), there is nothing to extract — pass it straight to
    MIL pooling. Used for the frozen-BEATs head-only path.
    """

    def __init__(self, out_dim: int = 768) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B, T, D); tolerate (B, 1, D, T) stored like a log-mel
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1).transpose(1, 2)
        return x.contiguous()


_NOT_WIRED = {"panns", "cnn14", "ast"}


def build_backbone(name: str = "conv", **kwargs) -> nn.Module:
    if name == "conv":
        return ConvFrameBackbone(**kwargs)
    if name == "mfcc_bilstm":
        return MFCCBiLSTMBackbone(**kwargs)
    if name in ("passthrough", "beats"):
        # "beats" trains on precomputed frozen-BEATs frame embeddings (see
        # models.beats_extractor); the backbone itself is an identity here.
        return PassthroughBackbone(**kwargs)
    if name in _NOT_WIRED:
        raise NotImplementedError(
            f"backbone {name!r} needs external pretrained weights and an adapter "
            f"emitting frame embeddings (B, T', D). See references [panns]."
        )
    raise ValueError(f"unknown backbone {name!r}")
