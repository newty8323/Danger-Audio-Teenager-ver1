"""Tiny student models for BEATs distillation (violence trigger, always-on candidate).

Student takes the SAME raw 16 kHz waveform as the BEATs teacher (alignment), computes its
own log-mel internally, and emits a 256-d embedding (for feature distillation vs the
teacher's projection) + 4 violence logits. Target ~1-3M params so it can run continuously
on a low-power core / serve as the on-device gate.

Kept deliberately simple and self-contained in distill/ (no dependency on src/ model code).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class LogMel(nn.Module):
    """Raw waveform (B, N) @16kHz -> log-mel (B, 1, n_mels, T). Fixed, non-trainable."""

    def __init__(self, sr: int = 16000, n_fft: int = 400, hop: int = 160, n_mels: int = 64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        m = self.mel(wav)                       # (B, n_mels, T)
        m = torch.log(m.clamp_min(1e-6))
        return m.unsqueeze(1)                   # (B, 1, n_mels, T)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class TinyMelCNN(nn.Module):
    """~1-2M param log-mel CNN. forward -> {"logits": (B, C), "embeddings": (B, emb_dim)}.

    emb_dim matches the teacher projection (256) so feature distillation needs no adapter.
    The loss itself is cosine distance, not MSE — see train_distill.py:155."""

    def __init__(self, num_classes: int = 4, n_mels: int = 64, widths=(32, 64, 128), emb_dim: int = 256):
        super().__init__()
        self.logmel = LogMel(n_mels=n_mels)
        self.bn_in = nn.BatchNorm2d(1)
        chans = [1, *widths]
        self.features = nn.Sequential(*[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(widths[-1], emb_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, wav: torch.Tensor, return_projection: bool = True) -> dict:
        x = self.bn_in(self.logmel(wav))
        x = self.features(x)
        x = self.pool(x).flatten(1)             # (B, widths[-1])
        emb = self.proj(x)                      # (B, emb_dim)
        out = {"logits": self.classifier(emb)}
        if return_projection:
            out["embeddings"] = emb
        return out

    def num_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6


if __name__ == "__main__":  # smoke
    m = TinyMelCNN()
    x = torch.randn(2, 160000)
    o = m(x)
    print("params(M):", round(m.num_params(), 3),
          "logits:", tuple(o["logits"].shape), "emb:", tuple(o["embeddings"].shape))
