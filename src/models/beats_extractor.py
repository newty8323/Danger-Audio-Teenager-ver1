"""BEATs frozen frame-embedding extractor (spec §5 primary backbone, strategy A).

Wraps the vendored BEATs (MIT, microsoft/unilm) as a feature extractor: raw 16kHz
waveform -> (T, 768) frame embeddings. BEATs computes its own kaldi fbank
internally, so this consumes waveforms (not our log-mel). Intended use: precompute
BEATs frame embeddings as the per-clip feature, then train MIL-pool + heads on top
(frozen backbone = local/MPS-friendly). Full fine-tuning of BEATs needs a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.beats import BEATs, BEATsConfig

DEFAULT_CKPT = "weights/beats/BEATs_iter3_plus_AS2M.pt"


class BEATsExtractor:
    out_dim = 768

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT, device: str = "cpu") -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.predictor = None  # feature-extractor mode -> encoder frame outputs
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """(N,) or (B, N) float32 waveform in [-1, 1] @16kHz -> (B, T, 768) embeddings."""
        wav = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feats, _ = self.model.extract_features(wav.to(self.device))
        return feats.cpu().numpy()
