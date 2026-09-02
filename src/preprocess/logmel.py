"""Log-mel spectrogram extraction (spec §4).

STFT (n_fft=1024, hop=320, Hann) -> mel (128 bands, 50-8000 Hz) ->
log(mel + 1e-6). Output shape (1, n_mels, T).
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T

from preprocess.config import PreprocessConfig


class LogMelExtractor:
    def __init__(self, cfg: PreprocessConfig | None = None) -> None:
        self.cfg = cfg or PreprocessConfig()
        self.mel = T.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            win_length=self.cfg.win_length,
            hop_length=self.cfg.hop_length,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            power=2.0,
            center=True,
            window_fn=torch.hann_window,
        )
        self.mel.eval()

    @torch.no_grad()
    def __call__(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return log-mel of shape (1, n_mels, T) as float32 numpy."""
        wav = self._as_tensor(waveform)
        mel = self.mel(wav)  # (1, n_mels, T)
        logmel = torch.log(mel + self.cfg.log_offset)
        return logmel.squeeze(0).unsqueeze(0).to(torch.float32).cpu().numpy()

    @staticmethod
    def _as_tensor(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        else:
            wav = waveform.to(torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, N)
        return wav
