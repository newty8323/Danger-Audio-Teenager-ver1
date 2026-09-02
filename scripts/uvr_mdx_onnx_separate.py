#!/usr/bin/env python3
"""Standalone 2-stem UVR MDX-Net ONNX experiment for the Ver1 ASR branch.

The selected model predicts an instrumental stem; vocals are ``mixture -
instrumental``.  This utility intentionally does not alter the live application.
It writes an isolated-vocals WAV and a compact JSON report for comparison with
the current PyTorch MDX Extra Demucs front end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts/onnx/uvr_mdx_net_inst_hq_3/UVR-MDX-NET-Inst_HQ_3.onnx"
SAMPLE_RATE = 44_100
HOP = 1_024
DIM_T = 256


def _providers() -> list[str]:
    available = ort.get_available_providers()
    return (["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in available else ["CPUExecutionProvider"])


class UVRMDXNet:
    """MDX-Net ONNX wrapper with its STFT/ISTFT pre- and post-processing."""

    def __init__(self, model_path: Path):
        self.providers = _providers()
        self.session = ort.InferenceSession(str(model_path), providers=self.providers)
        shape = self.session.get_inputs()[0].shape
        self.dim_f = int(shape[2])
        self.dim_t = int(shape[3])
        # MDX-Net exports use ``dim_f = n_fft / 2``; one Nyquist bin is
        # reconstructed during ISTFT.  This supports both HQ-3 (3072/6144)
        # and the smaller MDXNET-3 (2048/4096) without a separate runner.
        self.n_fft = self.dim_f * 2
        self.chunk_samples = HOP * (self.dim_t - 1)
        self.window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)

    def _stft(self, waves: np.ndarray) -> np.ndarray:
        # [batch, 2, samples] -> [batch, 4, dim_f, dim_t]
        spectra = []
        for chunk in waves.reshape(-1, self.chunk_samples):
            spec = librosa.stft(chunk, n_fft=self.n_fft, hop_length=HOP, window=self.window,
                                center=True)
            spectra.append(np.stack((spec.real, spec.imag), axis=0))
        packed = np.asarray(spectra, dtype=np.float32)
        # [batch*2, 2, freq, time] -> [batch, 4, freq, time]
        packed = packed.reshape(-1, 2, 2, self.n_fft // 2 + 1, self.dim_t)
        return packed.reshape(-1, 4, self.n_fft // 2 + 1, self.dim_t)[:, :, :self.dim_f]

    def _istft(self, spectra: np.ndarray) -> np.ndarray:
        # Model output is a 4-channel (L-real/L-imag/R-real/R-imag) spectrum.
        pad = np.zeros((len(spectra), 4, self.n_fft // 2 + 1 - self.dim_f, self.dim_t), dtype=np.float32)
        full = np.concatenate((spectra, pad), axis=2).reshape(
            -1, 2, 2, self.n_fft // 2 + 1, self.dim_t
        )
        out = []
        for complex_parts in full.reshape(-1, 2, self.n_fft // 2 + 1, self.dim_t):
            spec = complex_parts[0] + 1j * complex_parts[1]
            out.append(librosa.istft(spec, hop_length=HOP, window=self.window,
                                     center=True, length=self.chunk_samples))
        return np.asarray(out, dtype=np.float32).reshape(-1, 2, self.chunk_samples)

    def separate(self, mix: np.ndarray) -> np.ndarray:
        """Return a stereo vocal stem for a [2, samples] 44.1 kHz waveform."""
        if mix.ndim != 2 or mix.shape[0] != 2:
            raise ValueError("mix must have shape [2, samples]")
        original_samples = mix.shape[-1]
        trim = self.n_fft // 2
        generated = self.chunk_samples - 2 * trim
        pad = (-original_samples) % generated
        padded = np.concatenate((np.zeros((2, trim), np.float32), mix,
                                 np.zeros((2, pad + trim), np.float32)), axis=1)
        windows = np.stack([padded[:, start:start + self.chunk_samples]
                            for start in range(0, original_samples + pad, generated)])
        predicted_instrumental = self._istft(
            self.session.run(None, {"input": self._stft(windows)})[0]
        )[:, :, trim:-trim].transpose(1, 0, 2).reshape(2, -1)[:, :original_samples]
        return (mix - predicted_instrumental).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/onnx/experiments")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--duration", type=float, default=4.0, help="input length in seconds")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    mix, _ = librosa.load(str(args.file), sr=SAMPLE_RATE, mono=False,
                          offset=max(0.0, args.start), duration=args.duration)
    if mix.ndim == 1:
        mix = np.stack((mix, mix))
    if not mix.size:
        raise ValueError("selected range contains no audio")
    separator = UVRMDXNet(args.model)
    started = time.perf_counter()
    vocals = separator.separate(np.asarray(mix[:2], dtype=np.float32))
    elapsed = time.perf_counter() - started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.file.stem}_{args.start:g}s_{args.duration:g}s_uvr-vocals"
    wav_path = args.out_dir / f"{stem}.wav"
    report_path = args.out_dir / f"{stem}.json"
    sf.write(wav_path, vocals.T, SAMPLE_RATE)
    report = {
        "input": str(args.file), "model": str(args.model), "providers": separator.providers,
        "start_sec": args.start, "duration_sec": len(mix[0]) / SAMPLE_RATE,
        "separation_ms": round(elapsed * 1000, 3),
        "rtf": round(elapsed / (len(mix[0]) / SAMPLE_RATE), 4),
        "vocals_wav": str(wav_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
