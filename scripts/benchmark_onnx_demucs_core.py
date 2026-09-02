#!/usr/bin/env python3
"""Run the verified Hybrid Demucs ONNX core directly on an audio file.

This is deliberately *not* an Android app.  It is the measurement step before
an APK exists:

    waveform -> STFT -> ONNX Runtime core -> mixture phase + ISTFT -> vocals

The STFT/ISTFT are ordinary signal processing, not a PyTorch Demucs fallback.
Only the exported ONNX graph performs learned inference.  The graph is fixed to
four seconds because that is the intended streaming hop for the experiment.
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
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SR = 44_100
BLOCK_SECONDS = 4.0
BLOCK = int(SR * BLOCK_SECONDS)
NFFT = 4096
HOP = 1024
PAD = HOP // 2 * 3
VOCALS_INDEX = 3


def _stft_for_hybrid_demucs(mix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the STFT preparation in the exported 14fc6a69 Hybrid Demucs."""
    wave = torch.from_numpy(mix[None].astype(np.float32, copy=False))
    length = wave.shape[-1]
    frames = int(np.ceil(length / HOP))
    right = PAD + frames * HOP - length
    padded = F.pad(wave, (PAD, right), mode="reflect")
    flat = padded.reshape(-1, padded.shape[-1])
    stft = torch.stft(
        flat, NFFT, hop_length=HOP, win_length=NFFT,
        window=torch.hann_window(NFFT), normalized=True, center=True,
        return_complex=True, pad_mode="reflect",
    ).reshape(1, 2, NFFT // 2 + 1, -1)
    # HDemucs removes the Nyquist bin and the two frame margins before inference.
    stft = stft[..., :-1, 2:2 + frames]
    magnitude = stft.abs()
    return stft, magnitude


def _reconstruct_vocals(
    mixture_spec: torch.Tensor,
    source_magnitudes: np.ndarray,
    source_waveforms: np.ndarray,
    original_length: int,
) -> np.ndarray:
    """Apply the checkpoint's zero-iteration mixture-phase reconstruction."""
    vocal_mag = torch.from_numpy(source_magnitudes[:, VOCALS_INDEX])
    vocal_time = torch.from_numpy(source_waveforms[:, VOCALS_INDEX])
    vocal_spec = mixture_spec * vocal_mag / mixture_spec.abs().clamp_min(1e-8)
    vocal_spec = F.pad(vocal_spec, (0, 0, 0, 1))  # restore the Nyquist bin
    vocal_spec = F.pad(vocal_spec, (2, 2))        # restore hybrid frame margins
    expected = HOP * int(np.ceil(original_length / HOP)) + 2 * PAD
    flat = vocal_spec.reshape(-1, vocal_spec.shape[-2], vocal_spec.shape[-1])
    frequency_audio = torch.istft(
        flat, NFFT, hop_length=HOP, win_length=NFFT,
        window=torch.hann_window(NFFT), normalized=True, length=expected,
        center=True,
    ).reshape(1, 2, expected)[..., PAD:PAD + original_length]
    return (frequency_audio + vocal_time).squeeze(0).numpy()


def _load_segment(path: Path, start: float) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SR, mono=False, offset=start, duration=BLOCK_SECONDS)
    if audio.ndim == 1:
        audio = np.repeat(audio[None], 2, axis=0)
    if audio.shape[0] > 2:
        audio = audio[:2]
    if audio.shape[-1] < BLOCK:
        audio = np.pad(audio, ((0, 0), (0, BLOCK - audio.shape[-1])))
    return np.asarray(audio[:, :BLOCK], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "artifacts/onnx/demucs_mdx_extra_q/hybrid_core_4s.fp16.onnx")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "artifacts/onnx/experiments")
    parser.add_argument("--provider", choices=("coreml", "cpu"), default="coreml")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                 if args.provider == "coreml" else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(args.model), providers=providers)
    mix = _load_segment(args.audio, args.start)

    started = time.perf_counter()
    mixture_spec, magnitude = _stft_for_hybrid_demucs(mix)
    stft_seconds = time.perf_counter() - started
    started = time.perf_counter()
    source_magnitudes, source_waveforms = session.run(None, {
        "mixture_magnitude": magnitude.numpy(), "mixture_waveform": mix[None],
    })
    onnx_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vocals = _reconstruct_vocals(mixture_spec, source_magnitudes, source_waveforms, BLOCK)
    istft_seconds = time.perf_counter() - started

    stem = args.audio.stem.replace("/", "_")
    wav_path = args.out_dir / f"{stem}_{args.start:g}s_onnx-demucs-vocals.wav"
    sf.write(wav_path, vocals.T, SR)
    total = stft_seconds + onnx_seconds + istft_seconds
    report = {
        "audio": str(args.audio), "start_seconds": args.start,
        "input_seconds": BLOCK_SECONDS, "model": str(args.model),
        "providers": session.get_providers(), "stft_ms": round(stft_seconds * 1000, 3),
        "onnx_core_ms": round(onnx_seconds * 1000, 3),
        "istft_ms": round(istft_seconds * 1000, 3),
        "total_ms": round(total * 1000, 3), "rtf": round(total / BLOCK_SECONDS, 4),
        "vocals_wav": str(wav_path),
    }
    report_path = wav_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
