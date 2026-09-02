#!/usr/bin/env python3
"""Place the verified ONNX model and streaming-context fixtures into assets."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_onnx_demucs_core import _load_segment, _reconstruct_vocals, _stft_for_hybrid_demucs


def _mono_demucs_input(audio: Path, start: float) -> tuple[np.ndarray, object, np.ndarray]:
    """Load one 4-second live interval exactly as the Android app receives it."""
    wave = _load_segment(audio, start)
    wave = np.repeat(wave.mean(axis=0, keepdims=True), 2, axis=0).astype(np.float32)
    mixture_spec, magnitude = _stft_for_hybrid_demucs(wave)
    return wave, mixture_spec, magnitude


def _reference_vocals(
    session: ort.InferenceSession, wave: np.ndarray, mixture_spec: object, magnitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    magnitudes, waveforms = session.run(None, {
        "mixture_magnitude": magnitude.numpy(), "mixture_waveform": wave[None],
    })
    vocals = _reconstruct_vocals(mixture_spec, magnitudes, waveforms, wave.shape[-1])
    frequency_only = _reconstruct_vocals(mixture_spec, magnitudes, np.zeros_like(waveforms), wave.shape[-1])
    vocals = np.repeat(vocals[:1], 2, axis=0)
    frequency_only = np.repeat(frequency_only[:1], 2, axis=0)
    waveform_only = np.repeat(waveforms[0, 3, :1], 2, axis=0)
    return vocals, frequency_only, waveform_only, (magnitudes, waveforms)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--start", type=float, default=5.0)
    p.add_argument("--model", type=Path,
                   default=ROOT / "artifacts/onnx/demucs_mdx_extra_q/hybrid_core_4s.fp16.onnx")
    args = p.parse_args()
    assets = Path(__file__).parent / "app/src/main/assets"
    model = args.model
    if not model.is_file() or not args.audio.is_file():
        raise FileNotFoundError("model or audio file is missing")
    # The benchmark runs Demucs for the newest 4-second interval only. The
    # preceding interval is stored as a verified vocal stem, mirroring the
    # rolling 8-second ASR buffer used by the eventual live app.
    if args.start < 4:
        raise ValueError("--start must be at least 4 seconds for 8-second context")
    previous_wave, previous_spec, previous_magnitude = _mono_demucs_input(args.audio, args.start - 4)
    wave, mixture_spec, magnitude = _mono_demucs_input(args.audio, args.start)
    shutil.copy2(model, assets / "model.onnx")
    magnitude.numpy().astype("<f4").tofile(assets / "fixture_magnitude.f32")
    wave.astype("<f4").tofile(assets / "fixture_waveform.f32")
    # Android emulator uses ORT's CPU execution path.  Use the same provider
    # for the reference fixture: CoreML partitions this hybrid graph
    # differently and is not a valid numerical reference for Android.
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    previous_vocals, _, _, _ = _reference_vocals(
        session, previous_wave, previous_spec, previous_magnitude
    )
    vocals, frequency_only, waveform_only, (magnitudes, waveforms) = _reference_vocals(
        session, wave, mixture_spec, magnitude
    )
    magnitudes.astype("<f4").tofile(assets / "fixture_expected_source_magnitudes.f32")
    waveforms.astype("<f4").tofile(assets / "fixture_expected_source_waveforms.f32")
    frequency_only.astype("<f4").tofile(assets / "fixture_expected_frequency.f32")
    waveform_only.astype("<f4").tofile(assets / "fixture_expected_waveform_branch.f32")
    vocals.astype("<f4").tofile(assets / "fixture_expected_vocals.f32")
    previous_vocals.astype("<f4").tofile(assets / "fixture_previous_vocals.f32")
    print(
        f"Prepared Android assets: retained context {args.start - 4:g}–{args.start:g}s, "
        f"new interval {args.start:g}–{args.start + 4:g}s"
    )


if __name__ == "__main__":
    main()
