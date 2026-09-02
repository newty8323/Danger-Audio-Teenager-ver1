#!/usr/bin/env python3
"""Compare the Android FP32 and candidate Demucs cores on identical audio windows.

This is an output-equivalence gate, not an ASR-quality claim.  It reports
separate magnitude/waveform RMSE values and writes both reconstructed vocal WAVs
so that the user can listen before accepting the smaller APK.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from benchmark_onnx_demucs_core import (
    BLOCK,
    ROOT,
    _load_segment,
    _reconstruct_vocals,
    _stft_for_hybrid_demucs,
)


ASSETS = ROOT / "android-onnx-benchmark/app/src/main/assets"


def run(path: Path, magnitude: np.ndarray, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    magnitude_out, waveform_out = session.run(None, {
        "mixture_magnitude": magnitude,
        "mixture_waveform": waveform,
    })
    return magnitude_out, waveform_out, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--baseline", type=Path, default=ASSETS / "demucs_4s.onnx")
    parser.add_argument("--candidate", type=Path, default=ASSETS / "demucs_4s.int8.onnx")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/onnx/experiments/int8-demucs")
    args = parser.parse_args()
    missing = [str(path) for path in (args.audio, args.baseline, args.candidate) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing file(s):\n  - " + "\n  - ".join(missing))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mix = _load_segment(args.audio, args.start)
    spec, magnitude = _stft_for_hybrid_demucs(mix)
    inputs = magnitude.numpy(), mix[None]
    base_mag, base_wave, base_ms = run(args.baseline, *inputs)
    candidate_mag, candidate_wave, candidate_ms = run(args.candidate, *inputs)
    base_vocal = _reconstruct_vocals(spec, base_mag, base_wave, BLOCK)
    candidate_vocal = _reconstruct_vocals(spec, candidate_mag, candidate_wave, BLOCK)
    stem = args.audio.stem.replace("/", "_") + f"_{args.start:g}s"
    base_wav = args.out_dir / f"{stem}_fp32-vocals.wav"
    candidate_wav = args.out_dir / f"{stem}_candidate-vocals.wav"
    sf.write(base_wav, base_vocal.T, 44_100)
    sf.write(candidate_wav, candidate_vocal.T, 44_100)
    report = {
        "audio": str(args.audio), "start_seconds": args.start,
        "fp32_onnx_ms": round(base_ms, 2), "candidate_onnx_ms": round(candidate_ms, 2),
        "magnitude_rmse": float(np.sqrt(np.mean((base_mag - candidate_mag) ** 2))),
        "waveform_rmse": float(np.sqrt(np.mean((base_wave - candidate_wave) ** 2))),
        "vocal_rmse": float(np.sqrt(np.mean((base_vocal - candidate_vocal) ** 2))),
        "fp32_vocals": str(base_wav), "candidate_vocals": str(candidate_wav),
    }
    report_path = args.out_dir / f"{stem}_comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
