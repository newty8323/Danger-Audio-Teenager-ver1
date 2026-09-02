#!/usr/bin/env python3
"""Compare the verified PyTorch 14fc6a69 separator with its ONNX core.

Writes both vocal stems and a JSON result.  Optionally transcribes both stems
with the same Whisper Base model; this is the quality check needed before any
Android work starts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from benchmark_onnx_demucs_core import (
    ROOT, SR, BLOCK, VOCALS_INDEX, _load_segment, _reconstruct_vocals,
    _stft_for_hybrid_demucs,
)


def _transcribe(wav: Path) -> tuple[str, float]:
    cli = ROOT / "artifacts/ver1/bin/whisper-cli"
    model = ROOT / "artifacts/ver1/whisper/ggml-base.bin"
    started = time.perf_counter()
    done = subprocess.run([str(cli), "-m", str(model), "-f", str(wav), "-l", "ko", "-nt", "-np"],
                          capture_output=True, check=True)
    text = "\n".join(line.removeprefix("-").strip() for line in
                     done.stdout.decode("utf-8", errors="replace").splitlines() if line.strip())
    return text, time.perf_counter() - started


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--whisper", action="store_true")
    p.add_argument("--model", type=Path,
                   default=ROOT / "artifacts/onnx/demucs_mdx_extra_q/hybrid_core_4s.fp16.onnx")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/onnx/parity")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mix = _load_segment(args.audio, args.start)

    import sys
    sys.path.insert(0, str(ROOT / "vendor/demucs-python"))
    from demucs.pretrained import get_model
    reference_model = get_model("14fc6a69", repo=ROOT / "artifacts/ver1/demucs-quantized").eval()
    with torch.inference_mode():
        started = time.perf_counter()
        reference = reference_model(torch.from_numpy(mix[None]))[0, VOCALS_INDEX].numpy()
        pytorch_seconds = time.perf_counter() - started

    session = ort.InferenceSession(
        str(args.model),
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    spec, mag = _stft_for_hybrid_demucs(mix)
    started = time.perf_counter()
    mags, waves = session.run(None, {"mixture_magnitude": mag.numpy(), "mixture_waveform": mix[None]})
    onnx_seconds = time.perf_counter() - started
    candidate = _reconstruct_vocals(spec, mags, waves, BLOCK)
    error = reference - candidate
    sdr = 10 * np.log10((np.mean(reference ** 2) + 1e-12) / (np.mean(error ** 2) + 1e-12))
    tag = f"{args.audio.stem}_{args.start:g}s"
    model_tag = args.model.stem
    reference_wav = args.out_dir / f"{tag}_pytorch-vocals.wav"
    onnx_wav = args.out_dir / f"{tag}_{model_tag}_vocals.wav"
    sf.write(reference_wav, reference.T, SR)
    sf.write(onnx_wav, candidate.T, SR)
    report = {"audio": str(args.audio), "start_seconds": args.start,
              "pytorch_vocals_wav": str(reference_wav), "onnx_vocals_wav": str(onnx_wav),
              "onnx_model": str(args.model), "pytorch_model_ms": round(pytorch_seconds * 1000, 3),
              "onnx_core_ms": round(onnx_seconds * 1000, 3),
              "max_abs_error": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error ** 2))),
              "sdr_db": float(sdr)}
    if args.whisper:
        report["pytorch_transcript"], pt = _transcribe(reference_wav)
        report["onnx_transcript"], ot = _transcribe(onnx_wav)
        report["pytorch_whisper_ms"] = round(pt * 1000, 3)
        report["onnx_whisper_ms"] = round(ot * 1000, 3)
    path = args.out_dir / f"{tag}_parity.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
