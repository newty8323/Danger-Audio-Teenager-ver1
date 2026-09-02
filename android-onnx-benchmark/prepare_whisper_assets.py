#!/usr/bin/env python3
"""Prepare the fixed Whisper Base ONNX assets used by the Android bench.

The decoder export has external weight data. Android assets must contain one
self-contained ONNX file, so this script first embeds that data and then adds
the encoder, the exact Whisper 80-band mel filterbank and the decoder vocab.
It does not run inference.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from transformers import WhisperFeatureExtractor


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "app/src/main/assets"
ONNX_DIR = ROOT / "artifacts/onnx/whisper_base_fp16"
HF_SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--openai--whisper-base/snapshots"
)


def _snapshot() -> Path:
    choices = sorted(HF_SNAPSHOT.glob("*"))
    if not choices:
        raise FileNotFoundError("Whisper Base tokenizer cache was not found")
    return choices[-1]


def main() -> None:
    # Exact original 30-second Whisper Base ONNX graphs.  This is the
    # accuracy baseline; no encoder sequence is shortened or rewritten.
    encoder = ONNX_DIR / "encoder.fp32.onnx"
    decoder = ONNX_DIR / "decoder_initial.fp32.onnx"
    if not all(path.is_file() for path in (encoder, decoder)):
        raise FileNotFoundError("Export the original FP32 Whisper Base ONNX graphs first")

    ASSETS.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (encoder, "whisper_encoder_30s.onnx"),
        (decoder, "whisper_decoder_30s.onnx"),
    ):
        # Keep decoders as one file so Ort can load them after asset copying.
        model = onnx.load_model(str(source), load_external_data=True)
        for initializer in model.graph.initializer:
            initializer.data_location = onnx.TensorProto.DEFAULT
            del initializer.external_data[:]
        onnx.save_model(model, str(ASSETS / destination))

    # The checkpoint is already cached locally; do not make a network request
    # while preparing an offline Android experiment.
    filters = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-base", local_files_only=True
    ).mel_filters
    # Android indexes [mel_band, fft_bin]; HF stores [fft_bin, mel_band].
    np.asarray(filters.T, dtype="<f4").tofile(ASSETS / "whisper_mel_filters.f32")
    shutil.copy2(_snapshot() / "vocab.json", ASSETS / "whisper_vocab.json")

    # Cached FP16 assets belong to the previous experiment and would add about
    # 230 MiB to the APK while the current runner does not use them.
    for stale in (
        "whisper_encoder.onnx",
        "whisper_decoder.onnx",
        "whisper_encoder_8s.onnx",
        "whisper_decoder_8s.onnx",
        "whisper_encoder_fp16.onnx",
        "whisper_decoder_initial_cache_fp16.onnx",
        "whisper_decoder_cached_fp16.onnx",
    ):
        (ASSETS / stale).unlink(missing_ok=True)

    print("Whisper Android original 30-second FP32 assets are ready:")
    for name in ("whisper_encoder_30s.onnx", "whisper_decoder_30s.onnx", "whisper_mel_filters.f32", "whisper_vocab.json"):
        path = ASSETS / name
        print(f"  {name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
