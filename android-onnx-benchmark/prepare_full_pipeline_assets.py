#!/usr/bin/env python3
"""Package verified ONNX models and metadata for the Android live-pipeline app.

This is deliberately separate from the fixed-fixture benchmark preparation:
the live app has no fixture waveforms or reference tensors, only the models and
the metadata required for local Stage-1/2 inference.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "app/src/main/assets"

FILES = {
    ROOT / "artifacts/onnx/demucs_mdx_extra_q/hybrid_core_4s.fp32.android.onnx": "demucs_4s.onnx",
    ROOT / "artifacts/onnx/whisper_base_fp16/encoder.fp32.onnx": "whisper_encoder_30s.onnx",
    ROOT / "artifacts/onnx/whisper_base_fp16/decoder_initial.fp32.onnx": "whisper_decoder_30s.onnx",
    ROOT / "artifacts/onnx/ced_mini_vio/model.fp32.onnx": "ced_mini_vio.onnx",
    ROOT / "artifacts/onnx/koelectra_small_harm_asraug_slang/model.int8.onnx": "koelectra_harm.int8.onnx",
    ROOT / "artifacts/koelectra_small_harm_asraug_slang/tokenizer.json": "koelectra_tokenizer.json",
    ROOT / "artifacts/koelectra_small_harm_asraug_slang/cats.json": "koelectra_cats.json",
    ROOT / "artifacts/cascade_thresholds.json": "cascade_thresholds.json",
}


def _pack(source: Path, destination: Path) -> None:
    """Embed ONNX external weights so Android can load one copied asset file."""
    if source.suffix != ".onnx":
        shutil.copy2(source, destination)
        return
    import onnx

    model = onnx.load_model(str(source), load_external_data=True)
    for initializer in model.graph.initializer:
        initializer.data_location = onnx.TensorProto.DEFAULT
        del initializer.external_data[:]
    onnx.save_model(model, str(destination))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    missing = [str(source) for source in FILES if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing full-pipeline asset(s):\n  - " + "\n  - ".join(missing))
    for source, name in FILES.items():
        _pack(source, ASSETS / name)
    # Must match CEDRawBackbone's torchaudio MelSpectrogram defaults exactly:
    # 16 kHz, n_fft/win=512, hop=160, 64 bands, HTK scale, no normalization.
    import torchaudio.functional as AF
    filters = AF.melscale_fbanks(
        n_freqs=257, f_min=0.0, f_max=8000.0, n_mels=64,
        sample_rate=16_000, norm=None, mel_scale="htk",
    )
    np.asarray(filters.T, dtype="<f4").tofile(ASSETS / "ced_mel_filters.f32")
    print("Android live-pipeline assets are ready:")
    for name in FILES.values():
        path = ASSETS / name
        print(f"  {name}: {path.stat().st_size / 1024 / 1024:.1f} MiB")
    print("  ced_mel_filters.f32: 0.1 MiB")


if __name__ == "__main__":
    main()
