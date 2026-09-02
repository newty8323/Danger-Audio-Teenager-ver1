#!/usr/bin/env python3
"""Create a separately stored INT8 Demucs candidate for Android.

This script never alters the verified FP32 baseline asset.  It uses representative
audio only to calibrate learned Demucs layers, then writes a new candidate as
``demucs_4s.int8.onnx``.  The Android Gradle ``demucs-int8`` profile packages
that file in place of the FP32 file; the baseline APK remains reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from benchmark_onnx_demucs_core import ROOT, _load_segment, _stft_for_hybrid_demucs


ANDROID_ASSETS = ROOT / "android-onnx-benchmark/app/src/main/assets"
BASELINE = ANDROID_ASSETS / "demucs_4s.onnx"
CANDIDATE = ANDROID_ASSETS / "demucs_4s.int8.onnx"


class AudioCalibrationReader(CalibrationDataReader):
    def __init__(self, files: list[Path]):
        self.items: list[dict[str, np.ndarray]] = []
        for path in files:
            # Multiple locations prevent the calibration from representing only
            # a silent introduction.  Missing later samples are zero-padded by
            # the shared loader, which is still a valid four-second input.
            for start in (0.0, 5.0, 20.0):
                wave = _load_segment(path, start)
                _, magnitude = _stft_for_hybrid_demucs(wave)
                self.items.append({
                    "mixture_magnitude": magnitude.numpy(),
                    "mixture_waveform": wave[None],
                })
        self.index = 0

    def get_next(self):
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path,
                        help="two or more representative speech/music files for calibration")
    parser.add_argument("--model", type=Path, default=BASELINE,
                        help="verified FP32 Android baseline (never modified)")
    parser.add_argument("--out", type=Path, default=CANDIDATE,
                        help="new INT8 candidate file; must not be the baseline")
    args = parser.parse_args()
    missing = [str(path) for path in [args.model, *args.audio] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing file(s):\n  - " + "\n  - ".join(missing))
    if args.model.resolve() == args.out.resolve():
        raise ValueError("Refusing to overwrite the FP32 baseline. Choose a different --out path.")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        str(args.model), str(args.out), AudioCalibrationReader(args.audio),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm", "LSTM"],
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "AddQDQPairToWeight": True,
            "DedicatedQDQPair": True,
        },
    )
    print("INT8 Demucs candidate created; baseline is unchanged.")
    for label, path in (("baseline", args.model), ("candidate", args.out)):
        print(f"{label}: {path}\n  {path.stat().st_size / 1024 / 1024:.1f} MiB\n  sha256={digest(path)}")
    print("Next: compare the two outputs before installing the demucs-int8 APK.")


if __name__ == "__main__":
    main()
