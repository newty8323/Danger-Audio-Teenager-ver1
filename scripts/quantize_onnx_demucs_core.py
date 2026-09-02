#!/usr/bin/env python3
"""Create an INT8 QDQ candidate from the verified FP32 Hybrid Demucs ONNX core.

This does not replace the FP16 reference.  It only creates a candidate which
must pass ``compare_demucs_onnx_parity.py`` before it can be considered for an
Android build.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from benchmark_onnx_demucs_core import ROOT, _load_segment, _stft_for_hybrid_demucs


class AudioCalibrationReader(CalibrationDataReader):
    def __init__(self, files: list[Path]):
        self.items = []
        for path in files:
            # Two sections per file give the calibrator speech/music variation
            # without treating test output as calibration data.
            for start in (0.0, 5.0):
                wave = _load_segment(path, start)
                _, magnitude = _stft_for_hybrid_demucs(wave)
                self.items.append({"mixture_magnitude": magnitude.numpy(),
                                   "mixture_waveform": wave[None]})
        self.index = 0

    def get_next(self):
        if self.index == len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+", type=Path, help="representative files for calibration")
    p.add_argument("--model", type=Path,
                   default=ROOT / "artifacts/onnx/demucs_mdx_extra_q/hybrid_core_4s.fp32.onnx")
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts/onnx/demucs_mdx_extra_q/hybrid_core_4s.int8-qdq.onnx")
    args = p.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    missing = [str(x) for x in args.audio if not x.is_file()]
    if missing:
        raise FileNotFoundError("missing calibration audio: " + ", ".join(missing))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.model), str(args.out), AudioCalibrationReader(args.audio),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, per_channel=True,
        # Quantize learned heavy layers only.  Quantizing reshape/transpose and
        # signal-shape plumbing is neither useful on a phone nor supported for
        # every shared Hybrid Demucs initializer in ORT.
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm", "LSTM"],
        extra_options={
            "ActivationSymmetric": False, "WeightSymmetric": True,
            # Hybrid Demucs reuses convolution/LSTM weights in several graph
            # paths. Give each consumer its own QDQ pair instead of asking ORT
            # to share a quantization parameter for a weight initializer.
            "AddQDQPairToWeight": True, "DedicatedQDQPair": True,
        },
    )
    print(f"INT8 candidate: {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
