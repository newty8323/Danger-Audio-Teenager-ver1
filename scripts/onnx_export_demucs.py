#!/usr/bin/env python3
"""Export the selected 4 s MDX Extra Demucs separator to ONNX.

``14fc6a69`` is a *hybrid* Demucs: it has both waveform and spectrogram branches.
ONNX Runtime Mobile cannot reliably export the complex STFT/ISTFT section as part
of the same graph.  This exporter therefore writes the learned core only.  The
Android runner supplies a magnitude spectrogram and the original time waveform,
then performs STFT, mixture-phase reconstruction, ISTFT and overlap/add outside
ONNX.  No learned layer or weight is replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16


ROOT = Path(__file__).resolve().parents[1]
DEMUCS_PACKAGE = ROOT / "vendor/demucs-python"
DEFAULT_REPO = ROOT / "artifacts/ver1/demucs-quantized"
DEFAULT_OUT = ROOT / "artifacts/onnx/demucs_mdx_extra_q"
SAMPLE_RATE = 44_100
BLOCK_SAMPLES = SAMPLE_RATE * 4


class HybridDemucsCore(torch.nn.Module):
    """The exact learned portion of ``HDemucs.forward`` without complex DSP.

    Outputs are the four-source frequency magnitudes and waveform branch.  The
    caller reconstructs each source with the mixture phase, exactly matching the
    model's ``wiener_iters=0`` behaviour.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, magnitude: torch.Tensor, mix_time: torch.Tensor):
        model = self.model
        x = magnitude
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix_time
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(model.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(model.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = model.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and model.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = model.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + model.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        xt = torch.zeros_like(x)
        for idx, decode in enumerate(model.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = model.depth - len(model.tdecoder)
            if idx >= offset:
                tdec = model.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        sources = len(model.sources)
        freq_magnitude = x.view(x.shape[0], sources, -1, x.shape[-2], x.shape[-1])
        freq_magnitude = freq_magnitude * std[:, None] + mean[:, None]
        time_sources = xt.view(xt.shape[0], sources, -1, mix_time.shape[-1])
        time_sources = time_sources * stdt[:, None] + meant[:, None]
        return freq_magnitude, time_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model", default="14fc6a69")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-fp16", action="store_true",
                        help="skip the Android-oriented FP16 copy")
    args = parser.parse_args()

    sys.path.insert(0, str(DEMUCS_PACKAGE))
    from demucs.pretrained import get_model

    model = get_model(args.model, repo=args.repo).eval()
    wrapper = HybridDemucsCore(model).eval()
    # A deterministic non-silent input detects a channel/source-order mistake.
    frame = torch.linspace(-0.05, 0.05, BLOCK_SAMPLES, dtype=torch.float32)
    example = torch.stack((frame, frame * 0.7), dim=0)[None]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "hybrid_core_4s.fp32.onnx"
    # The spectrogram is deliberately calculated outside ONNX.  This is the
    # identical input that HDemucs.forward creates before its learned layers.
    with torch.inference_mode():
        z = model._spec(example)
        magnitude = model._magnitude(z)
    try:
        torch.onnx.export(
            wrapper, (magnitude, example), str(model_path),
            input_names=["mixture_magnitude", "mixture_waveform"],
            output_names=["source_magnitudes", "source_waveforms"],
            # The newer exporter is tried first because it supports more FFT/STFT
            # decompositions than the legacy exporter.
            opset_version=18, do_constant_folding=True, dynamo=True,
        )
    except Exception as error:
        report = {
            "status": "not_exported",
            "source_repo": str(args.repo),
            "signature": args.model,
            "reason": str(error).splitlines()[0],
            "detail": str(error)[-3000:],
            "next_step": (
                "inspect the detailed export error; do not replace the verified 14fc6a69 "
                "checkpoint with a lower-quality separator"
            ),
        }
        (args.out_dir / "export_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return

    with torch.inference_mode():
        torch_magnitudes, torch_waveforms = wrapper(magnitude, example)
        torch_magnitudes = torch_magnitudes.numpy()
        torch_waveforms = torch_waveforms.numpy()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_magnitudes, ort_waveforms = session.run(None, {
        "mixture_magnitude": magnitude.numpy(), "mixture_waveform": example.numpy(),
    })
    # End-to-end parity: keep the deliberately external complex DSP in PyTorch
    # for this verification only.  Android implements these same four steps
    # (STFT -> mixture-phase mask -> ISTFT -> overlap/add) outside ONNX.
    with torch.inference_mode():
        ort_magnitude_t = torch.from_numpy(ort_magnitudes)
        ort_waveform_t = torch.from_numpy(ort_waveforms)
        separated_spec = model._mask(z, ort_magnitude_t)
        reconstructed = model._ispec(separated_spec, example.shape[-1]) + ort_waveform_t
        full_reference = model(example)
    report = {
        "source_repo": str(args.repo),
        "signature": args.model,
        "sample_rate": SAMPLE_RATE,
        "input_magnitude_shape": list(magnitude.shape),
        "input_waveform_shape": [1, 2, BLOCK_SAMPLES],
        "source_magnitudes_shape": list(ort_magnitudes.shape),
        "source_waveforms_shape": list(ort_waveforms.shape),
        "onnx": str(model_path),
        "onnx_bytes": model_path.stat().st_size,
        "core_magnitude_max_abs_difference": float(np.max(np.abs(torch_magnitudes - ort_magnitudes))),
        "core_waveform_max_abs_difference": float(np.max(np.abs(torch_waveforms - ort_waveforms))),
        "full_separator_max_abs_difference": float(
            (full_reference - reconstructed).abs().max().item()
        ),
    }
    if not args.no_fp16:
        # Keep the application inputs FP32: audio capture, STFT and Android FFT
        # remain ordinary float calculations.  Only the learned operations and
        # weights become FP16, which cuts the expanded ONNX weights in half.
        fp16_path = args.out_dir / "hybrid_core_4s.fp16.onnx"
        fp16_model = convert_float_to_float16(
            onnx.load_model(str(model_path), load_external_data=True),
            keep_io_types=True,
        )
        # ONNX Runtime Mobile accepts a single asset most reliably.  Keeping
        # weights inline also avoids an ORT parser issue with tiny INT64 shape
        # tensors stored in an external-data sidecar.
        for initializer in fp16_model.graph.initializer:
            initializer.data_location = onnx.TensorProto.DEFAULT
            del initializer.external_data[:]
        onnx.save_model(fp16_model, str(fp16_path))
        fp16_session = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
        fp16_magnitudes, fp16_waveforms = fp16_session.run(None, {
            "mixture_magnitude": magnitude.numpy(), "mixture_waveform": example.numpy(),
        })
        report.update({
            "fp16_onnx": str(fp16_path),
            "fp16_total_bytes": fp16_path.stat().st_size,
            "fp16_core_magnitude_max_abs_difference": float(np.max(np.abs(torch_magnitudes - fp16_magnitudes))),
            "fp16_core_waveform_max_abs_difference": float(np.max(np.abs(torch_waveforms - fp16_waveforms))),
        })
    (args.out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
