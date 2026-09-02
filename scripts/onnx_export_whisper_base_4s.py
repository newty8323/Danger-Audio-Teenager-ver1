#!/usr/bin/env python3
"""Export a short-window FP32 Whisper Base ONNX pair for Android benchmarking.

Standard Whisper enforces a fixed 30-second (3,000-frame) encoder input even
when the live pipeline supplies only four seconds of audible vocal audio.  The
weights themselves support shorter sequences: this wrapper uses the required
leading encoder positional embeddings after Whisper's two convolution layers.
It does not retrain, prune, or alter any learned parameter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/onnx"


class WhisperShortEncoder(torch.nn.Module):
    """Whisper encoder without the library's fixed-3,000-frame guard."""

    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.encoder.conv1(input_features))
        hidden = F.gelu(self.encoder.conv2(hidden)).permute(0, 2, 1)
        # The standard 30-second encoder uses the identical leading position
        # embeddings. The input length determines how many are used here.
        hidden = hidden + self.encoder.embed_positions.weight[: hidden.shape[1]]
        for layer in self.encoder.layers:
            hidden = layer(hidden, None)
        return self.encoder.layer_norm(hidden)


class WhisperDecoder(torch.nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        ).last_hidden_state
        return self.proj_out(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/whisper-base")
    parser.add_argument("--window-seconds", type=int, choices=(4, 8), default=4)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    mel_frames = args.window_seconds * 100
    encoder_frames = mel_frames // 2
    out_dir = args.out_dir or DEFAULT_OUT / f"whisper_base_{args.window_seconds}s_fp32"

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, local_files_only=True
    ).eval().float()
    model.config._attn_implementation = "eager"
    model.model.decoder.config._attn_implementation = "eager"
    encoder = WhisperShortEncoder(model).eval()
    decoder = WhisperDecoder(model).eval()
    features = torch.zeros((1, 80, mel_frames), dtype=torch.float32)
    tokens = torch.tensor([[50258, 50264, 50359, 50363]], dtype=torch.long)

    with torch.no_grad():
        encoded = encoder(features)
        reference_logits = decoder(tokens, encoded)

    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = out_dir / f"encoder_{args.window_seconds}s.fp32.onnx"
    decoder_path = out_dir / f"decoder_{args.window_seconds}s.fp32.onnx"
    torch.onnx.export(
        encoder, features, str(encoder_path),
        input_names=["input_features"], output_names=["encoder_hidden_states"],
        opset_version=18, do_constant_folding=True, dynamo=False,
    )
    torch.onnx.export(
        decoder, (tokens, encoded), str(decoder_path),
        input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
        dynamic_axes={"input_ids": {1: "token_count"}, "logits": {1: "token_count"}},
        opset_version=18, do_constant_folding=True, dynamo=True,
    )

    # A single CPU parity check validates only export correctness; it is not a
    # speed benchmark and does not claim Android-device performance.
    enc_ort = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    ort_encoded = enc_ort.run(None, {"input_features": features.numpy()})[0]
    dec_ort = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    ort_logits = dec_ort.run(None, {
        "input_ids": tokens.numpy().astype(np.int64),
        "encoder_hidden_states": ort_encoded,
    })[0]
    report = {
        "model_id": args.model_id,
        "precision": "FP32 (unquantized)",
        "window_seconds": args.window_seconds,
        "input_mel_frames": mel_frames,
        "encoder_output_frames": encoder_frames,
        "encoder": str(encoder_path),
        "decoder": str(decoder_path),
        "encoder_bytes": encoder_path.stat().st_size,
        "decoder_bytes": decoder_path.stat().st_size,
        "encoder_shape": list(ort_encoded.shape),
        "decoder_max_abs_difference": float(
            np.max(np.abs(reference_logits.numpy() - ort_logits))
        ),
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
