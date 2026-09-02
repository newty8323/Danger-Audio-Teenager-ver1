#!/usr/bin/env python3
"""Export Whisper Base encoder and initial decoder ONNX graphs.

This is the first, independently verifiable part of the Whisper conversion.  It
exports the fixed 30-second log-Mel encoder and a decoder graph that accepts an
arbitrary token prefix.  A separate cached-decoder graph is intentionally not
claimed here: Transformers 5 uses a new ``EncoderDecoderCache`` representation,
so the cache tensor interface must be exported and parity-tested independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from torch.export import Dim


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts/onnx/whisper_base_fp16"


class WhisperEncoder(torch.nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features).last_hidden_state


class WhisperDecoderInitial(torch.nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        ).last_hidden_state
        return self.proj_out(hidden)


def _cache_tensors(cache: EncoderDecoderCache) -> tuple[torch.Tensor, ...]:
    """Flatten Whisper's six self/cross-attention KV pairs into ONNX tensors."""
    values: list[torch.Tensor] = []
    for layer in range(6):
        self_layer = cache.self_attention_cache.layers[layer]
        cross_layer = cache.cross_attention_cache.layers[layer]
        values.extend((self_layer.keys, self_layer.values, cross_layer.keys, cross_layer.values))
    return tuple(values)


class WhisperDecoderInitialWithCache(torch.nn.Module):
    """First decoder call: prompt + encoder state → logits and all KV tensors."""
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
        )
        return (self.proj_out(output.last_hidden_state), *_cache_tensors(output.past_key_values))


class WhisperDecoderCached(torch.nn.Module):
    """One-token decoder call with explicit self/cross-attention KV cache."""
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    @staticmethod
    def _self_attention(attn: torch.nn.Module, hidden: torch.Tensor, past_key: torch.Tensor, past_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        query = (attn.q_proj(hidden) * attn.scaling).view(shape).transpose(1, 2).contiguous()
        key = attn.k_proj(hidden).view(shape).transpose(1, 2).contiguous()
        value = attn.v_proj(hidden).view(shape).transpose(1, 2).contiguous()
        key = torch.cat((past_key, key), dim=2)
        value = torch.cat((past_value, value), dim=2)
        weights = torch.nn.functional.softmax(torch.matmul(query, key.transpose(2, 3)), dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(*hidden.shape[:-1], -1)
        return attn.out_proj(output), key, value

    @staticmethod
    def _cross_attention(attn: torch.nn.Module, hidden: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        query = (attn.q_proj(hidden) * attn.scaling).view(shape).transpose(1, 2).contiguous()
        weights = torch.nn.functional.softmax(torch.matmul(query, key.transpose(2, 3)), dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(*hidden.shape[:-1], -1)
        return attn.out_proj(output)

    # Explicit parameters (rather than ``*past``) are needed so the Dynamo
    # exporter can attach a dynamic sequence axis to all 24 cache tensors.
    def forward(
        self, input_ids: torch.Tensor,
        past_self_key_0: torch.Tensor, past_self_value_0: torch.Tensor, past_cross_key_0: torch.Tensor, past_cross_value_0: torch.Tensor,
        past_self_key_1: torch.Tensor, past_self_value_1: torch.Tensor, past_cross_key_1: torch.Tensor, past_cross_value_1: torch.Tensor,
        past_self_key_2: torch.Tensor, past_self_value_2: torch.Tensor, past_cross_key_2: torch.Tensor, past_cross_value_2: torch.Tensor,
        past_self_key_3: torch.Tensor, past_self_value_3: torch.Tensor, past_cross_key_3: torch.Tensor, past_cross_value_3: torch.Tensor,
        past_self_key_4: torch.Tensor, past_self_value_4: torch.Tensor, past_cross_key_4: torch.Tensor, past_cross_value_4: torch.Tensor,
        past_self_key_5: torch.Tensor, past_self_value_5: torch.Tensor, past_cross_key_5: torch.Tensor, past_cross_value_5: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Cross-attention K/V are generated by the prompt graph, so this
        # one-token graph has no encoder-state input at all.
        past = (
            past_self_key_0, past_self_value_0, past_cross_key_0, past_cross_value_0,
            past_self_key_1, past_self_value_1, past_cross_key_1, past_cross_value_1,
            past_self_key_2, past_self_value_2, past_cross_key_2, past_cross_value_2,
            past_self_key_3, past_self_value_3, past_cross_key_3, past_cross_value_3,
            past_self_key_4, past_self_value_4, past_cross_key_4, past_cross_value_4,
            past_self_key_5, past_self_value_5, past_cross_key_5, past_cross_value_5,
        )
        past_length = past[0].shape[2]
        hidden = self.decoder.embed_tokens(input_ids) + self.decoder.embed_positions(input_ids, past_length)
        presents: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.decoder.layers):
            base = layer_index * 4
            residual = hidden
            self_out, self_key, self_value = self._self_attention(
                layer.self_attn, layer.self_attn_layer_norm(hidden), past[base], past[base + 1]
            )
            hidden = residual + self_out
            residual = hidden
            cross_out = self._cross_attention(
                layer.encoder_attn, layer.encoder_attn_layer_norm(hidden), past[base + 2], past[base + 3]
            )
            hidden = residual + cross_out
            residual = hidden
            hidden = layer.final_layer_norm(hidden)
            hidden = layer.fc2(layer.activation_fn(layer.fc1(hidden)))
            hidden = residual + hidden
            presents.extend((self_key, self_value, past[base + 2], past[base + 3]))
        hidden = self.decoder.layer_norm(hidden)
        return (self.proj_out(hidden), *presents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/whisper-base")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # This is the *unquantized FP16* mobile candidate.  It keeps every Whisper
    # weight but halves storage compared with the HF FP32 checkpoint; CoreML is
    # the intended execution provider on Apple Silicon.
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id).eval().half()
    # SDPA turns ``is_causal`` into a symbolic boolean when exporting the
    # one-token cache graph. Eager attention keeps that control value concrete
    # and lets the modern exporter retain the compact FP16 external weights.
    model.config._attn_implementation = "eager"
    model.model.decoder.config._attn_implementation = "eager"
    encoder = WhisperEncoder(model).eval()
    decoder = WhisperDecoderInitial(model).eval()
    decoder_initial_cache = WhisperDecoderInitialWithCache(model).eval()
    decoder_cached = WhisperDecoderCached(model).eval()
    features = torch.zeros((1, 80, 3000), dtype=torch.float16)
    # Standard Korean task-prefix tokens: <|startoftranscript|><|ko|><|transcribe|>.
    token_ids = torch.tensor([[50258, 50264, 50359]], dtype=torch.long)
    # ``torch.onnx.export`` traces decoder weights through autograd internals;
    # no_grad avoids retaining activations while keeping the encoder output a
    # normal tensor that the tracer can consume.
    with torch.no_grad():
        torch_encoded = encoder(features)
        torch_logits = decoder(token_ids, torch_encoded)
        initial_cache_outputs = decoder_initial_cache(token_ids, torch_encoded)
        next_token = torch.tensor([[100]], dtype=torch.long)
        cached_outputs = decoder_cached(next_token, *initial_cache_outputs[1:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.out_dir / "encoder.fp16.onnx"
    decoder_path = args.out_dir / "decoder_initial.fp16.onnx"
    decoder_initial_cache_path = args.out_dir / "decoder_initial_cache.fp32.onnx"
    decoder_cached_path = args.out_dir / "decoder_cached.fp32.onnx"
    torch.onnx.export(
        encoder, features, str(encoder_path),
        input_names=["input_features"], output_names=["encoder_hidden_states"],
        opset_version=18, do_constant_folding=True, dynamo=False,
    )
    torch.onnx.export(
        decoder, (token_ids, torch_encoded), str(decoder_path),
        input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
        dynamic_axes={"input_ids": {1: "token_count"}, "logits": {1: "token_count"}},
        # The legacy exporter cannot lower Whisper's causal-mask ``aten::diff``.
        # Use the new exporter for this graph; it has a decomposition for it.
        opset_version=18, do_constant_folding=True, dynamo=True,
    )
    # With populated cross-attention KV tensors, encoder_hidden_states is not
    # consumed by the one-token graph and the exporter correctly prunes it.
    cache_input_names = ["input_ids"] + [
        f"past_{kind}_{kv}_{layer}"
        for layer in range(6)
        for kind in ("self", "cross")
        for kv in ("key", "value")
    ]
    cache_output_names = ["logits"] + [
        f"present_{kind}_{kv}_{layer}"
        for layer in range(6)
        for kind in ("self", "cross")
        for kv in ("key", "value")
    ]
    torch.onnx.export(
        decoder_initial_cache,
        (token_ids, torch_encoded),
        str(decoder_initial_cache_path),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=cache_output_names,
        dynamic_axes={"input_ids": {1: "prompt_tokens"}, "logits": {1: "prompt_tokens"}},
        opset_version=18, do_constant_folding=True, dynamo=True,
    )
    torch.onnx.export(
        decoder_cached,
        (next_token, *initial_cache_outputs[1:]),
        str(decoder_cached_path),
        input_names=cache_input_names,
        output_names=cache_output_names,
        # Explicit Dynamo constraints keep all six self-cache layers on one
        # shared variable-length axis. ``dynamic_axes`` mis-associates some
        # cache arguments under Transformers 5's nested export graph.
        dynamic_shapes=(
            {},  # cached decoder always receives exactly one new token
            *(
                {2: Dim("past_tokens", min=1, max=447)} if "past_self" in name else {}
                for name in cache_input_names[1:]
            ),
        ),
        opset_version=18, do_constant_folding=True, dynamo=True,
    )
    # CoreML is available on this Mac.  Use it for the parity smoke test and
    # leave provider selection to the benchmark script for an explicit record.
    providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                 if "CoreMLExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    enc_session = ort.InferenceSession(str(encoder_path), providers=providers)
    ort_encoded = enc_session.run(None, {"input_features": features.numpy()})[0]
    dec_session = ort.InferenceSession(str(decoder_path), providers=providers)
    ort_logits = dec_session.run(None, {
        "input_ids": token_ids.numpy().astype(np.int64),
        "encoder_hidden_states": ort_encoded,
    })[0]
    report = {
        "model_id": args.model_id,
        "precision": "FP16 (unquantized)",
        "providers": providers,
        "encoder": str(encoder_path),
        "decoder_initial": str(decoder_path),
        "decoder_initial_cache": str(decoder_initial_cache_path),
        "decoder_cached": str(decoder_cached_path),
        "encoder_bytes": encoder_path.stat().st_size,
        "decoder_initial_bytes": decoder_path.stat().st_size,
        "encoder_max_abs_difference": float(np.max(np.abs(torch_encoded.numpy() - ort_encoded))),
        "decoder_initial_max_abs_difference": float(np.max(np.abs(torch_logits.numpy() - ort_logits))),
        "cache_status": "complete: initial prompt graph plus one-token graph with explicit 6-layer self/cross KV tensors",
    }
    (args.out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
