#!/usr/bin/env python3
"""Export the local KoELECTRA harm classifier and verify ONNX Runtime parity.

The source checkpoint stays untouched.  Outputs are written to artifacts/onnx/ and
are intentionally excluded from the application pipeline until their file benchmark
has been reviewed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts/koelectra_small_harm_asraug_slang"
DEFAULT_OUT = ROOT / "artifacts/onnx/koelectra_small_harm_asraug_slang"


class ClassifierLogits(torch.nn.Module):
    """Keep the exported graph's output stable even if HF returns a dataclass."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_dir), local_files_only=True
    ).eval()
    wrapped = ClassifierLogits(model).eval()
    batch = tokenizer(
        ["이 새끼야, 당장 꺼져.", "오늘 날씨가 좋다."],
        padding=True, truncation=True, max_length=128, return_tensors="pt",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.out_dir / "model.fp32.onnx"
    int8_path = args.out_dir / "model.int8.onnx"

    torch.onnx.export(
        wrapped,
        (batch["input_ids"], batch["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        # The torch.export-based exporter currently stores an incorrect symbolic
        # intermediate shape for this Electra checkpoint; ORT cannot quantize that
        # graph.  The established TorchScript exporter produces a valid opset-17
        # graph and preserves the same dynamic batch/sequence interface.
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    with torch.inference_mode():
        torch_logits = wrapped(batch["input_ids"], batch["attention_mask"]).numpy()
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(None, {
        "input_ids": batch["input_ids"].numpy().astype(np.int64),
        "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
    })[0]
    maximum_absolute_difference = float(np.max(np.abs(torch_logits - ort_logits)))
    report = {
        "source": str(args.model_dir),
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "verification_max_abs_logit_difference": maximum_absolute_difference,
        "samples": ["이 새끼야, 당장 꺼져.", "오늘 날씨가 좋다."],
    }
    (args.out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
