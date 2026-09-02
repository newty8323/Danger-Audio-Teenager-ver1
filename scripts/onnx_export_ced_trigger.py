#!/usr/bin/env python3
"""Export the trained CED-mini acoustic harm trigger for Android ONNX Runtime.

Android performs the fixed 16 kHz / 64-band log-Mel transform.  Keeping that
DSP outside ONNX makes the model input explicit and avoids relying on a
platform-specific audio operator.  The exported graph contains the fine-tuned
CED encoder, MIL pooling, and four-class harm classifier exactly as trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/onnx/ced_mini_vio"


class CEDTriggerLogits(torch.nn.Module):
    def __init__(self, trained: torch.nn.Module):
        super().__init__()
        self.encoder = trained.backbone.ced
        self.pool = trained.pool
        self.classifier = trained.classifier

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        frames = self.encoder(input_values=log_mel).logits
        pooled, _ = self.pool(frames)
        return self.classifier(pooled)


def main() -> None:
    import sys

    for path in (ROOT / ".autorun", ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from cascade.pipeline import load_trigger

    trained = load_trigger(int8=False).eval()
    model = CEDTriggerLogits(trained).eval()
    # 4 seconds at 16 kHz with n_fft=512, hop=160, center=True gives 401 frames.
    features = torch.zeros((1, 64, 401), dtype=torch.float32)
    with torch.no_grad():
        reference = model(features).numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "model.fp32.onnx"
    torch.onnx.export(
        model, features, str(destination),
        input_names=["log_mel"], output_names=["logits"],
        dynamic_axes={"log_mel": {2: "frames"}},
        opset_version=18, do_constant_folding=True, dynamo=False,
    )
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"log_mel": features.numpy()})[0]
    report = {
        "checkpoint": str(ROOT / "ckpt_ced_mini_vio/best.ckpt"),
        "input": "16 kHz mono log-Mel, 64 bands, n_fft=512, hop=160, amplitude dB top_db=120",
        "output": "four harm logits; Android applies sigmoid then max",
        "model": str(destination),
        "bytes": destination.stat().st_size,
        "max_abs_logit_difference": float(np.max(np.abs(reference - actual))),
    }
    (OUT / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
