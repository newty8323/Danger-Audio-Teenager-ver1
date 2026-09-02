#!/usr/bin/env python3
"""Inline an ONNX external-data model for Android asset packaging."""
from __future__ import annotations
import argparse
from pathlib import Path
import onnx

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    model = onnx.load_model(str(args.source), load_external_data=True)
    for initializer in model.graph.initializer:
        initializer.data_location = onnx.TensorProto.DEFAULT
        del initializer.external_data[:]
    onnx.save_model(model, str(args.output))
    print(f"{args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")

if __name__ == "__main__":
    main()
