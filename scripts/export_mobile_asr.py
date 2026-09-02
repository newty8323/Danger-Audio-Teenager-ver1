"""Export the trained Whisper student to CTranslate2 int8 and enforce the size budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def directory_bytes(path: str | Path) -> int:
    root = Path(path)
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def budget_report(asr_bytes: int, other_model_mb: float, total_budget_mb: float) -> dict:
    asr_mb = asr_bytes / 1e6
    total_mb = asr_mb + other_model_mb
    return {
        "asr_mb": round(asr_mb, 3),
        "other_models_mb": round(other_model_mb, 3),
        "total_model_mb": round(total_mb, 3),
        "budget_mb": round(total_budget_mb, 3),
        "within_budget": total_mb <= total_budget_mb,
        "remaining_mb": round(total_budget_mb - total_mb, 3),
        "note": "model files only; peak RAM and app/runtime binaries are measured separately",
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/mobile_asr_whisper_tiny_int8")
    parser.add_argument("--other-model-mb", type=float, default=38.0,
                        help="CED-mini 10MB + KoELECTRA 28MB")
    parser.add_argument("--total-budget-mb", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    from ctranslate2.converters import TransformersConverter

    output = Path(args.output).resolve()
    TransformersConverter(args.checkpoint, low_cpu_mem_usage=True).convert(
        str(output), quantization="int8", force=args.force
    )
    report = budget_report(directory_bytes(output), args.other_model_mb, args.total_budget_mb)
    (output / "mobile_budget.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["within_budget"]:
        raise SystemExit("export succeeded, but the total model-file budget exceeds the limit")


if __name__ == "__main__":
    main()

