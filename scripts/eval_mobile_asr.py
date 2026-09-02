"""Evaluate a mobile ASR candidate separately on all four required domains.

Run a Hugging Face checkpoint:
  PYTHONPATH=src uv run --group nlp --group asr python scripts/eval_mobile_asr.py \
    --model data_dl/mobile_asr/whisper-tiny-ko-multidomain/final

Run an exported CTranslate2 int8 checkpoint:
  ... --model artifacts/mobile_asr_int8 --backend faster-whisper
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mobile_asr.data import SR, load_audio
from mobile_asr.manifest import domain_counts, load_manifest
from mobile_asr.metrics import evaluate_predictions


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/mobile_asr.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("transformers", "faster-whisper"),
                        default="transformers")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--language", default="ko",
                        help="ko for the Korean-only target; 'auto' for code-switch ablation")
    parser.add_argument("--output", default="data_dl/mobile_asr/evaluation.json")
    return parser.parse_args()


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersBackend:
    def __init__(self, model: str, device: str, language: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model).to(device).eval()
        self.device = device
        self.language = language

    @torch.no_grad()
    def transcribe(self, wav) -> tuple[str, dict]:
        features = self.processor.feature_extractor(
            wav, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        kwargs = {"task": "transcribe"}
        if self.language != "auto":
            kwargs["language"] = self.language
        ids = self.model.generate(**features, **kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip(), {}


class FasterWhisperBackend:
    def __init__(self, model: str, device: str, language: str):
        from faster_whisper import WhisperModel

        ct = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model, device=device, compute_type=ct)
        self.language = None if language == "auto" else language

    def transcribe(self, wav) -> tuple[str, dict]:
        segments, info = self.model.transcribe(
            wav,
            language=self.language,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            multilingual=self.language is None,
        )
        segments = list(segments)
        text = "".join(segment.text for segment in segments).strip()
        metadata = {
            "language": info.language,
            "language_probability": info.language_probability,
            "avg_logprob": (
                sum(segment.avg_logprob for segment in segments) / len(segments)
                if segments else None
            ),
            "max_no_speech_prob": (
                max(segment.no_speech_prob for segment in segments) if segments else None
            ),
            "max_compression_ratio": (
                max(segment.compression_ratio for segment in segments) if segments else None
            ),
        }
        return text, metadata


def _model_bytes(path: str) -> int | None:
    root = Path(path)
    if not root.exists():
        return None
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def main() -> None:
    args = _parse()
    rows = [row for row in load_manifest(args.manifest) if row.split == args.split]
    if not rows:
        raise SystemExit(f"manifest has no {args.split} rows")
    device = _device(args.device)
    if args.backend == "faster-whisper" and device == "mps":
        device = "cpu"  # CTranslate2 supports CPU/CUDA, not Apple MPS.
    backend = (
        TransformersBackend(args.model, device, args.language)
        if args.backend == "transformers"
        else FasterWhisperBackend(args.model, device, args.language)
    )
    print(f"[mobile-asr-eval] {args.split} {domain_counts(rows)} · {device}", flush=True)
    hypotheses: dict[str, str] = {}
    confidence: dict[str, dict] = {}
    audio_sec = 0.0
    started = time.time()
    for index, row in enumerate(rows, 1):
        wav = load_audio(row.audio, max_seconds=30.0, random_crop=False)
        audio_sec += len(wav) / SR
        hypotheses[row.item_id], confidence[row.item_id] = backend.transcribe(wav)
        if index % 25 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    elapsed = time.time() - started
    result = {
        "model": args.model,
        "backend": args.backend,
        "split": args.split,
        "device": device,
        "model_bytes": _model_bytes(args.model),
        "model_mb": round(_model_bytes(args.model) / 1e6, 2) if _model_bytes(args.model) else None,
        "audio_sec": round(audio_sec, 2),
        "elapsed_sec": round(elapsed, 2),
        "rtf": round(elapsed / max(audio_sec, 1e-9), 4),
        "metrics": evaluate_predictions(rows, hypotheses),
        "samples": [
            {"id": row.item_id, "domain": row.domain, "ref": row.text,
             "hyp": hypotheses[row.item_id], "confidence": confidence[row.item_id]}
            for row in rows[:50]
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"[mobile-asr-eval] RTF={result['rtf']} size={result['model_mb']}MB -> {output}")


if __name__ == "__main__":
    main()

