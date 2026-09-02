"""Generate sequence-level distillation targets with a large offline Whisper teacher.

Only rows that omit both ``text`` and ``teacher_text`` are transcribed.  ``no_speech`` rows
remain human-labelled empty targets; otherwise the teacher could teach its own hallucinations.
The output is a new manifest and never overwrites the source manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobile_asr.data import load_audio


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--language", default="ko")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise SystemExit("--output must differ from --input")
    raw_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    from faster_whisper import WhisperModel

    compute = "int8_float16" if args.device == "cuda" else "int8"
    teacher = WhisperModel(args.model, device=args.device, compute_type=compute)
    completed = 0
    for index, row in enumerate(raw_rows, 1):
        if row.get("domain") == "no_speech":
            row.setdefault("text", "")
            continue
        if row.get("text") is not None or row.get("teacher_text") is not None:
            continue
        audio = Path(str(row["audio"])).expanduser()
        if not audio.is_absolute():
            audio = source.parent / audio
        wav = load_audio(audio.resolve(), max_seconds=30.0, random_crop=False)
        segments, info = teacher.transcribe(
            wav,
            language=None if args.language == "auto" else args.language,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            multilingual=args.language == "auto",
        )
        segments = list(segments)
        row["teacher_text"] = "".join(segment.text for segment in segments).strip()
        row["teacher"] = {
            "model": args.model,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 6),
            "avg_logprob": round(
                sum(segment.avg_logprob for segment in segments) / len(segments), 6
            ) if segments else None,
        }
        completed += 1
        if completed % 25 == 0:
            print(f"[teacher] labelled {completed} (row {index}/{len(raw_rows)})", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_rows) + "\n",
        encoding="utf-8",
    )
    print(f"[teacher] added {completed} targets -> {output}")


if __name__ == "__main__":
    main()

