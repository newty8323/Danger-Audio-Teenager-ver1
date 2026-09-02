"""Build a source-disjoint 115-item public-data ASR experiment set.

The set is intentionally small enough for a first Mac/phone-oriented comparison:
50 clean Korean utterances, 25 movie-like mixtures, 20 Korean singing windows, and
20 non-speech environmental sounds.  Raw and generated audio remain under ``data_dl``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000
ESC50_AUDIO_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{name}"
SPLIT_SPEAKERS = {
    "train": (104, 105, 112, 118, 121, 126),
    "val": (132, 137),
    "test": (147, 149),
}
GENERAL_PER_SPEAKER = 5
MOVIE_PER_SPEAKER = {
    "train": (3, 3, 3, 2, 2, 2),
    "val": (3, 2),
    "test": (3, 2),
}
SONGS = {
    "kr007a": ("train", ((0,), (1,), (2,), (3,))),
    "kr011a": ("train", ((0, 1), (2, 3), (4, 5), (6, 7))),
    "kr024a": ("train", ((0,), (1,), (2,), (3,))),
    "kr027a": ("val", ((0,), (1,), (3,), (5,))),
    "kr028a": ("test", ((0, 1), (2, 3), (4, 5), (6, 7))),
}
ESC_PLAN = {
    "train": (
        "rain", "sea_waves", "crackling_fire", "crickets", "clock_tick", "helicopter",
        "chainsaw", "engine", "train", "thunderstorm", "vacuum_cleaner", "washing_machine",
    ),
    "val": ("door_wood_knock", "can_opening", "mouse_click", "keyboard_typing"),
    "test": ("glass_breaking", "siren", "car_horn", "fireworks"),
}
ESC_FOLD = {"train": 1, "val": 4, "test": 5}


def _resample(wav: np.ndarray, source_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if source_sr != SR:
        divisor = math.gcd(source_sr, SR)
        wav = resample_poly(wav, SR // divisor, source_sr // divisor).astype(np.float32)
    return wav


def _read_audio(source: Path | bytes) -> np.ndarray:
    wav, sample_rate = sf.read(BytesIO(source) if isinstance(source, bytes) else source,
                               dtype="float32", always_2d=True)
    return _resample(wav, sample_rate)


def _write_audio(path: Path, wav: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    sf.write(path, wav, SR, subtype="PCM_16")
    return len(wav) / SR


def _mix_snr(speech: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    if len(background) < len(speech):
        background = np.tile(background, math.ceil(len(speech) / len(background)))
    background = background[:len(speech)]
    speech_rms = float(np.sqrt(np.mean(np.square(speech)) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(np.square(background)) + 1e-12))
    scale = speech_rms / (noise_rms * 10 ** (snr_db / 20))
    return speech + background * scale


def _select_esc_rows(metadata: Path) -> dict[str, list[dict[str, str]]]:
    rows = list(csv.DictReader(metadata.open(encoding="utf-8")))
    selected: dict[str, list[dict[str, str]]] = {}
    for split, categories in ESC_PLAN.items():
        fold = str(ESC_FOLD[split])
        chosen = []
        for category in categories:
            candidates = sorted(
                (row for row in rows if row["fold"] == fold and row["category"] == category),
                key=lambda row: row["filename"],
            )
            if not candidates:
                raise RuntimeError(f"ESC-50 has no {category!r} item in fold {fold}")
            chosen.append(candidates[0])
        selected[split] = chosen
    return selected


def _fetch_esc_audio(rows: dict[str, list[dict[str, str]]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for row in (row for split_rows in rows.values() for row in split_rows):
        destination = output / row["filename"]
        if destination.is_file():
            continue
        print(f"[get] ESC-50 {row['category']}: {row['filename']}")
        request = urllib.request.Request(
            ESC50_AUDIO_URL.format(name=row["filename"]),
            headers={"User-Agent": "mobile-asr-smallset/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())


def _song_windows(
    stem: str, groups: tuple[tuple[int, ...], ...], csd: Path
) -> list[tuple[np.ndarray, str, float, float]]:
    wav = _read_audio(csd / "wav" / f"{stem}.wav")
    rows = list(csv.DictReader((csd / "csv" / f"{stem}.csv").open(encoding="utf-8")))
    lyric = (csd / "lyric" / f"{stem}.txt").read_text(encoding="utf-8")
    char_positions = [match.start() for match in re.finditer(r"[가-힣]", lyric)]
    if len(char_positions) != len(rows):
        raise RuntimeError(
            f"CSD alignment mismatch for {stem}: {len(rows)} != {len(char_positions)}"
        )
    line_ranges: list[tuple[int, int, str]] = []
    syllable_index = 0
    for line in lyric.splitlines():
        text = line.strip()
        syllables = len(re.findall(r"[가-힣]", text))
        if syllables:
            line_ranges.append((syllable_index, syllable_index + syllables, text))
            syllable_index += syllables
    if syllable_index != len(rows):
        raise RuntimeError(f"CSD line mapping mismatch for {stem}")

    duration = len(wav) / SR
    clips = []
    for group in groups:
        if tuple(range(group[0], group[-1] + 1)) != group:
            raise RuntimeError(f"non-contiguous CSD line group for {stem}: {group}")
        selected = [line_ranges[index] for index in group]
        first, last = selected[0][0], selected[-1][1] - 1
        start = max(0.0, float(rows[first]["start"]) - 0.25)
        end = min(duration, float(rows[last]["end"]) + 0.25)
        text = " ".join(line[2] for line in selected)
        sample_start, sample_end = round(start * SR), round(end * SR)
        clips.append((wav[sample_start:sample_end], text, float(start), end))
    return clips


def _row(item_id: str, audio: Path, text: str, domain: str, split: str, source_id: str,
         **extra: object) -> dict[str, object]:
    return {
        "id": item_id,
        "audio": str(audio),
        "text": text,
        "domain": domain,
        "split": split,
        "source_id": source_id,
        "harm_terms": [],
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("data_dl/mobile_asr/sources"))
    parser.add_argument("--output", type=Path, default=Path("data_dl/mobile_asr/smallset"))
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    zeroth_path = args.sources / "zeroth_test.parquet"
    esc_metadata = args.sources / "esc50.csv"
    csd = args.sources / "csd"
    required = [zeroth_path, esc_metadata, *(csd / "wav" / f"{stem}.wav" for stem in SONGS)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source files are missing: {missing}")

    clips = args.output / "clips"
    esc_source = args.sources / "esc50_audio"
    esc_rows = _select_esc_rows(esc_metadata)
    _fetch_esc_audio(esc_rows, esc_source)
    esc_by_split = {
        split: [(row, _read_audio(esc_source / row["filename"])) for row in rows]
        for split, rows in esc_rows.items()
    }

    all_zeroth = pq.read_table(zeroth_path).to_pylist()
    eligible: dict[int, list[dict]] = defaultdict(list)
    for row in all_zeroth:
        duration = sf.info(BytesIO(row["audio"]["bytes"])).duration
        if duration <= 10.0:
            eligible[int(row["speaker_id"])].append(row)
    rng = random.Random(args.seed)
    for speaker_rows in eligible.values():
        rng.shuffle(speaker_rows)

    manifest_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for split, speakers in SPLIT_SPEAKERS.items():
        for speaker_index, speaker in enumerate(speakers):
            movie_count = MOVIE_PER_SPEAKER[split][speaker_index]
            needed = GENERAL_PER_SPEAKER + movie_count
            if len(eligible[speaker]) < needed:
                raise RuntimeError(
                    f"speaker {speaker} has only {len(eligible[speaker])} short utterances"
                )
            selected = eligible[speaker][:needed]
            for source in selected[:GENERAL_PER_SPEAKER]:
                item_id = f"general-z{source['id']}"
                destination = clips / "general" / f"{item_id}.wav"
                duration = _write_audio(destination, _read_audio(source["audio"]["bytes"]))
                manifest_rows.append(_row(
                    item_id, destination, source["text"], "general", split,
                    f"zeroth-speaker-{speaker}", dataset="Zeroth-Korean",
                ))
                review_rows.append({"id": item_id, "domain": "general", "split": split,
                                    "duration_s": f"{duration:.3f}", "text": source["text"],
                                    "source": source["id"], "snr_db": ""})
            for row_index, source in enumerate(selected[GENERAL_PER_SPEAKER:]):
                background_row, background = esc_by_split[split][
                    (speaker_index + row_index) % len(esc_by_split[split])
                ]
                snr_db = (0.0, 5.0, 10.0)[row_index % 3]
                speech = _read_audio(source["audio"]["bytes"])
                mixed = _mix_snr(speech, background, snr_db)
                item_id = f"movie-z{source['id']}-snr{int(snr_db)}"
                destination = clips / "movie" / f"{item_id}.wav"
                duration = _write_audio(destination, mixed)
                manifest_rows.append(_row(
                    item_id, destination, source["text"], "movie", split,
                    f"zeroth-speaker-{speaker}", dataset="Zeroth-Korean+ESC-50",
                    background=background_row["filename"], snr_db=snr_db,
                ))
                review_rows.append({"id": item_id, "domain": "movie", "split": split,
                                    "duration_s": f"{duration:.3f}", "text": source["text"],
                                    "source": f"{source['id']}+{background_row['filename']}",
                                    "snr_db": snr_db})

    for stem, (split, groups) in SONGS.items():
        for index, (wav, text, start, end) in enumerate(_song_windows(stem, groups, csd), 1):
            item_id = f"song-{stem}-{index:02d}"
            destination = clips / "song" / f"{item_id}.wav"
            duration = _write_audio(destination, wav)
            manifest_rows.append(_row(
                item_id, destination, text, "song", split, f"csd-song-{stem[:5]}",
                dataset="CSD-1.1", start_s=round(start, 4), end_s=round(end, 4),
            ))
            review_rows.append({"id": item_id, "domain": "song", "split": split,
                                "duration_s": f"{duration:.3f}", "text": text,
                                "source": f"{stem}:{start:.2f}-{end:.2f}", "snr_db": ""})

    for split, rows in esc_rows.items():
        for row in rows:
            item_id = f"no-speech-{Path(row['filename']).stem}"
            destination = clips / "no_speech" / f"{item_id}.wav"
            duration = _write_audio(destination, _read_audio(esc_source / row["filename"]))
            manifest_rows.append(_row(
                item_id, destination, "", "no_speech", split,
                f"esc50-source-{row['src_file']}", dataset="ESC-50", category=row["category"],
            ))
            review_rows.append({"id": item_id, "domain": "no_speech", "split": split,
                                "duration_s": f"{duration:.3f}", "text": "",
                                "source": f"{row['filename']} ({row['category']})", "snr_db": ""})

    manifest_rows.sort(key=lambda row: (str(row["split"]), str(row["domain"]), str(row["id"])))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "mobile_asr_small.jsonl"
    for row in manifest_rows:
        row["audio"] = str(Path(row["audio"]).relative_to(args.output))
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    review = args.output / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "id", "domain", "split", "duration_s", "text", "source", "snr_db",
        ))
        writer.writeheader()
        writer.writerows(review_rows)
    counts = Counter((str(row["split"]), str(row["domain"])) for row in manifest_rows)
    stats = {
        "rows": len(manifest_rows),
        "counts": {split: {domain: counts[(split, domain)] for domain in
                            ("general", "movie", "song", "no_speech")}
                   for split in ("train", "val", "test")},
        "seed": args.seed,
    }
    (args.output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "SOURCES.md").write_text(
        "# Public sources and limits\n\n"
        "- Zeroth-Korean test: OpenSLR SLR40, CC BY 4.0.\n"
        "- CSD 1.1 Korean singing: Zenodo 4916302, CC BY-NC-SA 4.0.\n"
        "- ESC-50 environmental sounds: CC BY-NC 3.0.\n\n"
        "Movie-like rows are synthetic mixtures of source-disjoint Zeroth speech and ESC-50 "
        "effects at 0, 5, or 10 dB SNR. CSD consists of children's songs by one singer. "
        "This first set measures domain robustness and false transcripts; it does not yet "
        "represent commercial films, accompanied K-pop, or harmful-language recall.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[done] manifest={manifest} review={review}")


if __name__ == "__main__":
    main()
