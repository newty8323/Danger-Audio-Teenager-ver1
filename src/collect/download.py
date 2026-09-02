"""Fetch AudioSet segment audio from YouTube (spec §3, §6.4 ethics).

For each manifest record: resolve the YouTube audio stream with yt-dlp, then cut
the 10s segment to 16 kHz mono wav with ffmpeg. Many AudioSet videos are gone
(deleted / private / geo-blocked), so per-clip failure is expected and tallied,
not fatal.

This step hits the network and is user-run (heavy). Only violence/confusable
sources are collected this way; adult sources are out of scope here (spec §6.4).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from datasets.manifest import ClipRecord, read_manifest


class DownloadStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"  # already on disk
    UNAVAILABLE = "unavailable"  # yt-dlp could not resolve (gone/private/geo)
    FAILED = "failed"  # ffmpeg cut failed


class ToolMissingError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise ToolMissingError(f"{tool} not found on PATH")
    return path


def youtube_url(ytid: str) -> str:
    return f"https://www.youtube.com/watch?v={ytid}"


def download_segment(
    ytid: str,
    start: float,
    duration: float,
    out_path: str | Path,
    sample_rate: int = 16_000,
) -> DownloadStatus:
    """Download one segment to ``out_path`` (16 kHz mono wav)."""
    out_path = Path(out_path)
    if out_path.exists():
        return DownloadStatus.SKIPPED

    ytdlp = _require("yt-dlp")
    ffmpeg = _require("ffmpeg")

    # bestaudio/best: fall back to a muxed stream when no standalone audio exists
    # (ffmpeg still extracts audio) — recovers some otherwise-unavailable clips.
    resolve = subprocess.run(
        [ytdlp, "-f", "bestaudio/best", "-g", youtube_url(ytid)],
        capture_output=True, text=True,
    )
    if resolve.returncode != 0 or not resolve.stdout.strip():
        return DownloadStatus.UNAVAILABLE
    stream_url = resolve.stdout.strip().splitlines()[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cut = subprocess.run(
        [
            ffmpeg, "-nostdin", "-loglevel", "error",
            "-ss", f"{start}", "-i", stream_url, "-t", f"{duration}",
            "-ac", "1", "-ar", str(sample_rate), "-y", str(out_path),
        ],
        capture_output=True,
    )
    if cut.returncode != 0:
        out_path.unlink(missing_ok=True)  # don't leave a truncated file
        return DownloadStatus.FAILED
    return DownloadStatus.OK


@dataclass
class DownloadReport:
    counts: dict[DownloadStatus, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)  # clip_ids not obtained

    def add(self, clip_id: str, status: DownloadStatus) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1
        if status in (DownloadStatus.UNAVAILABLE, DownloadStatus.FAILED):
            self.failures.append(clip_id)

    @property
    def n_ok(self) -> int:
        return self.counts.get(DownloadStatus.OK, 0)


def download_manifest(
    records: list[ClipRecord],
    clips_dir: str | Path,
    sample_rate: int = 16_000,
) -> DownloadReport:
    clips_dir = Path(clips_dir)
    report = DownloadReport()
    for r in records:
        out = clips_dir / f"{r.clip_id}.wav"
        status = download_segment(r.source_id, r.start_sec, r.duration, out, sample_rate)
        report.add(r.clip_id, status)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download AudioSet segment audio from YouTube.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--sample-rate", type=int, default=16_000)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    records = read_manifest(args.manifest)
    report = download_manifest(records, args.clips_dir, args.sample_rate)
    summary = ", ".join(f"{k.value}={v}" for k, v in sorted(report.counts.items()))
    print(f"downloaded {report.n_ok}/{len(records)} ({summary})")


if __name__ == "__main__":
    main()
