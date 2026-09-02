"""Download selected files from the large CSD ZIP without fetching the whole archive.

CSD 1.1 is distributed as a ~1.85 GB ZIP on Zenodo.  The server supports byte ranges,
so this script reads the ZIP directory and extracts only the Korean songs needed by the
small mobile-ASR experiment.  It verifies both the uncompressed size and CRC32.
"""
from __future__ import annotations

import argparse
import binascii
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = "https://zenodo.org/records/4916302/files/CSD.zip?download=1"
DEFAULT_SONGS = ("kr007a", "kr011a", "kr024a", "kr027a", "kr028a")


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression: int
    crc32: int
    compressed_size: int
    size: int
    local_header_offset: int


def _request(url: str, start: int, end: int, *, attempts: int = 5) -> bytes:
    expected = end - start + 1
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
                content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise RuntimeError(f"server ignored byte range {start}-{end}: {content_range!r}")
            if len(data) != expected:
                raise RuntimeError(f"short range response: expected {expected}, got {len(data)}")
            return data
        except (OSError, RuntimeError, urllib.error.HTTPError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"range download failed for {start}-{end}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content_range = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
    if not match:
        raise RuntimeError(f"server does not advertise a ranged file size: {content_range!r}")
    return int(match.group(1))


def _central_directory(url: str) -> list[ZipEntry]:
    size = _remote_size(url)
    tail_start = max(0, size - 65_557)
    tail = _request(url, tail_start, size - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    _, _, _, _, total, directory_size, directory_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_at
    )
    directory = _request(url, directory_offset, directory_offset + directory_size - 1)
    entries: list[ZipEntry] = []
    offset = 0
    while offset < len(directory):
        if directory[offset : offset + 4] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory entry at byte {offset}")
        fields = struct.unpack_from("<4s6H3L5H2L", directory, offset)
        name_len, extra_len, comment_len = fields[10:13]
        name_start = offset + 46
        name = directory[name_start : name_start + name_len].decode("utf-8")
        entries.append(
            ZipEntry(
                name=name,
                compression=fields[4],
                crc32=fields[7],
                compressed_size=fields[8],
                size=fields[9],
                local_header_offset=fields[-1],
            )
        )
        offset = name_start + name_len + extra_len + comment_len
    if len(entries) != total:
        raise RuntimeError(f"expected {total} ZIP entries, parsed {len(entries)}")
    return entries


def _extract(url: str, entry: ZipEntry) -> bytes:
    header = _request(url, entry.local_header_offset, entry.local_header_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {entry.name}")
    name_len, extra_len = fields[-2:]
    data_start = entry.local_header_offset + 30 + name_len + extra_len
    compressed = _request(url, data_start, data_start + entry.compressed_size - 1)
    if entry.compression == 0:
        data = compressed
    elif entry.compression == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported ZIP method {entry.compression} for {entry.name}")
    if len(data) != entry.size:
        raise RuntimeError(f"size mismatch for {entry.name}: {len(data)} != {entry.size}")
    if binascii.crc32(data) & 0xFFFFFFFF != entry.crc32:
        raise RuntimeError(f"CRC mismatch for {entry.name}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data_dl/mobile_asr/sources/csd"))
    parser.add_argument("--songs", nargs="+", default=list(DEFAULT_SONGS))
    args = parser.parse_args()

    wanted = {
        f"CSD/korean/{folder}/{song}.{suffix}"
        for song in args.songs
        for folder, suffix in (("wav", "wav"), ("csv", "csv"), ("lyric", "txt"))
    }
    entries = {entry.name: entry for entry in _central_directory(args.url)}
    missing = sorted(wanted - entries.keys())
    if missing:
        raise RuntimeError(f"files absent from CSD ZIP: {missing}")
    for name in sorted(wanted):
        relative = Path(name).relative_to("CSD/korean")
        destination = args.output / relative
        entry = entries[name]
        if destination.is_file() and destination.stat().st_size == entry.size:
            print(f"[skip] {destination}")
            continue
        print(f"[get]  {name} ({entry.compressed_size / 1_000_000:.2f} MB)")
        data = _extract(args.url, entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"[done] {len(wanted)} files under {args.output}")


if __name__ == "__main__":
    main()
