"""Validate the four-domain ASR manifest before spending training compute."""
from __future__ import annotations

import argparse
import json

from mobile_asr.manifest import SPLITS, domain_counts, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="data/manifests/mobile_asr.jsonl")
    parser.add_argument("--allow-missing-audio", action="store_true")
    args = parser.parse_args()
    rows = load_manifest(args.manifest, require_audio=not args.allow_missing_audio)
    report = {"rows": len(rows), "splits": {split: domain_counts(rows, split) for split in SPLITS}}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

