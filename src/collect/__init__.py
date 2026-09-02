"""Data collection: AudioSet manifest building + YouTube segment download (spec §3)."""

from collect.audioset import (
    AudioSetSegment,
    build_manifest,
    describe_label_map,
    invert_label_map,
    load_label_map,
    parse_ontology,
    parse_segments_csv,
    select_segments,
    validate_label_map,
)
from collect.download import (
    DownloadReport,
    DownloadStatus,
    ToolMissingError,
    download_manifest,
    download_segment,
)

__all__ = [
    "AudioSetSegment",
    "parse_ontology",
    "parse_segments_csv",
    "load_label_map",
    "invert_label_map",
    "validate_label_map",
    "describe_label_map",
    "select_segments",
    "build_manifest",
    "DownloadStatus",
    "DownloadReport",
    "ToolMissingError",
    "download_segment",
    "download_manifest",
]
