import json

from collect import audioset
from datasets.manifest import validate_manifest
from datasets.taxonomy import load_taxonomy

# A tiny AudioSet-style CSV (3 comment lines + rows; quoted comma-joined labels).
SAMPLE_CSV = (
    "# Segments csv\n"
    "# num_ytids=3\n"
    "# YTID, start_seconds, end_seconds, positive_labels\n"
    'abc123, 30.000, 40.000, "/m/03qc9zr,/m/09x0r"\n'  # Screaming (+ unrelated)
    '-xyz789, 0.000, 10.000, "/m/032s66"\n'            # Gunshot; ytid starts with '-'
    'nomatch1, 5.000, 15.000, "/t/dd00099"\n'          # no target label -> dropped
)

# Mirror of a couple of config classes so the test doesn't depend on config edits.
LABEL_MAP = {
    "vio_scream": ["/m/03qc9zr"],
    "vio_gunshot": ["/m/032s66", "/m/04zjc"],
}


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_parse_segments_csv(tmp_path):
    path = _write(tmp_path, "seg.csv", SAMPLE_CSV)
    segs = list(audioset.parse_segments_csv(path))
    assert len(segs) == 3
    assert segs[0].ytid == "abc123" and segs[0].start == 30.0 and segs[0].duration == 10.0
    assert segs[0].mids == ("/m/03qc9zr", "/m/09x0r")
    assert segs[1].ytid == "-xyz789"  # leading-dash ytid preserved


def test_invert_label_map():
    inv = audioset.invert_label_map(LABEL_MAP)
    assert inv["/m/03qc9zr"] == {"vio_scream"}
    assert inv["/m/04zjc"] == {"vio_gunshot"}


def test_select_segments_maps_and_drops_nonmatching(tmp_path):
    segs = list(audioset.parse_segments_csv(_write(tmp_path, "seg.csv", SAMPLE_CSV)))
    recs = audioset.select_segments(segs, LABEL_MAP)
    ids = {r.clip_id for r in recs}
    assert ids == {"abc123_30", "-xyz789_0"}  # nomatch1 dropped
    scream = next(r for r in recs if r.clip_id == "abc123_30")
    assert scream.labels == ["vio_scream"]
    assert scream.source == "audioset" and scream.source_id == "abc123"
    assert scream.label_confidence == "weak"


def test_select_segments_records_validate_against_taxonomy(tmp_path):
    segs = list(audioset.parse_segments_csv(_write(tmp_path, "seg.csv", SAMPLE_CSV)))
    recs = audioset.select_segments(segs, LABEL_MAP)
    assert validate_manifest(recs, load_taxonomy()) == []


def test_per_class_cap():
    segs = [
        audioset.AudioSetSegment(f"v{i}", 0.0, 10.0, ("/m/03qc9zr",)) for i in range(5)
    ]
    recs = audioset.select_segments(segs, LABEL_MAP, per_class_cap=2)
    assert len(recs) == 2


def test_validate_label_map_flags_unknown_mid():
    ontology = {"/m/03qc9zr": "Screaming"}  # missing /m/032s66, /m/04zjc
    unknown = audioset.validate_label_map(LABEL_MAP, ontology)
    assert set(unknown) == {"/m/032s66", "/m/04zjc"}


def test_describe_label_map_shows_ontology_names():
    ontology = {"/m/03qc9zr": "Screaming", "/m/032s66": "Gunshot, gunfire"}
    rows = audioset.describe_label_map(LABEL_MAP, ontology)
    assert ("vio_scream", "/m/03qc9zr", "Screaming") in rows
    assert ("vio_gunshot", "/m/032s66", "Gunshot, gunfire") in rows
    assert ("vio_gunshot", "/m/04zjc", "<MISSING>") in rows  # not in ontology


def test_clip_id_preserves_fractional_start():
    segs = [audioset.AudioSetSegment("vid", 30.2, 40.2, ("/m/03qc9zr",))]
    recs = audioset.select_segments(segs, LABEL_MAP)
    assert recs[0].clip_id == "vid_30.2"  # lossless, no collision with 30.4


def test_parse_ontology(tmp_path):
    data = [{"id": "/m/03qc9zr", "name": "Screaming"}, {"id": "/m/032s66", "name": "Gunshot"}]
    path = _write(tmp_path, "ont.json", json.dumps(data))
    ont = audioset.parse_ontology(path)
    assert ont["/m/03qc9zr"] == "Screaming"


def test_shipped_label_map_is_valid_taxonomy():
    # Every class in the shipped config must exist in the taxonomy.
    tax = load_taxonomy()
    label_map = audioset.load_label_map("configs/data/audioset_labels.yaml")
    for cls in label_map:
        assert cls in tax.categories, f"{cls} not in taxonomy"


def test_build_manifest_roundtrip(tmp_path):
    from datasets.manifest import read_manifest

    csv_path = _write(tmp_path, "seg.csv", SAMPLE_CSV)
    map_path = tmp_path / "map.yaml"
    map_path.write_text("version: t\nmap:\n  vio_scream: [/m/03qc9zr]\n")
    out = tmp_path / "manifest.jsonl"
    recs = audioset.build_manifest([csv_path], map_path, out)
    assert len(recs) == 1
    assert read_manifest(out)[0].labels == ["vio_scream"]
