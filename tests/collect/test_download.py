import pytest

from collect import download
from collect.download import DownloadStatus
from datasets.manifest import ClipRecord


class FakeProc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def _patch_tools(monkeypatch, present=True):
    monkeypatch.setattr(download.shutil, "which", lambda t: ("/usr/bin/" + t) if present else None)


def test_skips_existing_file(tmp_path, monkeypatch):
    out = tmp_path / "clip.wav"
    out.write_bytes(b"x")
    # Should not even look for tools when the file already exists.
    monkeypatch.setattr(download.shutil, "which", lambda t: pytest.fail("tool lookup"))
    assert download.download_segment("yt", 0.0, 10.0, out) == DownloadStatus.SKIPPED


def test_missing_tool_raises(tmp_path, monkeypatch):
    _patch_tools(monkeypatch, present=False)
    with pytest.raises(download.ToolMissingError):
        download.download_segment("yt", 0.0, 10.0, tmp_path / "c.wav")


def test_ok_path(tmp_path, monkeypatch):
    _patch_tools(monkeypatch)

    def fake_run(cmd, **kw):
        if "-g" in cmd:  # yt-dlp resolve
            return FakeProc(0, "https://stream/url\n")
        return FakeProc(0)  # ffmpeg cut

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    assert download.download_segment("yt", 0.0, 10.0, tmp_path / "c.wav") == DownloadStatus.OK


def test_unavailable_when_resolve_fails(tmp_path, monkeypatch):
    _patch_tools(monkeypatch)
    monkeypatch.setattr(download.subprocess, "run", lambda cmd, **kw: FakeProc(1, ""))
    status = download.download_segment("yt", 0.0, 10.0, tmp_path / "c.wav")
    assert status == DownloadStatus.UNAVAILABLE


def test_failed_when_cut_fails(tmp_path, monkeypatch):
    _patch_tools(monkeypatch)

    def fake_run(cmd, **kw):
        return FakeProc(0, "https://stream/url\n") if "-g" in cmd else FakeProc(1)

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    out = tmp_path / "c.wav"
    assert download.download_segment("yt", 0.0, 10.0, out) == DownloadStatus.FAILED
    assert not out.exists()  # truncated file cleaned up


def test_download_manifest_tallies(tmp_path, monkeypatch):
    recs = [
        ClipRecord(f"c{i}", "audioset", f"v{i}", 0.0, 10.0, ["vio_scream"], "weak", "train")
        for i in range(3)
    ]
    seq = iter([DownloadStatus.OK, DownloadStatus.UNAVAILABLE, DownloadStatus.OK])
    monkeypatch.setattr(download, "download_segment", lambda *a, **k: next(seq))
    report = download.download_manifest(recs, tmp_path / "clips")
    assert report.n_ok == 2
    assert report.counts[DownloadStatus.UNAVAILABLE] == 1
    assert report.failures == ["c1"]
