"""Missing model weights must produce one actionable message, not a torch traceback."""
import pytest

from cascade import pipeline as P


def test_missing_artifacts_lists_absent_files(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_ROOT", tmp_path)
    missing = P.missing_artifacts(text=True)
    assert len(missing) == 2
    assert any("violence trigger" in m for m in missing)
    assert any("text classifier" in m for m in missing)


def test_missing_artifacts_skips_text_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_ROOT", tmp_path)
    assert len(P.missing_artifacts(text=False)) == 1


def _make_ready(root):
    (root / "ckpt_ced_mini_vio").mkdir(exist_ok=True)
    (root / "ckpt_ced_mini_vio/best.ckpt").write_bytes(b"x")
    d = root / P.TEXT_MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (d / f).write_bytes(b"{}")
    return d


def test_missing_artifacts_empty_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_ROOT", tmp_path)
    _make_ready(tmp_path)
    assert P.missing_artifacts(text=True) == []


def test_empty_model_dir_is_reported_as_incomplete(monkeypatch, tmp_path):
    """An extracted-but-empty dir made transformers claim the PATH was a bad Hub repo id."""
    monkeypatch.setattr(P, "_ROOT", tmp_path)
    (tmp_path / "ckpt_ced_mini_vio").mkdir()
    (tmp_path / "ckpt_ced_mini_vio/best.ckpt").write_bytes(b"x")
    (tmp_path / P.TEXT_MODEL_DIR).mkdir(parents=True)
    (missing,) = P.missing_artifacts(text=True)
    assert "incomplete" in missing and "config.json" in missing and "(empty)" in missing


def test_model_dir_without_weights_is_incomplete(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_bytes(b"{}")
    (d / "tokenizer_config.json").write_bytes(b"{}")
    problem = P._hf_dir_problem(d)
    assert problem and "model.safetensors" in problem


def test_require_hf_dir_error_mentions_the_fix(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    with pytest.raises(FileNotFoundError) as e:
        P._require_hf_dir(d, "text classifier")
    msg = str(e.value)
    assert "text classifier" in msg and "fetch_data.sh --models" in msg


def test_require_error_names_the_file_and_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        P._require(tmp_path / "nope.ckpt", "violence trigger checkpoint")
    msg = str(e.value)
    assert "violence trigger checkpoint" in msg
    assert "nope.ckpt" in msg
    assert "fetch_data.sh --models" in msg


def test_preflight_exits_with_guidance(monkeypatch, tmp_path):
    from app import main as M
    monkeypatch.setattr(P, "_ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        M._preflight(text_enabled=True)
    assert "fetch_data.sh --models" in str(e.value)
