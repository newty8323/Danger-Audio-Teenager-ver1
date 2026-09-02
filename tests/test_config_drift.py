"""Guard against drift between the 'mirror' yaml configs and their dataclasses.

These four configs are not yet loaded by the CLIs (they use dataclass defaults;
hydra wiring is a pre-experiment task). Until then they can silently disagree
with the dataclasses. This test fails if an edit to either side diverges.
"""

import yaml

from losses.combined import LossConfig
from models.harm_model import ModelConfig
from preprocess.config import PreprocessConfig
from training.config import DEFAULT_CURRICULUM, TrainConfig


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _assert_matches(yaml_dict, cfg, keys):
    for k in keys:
        assert k in yaml_dict, f"missing key {k!r} in yaml"
        assert yaml_dict[k] == getattr(cfg, k), (
            f"{k}: yaml={yaml_dict[k]!r} != dataclass={getattr(cfg, k)!r}"
        )


def test_preprocess_yaml_matches_dataclass():
    y = _load("configs/data/preprocess.yaml")
    _assert_matches(y, PreprocessConfig(), [
        "sample_rate", "clip_seconds", "mono", "rms_gate_dbfs", "n_fft",
        "hop_length", "win_length", "n_mels", "fmin", "fmax", "log_offset",
    ])


def test_model_yaml_matches_dataclass():
    y = _load("configs/model/harm_model.yaml")
    _assert_matches(y, ModelConfig(), [
        "backbone", "backbone_out_dim", "attn_dim", "classifier_hidden",
        "dropout", "proj_dim",
    ])


def test_loss_yaml_matches_dataclass():
    y = _load("configs/train/loss.yaml")
    _assert_matches(y, LossConfig(), ["gamma", "mu", "temperature", "jaccard_threshold"])


def test_train_yaml_matches_dataclass():
    y = _load("configs/train/train.yaml")
    cfg = TrainConfig()
    _assert_matches(y, cfg, [
        "seed", "batch_size", "grad_accum_steps", "lr_heads", "lr_backbone",
        "weight_decay", "layer_decay", "warmup_pct", "patience", "amp",
        "time_guard_hours", "time_guard_check_steps", "device", "num_workers", "ckpt_dir",
    ])
    # curriculum: list of dicts mirrors DEFAULT_CURRICULUM
    assert len(y["curriculum"]) == len(DEFAULT_CURRICULUM)
    for stage_yaml, stage in zip(y["curriculum"], DEFAULT_CURRICULUM, strict=True):
        assert stage_yaml["name"] == stage.name
        assert stage_yaml["epochs"] == stage.epochs
        assert stage_yaml["freeze_backbone"] == stage.freeze_backbone
        assert stage_yaml["use_supcon"] == stage.use_supcon
