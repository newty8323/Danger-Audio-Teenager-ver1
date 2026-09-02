"""Dump BEATs (Model B / full) test probs+labels to .npz for baseline bootstrap comparison."""
import os, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")
import combined_data as CD
CD.VIOLENCE = "data_dl/manifests/violence_v2.jsonl"; CD.GAMBLING = "data_dl/manifests/gambling.jsonl"
from datasets.taxonomy import load_taxonomy
from evaluate import predict
from models.beats_finetune import build_finetune_model
from preprocess.config import PreprocessConfig
from training.trainer import resolve_device
from train_beats_finetune import RawAudioDataset, _has_clip

FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
# Require BOTH wav (BEATs input) AND log-mel feature, so the test set matches the baselines
# exactly (precompute drops ~237 near-silent clips per spec §4) -> identical clip order -> arrays align.
def _has_both(cid):
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")

tax = load_taxonomy()
_, _, te = CD.build_combined_records(exists_fn=_has_both)
loader = DataLoader(RawAudioDataset(te, tax, PreprocessConfig(), train=False), batch_size=8)
device = resolve_device("auto")
model = build_finetune_model(tax.num_classes, head_ckpt=None,
                             beats_ckpt=os.environ["BEATS_CKPT"], unfreeze_top_k=4)
model.load_state_dict(torch.load("ckpt_p2_full/best.ckpt", map_location="cpu", weights_only=False)["model"])
model.to(device)
probs, labels = predict(model, loader, device)
np.savez("data_dl/artifacts/probs_beats.npz", probs=probs, labels=labels)
print(f"saved BEATs probs {probs.shape} -> data_dl/artifacts/probs_beats.npz")
