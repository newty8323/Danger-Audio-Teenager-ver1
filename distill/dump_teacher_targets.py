"""Dump BEATs teacher soft targets for distillation. For every clip (train/val/test) runs
the fine-tuned full BEATs and saves the 4 violence LOGITS + 256-d projection embedding +
the hard violence label. Order matches combined_data records; clip_ids saved for keyed lookup.

Env: FULL_CKPT (default ckpt_beats_finetune_top4/best.ckpt), BEATS_CKPT, CLIP_DIR.
Out: distill/teacher_targets_{train,val,test}.npz  (clip_ids, vio_logits[N,4], emb[N,256], hard[N,4])
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")
import combined_data as CD
CD.VIOLENCE = os.environ.get("VIOLENCE_MANIFEST", "data_dl/manifests/violence_v2.jsonl")
CD.GAMBLING = os.environ.get("GAMBLING_MANIFEST", "data_dl/manifests/gambling.jsonl")

from datasets.taxonomy import load_taxonomy
from preprocess.config import PreprocessConfig
from models.beats_finetune import build_finetune_model
from train_beats_finetune import RawAudioDataset, _has_clip

FULL_CKPT = os.environ.get("FULL_CKPT", "ckpt_beats_finetune_top4/best.ckpt")
BEATS_CKPT = os.environ.get("BEATS_CKPT", "data_dl/weights/BEATs_iter3_plus_AS2M.pt")
FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
VIO = ["vio_scream", "vio_impact", "vio_gunshot", "vio_verbal"]
OUT = Path(__file__).resolve().parent


def _has_both(cid):
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


@torch.no_grad()
def _dump(model, records, tax, vi, cfg_pp, device, name):
    loader = DataLoader(RawAudioDataset(records, tax, cfg_pp, train=False), batch_size=8, shuffle=False)
    logits, embs = [], []
    for x, _ in loader:
        out = model(x.to(device), return_projection=True)
        logits.append(out["logits"][:, vi].float().cpu().numpy())
        embs.append(out["embeddings"].float().cpu().numpy())
    logits = np.concatenate(logits); embs = np.concatenate(embs)
    hard = np.stack([tax.encode(r.labels)[vi] for r in records]).astype(np.float32)
    cids = np.array([r.clip_id for r in records])
    path = OUT / f"teacher_targets_{name}.npz"
    np.savez(path, clip_ids=cids, vio_logits=logits, emb=embs, hard=hard)
    pos = int(hard.max(1).sum())
    print(f"[teacher] {name}: {len(records)} clips (vio-pos {pos}) -> {path.name} "
          f"logits{logits.shape} emb{embs.shape}", flush=True)


def main():
    tax = load_taxonomy()  # v1.0 23-class (teacher space)
    vi = [tax.index_of(c) for c in VIO]
    cfg_pp = PreprocessConfig()
    tr, va, te = CD.build_combined_records(exists_fn=_has_both)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_finetune_model(tax.num_classes, head_ckpt=None, beats_ckpt=BEATS_CKPT, use_layers=None)
    model.load_state_dict(torch.load(FULL_CKPT, map_location="cpu", weights_only=False)["model"], strict=True)
    model.to(device).eval()
    print(f"[teacher] loaded {FULL_CKPT} on {device}; vio idx {vi}", flush=True)
    for name, recs in (("train", tr), ("val", va), ("test", te)):
        _dump(model, recs, tax, vi, cfg_pp, device, name)


if __name__ == "__main__":
    main()
