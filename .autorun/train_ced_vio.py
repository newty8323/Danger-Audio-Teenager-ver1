"""CED-mini VIOLENCE-ONLY fine-tune — candidate to replace BEATs (90M) as the acoustic
trigger with a 9x smaller backbone (10M, AudioSet mAP 49.0 > BEATs 48.6). See model_light.md.

Same recipe/split as the BEATs experiments (top-4 blocks unfrozen, focal-BCE + SupCon,
batch 8 x accum 4, class-balanced sampler, violence-only taxonomy v2.0-vio, non-violence
clips = all-zero negatives). Heads are NEW-init (CED dim 256 != BEATs 768 -> no warm-start;
noted as an honest difference vs the BEATs baseline which warm-started from a 23-class head).

Backbone: HF `mispeech/ced-mini` encoder. Raw 16kHz wav -> GPU mel (identical params to CED's
feature extractor: n_fft 512, win 512, hop 160, n_mels 64, AmplitudeToDB top_db=120) ->
CedModel.forward -> token sequence (B, N', 256) -> our MIL attention + heads.

Test split identical to probs_beats*.npz -> comparable via .autorun/compare_vio.py.
Env: CED_ID (default mispeech/ced-mini), UNFREEZE_TOP_K, EPOCHS, SEED, CKPT_DIR, OUT_PROBS, WANDB_*.
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torchaudio.transforms as AT
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")
import combined_data as CD
CD.VIOLENCE = os.environ.get("VIOLENCE_MANIFEST", "data_dl/manifests/violence_v2.jsonl")
CD.GAMBLING = os.environ.get("GAMBLING_MANIFEST", "data_dl/manifests/gambling.jsonl")

from datasets.sampler import BalancedBatchSampler
from datasets.taxonomy import load_taxonomy
from evaluate import predict
from losses.combined import CombinedLoss, LossConfig
from models.harm_model import HarmModel, ModelConfig
from preprocess.config import PreprocessConfig
from training.config import CurriculumStage, TrainConfig
from training.trainer import Trainer, resolve_device
from train_beats_finetune import RawAudioDataset, _class_alpha, _has_clip

CLIP_DIR = os.environ.get("CLIP_DIR", "data_dl/clips")
FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
CED_ID = os.environ.get("CED_ID", "mispeech/ced-mini")
TAX_CFG = os.environ.get("TAXONOMY_CFG", str(_ROOT / "configs/data/classes_vio.yaml"))
TAG = os.environ.get("TAG", CED_ID.split("/")[-1])          # e.g. ced-mini
UNFREEZE_TOP_K = int(os.environ.get("UNFREEZE_TOP_K", "4"))
CKPT_DIR = os.environ.get("CKPT_DIR", f"./ckpt_{TAG.replace('-', '_')}_vio")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
SEED = int(os.environ.get("SEED", "42"))
OUT_PROBS = os.environ.get("OUT_PROBS", f"data_dl/artifacts/probs_{TAG.replace('-', '_')}.npz")


class CEDRawBackbone(nn.Module):
    """Raw 16 kHz waveform (B, N) -> CED token embeddings (B, N', 256).

    Mel computed on-GPU with the exact params of CED's HF feature extractor, then the
    pretrained CedModel encoder. Only the top ``unfreeze_top_k`` transformer blocks
    (+ final norm) are trainable — mirrors BEATs strategy B."""

    manages_own_freezing = True

    def __init__(self, model_id: str = "mispeech/ced-mini", unfreeze_top_k: int = 4):
        super().__init__()
        # NOTE: must load via ForAudioClassification and take .encoder — the checkpoint keys
        # are prefixed "encoder.", so bare AutoModel silently random-inits EVERYTHING.
        from transformers import AutoModelForAudioClassification
        full = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True)
        self.ced = full.encoder
        self.out_dim = self.ced.config.embed_dim
        # identical to feature_extraction_ced.py defaults (verified against cached source)
        self.mel = AT.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512,
                                     hop_length=160, n_mels=64, f_min=0, center=True)
        self.to_db = AT.AmplitudeToDB(top_db=120)
        self.unfreeze_top_k = unfreeze_top_k
        self._set_trainable()

    def _set_trainable(self):
        for p in self.ced.parameters():
            p.requires_grad = False
        blocks = self.ced.blocks
        for blk in blocks[len(blocks) - self.unfreeze_top_k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.ced.norm.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return sum(p.numel() for p in self.ced.parameters() if p.requires_grad)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # feature extraction is fixed
            feats = self.to_db(self.mel(waveform))          # (B, 64, T)
        out = self.ced(input_values=feats)                   # token sequence in .logits
        return out.logits.contiguous()                       # (B, N', 256)


def _has_both(cid):  # match probs_beats*.npz test set exactly
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


def _strip(records, tax):
    keep = set(tax.all_classes)
    return [replace(r, labels=[l for l in r.labels if l in keep]) for r in records]


def _ap(s, y):
    o = np.argsort(-s); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    p = tp / np.maximum(tp + fp, 1); r = tp / max(yy.sum(), 1); a = 0.0; prev = 0.0
    for pp, rr in zip(p, r):
        a += pp * (rr - prev); prev = rr
    return float(a)


def _val_metrics(model, loader, tax, device):
    probs, labels = predict(model, loader, device)
    m = {}
    for i, c in enumerate(tax.all_classes):
        m[f"val/ap_{c}"] = _ap(probs[:, i], labels[:, i].astype(int))
    y = labels.max(axis=1).astype(int); s = probs.max(axis=1)
    m["val/anyvio_ap"] = _ap(s, y)
    neg = np.sort(s[y == 0])[::-1]; pos = s[y == 1]
    for f in (0.01, 0.05, 0.10):
        if len(neg) == 0 or len(pos) == 0:
            m[f"val/anyvio_recall@fpr{int(f*100)}"] = float("nan"); continue
        k = max(0, int(np.floor(f * len(neg))) - 1); thr = neg[min(k, len(neg) - 1)]
        m[f"val/anyvio_recall@fpr{int(f*100)}"] = float((pos >= thr).mean())
    return m


def _make_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        print("(no WANDB_API_KEY — skipping wandb)", flush=True); return None
    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "audio-harm"),
               group=os.environ.get("WANDB_GROUP", "ced"),
               id=os.environ.get("WANDB_RUN_ID", f"{TAG}-vio"), resume="allow",
               config={"task": "violence-only", "backbone": CED_ID,
                       "unfreeze_top_k": UNFREEZE_TOP_K, "epochs": EPOCHS, "seed": SEED})
    return wandb.log


def main():
    torch.manual_seed(SEED)
    tax = load_taxonomy(TAX_CFG); cfg_pp = PreprocessConfig()
    tr, va, te = CD.build_combined_records(exists_fn=_has_both)
    tr, va, te = _strip(tr, tax), _strip(va, tax), _strip(te, tax)
    print(f"[{TAG}] classes={tax.all_classes} | train {len(tr)} val {len(va)} test {len(te)} "
          f"(test vio-pos {sum(1 for r in te if r.labels)}) seed={SEED}", flush=True)
    device = resolve_device("auto")

    backbone = CEDRawBackbone(CED_ID, unfreeze_top_k=UNFREEZE_TOP_K)
    model = HarmModel(tax.num_classes, ModelConfig(backbone="passthrough",
                                                   backbone_out_dim=backbone.out_dim))
    model.backbone = backbone  # heads NEW-init at dim 256 (no 768-head warm-start possible)
    total = sum(p.numel() for p in backbone.ced.parameters()) / 1e6
    print(f"[{TAG}] backbone {total:.1f}M (dim {backbone.out_dim}) | "
          f"trainable top-{UNFREEZE_TOP_K} {backbone.trainable_parameters()/1e6:.1f}M", flush=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    alpha, _ = _class_alpha(tr, tax, device)
    cfg = TrainConfig(device="auto", batch_size=8, grad_accum_steps=4, num_workers=2,
                      lr_heads=1e-4, lr_backbone=1e-5, layer_decay=1.0, warmup_pct=0.05, patience=8,
                      amp=use_bf16, amp_dtype="bf16", ckpt_dir=CKPT_DIR, seed=SEED,
                      curriculum=(CurriculumStage("finetune", EPOCHS, freeze_backbone=False, use_supcon=True),))
    sampler = BalancedBatchSampler(tr, tax, cfg.batch_size, seed=cfg.seed)
    tl = DataLoader(RawAudioDataset(tr, tax, cfg_pp, train=True, seed=cfg.seed),
                    batch_sampler=sampler, num_workers=cfg.num_workers)
    vl = DataLoader(RawAudioDataset(va, tax, cfg_pp, train=False), batch_size=8)
    tel = DataLoader(RawAudioDataset(te, tax, cfg_pp, train=False), batch_size=8)

    wandb_log = _make_wandb()

    def _on_epoch(info):
        extra = _val_metrics(model, vl, tax, device)
        if wandb_log is not None:
            wandb_log({**info, **extra})
        print(f"[{TAG}] ep{info['epoch']} val_map={info['val_map']:.3f} "
              f"anyvio_ap={extra['val/anyvio_ap']:.3f} "
              f"R@1%={extra['val/anyvio_recall@fpr1']:.3f} "
              f"R@5%={extra['val/anyvio_recall@fpr5']:.3f} "
              f"R@10%={extra['val/anyvio_recall@fpr10']:.3f}", flush=True)

    trainer = Trainer(model, CombinedLoss(LossConfig(), alpha=alpha), cfg, on_epoch=_on_epoch)
    res = trainer.fit(tl, vl, resume="auto")
    print(f"[{TAG}] status={res.status} best_val_mAP={res.best_metric:.3f}", flush=True)
    probs, labels = predict(trainer.model, tel, device)
    Path(OUT_PROBS).parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PROBS, probs=probs, labels=labels)
    print(f"[{TAG}] saved test probs {probs.shape} -> {OUT_PROBS}", flush=True)


if __name__ == "__main__":
    main()
