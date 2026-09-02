"""VIOLENCE-ONLY BEATs (+ optional depth reduction) — the real deployment task.

Taxonomy = configs/data/classes_vio.yaml (4 violence classes, v2.0-vio). Every non-violence
clip (former gmb + former confusables) is stripped to an ALL-ZERO negative, so the backbone
learns the easier "violence vs everything" representation — the mechanism by which a
depth-reduced (lightweight) backbone can stay a viable trigger. Warm-starts MIL/proj/heads
from the strong 23-class head_ckpt, remapping the violence rows of the final classifier by
NAME (23-class -> 4-class). Same recipe as the depth experiment; only the task differs.

Test split is the SAME (_has_both: wav+feature) as probs_beats.npz, so the violence-trigger
recall@FPR is comparable across models.

Env: USE_LAYERS (unset=full 12; 6/8 for depth), TAXONOMY_CFG (default classes_vio.yaml),
     CLIP_DIR, BEATS_CKPT, HEAD_CKPT, VIOLENCE_MANIFEST, GAMBLING_MANIFEST, EPOCHS, SEED, WANDB_* .
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
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
from models.beats_finetune import BEATsRawBackbone
from models.harm_model import HarmModel, ModelConfig
from preprocess.config import PreprocessConfig
from training.config import CurriculumStage, TrainConfig
from training.trainer import Trainer, resolve_device
from train_beats_finetune import RawAudioDataset, _class_alpha, _has_clip

CLIP_DIR = os.environ.get("CLIP_DIR", "data_dl/clips")
FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
BEATS_CKPT = os.environ["BEATS_CKPT"]
HEAD_CKPT = os.environ["HEAD_CKPT"]
TAX_CFG = os.environ.get("TAXONOMY_CFG", str(_ROOT / "configs/data/classes_vio.yaml"))
V1_CFG = str(_ROOT / "configs/data/classes.yaml")
_ul = os.environ.get("USE_LAYERS", "").strip()
USE_LAYERS = int(_ul) if _ul else None
TAG = f"L{USE_LAYERS}" if USE_LAYERS else "full"
UNFREEZE_TOP_K = int(os.environ.get("UNFREEZE_TOP_K", "4"))
CKPT_DIR = os.environ.get("CKPT_DIR", f"./ckpt_beats_vio_{TAG}")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
SEED = int(os.environ.get("SEED", "42"))
OUT_PROBS = os.environ.get("OUT_PROBS", f"data_dl/artifacts/probs_beats_vio_{TAG}.npz")


def _has_both(cid):  # match probs_beats.npz test set exactly
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


def _strip(records, tax):
    """Keep only labels present in the (violence-only) taxonomy; others -> [] (all-zero
    negative). ClipRecord is a mutable dataclass, but replace() keeps it clean."""
    keep = set(tax.all_classes)
    return [replace(r, labels=[l for l in r.labels if l in keep]) for r in records]


def _warmstart_head(num_classes, tax, device):
    """HarmModel(num_classes) warm-started from the 23-class head_ckpt: shape-matching
    params load directly; the final classifier layer (classifier.net.3) is remapped
    row-by-row by CLASS NAME (v1.0 index -> v2 row). Backbone swapped to BEATs after."""
    model = HarmModel(num_classes, ModelConfig(backbone="passthrough", backbone_out_dim=768))
    old = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)["model"]
    old_tax = load_taxonomy(V1_CFG)
    new_sd = model.state_dict()
    HEAD_W, HEAD_B = "classifier.net.3.weight", "classifier.net.3.bias"
    loaded, remapped = 0, 0
    for k, v in new_sd.items():
        if k in (HEAD_W, HEAD_B):
            src = old[k]
            for j, name in enumerate(tax.all_classes):
                v[j] = src[old_tax.index_of(name)]
            remapped += 1
        elif k in old and old[k].shape == v.shape:
            new_sd[k] = old[k]; loaded += 1
    model.load_state_dict(new_sd)
    print(f"[vio-{TAG}] warm-start: {loaded} tensors direct, {remapped} classifier layers "
          f"name-remapped ({old_tax.num_classes}->{num_classes})", flush=True)
    model.backbone = BEATsRawBackbone(BEATS_CKPT, unfreeze_top_k=UNFREEZE_TOP_K, use_layers=USE_LAYERS)
    return model


def _ap(s, y):
    o = np.argsort(-s); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    p = tp / np.maximum(tp + fp, 1); r = tp / max(yy.sum(), 1); a = 0.0; prev = 0.0
    for pp, rr in zip(p, r):
        a += pp * (rr - prev); prev = rr
    return float(a)


def _val_metrics(model, loader, tax, device):
    """Taxonomy-invariant val metrics so wandb curves are COMPARABLE across runs
    (23-class vs 4-class, L6 vs L8 vs full): per-class vio AP, any-violence AP, and
    any-violence recall@fixed-FPR (the deployment trigger metric)."""
    probs, labels = predict(model, loader, device)
    m = {}
    for i, c in enumerate(tax.all_classes):
        m[f"val/ap_{c}"] = _ap(probs[:, i], labels[:, i].astype(int))
    y = labels.max(axis=1).astype(int)         # all cols are violence in vio taxonomy
    s = probs.max(axis=1)
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
               group=os.environ.get("WANDB_GROUP", "beats-vio"),
               id=os.environ.get("WANDB_RUN_ID", f"beats-vio-{TAG}"), resume="allow",
               config={"task": "violence-only", "use_layers": USE_LAYERS, "tag": TAG,
                       "unfreeze_top_k": UNFREEZE_TOP_K, "epochs": EPOCHS, "seed": SEED})
    return wandb.log


def main():
    torch.manual_seed(SEED)
    tax = load_taxonomy(TAX_CFG); cfg_pp = PreprocessConfig()
    assert tax.num_classes == len(tax.harm_classes), "vio taxonomy should be all-harm"
    tr, va, te = CD.build_combined_records(exists_fn=_has_both)
    tr, va, te = _strip(tr, tax), _strip(va, tax), _strip(te, tax)
    npos = sum(1 for r in te if r.labels)
    print(f"[vio-{TAG}] classes={tax.all_classes} | train {len(tr)} val {len(va)} test {len(te)} "
          f"(test vio-pos {npos}) seed={SEED}", flush=True)
    device = resolve_device("auto")

    model = _warmstart_head(tax.num_classes, tax, device)
    nlayers = len(model.backbone.beats.encoder.layers)
    bb = sum(p.numel() for p in model.backbone.beats.parameters()) / 1e6
    print(f"[vio-{TAG}] BEATs {nlayers}/12 layers | backbone {bb:.1f}M | "
          f"trainable top-{UNFREEZE_TOP_K} {model.backbone.trainable_parameters()/1e6:.1f}M", flush=True)

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
        extra = _val_metrics(model, vl, device=device, tax=tax)
        merged = {**info, **extra}
        if wandb_log is not None:
            wandb_log(merged)
        print(f"[vio-{TAG}] ep{info['epoch']} val_map={info['val_map']:.3f} "
              f"anyvio_ap={extra['val/anyvio_ap']:.3f} "
              f"R@1%={extra['val/anyvio_recall@fpr1']:.3f} "
              f"R@5%={extra['val/anyvio_recall@fpr5']:.3f} "
              f"R@10%={extra['val/anyvio_recall@fpr10']:.3f}", flush=True)

    trainer = Trainer(model, CombinedLoss(LossConfig(), alpha=alpha), cfg, on_epoch=_on_epoch)
    res = trainer.fit(tl, vl, resume="auto")
    print(f"[vio-{TAG}] status={res.status} best_val_mAP={res.best_metric:.3f}", flush=True)
    probs, labels = predict(trainer.model, tel, device)
    Path(OUT_PROBS).parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PROBS, probs=probs, labels=labels)
    print(f"[vio-{TAG}] saved test probs {probs.shape} -> {OUT_PROBS}", flush=True)


if __name__ == "__main__":
    main()
