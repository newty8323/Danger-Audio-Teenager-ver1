"""int8 dynamic PTQ of the fine-tuned CED-mini violence trigger — size + accuracy retention
on the same 908-clip test set. Dumps probs_ced_mini_int8.npz for
compare_vio.py. Env: CKPT (default ckpt_ced_mini_vio/best.ckpt), LIMIT (0=all)."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np, torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / ".autorun"))
os.environ.setdefault("CLIP_DIR", "data_dl/clips")
import combined_data as CD
CD.VIOLENCE = os.environ.get("VIOLENCE_MANIFEST", "data_dl/manifests/violence_v2.jsonl")
CD.GAMBLING = os.environ.get("GAMBLING_MANIFEST", "data_dl/manifests/gambling.jsonl")

from torch.utils.data import DataLoader
from datasets.taxonomy import load_taxonomy
from preprocess.config import PreprocessConfig
from models.harm_model import HarmModel, ModelConfig
from train_beats_finetune import RawAudioDataset, _has_clip
from train_ced_vio import CEDRawBackbone, _strip, _ap

CKPT = os.environ.get("CKPT", "ckpt_ced_mini_vio/best.ckpt")
FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
LIMIT = int(os.environ.get("LIMIT", "0"))


def _has_both(cid):
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


def _size_mb(model):
    b = 0
    for v in model.state_dict().values():
        if hasattr(v, "numel"):
            b += v.numel() * v.element_size()
        elif isinstance(v, tuple):
            for t in v:
                if hasattr(t, "numel"):
                    b += t.numel() * t.element_size()
    return b / 1e6


@torch.no_grad()
def _infer(model, loader):
    model.eval(); ps, ys = [], []
    t0 = time.time()
    for x, y in loader:
        out = model(x, return_projection=False)
        ps.append(torch.sigmoid(out["logits"]).float().numpy()); ys.append(np.asarray(y))
    return np.concatenate(ps), np.concatenate(ys), time.time() - t0


def _pick_quantized_engine() -> str:
    """fbgemm (x86) is preferred; Apple Silicon / ARM only ships qnnpack."""
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("fbgemm", "qnnpack"):
        if engine in supported:
            return engine
    raise RuntimeError(f"no int8 engine available; supported={supported}")


def main():
    engine = _pick_quantized_engine()
    torch.backends.quantized.engine = engine
    print(f"[quantize] engine={engine}")
    tax = load_taxonomy(str(_ROOT / "configs/data/classes_vio.yaml"))
    cfg_pp = PreprocessConfig()
    _, _, te = CD.build_combined_records(exists_fn=_has_both)
    te = _strip(te, tax)
    if LIMIT:
        te = te[:LIMIT]
    tel = DataLoader(RawAudioDataset(te, tax, cfg_pp, train=False), batch_size=8)
    print(f"[quant-ced] test {len(te)} (vio-pos {sum(1 for r in te if r.labels)})", flush=True)

    bb = CEDRawBackbone()
    model = HarmModel(tax.num_classes, ModelConfig(backbone="passthrough", backbone_out_dim=bb.out_dim))
    model.backbone = bb
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    fp32_mb = _size_mb(model)
    print(f"[quant-ced] loaded {CKPT} ({fp32_mb:.0f} MB fp32); fp32 eval (CPU)…", flush=True)
    p32, y, dt32 = _infer(model, tel)
    np.savez("data_dl/artifacts/probs_ced_mini_fp32cpu.npz", probs=p32, labels=y)

    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    int8_mb = _size_mb(q)
    print(f"[quant-ced] int8 ({int8_mb:.0f} MB); int8 eval (CPU)…", flush=True)
    p8, _, dt8 = _infer(q, tel)
    np.savez("data_dl/artifacts/probs_ced_mini_int8.npz", probs=p8, labels=y)

    yv = y.max(1).astype(int)
    a32, a8 = _ap(p32.max(1), yv), _ap(p8.max(1), yv)
    print("\n=========== CED-mini QUANTIZATION ===========")
    print(f" size:  fp32 {fp32_mb:.0f} MB -> int8 {int8_mb:.0f} MB  ({fp32_mb/max(int8_mb,1e-9):.1f}x)")
    print(f" cpu:   fp32 {dt32:.0f}s -> int8 {dt8:.0f}s ({len(te)} clips)")
    print(f" any-vio AP:  fp32 {a32:.3f} -> int8 {a8:.3f}  (Δ {a8-a32:+.3f})")
    print(f" mean|Δprob|: {float(np.abs(p32-p8).mean()):.4f}")
    print(" dumped: probs_ced_mini_int8.npz")
    print("=============================================")


if __name__ == "__main__":
    main()
