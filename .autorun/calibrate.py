"""Track 1 — calibration of the best BEATs model (ckpt_p2_full).
Temperature scaling fit on VAL, evaluated on TEST: is the "%" trustworthy?
Reports ECE (Expected Calibration Error) + reliability bins over harm-class predictions,
before vs after scaling. Multi-label: one global temperature T (p = sigmoid(logit/T))."""
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
from models.beats_finetune import build_finetune_model
from preprocess.config import PreprocessConfig
from training.trainer import resolve_device
from train_beats_finetune import RawAudioDataset, _has_clip

FEATURE_ROOT = "data_dl/features"
HARM = ["vio_scream", "vio_impact", "vio_gunshot", "vio_verbal", "gmb_machine", "gmb_table"]


def _has_both(cid):
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")


@torch.no_grad()
def logits_of(model, loader, device):
    model.eval(); L, Y = [], []
    for x, y in loader:
        out = model(x.to(device), return_projection=False)
        L.append(out["logits"].float().cpu().numpy()); Y.append(np.asarray(y))
    return np.concatenate(L), np.concatenate(Y)


def ece(probs, labels, n_bins=10):
    """ECE over all (clip,class) pairs: bin by predicted prob, |avg_conf - accuracy|."""
    p = probs.ravel(); y = labels.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0; rows = []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            rows.append((bins[i], 0, np.nan, np.nan)); continue
        conf = p[m].mean(); acc = y[m].mean(); w = m.mean()
        e += w * abs(conf - acc); rows.append((bins[i], int(m.sum()), conf, acc))
    return e, rows


def main():
    tax = load_taxonomy()
    _, va, te = CD.build_combined_records(exists_fn=_has_both)
    device = resolve_device("auto")
    model = build_finetune_model(tax.num_classes, head_ckpt=None,
                                 beats_ckpt=os.environ["BEATS_CKPT"], unfreeze_top_k=4)
    model.load_state_dict(torch.load("ckpt_p2_full/best.ckpt", map_location="cpu", weights_only=False)["model"])
    model.to(device)
    vl = DataLoader(RawAudioDataset(va, tax, PreprocessConfig(), train=False), batch_size=16)
    tel = DataLoader(RawAudioDataset(te, tax, PreprocessConfig(), train=False), batch_size=16)
    print(f"val {len(va)} test {len(te)}", flush=True)
    Lv, Yv = logits_of(model, vl, device)
    Lt, Yt = logits_of(model, tel, device)

    hi = [tax.all_classes.index(c) for c in HARM]
    Lv, Yv, Lt, Yt = Lv[:, hi], Yv[:, hi], Lt[:, hi], Yt[:, hi]  # harm classes only

    # fit temperature T on val (minimize BCE-with-logits NLL)
    lv = torch.tensor(Lv, dtype=torch.float32); yv = torch.tensor(Yv, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    bce = torch.nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = bce(lv / logT.exp(), yv); loss.backward(); return loss
    opt.step(closure)
    T = float(logT.exp())

    def sig(x): return 1 / (1 + np.exp(-x))
    p_before = sig(Lt); p_after = sig(Lt / T)
    e0, r0 = ece(p_before, Yt); e1, r1 = ece(p_after, Yt)
    print(f"\nfitted temperature T = {T:.3f}  (T>1 = model was over-confident)")
    print(f"ECE (harm classes, test):  before {e0:.4f}  ->  after {e1:.4f}   ({'improved' if e1<e0 else 'worse'})")
    print(f"\nreliability (after scaling): bin  n   avg_conf  accuracy")
    for (lo, n, conf, acc) in r1:
        if n: print(f"  [{lo:.1f}-{lo+0.1:.1f})  {n:5d}   {conf:.3f}     {acc:.3f}   {'over' if conf>acc else 'under'}-confident")
    np.savez("data_dl/artifacts/calibration.npz", T=T, ece_before=e0, ece_after=e1)
    print("\nsaved data_dl/artifacts/calibration.npz")


if __name__ == "__main__":
    main()
