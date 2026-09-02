"""Phase 1.6 (나) — calibrate the decision threshold on REAL + held-out data.

Scores each sentence once (text_risk) and sweeps the flag threshold to expose the
real-world operating point: real-corpus false-positive rate (kor_unsmile clean) vs
real-speech recall (the 41 recordings) vs held-out synthetic F1 (470). Picks a
recommended threshold that keeps real FP low while preserving recall.

    uv run python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
import pandas as pd  # noqa: E402  (import HF stack before src to dodge src/datasets clash)
from huggingface_hub import hf_hub_download  # noqa: E402

sys.path.insert(0, str(_ROOT / "src"))
from text.harm_combined import score_text_all  # noqa: E402

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
OUT = _ROOT / "artifacts" / "calibration.json"


def _risks(texts):
    return [score_text_all(str(t)).text_risk for t in texts]


def main() -> None:
    # --- real corpus: clean subset (all benign) -> false positives ---
    p = hf_hub_download("smilegate-ai/kor_unsmile",
                        "data/valid-00000-of-00001.parquet", repo_type="dataset")
    clean = pd.read_parquet(p)
    clean = clean[clean["clean"] == 1]["문장"].tolist()
    print(f"scoring real-corpus clean ({len(clean)})...")
    corpus_risk = _risks(clean)

    # --- real speech 41: use saved transcripts + gold ---
    rs = json.loads((_ROOT / "artifacts" / "eval_real_speech.json").read_text())["records"]
    print(f"scoring real-speech transcripts ({len(rs)})...")
    rs_scored = [(score_text_all(r["asr"]).text_risk, r["gold"] != "safe") for r in rs]

    # --- held-out 470 synthetic ---
    ho_path = _ROOT / "configs/text/harm_language_testset.jsonl"
    rows = [json.loads(x) for x in ho_path.read_text().splitlines() if x.strip()]
    print(f"scoring held-out 470 ({len(rows)})...")
    ho = [(score_text_all(r["text"]).text_risk, r["label"] != "safe") for r in rows]

    def prf(scored, thr):
        tp = sum(1 for s, h in scored if h and s >= thr)
        fp = sum(1 for s, h in scored if (not h) and s >= thr)
        fn = sum(1 for s, h in scored if h and s < thr)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        return round(p_, 3), round(r_, 3), round(f1, 3)

    print(f"\n{'thr':>5s} | {'realFP%':>7s} | {'speech P/R/F1':>18s} | {'470 P/R/F1':>18s}")
    print("-" * 62)
    table = []
    for thr in THRESHOLDS:
        fp_rate = sum(1 for s in corpus_risk if s >= thr) / len(corpus_risk)
        sp, sr, sf = prf(rs_scored, thr)
        hp, hr, hf = prf(ho, thr)
        table.append({"threshold": thr, "real_corpus_fp_rate": round(fp_rate, 4),
                      "speech": [sp, sr, sf], "heldout470": [hp, hr, hf]})
        print(f"{thr:5.2f} | {fp_rate*100:6.1f}% | "
              f"{sp:.2f}/{sr:.2f}/{sf:.2f}       | {hp:.2f}/{hr:.2f}/{hf:.2f}")

    # recommend: lowest threshold whose real FP <= 2%, else the one closest to 2%
    ok = [t for t in table if t["real_corpus_fp_rate"] <= 0.02]
    rec = (min(ok, key=lambda t: t["threshold"]) if ok
           else min(table, key=lambda t: abs(t["real_corpus_fp_rate"] - 0.02)))
    print(f"\nRECOMMENDED threshold: {rec['threshold']} "
          f"(real FP {rec['real_corpus_fp_rate']*100:.1f}%, "
          f"speech R {rec['speech'][1]}, 470 F1 {rec['heldout470'][2]})")
    OUT.write_text(json.dumps({"table": table, "recommended": rec}, ensure_ascii=False, indent=2))
    print(f"wrote {OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
