"""One-shot harm verdict for an audio file — "is this dangerous?"

    uv run python tools/predict.py path/to/audio.wav

Prints an overall risk verdict (SAFE / WARN / BLOCK), the risk score, and the top
harmful events with timing. Slides a 10s window over longer clips and reports the
worst window. Uses the frozen-BEATs model + fitted risk scorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets.taxonomy import load_taxonomy  # noqa: E402
from infer_stream import StreamRiskInference  # noqa: E402
from models.beats_extractor import BEATsExtractor  # noqa: E402
from models.harm_model import HarmModel  # noqa: E402
from preprocess.audio import load_audio  # noqa: E402
from preprocess.config import PreprocessConfig  # noqa: E402
from preprocess.pipeline import fix_length  # noqa: E402
from risk.policy import load_risk_policy  # noqa: E402
from risk.scorer import RiskScorer  # noqa: E402

CKPT = "artifacts/ckpt_beats_v2/best.ckpt"
RISK = "artifacts/risk_beats_v2.json"
_ORDER = {"safe": 0, "warn": 1, "block": 2}
_LABEL = {"safe": "🟢 SAFE (안전)", "warn": "🟡 WARN (주의)", "block": "🔴 BLOCK (위험/차단)"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Harm verdict for an audio file.")
    ap.add_argument("audio")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--risk", default=RISK)
    ap.add_argument("--asr", action="store_true",
                    help="also transcribe speech and check the words (needs 'asr' dep group)")
    ap.add_argument("--asr-model", default="small", help="whisper model (small/medium/large)")
    args = ap.parse_args()

    cfg = PreprocessConfig()
    tax = load_taxonomy()
    policy = load_risk_policy()
    extractor = BEATsExtractor(device="cpu")
    model = HarmModel.from_checkpoint(args.ckpt, tax.num_classes, map_location="cpu")
    model.eval()
    scorer = RiskScorer.from_policy(policy, tax).load_params(args.risk)

    @torch.no_grad()
    def predict_window(win: np.ndarray) -> np.ndarray:
        feats = extractor.extract(win)
        return torch.sigmoid(model(torch.from_numpy(feats), return_projection=False)["logits"])[0].numpy()

    wav = load_audio(args.audio, sample_rate=cfg.sample_rate)
    dur = len(wav) / cfg.sample_rate
    if len(wav) < cfg.clip_samples:
        wav = fix_length(wav, cfg.clip_samples)

    results = StreamRiskInference(tax, scorer, policy).run(wav, predict_window, clip_id="clip")
    worst = max(results, key=lambda r: _ORDER[r.risk_level] * 10 + r.risk_score)

    print(f"\n파일: {args.audio}  ({dur:.1f}s, {len(results)} windows)")
    print(f"판정: {_LABEL[worst.risk_level]}   risk score {worst.risk_score:.3f}  @ {worst.start_sec:.0f}s")
    harm = [e for e in worst.top_events if tax.is_harm(e["class"])]
    if harm:
        print("가장 유해로 의심되는 소리:")
        for e in harm:
            print(f"   - {e['class']:14s} {e['prob']:.2f}")
    else:
        print("유해 클래스 신호 낮음 (top 이벤트가 모두 무해/혼동 클래스).")
    if len(results) > 1:
        line = "  ".join(f"{r.start_sec:.0f}s:{r.risk_level[0].upper()}" for r in results)
        print(f"타임라인: {line}")

    # --- language layer: transcribe speech + judge the MEANING (multimodal) ---
    overall = worst.risk_score
    if args.asr:
        from text.asr import transcribe
        from text.harm_combined import score_text_all
        from text.harm_text import load_lexicon, vocabulary_prompt
        # prime Whisper with the harm vocabulary -> better Korean recall
        text = transcribe(args.audio, model=args.asr_model,
                          prompt=vocabulary_prompt(load_lexicon()))
        tr = score_text_all(text)   # lexicon + semantic (generalizes to implicit harm)
        print(f"\n[언어] 전사: '{text}'")
        if tr.text_risk > 0.05:
            src = f"렉시콘 {tr.lexicon_risk:.2f} / 의미 {tr.semantic_risk:.2f}"
            print(f"[언어] 위험 내용 감지: risk {tr.text_risk:.2f} · {tr.top_category} · ({src})")
        else:
            print("[언어] 위험한 내용 없음.")
        if tr.semantic_error:
            print(f"[언어] (의미 분류 미사용: {tr.semantic_error})")
        overall = max(overall, tr.text_risk)

    lvl = "block" if overall >= policy.tau_block else ("warn" if overall >= policy.tau_warn else "safe")
    print(f"\n=== 종합 판정 (음향+언어): {_LABEL[lvl]}   overall risk {overall:.3f} ===")
    print("주의: 음향모델은 폭력/도박(14클래스) 학습 · 성적 오디오는 윤리게이트. "
          "언어층은 한/영 렉시콘(설명가능) · 한국어 ASR 정확도는 모델 크기에 좌우.")


if __name__ == "__main__":
    main()
