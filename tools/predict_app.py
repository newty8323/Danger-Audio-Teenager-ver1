"""Interactive harm-prediction viewer (Streamlit).

Upload any audio clip -> frozen-BEATs model -> per-class probabilities, risk score
+ level, top events, log-mel spectrogram, and the MIL attention over time
(when the harmful sound occurs). Longer clips get a sliding-window risk timeline.

Run:
    uv sync --group annotator            # installs streamlit
    uv run streamlit run tools/predict_app.py

Requires a trained checkpoint + fitted risk params (produced by the frozen-BEATs
run): artifacts/ckpt_beats_v2/best.ckpt and artifacts/risk_beats_v2.json.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from datasets.taxonomy import load_taxonomy  # noqa: E402
from infer_stream import StreamRiskInference  # noqa: E402
from models.beats_extractor import BEATsExtractor  # noqa: E402
from models.harm_model import HarmModel  # noqa: E402
from preprocess.audio import load_audio  # noqa: E402
from preprocess.config import PreprocessConfig  # noqa: E402
from preprocess.logmel import LogMelExtractor  # noqa: E402
from preprocess.pipeline import fix_length  # noqa: E402
from risk.policy import BLOCK, SAFE, WARN, load_risk_policy, risk_level  # noqa: E402
from risk.scorer import RiskScorer  # noqa: E402

CKPT = "artifacts/ckpt_beats_v2/best.ckpt"
RISK = "artifacts/risk_beats_v2.json"
LEVEL_COLOR = {SAFE: "#1f7a5b", WARN: "#b26b00", BLOCK: "#b3261e"}
CAT_KO = {"threat": "위협/폭력", "gambling": "도박", "sexual": "성적", "drug": "약물"}


@st.cache_resource
def load_model():
    cfg = PreprocessConfig()
    tax = load_taxonomy()
    policy = load_risk_policy()
    extractor = BEATsExtractor(device="cpu")
    model = HarmModel.from_checkpoint(CKPT, tax.num_classes, map_location="cpu")
    model.eval()
    scorer = RiskScorer.from_policy(policy, tax).load_params(RISK)
    logmel = LogMelExtractor(cfg)
    return cfg, tax, policy, extractor, model, scorer, logmel


@torch.no_grad()
def predict_window(wav_10s, extractor, model):
    feats = extractor.extract(wav_10s)  # (1, T, 768)
    out = model(torch.from_numpy(feats), return_projection=False)
    probs = torch.sigmoid(out["logits"])[0].numpy()
    attn = out["attention"][0].numpy()
    return probs, attn


def prob_bars(probs, tax, k=8):
    order = np.argsort(-probs)[:k]
    rows = ""
    for i in order:
        name = tax.all_classes[i]
        harm = tax.is_harm(name)
        col = "#b3261e" if harm else "#5b6672"
        pct = float(probs[i]) * 100
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
            f'<div style="width:130px;font:12px ui-monospace,Menlo;color:{"#151a21" if harm else "#5b6672"}">'
            f'{name}{" ⚠" if harm else ""}</div>'
            f'<div style="flex:1;height:16px;background:#eceff2;border-radius:4px;overflow:hidden">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{col}"></div></div>'
            f'<div style="width:48px;text-align:right;font:12px ui-monospace">{probs[i]:.3f}</div></div>'
        )
    return rows


def spectrogram_img(logmel):
    lm = np.asarray(logmel, dtype=np.float32)
    lo, hi = float(lm.min()), float(lm.max())
    return np.flipud((lm - lo) / (hi - lo + 1e-9))


def main():
    st.set_page_config(page_title="Harm audio predictor", page_icon="🔊", layout="centered")
    st.title("🔊 유해 오디오 예측 뷰어")
    st.caption("frozen-BEATs 음향 모델 + 언어(음성→텍스트) 층 · 오디오를 올리면 "
               "클래스별 확률·위험도·시간축 어텐션과 종합 판정을 보여줍니다.")

    with st.sidebar:
        st.markdown("### 언어 분석 (음성→텍스트)")
        use_asr = st.checkbox("🗣 ASR로 말도 검사 (threat/도박/성적/약물)", value=False,
                              help="음성을 전사해 위험한 '내용'까지 잡습니다. "
                                   "차분히 말한 위협처럼 소리는 안전해도 말이 위험한 경우를 포착.")
        asr_model = st.selectbox("Whisper 모델", ["small", "medium", "large"], index=0,
                                 help="클수록 한국어 정확도↑·속도↓. 'asr' 의존성 그룹 필요.",
                                 disabled=not use_asr)

    if not Path(CKPT).exists() or not Path(RISK).exists():
        st.error(f"모델/리스크 아티팩트가 없습니다: {CKPT}, {RISK}")
        st.stop()
    cfg, tax, policy, extractor, model, scorer, logmel_ex = load_model()

    up = st.file_uploader("오디오 파일 (wav / mp3 / m4a / flac ...)",
                          type=["wav", "mp3", "m4a", "flac", "ogg", "opus"])
    if up is None:
        st.info("오디오 파일을 업로드하세요.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
        f.write(up.read())
        tmp = f.name
    st.audio(up)
    wav = load_audio(tmp, sample_rate=cfg.sample_rate)
    dur = len(wav) / cfg.sample_rate

    # whole-clip (first 10s window) prediction
    probs, attn = predict_window(fix_length(wav, cfg.clip_samples), extractor, model)
    risk = float(scorer.score(probs))
    level = risk_level(risk, policy)
    acoustic_max = risk  # worst acoustic risk across windows (updated by streaming below)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f'<div style="padding:16px;border-radius:12px;background:{LEVEL_COLOR[level]}18;'
            f'border:1px solid {LEVEL_COLOR[level]}55">'
            f'<div style="font:600 12px ui-monospace;letter-spacing:.1em;text-transform:uppercase;'
            f'color:{LEVEL_COLOR[level]}">risk level</div>'
            f'<div style="font-size:34px;font-weight:700;color:{LEVEL_COLOR[level]}">{level.upper()}</div>'
            f'<div style="font:13px ui-monospace;color:#5b6672">risk score {risk:.3f} · {dur:.1f}s</div></div>',
            unsafe_allow_html=True)
    with c2:
        top = np.argsort(-probs)[:3]
        st.markdown("**Top events**")
        for i in top:
            st.markdown(f"- `{tax.all_classes[i]}` — {probs[i]:.3f}")

    st.markdown("### 클래스별 확률 (상위 8, ⚠=유해)")
    st.markdown(f'<div>{prob_bars(probs, tax)}</div>', unsafe_allow_html=True)

    st.markdown("### log-mel 스펙트로그램")
    st.image(spectrogram_img(logmel_ex(fix_length(wav, cfg.clip_samples))[0]),
             use_container_width=True, clamp=True)

    st.markdown("### 시간축 어텐션 — 모델이 주목한 순간 (언제 유해음이?)")
    secs = np.linspace(0, min(dur, 10.0), len(attn))
    st.line_chart({"attention": attn}, x_label="초 (0–10s window)", y_label="attention")
    st.caption(f"어텐션 최대 지점 ≈ {secs[int(np.argmax(attn))]:.1f}s")

    if dur > cfg.clip_seconds + 0.5:
        st.markdown("### 스트리밍 위험도 타임라인 (10s 창, 5s stride)")
        infer = StreamRiskInference(tax, scorer, policy)
        results = infer.run(wav, lambda w: predict_window(w, extractor, model)[0], clip_id=up.name)
        if results:
            starts = [r.start_sec for r in results]
            acoustic_max = max(acoustic_max, max(r.risk_score for r in results))
            st.area_chart({"risk": [r.risk_score for r in results]},
                          x_label="window start (s)", y_label="risk")
            st.caption("levels: " + " · ".join(f"{s:.0f}s={r.risk_level}"
                       for s, r in zip(starts, results, strict=True)))

    # --- language layer: transcribe speech + check the words (multimodal) ---
    text_risk = 0.0
    if use_asr:
        st.markdown("### 🗣 언어 분석 (음성→텍스트, 의미 기반)")
        from text.harm_combined import score_text_all
        from text.harm_text import load_lexicon, vocabulary_prompt
        with st.spinner(f"Whisper({asr_model}) 전사 중…"):
            try:
                from text.asr import transcribe
                # prime Whisper with the harm vocabulary -> better Korean recall
                transcript = transcribe(tmp, model=asr_model,
                                        prompt=vocabulary_prompt(load_lexicon()))
            except ImportError:
                st.warning("Whisper가 설치돼 있지 않습니다:  `uv sync --group asr`")
                transcript = None
            except Exception as e:  # e.g. ffmpeg 미설치 → 친절한 메시지 (앱은 사용자 대면)
                st.warning(f"전사 실패: {e}\n(ffmpeg 필요:  `brew install ffmpeg`)")
                transcript = None
        if transcript is not None:
            st.markdown(f"> _{transcript or '(무음/전사 없음)'}_")
            with st.spinner("의미 기반 위험 분석 중…"):
                tr = score_text_all(transcript)   # lexicon + semantic
            text_risk = tr.text_risk
            if tr.text_risk > 0.05:
                c1, c2, c3 = st.columns(3)
                c1.metric("종합 text risk", f"{tr.text_risk:.2f}",
                          CAT_KO.get(tr.top_category, tr.top_category or "—"))
                c2.metric("렉시콘(키워드)", f"{tr.lexicon_risk:.2f}")
                c3.metric("의미(문맥)", f"{tr.semantic_risk:.2f}")
                if tr.lexicon.matched:
                    st.caption("키워드 매칭: " + " · ".join(
                        f"{CAT_KO.get(c, c)}[{', '.join(m)}]" for c, m in tr.lexicon.matched.items()))
                if tr.lexicon_risk >= 0.5 and tr.text_risk < 0.5:
                    st.caption("ℹ️ 키워드는 걸렸지만 문맥상 안전으로 판단(관용구/비난 등) → 억제됨.")
            else:
                st.success("위험한 내용 없음 (문맥 기반).")
            if tr.semantic_error:
                st.info(f"의미 분류 미사용 → 렉시콘만 사용: {tr.semantic_error}")

    # --- combined verdict (acoustic ⊕ language) ---
    overall = max(acoustic_max, text_risk)
    olevel = risk_level(overall, policy)
    st.markdown("### 🧩 종합 판정 (음향 + 언어)")
    st.markdown(
        f'<div style="padding:16px;border-radius:12px;background:{LEVEL_COLOR[olevel]}18;'
        f'border:1px solid {LEVEL_COLOR[olevel]}55">'
        f'<div style="font-size:30px;font-weight:700;color:{LEVEL_COLOR[olevel]}">'
        f'{olevel.upper()}</div>'
        f'<div style="font:13px ui-monospace;color:#5b6672">overall risk {overall:.3f} '
        f'= max(음향 {acoustic_max:.3f}, 언어 {text_risk:.3f})</div></div>',
        unsafe_allow_html=True)
    st.caption("음향모델은 폭력/도박(14클래스) 학습 · 성적 오디오는 윤리게이트. "
               "언어층은 한/영 렉시콘(설명가능) · 한국어 ASR 정확도는 Whisper 모델 크기에 좌우.")


if __name__ == "__main__":
    main()
