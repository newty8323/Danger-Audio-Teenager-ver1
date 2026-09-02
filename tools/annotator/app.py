"""Streamlit annotator for HNM review (spec §7 step 3).

Thin UI over ``mining.review.ReviewSession``: shows a candidate's waveform,
log-mel spectrogram, audio player, the model's top harm class + prob, and the
CLAP pseudo-label; one click records the decision. All review logic lives in
``src/mining/review.py`` (unit-tested); this file only renders it.

Run:
    uv run streamlit run tools/annotator/app.py -- \
        --queue data/mining/queue.jsonl --clips-dir data/clips \
        --decisions data/mining/decisions.jsonl

The `--` separates streamlit args from this script's args.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when launched via `streamlit run`.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from datasets.taxonomy import load_taxonomy  # noqa: E402
from mining.review import FALSE_POSITIVE, POSITIVE, REJECT, SKIP, ReviewSession  # noqa: E402
from preprocess.audio import load_audio  # noqa: E402
from preprocess.config import PreprocessConfig  # noqa: E402
from preprocess.logmel import LogMelExtractor  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--decisions", default="decisions.jsonl")
    p.add_argument("--classes", default=None)
    # streamlit passes its own argv; ignore unknowns
    args, _ = p.parse_known_args()
    return args


@st.cache_resource
def _load(queue: str, decisions: str):
    session = ReviewSession.from_queue(queue)
    if Path(decisions).exists():
        session.load_decisions(decisions)
    return session


def main() -> None:
    args = _parse_args()
    taxonomy = load_taxonomy(args.classes)
    session = _load(args.queue, args.decisions)
    session.taxonomy = taxonomy  # enable immediate label-kind validation in decide()
    extractor = LogMelExtractor(PreprocessConfig())

    st.title("HNM review")
    done, total = session.progress()
    st.progress(done / total if total else 0.0, text=f"{done}/{total} decided")

    pending = session.pending()
    if not pending:
        st.success("Queue complete.")
        session.save_decisions(args.decisions)
        st.stop()

    cand = pending[0]
    st.subheader(f"{cand.clip_id}")
    st.write(
        f"predicted **{cand.top_harm_class}** ({cand.top_harm_prob:.2f}) · "
        f"reason: {cand.reason} · CLAP: {cand.clap_pseudo_label or '—'}"
    )

    audio_path = Path(args.clips_dir) / f"{cand.clip_id}.wav"
    if audio_path.exists():
        st.audio(str(audio_path))
        wave = load_audio(str(audio_path))
        logmel = extractor(wave)[0]  # (F, T)
        st.image(_spec_image(logmel), caption="log-mel", use_container_width=True)
    else:
        st.warning(f"audio not found: {audio_path}")

    confusables = list(taxonomy.confusable_classes)
    harms = list(taxonomy.harm_classes)
    col1, col2, col3 = st.columns(3)
    with col1:
        fp_label = st.selectbox("confusable label", confusables, key="fp")
        if st.button("False positive"):
            _record(session, cand.clip_id, FALSE_POSITIVE, fp_label, args.decisions)
    with col2:
        pos_label = st.selectbox("harm label", harms, key="pos")
        if st.button("True harm"):
            _record(session, cand.clip_id, POSITIVE, pos_label, args.decisions)
    with col3:
        if st.button("Reject"):
            _record(session, cand.clip_id, REJECT, None, args.decisions)
        if st.button("Skip"):
            _record(session, cand.clip_id, SKIP, None, args.decisions)


def _record(session: ReviewSession, clip_id: str, action: str, label, decisions_path: str) -> None:
    session.decide(clip_id, action, label)
    session.save_decisions(decisions_path)
    st.rerun()


def _spec_image(logmel: np.ndarray) -> np.ndarray:
    lm = np.asarray(logmel, dtype=np.float32)
    lo, hi = float(lm.min()), float(lm.max())
    norm = (lm - lo) / (hi - lo + 1e-9)
    return np.flipud(norm)  # low freq at bottom


if __name__ == "__main__":
    main()
