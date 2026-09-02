"""Interactive BSRNN-vocals ONNX experiment.

Run from the repository root:
  uv run --group annotator --group nlp --group onnx streamlit run tools/bsrnn_vocal_experiment_app.py
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/onnx/bsrnn_vocals/bsrnn_vocals.onnx"
WHISPER = ROOT / "artifacts/ver1/bin/whisper-cli"
WHISPER_BASE = ROOT / "artifacts/ver1/whisper/ggml-base.bin"
SR = 44_100
N_FFT = 4_096
HOP = 1_024


@st.cache_resource
def load_separator(model_path: str):
    available = ort.get_available_providers()
    providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                 if "CoreMLExecutionProvider" in available else ["CPUExecutionProvider"])
    return ort.InferenceSession(model_path, providers=providers), providers


def separate_vocals(audio: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """Apply the model's [frames, 2049] vocal mask to a stereo STFT."""
    if audio.ndim == 1:
        audio = np.stack((audio, audio))
    audio = audio[:2].astype(np.float32, copy=False)
    spectra = [librosa.stft(channel, n_fft=N_FFT, hop_length=HOP) for channel in audio]
    # The ONNX model receives one magnitude spectrum per batch row.
    magnitudes = np.concatenate([np.abs(spec).T for spec in spectra], axis=0).astype(np.float32)
    mask = session.run(None, {"input": magnitudes})[0]
    if mask.shape != magnitudes.shape:
        raise RuntimeError(f"unexpected BSRNN output shape: {mask.shape}")
    frames = spectra[0].shape[1]
    vocals = []
    for index, spec in enumerate(spectra):
        estimated = spec * mask[index * frames:(index + 1) * frames].T
        vocals.append(librosa.istft(estimated, hop_length=HOP, length=audio.shape[1]))
    return np.asarray(vocals, dtype=np.float32)


def whisper(wav_path: Path) -> tuple[str, float]:
    started = time.perf_counter()
    result = subprocess.run(
        [str(WHISPER), "-m", str(WHISPER_BASE), "-f", str(wav_path),
         "-l", "ko", "-nt", "-np"], capture_output=True,
    )
    elapsed = time.perf_counter() - started
    output = result.stdout.decode("utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1000:])
    lines = [line.removeprefix("-").strip() for line in output.splitlines()]
    return "\n".join(line for line in lines if line), elapsed


def main() -> None:
    st.set_page_config(page_title="BSRNN 보컬 분리 실험", page_icon="🎙️", layout="wide")
    st.title("🎙️ BSRNN vocal ONNX 실험")
    st.caption("원음 → STFT → BSRNN ONNX 음성 마스크 → ISTFT → Whisper Base 비교")

    if not MODEL.is_file():
        st.error(f"BSRNN 모델이 없습니다: {MODEL}")
        st.stop()
    uploaded = st.file_uploader("음원 파일", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if uploaded is None:
        st.info("영화·노래·일반 발화 파일을 올린 뒤 같은 구간을 기존 Demucs 결과와 비교하세요.")
        st.stop()

    with tempfile.TemporaryDirectory(prefix="bsrnn-web-") as directory:
        root = Path(directory)
        source = root / uploaded.name
        source.write_bytes(uploaded.getvalue())
        full, _ = librosa.load(str(source), sr=SR, mono=False)
        if full.ndim == 1:
            full = np.stack((full, full))
        duration = len(full[0]) / SR
        left, right = st.columns(2)
        start = left.slider("시작 시각 (초)", 0.0, max(0.0, duration - 0.25), 0.0, 0.25)
        maximum = max(0.25, duration - start)
        length = right.slider("실험 구간 길이 (초)", 0.25, min(30.0, maximum), min(4.0, maximum), 0.25)
        run_asr = st.checkbox("분리 보컬을 Whisper Base로 받아쓰기", value=True)

        lo, hi = int(start * SR), int((start + length) * SR)
        selected = full[:, lo:hi]
        # Streamlit expects a NumPy audio array as [channels, samples].  Passing
        # the transposed [samples, channels] array makes it attempt to write
        # hundreds of thousands of WAV channels (the ushort header error).
        st.audio(selected, sample_rate=SR)
        if not st.button("BSRNN 보컬 분리 실행", type="primary"):
            st.stop()

        session, providers = load_separator(str(MODEL))
        with st.spinner("BSRNN ONNX로 보컬 분리 중…"):
            started = time.perf_counter()
            vocals = separate_vocals(selected, session)
            separation_seconds = time.perf_counter() - started
        vocal_path = root / "bsrnn-vocals.wav"
        sf.write(vocal_path, vocals.T, SR)

        a, b, c = st.columns(3)
        a.metric("모델 파일", f"{MODEL.stat().st_size / 1024 / 1024:.1f} MiB")
        b.metric("분리 시간", f"{separation_seconds:.3f}초")
        c.metric("분리 RTF", f"{separation_seconds / length:.3f}")
        st.caption("실행 제공자: " + " → ".join(providers))
        st.subheader("분리된 보컬")
        st.audio(vocals, sample_rate=SR)

        transcript, asr_seconds = "", 0.0
        if run_asr:
            with st.spinner("Whisper Base 받아쓰기 중…"):
                transcript, asr_seconds = whisper(vocal_path)
            st.subheader("Whisper Base 결과")
            st.code(transcript or "(전사 없음)", language=None)
            st.caption(f"Whisper 시간: {asr_seconds:.3f}초 · 분리+Whisper RTF: "
                       f"{(separation_seconds + asr_seconds) / length:.3f}")

        report = {
            "model": str(MODEL), "providers": providers, "start_sec": start,
            "duration_sec": length, "separation_sec": separation_seconds,
            "separation_rtf": separation_seconds / length, "whisper_sec": asr_seconds,
            "transcript": transcript,
        }
        st.download_button("실험 JSON 기록 내려받기", json.dumps(report, ensure_ascii=False, indent=2),
                           file_name="bsrnn_experiment.json", mime="application/json")


if __name__ == "__main__":
    main()
