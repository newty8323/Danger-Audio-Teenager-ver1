# Android ONNX Core Benchmark

This is intentionally a minimal emulator benchmark, not the harmful-audio app.
It loads the verified 4-second Hybrid Demucs FP16 ONNX core and the exact float
inputs generated on the Mac, then records Android ONNX Runtime load/inference
time and output shapes.

## Prepare assets

From the repository root:

```bash
PYTHONPATH="$PWD/scripts" uv run --group nlp --group onnx \
  python android-onnx-benchmark/prepare_assets.py \
  "/Users/00a1go/Downloads/드라마 더글로리 전재준의 찰진 속사포욕 웃긴명장면.mp3" \
  --start 5
```

Then open `android-onnx-benchmark` in Android Studio, create an Android Emulator
(API 35, ARM64 on Apple Silicon), and press **ONNX 본체 실행**.

The emulator validates Android compatibility only. Its timing is not a phone
performance result. The full app adds Android STFT/ISTFT, audio capture, Whisper
and KoELECTRA only after this core test succeeds.
