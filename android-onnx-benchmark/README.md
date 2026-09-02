# Android ONNX Live Pipeline

This project builds two separate Android APK profiles for a fair model-size
experiment. The application code and the Whisper Base, CED-mini, and KoELECTRA
models are identical; only the Demucs model precision differs.

| Profile | Demucs model | Demucs size | Purpose |
| --- | --- | ---: | --- |
| `baseline` | FP32 ONNX | 322 MiB | Accuracy reference; default |
| `demucs-fp16` | FP16 ONNX | 162 MiB | First smaller candidate |

The two Demucs files are never placed into the same release APK. Old fixture
waveforms and duplicate benchmark models are also excluded from both APKs.

## Baseline APK

From `android-onnx-benchmark`:

```bash
./gradlew :app:assembleDebug --offline -PmodelProfile=baseline
```

## Smaller FP16 Demucs APK

```bash
./gradlew :app:assembleDebug --offline -PmodelProfile=demucs-fp16
```

The output is `app/build/outputs/apk/debug/app-debug.apk`. Build and install one
profile at a time, then analyse the exact same media file on the phone. The first
screen displays the active model profile.

## Acceptance rule

Do not adopt the FP16 APK merely because it installs. Compare the same clips:

1. normal spoken Korean;
2. movie dialogue with background music;
3. profanity or threat dialogue; and
4. music/rap.

Keep the smaller profile only when its vocal stem, Whisper transcript, and final
`ALERT`/`SAFE` decision match the baseline for the target harmful clips.

## Why the QDQ INT8 candidate is not used

`scripts/quantize_android_demucs_int8.py` makes a separate experiment file and
never overwrites the baseline. For this Hybrid Demucs graph, however, generic QDQ
quantization keeps the large FP32 weights and adds QDQ tensors, so the result is
not smaller. It is retained only as a failed reproducibility experiment, not as
a build profile to deploy.

`demucs-fp16` is therefore the current compression candidate: it cuts the Demucs
file roughly in half while retaining nearly identical ONNX output on the checked
movie and music windows. Physical-phone timing and harmful-event recall must
still be measured before accepting it.
