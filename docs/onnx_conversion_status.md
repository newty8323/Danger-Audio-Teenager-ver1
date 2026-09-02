# Ver1 ONNX 변환 상태

기준일: 2026-09-01

목표 조합은 `양자화 MDX Extra Demucs → 무음 제거 → Whisper Base FP16 → KoELECTRA INT8`이다.
이 문서는 앱을 만들기 전에 ONNX 변환 결과가 실제로 실행·검증되었는지 기록한다.

| 구성 요소 | 결과 | 산출물 | 확인 방법 |
| --- | --- | --- | --- |
| KoELECTRA-small | 완료 | `artifacts/onnx/koelectra_small_harm_asraug_slang/model.int8.onnx` | PyTorch logits와 ONNX Runtime logits 비교 |
| Whisper Base encoder | 완료(초기 검증) | `artifacts/onnx/whisper_base_fp16/encoder.fp16.onnx` | 30초 log-Mel 입력으로 출력 비교 |
| Whisper Base initial decoder | 완료(초기 검증) | `artifacts/onnx/whisper_base_fp16/decoder_initial.fp16.onnx` + `.data` | 한국어 task prefix 입력으로 출력 비교 |
| Whisper cached decoder | 완료(초기 검증) | `artifacts/onnx/whisper_base_fp16/decoder_initial_cache.fp32.onnx`, `decoder_cached.fp32.onnx` | 6개 층의 self/cross KV 텐서를 명시적으로 입·출력하고 Mac CPU에서 1-token parity 확인 |
| MDX Extra Demucs | 변환 불가 | 없음 | 모델 내부의 복소수 STFT/ISTFT 및 텐서 조건문이 표준 ONNX exporter에서 지원되지 않음 |
| UVR MDX-Net Inst HQ 3 | 비교용 | `artifacts/onnx/uvr_mdx_net_inst_hq_3/UVR-MDX-NET-Inst_HQ_3.onnx` | ONNX Runtime으로 보컬/반주 2-stem 분리 |
| UVR MDXNET-3 9662 | 실시간 후보 | `artifacts/onnx/uvr_mdxnet_3_9662/UVR_MDXNET_3_9662.onnx` | 빠른 ONNX 보컬/반주 2-stem 분리 |

## 실제 수치

- KoELECTRA FP32 ONNX: 56,746,996 bytes
- KoELECTRA INT8 ONNX: 14,703,332 bytes
- KoELECTRA logit 최대 절대 차이: 0.1849
- Whisper Base FP16 encoder: 41,235,829 bytes
- Whisper Base FP16 initial decoder: 905,194 bytes + 외부 가중치 파일 약 198 MB
- Whisper encoder 최대 절대 차이: 0.4277
- Whisper initial decoder 최대 절대 차이: 0.03125

Whisper 수치는 FP16/CoreML 경로의 **숫자 단위** 비교일 뿐, 아직 실제 음원에서 토큰·문장 일치까지 확인한 결과가 아니다. Android 벤치마크는 prompt 1회 뒤 1-token KV-cache decoder를 사용한다. 단, 실제 기기에서의 속도·문장 일치는 별도로 측정해야 한다.

## 재현 명령

```bash
cd /Users/00a1go/Desktop/sht/Danger-Audio-Teenager-Ver1

uv run --group nlp --group onnx python scripts/onnx_export_koelectra.py
uv run --group nlp --group onnx python scripts/onnx_export_whisper_base.py
uv run --group nlp --group onnx python scripts/onnx_export_demucs.py
```

Demucs 명령은 현재 실패가 정상이다. 실패 원인을 숨기지 않고 재현하기 위한 스크립트이며, 그 결과를 모델 파일로 오인해서는 안 된다.

## 다음 결정

완전한 ONNX 파이프라인 벤치마크를 하려면, 다음 중 하나를 먼저 선택해야 한다.

1. Demucs만 현재 PyTorch/Metal 실행으로 유지하고, Whisper·KoELECTRA만 ONNX로 실제 4초 벤치마크한다.
2. Demucs와 동등한 역할의 ONNX 친화적 음성 분리 모델로 교체한 뒤, 분리 품질·ASR·유해 탐지 재평가를 한다.

현재 MDX Extra Demucs를 그대로 유지하면서 표준 ONNX만으로 전체 체인을 실행하는 것은 검증되지 않았으며, 이 상태에서 실시간 가능하다고 결론 내리면 안 된다.

## 대체 ONNX 분리기: UVR MDX-Net

`UVR_MDXNET_3_9662.onnx`는 29,704,436 bytes인 실시간 후보 보컬/반주 2-stem ONNX 모델이다. 기본 `ver1-uvr-onnx` 실행은 이 모델을 사용한다. 모델이 예측하는 것은 반주이며, 실험 도구는 `원음 - 예측 반주`로 보컬을 만든다.

다음 명령으로 원본 파일의 원하는 구간을 분리할 수 있다.

```bash
uv run --group nlp --group onnx python scripts/uvr_mdx_onnx_separate.py \
  "/경로/음원.mp3" --start 20 --duration 4
```

`묻고 더블로 가`의 20–24초 4초 구간에서 MDXNET-3은 705.3ms(RTF 0.1763)였고, 이전 HQ-3 모델은 3,650.8ms(RTF 0.9127)였다. Whisper·KoELECTRA 시간까지 더한 전체 파이프라인의 실시간성은 별도로 측정해야 한다.
