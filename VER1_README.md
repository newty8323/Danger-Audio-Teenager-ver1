# Danger Audio Teenager Ver1

Ver1은 ASR 정확도를 우선하는 로컬 기준 구현이다. GitHub 원본 저장소와 분리되어
있으며 다음 두 갈래를 같은 10초 창에서 실행한다.

```text
원본 16 kHz mono ──► CED-mini acoustic 유해음 판정 ───────────┐
                 └─► 원본 HTDemucs ─► 무음 제거              │
                                      └─► Whisper Base FP16   ├─► 기존 판정 결합
                                          └─► KoELECTRA+사전 ┘
                                                            └─► 선택적 Qwen 서버
```

원음은 acoustic 갈래와 서버 증거 음원에 그대로 사용한다. Demucs 보컬 분리본은
Whisper에만 들어간다. 두 모델은 순차 subprocess로 실행되므로 한 단계가 끝나면
작업 메모리가 반환된다.

## 포함 모델

- 원본 HTDemucs PyTorch 체크포인트: 80.2 MiB
- Whisper Base GGML FP16: 141.1 MiB
- 핵심 모델 합계: 221.3 MiB

여기서 FP16은 양자화 정수가 아니라 부동소수점 형식이다. 따라서 HTDemucs와
Whisper Base 모두 INT8/Q5/Q6 같은 정수 양자화를 적용하지 않은 기준 모델이다.
CED-mini와 KoELECTRA 등 기존 유해 탐지 모델은 위 합계에 포함하지 않았다. 이 두
기존 분류기도 부동소수점으로 비교하려면 실행 명령에 `--fp32`를 추가한다.

## 실행

최초 한 번 의존성을 준비한다.

```bash
cd /Users/00a1go/Desktop/sht/Danger-Audio-Teenager-Ver1
uv sync --group nlp --group asr
```

파일로 전체 유해 탐지를 시험한다.

```bash
./run_ver1.command \
  --source file \
  --file "/Users/00a1go/Downloads/테스트.mp3"
```

Mac 재생음을 실시간 감시한다.

```bash
./run_ver1.command --source audiotee
```

모든 경량화 모델을 사용하는 비교 모드는 다음과 같다.

```bash
./run_ver1.command --source audiotee --all-quantized
```

이 모드는 CED-mini·KoELECTRA INT8, 공식 MDX Extra 양자화 단일 모델(36.7 MiB),
Whisper Base Q6_K(61.7 MiB)를 사용한다. 현재 욕설 회귀 구간에서는 원본 조합이
`개새끼야`를 보존한 반면 이 조합은 `[놀람]`으로 인식했으므로, 속도 비교용이며
정확도 검증 없이 최종 모델로 채택하면 안 된다.

서버의 Qwen 단계까지 연결하려면 다음 옵션을 추가한다.

```bash
./run_ver1.command \
  --source audiotee \
  --server "http://서버-IP:8770/" \
  --upload-audio
```

대시보드는 기본적으로 <http://127.0.0.1:8765>에서 열린다. Ver1 기본 ASR 주기는
긴 문장이 합쳐지지 않도록 4초이며 `--text-every`로 바꿀 수 있다. CED의 10초
음향 분석 창은 그대로 유지된다.

## ONNX 전환 지점

향후 ONNX 실험에서는 `src/app/ver1_audio.py`의 두 경계만 교체하면 된다.

1. `_demucs_vocals()`: HTDemucs PyTorch subprocess를 ONNX Runtime 추론기로 교체
2. `_whisper()`: whisper.cpp Base FP16을 Whisper encoder/decoder ONNX 추론기로 교체

`CascadeEngine`, KoELECTRA 유해도 점수, acoustic 갈래, 사건 병합, 서버 전송 계약은
그대로 유지한다. 따라서 ONNX 전후 결과와 실행시간을 같은 UI에서 비교할 수 있다.
