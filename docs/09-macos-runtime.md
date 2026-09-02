# 9. macOS에서 통합 모델 실행하기

이 저장소는 재생 중인 시스템 오디오를 마이크 없이 받아 다음 경로로 처리한다.

```text
재생 오디오
  ├─ CED-mini int8 ──────────────────────► 음향 유해성 판정
  └─ Wiener 강화 → 원본/강화 Silero VAD 합집합
                  → Whisper-tiny 한국어 판별
                  → Moonshine-KR ASR → KoELECTRA 유해 발화 판정
                                      └► 사건 묶기·선택적 서버 2차 판정
```

macOS에서는 Windows 전용 DeepFilterNet 실행 파일을 사용하지 않는다. 대신 이 프로젝트에 내장한
Wiener 강화가 VAD 전과 선택 발화 구간에 적용된다. 한국어 누락을 줄이는 정책, 원본/강화 VAD
합집합, 원본/Wiener LID 확률 최대 결합은 Windows 실험과 동일하다.

## 준비물

- Apple Silicon macOS 14.2 이상 권장
- GitHub CLI `gh` 로그인: 두 비공개 저장소의 릴리스 모델을 받는 데 필요
- Python 3.11과 `uv`
- `audiotee` 실행 파일. macOS 오디오 캡처 권한을 표시하려면 반드시 서명돼 있어야 한다.

```bash
git clone https://github.com/newty8323/Danger-Audio-Teenager-Korean-Gate.git
cd Danger-Audio-Teenager-Korean-Gate
uv sync --group nlp

# CED-mini·KoELECTRA 체크포인트를 기존 연구 릴리스에서 받음.
bash scripts/fetch_data.sh --models

# 새 저장소 릴리스에서 Silero와 Whisper-tiny 한국어 LID 체크포인트를 받음.
bash scripts/fetch_language_gate_models.sh
```

`scripts/fetch_data.sh --models`는 원 연구 저장소 `soysaucecrab/Danger-Audio-Teenager`의
`data-v1` 릴리스를 참조한다. 따라서 실행 계정은 그 저장소의 협업자여야 한다.

## 재생음 캡처 도우미

```bash
git clone https://github.com/makeusabrew/audiotee.git
cd audiotee
swift build -c release
sudo cp .build/release/audiotee /usr/local/bin/
codesign --force --sign - /usr/local/bin/audiotee
```

처음 실행할 때 macOS의 오디오 캡처 권한을 허용한다. 이 권한은 마이크 권한과 별개이며,
이 앱은 마이크를 요청하지 않는다.

## 실행

```bash
uv run --group nlp python -m app.main \
  --language-gate \
  --language-gate-vad artifacts/language_gate/silero_vad.jit \
  --language-gate-checkpoint artifacts/language_gate/whisper_tiny_encoder_lid.pt
```

기본 입력은 macOS 재생음이다. 대시보드는 <http://127.0.0.1:8765>에서 확인할 수 있다.
콘솔만 쓰려면 `--no-ui`를 붙인다. 캡처 전에 다음 명령으로 오디오가 실제 들어오는지 확인한다.

```bash
uv run --group nlp python -m app.sources --seconds 5
```

`peak`과 `rms`가 0에 가깝다면 모델 문제가 아니라 캡처 권한·서명·출력 장치 문제다. Bluetooth나
AirPlay 출력이 잡히지 않으면 내장 스피커로 먼저 확인하거나 BlackHole을 이용한 loopback 장치를
`--source device`로 지정한다.

## 모델 파일과 용량

| 용도 | 파일 | 크기 |
|---|---|---:|
| 음향 유해성 | CED-mini int8 | 10MB |
| 한국어 ASR | Moonshine-tiny-ko | 약 27MB |
| 유해 문장 판별 | KoELECTRA-small int8 | 28MB |
| 음성 구간 탐지 | Silero VAD | 2.27MB |
| 한국어/비한국어 판별 | Whisper-tiny LID | 16.48MB |
| 음성 강화 | Wiener | 모델 파일 0MB |

새 언어 게이트의 macOS 추가 파일은 약 **18.75MB**다. 전체 파일 용량 단순 합계는 약
83.75MB이며, 런타임·토크나이저·활성 메모리는 포함하지 않는다.

## 검증 범위

통합된 한국어 게이트는 실제 영화 음성 스모크 테스트에서 원본 2.848초 중 2.824초를 보존했다.
원본/Wiener 한국어 확률은 0.568580/0.974427이었고, 강화 전·후 VAD 타임스탬프를 합쳐
한국어 발화가 통과하도록 확인했다. 상세 수치와 한계는 [08장](08-korean-language-gate.md)에
기록돼 있다.
