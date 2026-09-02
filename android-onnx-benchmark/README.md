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

## 연속 실시간성 시험

앱의 **연속 시험 시작**은 다른 앱에서 재생되는 소리를 4초씩 계속 받아 현재 전체 파이프라인으로 처리하고, 각 창의 결과를 즉시 기기 내부에 저장한다. 앱이 비정상 종료되더라도 종료 직전까지 기록된 `windows.jsonl`과 `windows.csv`는 남는다.

### 권장 시험 순서

1. 휴대폰을 재부팅하고 배터리 절약 모드를 끈다.
2. 정확한 배터리 소모를 재려면 충전기를 분리한다. 충전 상태로 시험했다면 결과에 반드시 표시한다.
3. 앱에서 콘텐츠 종류를 고르고 콘텐츠명·회차를 적는다.
4. 처음에는 `5`분으로 기능을 확인한다.
5. 본 시험에서는 `60`분을 입력하고 **연속 시험 시작**을 누른다.
6. Android의 재생음 캡처 권한을 허용한 뒤 뉴스 또는 영화를 재생한다.
7. 계획 시간이 끝나면 앱이 자동으로 멈춘다. 중간에 멈추려면 **중지**를 누른다.
8. **마지막 시험 결과 공유**를 눌러 `summary.txt`, `windows.csv`, `windows.jsonl`을 Mac으로 보낸다.

뉴스 1시간과 영화·드라마 1시간은 서로 다른 세션으로 실행한다. 모델 프로필을 비교할 때는 같은 휴대폰, 같은 콘텐츠 구간, 같은 음량, 같은 화면 상태, 같은 서버 설정을 유지한다. 한 시험 직후 바로 다음 시험을 시작하면 발열 조건이 달라지므로, 시작 온도가 비슷해질 때까지 기기를 식힌 뒤 반복한다.

일부 앱은 저작권·보안 정책으로 Android 재생음 캡처를 막는다. 이 경우 로그의 `input_rms`가 계속 0에 가깝게 나타난다. 이것은 모델이 무음을 판단한 결과가 아니라 입력을 받지 못한 것이므로 해당 세션을 폐기하고, 캡처를 허용하는 플레이어나 앱으로 같은 파일을 재생한다.

### 생성되는 결과

결과는 앱 내부의 `files/benchmarks/<세션 시각>/`에 저장된다.

- `summary.txt`: 논문 표에 옮길 수 있는 세션 전체 요약
- `windows.csv`: 4초 창별 수치와 받아쓰기 결과
- `windows.jsonl`: 세션 시작·창·서버 응답·오류·종료 이벤트의 원본 로그

주요 계측 항목은 다음과 같다.

| 구분 | 기록 값 |
| --- | --- |
| 실시간성 | 전체 처리시간, 평균·p50·p95·p99, RTF, 캡처 간격, 대기시간, 최대 대기열, 미처리 창 |
| 모델 단계 | CED, Demucs, Whisper 전체·log-Mel·encoder·decoder, KoELECTRA 시간 |
| 탐지 출력 | CED 점수, KoELECTRA 점수, ALERT, 받아쓰기, 빈 문장, 반복 환각 |
| 자원 | 프로세스 CPU, 기기 전체 CPU 용량 대비 비율, PSS, Java heap |
| 배터리 | 잔량, 순간 전류, charge counter(µAh), energy counter(nWh), 충전 연결 여부 |
| 발열 | 배터리 온도, Android thermal status, MODERATE 이상 창 수, 후반/초반 처리시간 비율 |
| 서버 | 전송 수, 성공·실패·대기 수, 응답시간 |

`cpu_percent_one_core`는 한 코어를 완전히 쓰면 100%이며 여러 코어를 쓰면 100%를 넘을 수 있다. `cpu_percent_device`는 이 값을 논리 코어 수로 나눈 참고값이다. `battery_charge_counter_uah`와 `battery_energy_counter_nwh`는 제조사가 제공하지 않으면 측정 불가로 남는다.

`thermal_status`는 Android가 제공하는 열 압력 단계다. `0=NONE`, `1=LIGHT`, `2=MODERATE`, `3=SEVERE`, `4=CRITICAL`, `5=EMERGENCY`, `6=SHUTDOWN`이다. 일반 앱은 실제 CPU 클럭 제한 여부를 직접 확정할 수 없으므로, 이 프로젝트는 `MODERATE` 이상 창 수와 후반/초반 처리시간 비율을 함께 사용해 쓰로틀링 가능성을 해석한다.

### 실시간 통과 기준

한 번의 빠른 실행만으로 통과시키지 않고 다음 조건을 함께 확인한다.

- p95 RTF가 1 미만이다.
- p95 전체 처리시간이 4,000 ms 미만이다.
- 최대 대기열이 지속적으로 증가하지 않고 미처리 창이 0이다.
- 모델 오류가 0이다.
- 시험 후반에도 처리시간이 크게 증가하지 않는다.
- 배터리 소모와 thermal 상태가 실제 상시 사용에 감당 가능한 범위다.

뉴스에는 유해 발화가 적어도 실시간성과 오탐, ASR 환각을 측정하는 데 필요하다. 영화·드라마는 배경음악과 효과음 속 대사의 처리 성능을 본다. 다만 이 두 연속 시험만으로 유해 탐지 정확도나 재현율을 계산할 수는 없다. 정확도는 정답 구간을 표시한 별도의 유해·안전 평가 세트로 측정해야 한다.
