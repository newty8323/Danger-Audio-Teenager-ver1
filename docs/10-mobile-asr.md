# 10. 100MB 이하 한국어 통합 ASR 실험

## 목표

발화 탐지기가 노래나 작은 영화 대사를 제거하지 않도록, 재생 중인 모든 비무음 오디오를
작은 ASR에 입력한다. 스마트폰에 탑재되는 모델 파일의 합은 100MB 이하여야 한다.

```
원음 ─┬─ CED-mini int8(약 10MB) ─► 폭력 음향
      └─ 한국어 특화 Whisper-tiny int8(목표 약 40MB)
            └─ KoELECTRA-small int8(약 28MB) ─► 유해 언어
```

예상 모델 파일 합은 약 78MB다. 이 값은 실행 RAM이나 Android 런타임 크기를 포함하지
않으므로 최종 스마트폰에서는 모델 파일, 최대 RAM, RTF, 배터리를 각각 측정해야 한다.

## 왜 네 도메인을 분리하는가

노래 성능만 높아지고 일반 발화가 나빠지는 현상을 전체 평균 하나가 숨기지 못하게 한다.

| domain | 의미 | 학습 정답 |
|---|---|---|
| `general` | 뉴스, 낭독, 일상 대화, 전화 음질, 사투리 | 실제 발화 |
| `movie` | 배경음악·효과음·겹침이 있는 영화/드라마 대사 | 실제 대사 |
| `song` | 한국어 노래, 랩, 한영 혼합 가사 | 실제 가사 또는 검수된 teacher 출력 |
| `no_speech` | 반주 음악과 음향 효과만 있는 구간 | 빈 문자열 |

초기 학습 샘플링 비율은 일반 50%, 영화 25%, 노래 15%, 무발화 10%다. 이것은 데이터의
저장 비율이 아니라 학습 배치에서 뽑히는 비율이다. 일반 발화를 절반 유지해 도메인 특화로
인한 망각을 줄인다.

## manifest

예시는 `configs/asr/mobile_asr.example.jsonl`에 있다. 실제 실험 파일은
`data/manifests/mobile_asr.jsonl`로 만든다.

### 공개 데이터 소규모 실험 세트

첫 모델 비교용 115개 세트는 다음처럼 만든다. 원본과 생성 WAV는 모두 Git에서
제외된 `data_dl/mobile_asr/` 아래에 남는다.

```bash
uv run python scripts/fetch_csd_subset.py
PYTHONPATH=src uv run python scripts/build_mobile_asr_smallset.py
PYTHONPATH=src uv run python scripts/validate_mobile_asr_manifest.py \
  data_dl/mobile_asr/smallset/mobile_asr_small.jsonl
```

구성은 일반 한국어 발화 50개, Zeroth 발화와 ESC-50 효과음을 0/5/10 dB로 섞은
영화형 25개, CSD 한국어 노래 20개, 비음성 효과음 20개다. 학습/검증/시험은
각각 69/23/23개이며, Zeroth 화자·CSD 곡·ESC-50 원본이 분할 사이에서 겹치지
않는다. `review.csv`는 사람이 음원과 정답을 대조하기 위한 표다.

이 세트의 역할은 학습 완료가 아니라 후보 모델을 동일 조건에서 빠르게 거르는
것이다. CSD는 한 명의 가수가 부른 동요이고 영화형은 합성 혼합이므로, 상업 영화,
반주가 있는 대중가요, 유해어 회상률은 다음 실제 자료 단계에서 별도로 검증해야 한다.

```json
{"id":"song-001","audio":"../clips/song-001.wav","text":"실제 가사","domain":"song","split":"train","source_id":"song-A","harm_terms":["유해어"]}
```

- `source_id`가 같은 화자 세션·영화·곡은 여러 split에 걸치면 안 된다.
- 사람이 확인한 `text`가 최우선이다.
- 정답이 없을 때만 큰 teacher의 `teacher_text`를 사용한다.
- `no_speech`의 `text`는 반드시 빈 문자열이다.
- 저작권이 있는 영화·음악 원본은 저장소나 공개 데이터셋으로 재배포하지 않는다.

검증:

```bash
PYTHONPATH=src uv run --group nlp --group asr python \
  scripts/validate_mobile_asr_manifest.py data/manifests/mobile_asr.jsonl
```

## 큰 teacher로 sequence-level 증류

teacher는 학습 데이터를 만드는 컴퓨터에서만 실행하고 스마트폰에는 넣지 않는다. 사람이
작성한 정답이 없는 행만 large-v3-turbo로 전사한 다음 일부를 직접 검수한다. 무발화 음악은
teacher가 환각한 문장을 학습하지 않도록 항상 사람이 지정한 빈 정답을 사용한다.

```bash
PYTHONPATH=src uv run --group nlp --group asr python \
  scripts/make_mobile_asr_teacher_labels.py \
  --input data/manifests/mobile_asr_unlabelled.jsonl \
  --output data/manifests/mobile_asr.jsonl \
  --model large-v3-turbo \
  --device cuda
```

## 기준선 평가

학습 전에 기본 Whisper-tiny를 같은 test split에 저장해 둔다. 네 도메인의 CER, 무발화
허위 전사율, 유해 단어 보존율, RTF를 모두 기록한다.

```bash
PYTHONPATH=src uv run --group nlp --group asr python scripts/eval_mobile_asr.py \
  --manifest data/manifests/mobile_asr.jsonl \
  --model openai/whisper-tiny \
  --backend transformers \
  --split test \
  --output data_dl/mobile_asr/baseline.json
```

## student 학습

학습 설정은 `configs/asr/mobile_whisper_tiny.yaml`에 고정한다. 체크포인트가 있으면
`resume: auto`로 가장 최근 지점부터 이어서 시작한다.

```bash
PYTHONPATH=src uv run --group nlp --group asr python scripts/train_mobile_asr.py
```

데이터 위치나 배치 크기는 Hydra override로 바꾼다.

```bash
PYTHONPATH=src uv run --group nlp --group asr python scripts/train_mobile_asr.py \
  manifest=/data/mobile_asr.jsonl train.batch_size=8 device=cuda
```

## 평가와 int8 변환

```bash
PYTHONPATH=src uv run --group nlp --group asr python scripts/eval_mobile_asr.py \
  --model data_dl/mobile_asr/whisper-tiny-ko-multidomain/final \
  --backend transformers \
  --output data_dl/mobile_asr/finetuned.json

PYTHONPATH=src uv run --group nlp --group asr python scripts/export_mobile_asr.py \
  --checkpoint data_dl/mobile_asr/whisper-tiny-ko-multidomain/final \
  --output artifacts/mobile_asr_whisper_tiny_int8

PYTHONPATH=src uv run --group nlp --group asr python scripts/eval_mobile_asr.py \
  --model artifacts/mobile_asr_whisper_tiny_int8 \
  --backend faster-whisper \
  --output data_dl/mobile_asr/int8.json
```

내보내기 스크립트는 ASR과 기존 CED·KoELECTRA 모델 파일의 합이 100MB를 넘으면 실패로
종료한다. CTranslate2 결과는 현재 Mac 앱의 정확도·크기 실험용이다. Android 최종판은 같은
student 체크포인트를 모바일 지원 런타임으로 변환한 뒤 실제 기기에서 다시 측정해야 한다.

## 초기 채택 기준

1. 일반 발화 CER가 기본 Whisper-tiny보다 절대 2%p 이상 나빠지지 않는다.
2. 영화와 노래 CER가 각각 개선된다. 한 영역의 개선으로 다른 영역의 실패를 상쇄하지 않는다.
3. 유해 단어 보존율이 개선된다.
4. `no_speech` 허위 전사율이 증가하지 않는다.
5. CED·ASR·KoELECTRA 모델 파일 합이 100MB 이하다.
6. 목표 스마트폰에서 RTF가 1 이하고 최대 RAM·발열이 허용 범위다.

위 조건을 만족하지 못하면 모델을 채택하지 않고 데이터 구성, teacher 정답 검수율, 도메인
샘플링 비율을 바꿔 다시 실험한다.
