# 6. 파일 지도 — 어떤 파일이 뭘 하나

레포를 처음 열면 파일이 150개쯤 보입니다. 전부 읽을 필요는 없습니다.
**어떤 순서로 읽으면 되는지**부터 정리하고, 그다음에 전체 목록을 폴더별로 설명합니다.

---

> 이 문서는 **어디에 무엇이 있는지**를 알려주는 지도입니다.
> 각 파일 안의 코드를 줄 단위로 읽으려면 → **[07-code.md](07-code.md)**

## 처음 읽을 5개 파일

이 다섯 개만 읽으면 "소리가 들어와서 판단이 나오기까지"가 다 보입니다.

| 순서 | 파일 | 왜 이걸 먼저 |
|---|---|---|
| 1 | `src/preprocess/logmel.py` | 소리가 어떻게 숫자(사진 같은 것)로 바뀌는지 |
| 2 | `src/models/harm_model.py` | 모델 전체 조립도. 90줄이라 한 번에 읽힙니다 |
| 3 | `src/models/pooling.py` | 10초 중 **어디를 봤는지** 정하는 부분 (MIL 어텐션) |
| 4 | `src/training/trainer.py` | 학습 루프. 이 프로젝트 코드의 심장 |
| 5 | `.autorun/train_ced_vio.py` | 위 부품들을 실제로 조립해 채택 모델을 만드는 스크립트 |

---

## 폴더 구조 한눈에

```
src/         재사용되는 부품 (라이브러리) — "무엇을 만들 수 있나" (src/cascade 판정, src/app 실시간 앱 포함)
.autorun/    그 부품으로 채택 모델을 실제로 만든 스크립트 — "무엇을 만들었나"
distill/     1층 게이트 만들기 (증류)
scripts/     데이터 조립, 텍스트 분류기, 통계 검정 도구
configs/     설정 파일 — 라벨 체계, 위험도 정책, 학습 하이퍼파라미터
tools/       사람이 눈으로 보는 도구 (뷰어, 라벨링 UI)
tests/       단위 테스트 333개
docs/        지금 읽고 있는 문서들
data/        매니페스트 (실제 오디오는 fetch_data.sh로 data_dl/에 받습니다)
artifacts/   학습 산출물 (텍스트 분류기 가중치 등)
```

**`src/`와 `.autorun/`의 관계가 핵심입니다.**
`src/`는 부품 상자이고, `.autorun/`은 그 부품으로 만든 완성품입니다.
같은 부품으로 BEATs 버전도, CED 버전도 만들 수 있게 나눠놨습니다.

---

## `src/` — 재사용되는 부품

### `src/preprocess/` — 소리를 숫자로

| 파일 | 하는 일 |
|---|---|
| `audio.py` | 파일을 16kHz 모노로 디코딩(ffmpeg 경유) + 너무 조용한 클립 걸러내기 |
| `logmel.py` | STFT(창 1024, 홉 320) → 멜 128밴드(50–8000Hz) → 로그. **소리 → 사진** |
| `normalize.py` | 멜 밴드별 평균/표준편차. **학습셋에서만** 계산해 파일로 저장 |
| `pipeline.py` | 위 셋을 이어붙인 한 클립 처리 전체 |
| `precompute.py` | 전 클립을 미리 계산해 `.npy`로 저장 (학습 때마다 다시 안 하려고) |
| `config.py` | 위 숫자들의 단일 출처. `configs/data/preprocess.yaml`과 짝 |
| `paths.py` | 특징 파일 경로 규칙. 쓰는 쪽과 읽는 쪽이 어긋나지 않도록 분리 |

### `src/datasets/` — 데이터를 모델에 먹이는 부분

| 파일 | 하는 일 |
|---|---|
| `taxonomy.py` | 클래스 목록과 순서. `configs/data/classes*.yaml`에서 읽음 |
| `manifest.py` | 클립 한 줄 = JSON 한 줄. 라벨·출처·시간 정보 |
| `splits.py` | **`source_id` 단위 분할** — 같은 영상 조각이 학습/평가에 흩어지지 않게 |
| `dataset.py` | `.npy` 로드 + 정규화 + 멀티핫 라벨 → 파이토치 Dataset |
| `sampler.py` | 배치마다 유해:무해를 대략 1:1로 강제 (양성이 드물어서) |
| `augment.py` | 데이터 증강. 앞쪽은 로그-멜 위(SpecAugment 계열), 뒤쪽은 파형 위(게인·시프트·노이즈·mixup) |

> 증강은 **부품으로만 있고, 채택 경로에서는 꺼져 있습니다.**
> `scripts/train_beats_finetune.py`의 `AUGMENT` 환경변수 기본값이 `0`입니다
> (mixup이 총성 같은 짧은 소리를 뭉갠다는 이유 — [03장](03-training.md) ⑥).
> 켜보려면 `AUGMENT=1`로 돌리면 됩니다.

### `src/models/` — 모델 구조

| 파일 | 하는 일 |
|---|---|
| `harm_model.py` | **전체 조립도**: 로그-멜 → 백본 → 어텐션 → 머리 → 확률 |
| `pooling.py` | MIL 어텐션. 프레임마다 가중치를 매겨 평균 → **어디를 들었는지**가 남음 |
| `heads.py` | 분류 머리(클래스 수만큼의 **로짓** — 확률로 바꾸는 sigmoid는 손실/추론 쪽에서) + 투영 머리(대조학습용 임베딩) |
| `backbones.py` | 가벼운 백본 고르는 곳 (`conv`, `mfcc_bilstm`). `"beats"`는 여기서 모델이 아니라 **미리 뽑아둔 BEATs 특징을 그대로 통과**시키는 통로입니다 |
| `beats/` | BEATs 원본 코드 (Microsoft, MIT 라이선스, 그대로 가져옴) |
| `beats_extractor.py` | BEATs를 **얼려서** 특징 추출기로만 쓰는 경로 |
| `beats_finetune.py` | BEATs를 **학습 가능하게** 여는 경로 (위쪽 몇 블록만) |

### `src/losses/` — 무엇을 최소화할 것인가

| 파일 | 하는 일 |
|---|---|
| `focal.py` | focal-BCE (γ=2). 쉬운 샘플의 비중을 낮춰 드문 양성에 집중 |
| `supcon.py` | 멀티라벨 대조학습. 라벨이 Jaccard 0.5 이상 겹치면 "같은 편" |
| `combined.py` | `focal + 0.2 × SupCon`. 실제로 쓰는 최종 목적함수 |

### `src/training/` — 학습 루프

| 파일 | 하는 일 |
|---|---|
| `trainer.py` | **학습 루프 본체.** 기울기 누적, AMP, 커리큘럼(머리만 → 전체), 조기종료 |
| `optim.py` | AdamW + 머리/백본 학습률 분리(1e-4 / 1e-5) |
| `checkpoint.py` | 저장·재시작. 모델+옵티마이저+에폭+난수 상태까지 담아야 이어집니다 |
| `metrics.py` | AP·AUROC·recall@FPR를 numpy만으로 구현 (sklearn 의존 제거) |
| `config.py` | 학습 하이퍼파라미터. `configs/train/train.yaml`과 짝 |

### `src/risk/` — 확률을 "위험도"로

| 파일 | 하는 일 |
|---|---|
| `policy.py` | 클래스별 가중치·문턱·등급 정의. **모델 코드와 완전히 분리** |
| `scorer.py` | 클래스별 확률 → 위험도 하나 (가중 최대 + 가중 합) |
| `stream.py` | 시간에 따라 위험도를 부드럽게 추적하고 등급을 올리고 내림 |
| `fit.py` | 위 산출기의 계수를 검증셋에서 맞춰 파일로 저장 |

> ⚠️ `risk/`는 아직 **옛 9클래스 체계**(성적 3 + 폭력 4 + 도박 2, `configs/risk_policy/default.yaml`)를
> 그대로 씁니다. 지금 소리 갈래가 쓰는 폭력 4종(`classes_vio.yaml`)과 맞춰져 있지 않습니다 — 남은 정리 과제입니다.

### `src/text/` — 말 갈래

| 파일 | 하는 일 |
|---|---|
| `asr.py` | 음성 → 텍스트 (Whisper 경유) |
| `harm_learned.py` | **학습된 텍스트 분류기** (e5 임베딩 + MLP). 온디바이스 KoELECTRA의 선생 |
| `harm_text.py` | 어휘 목록 기반 채점. 정확하지만 목록에 없는 표현은 못 잡음 |
| `harm_semantic.py` | 의미 기반. 목록에 없어도 잡히지만 오탐이 늘어남 |
| `fuzzy_lexicon.py` | 받아쓰기가 자모 1–2개 틀렸을 때 복구 |
| `harm_toxicity.py` | 공개 혐오표현 탐지 모델 붙이기 |
| `harm_combined.py` | 위 신호들을 하나의 판정으로 합침 |

> 말 갈래는 역사가 층층이 쌓여 있습니다. **어휘 → 의미 → 학습된 분류기** 순으로
> 발전했고, 지금 주력은 `harm_learned.py`입니다. 앞의 것들은 비교 기준으로 남아 있습니다.

### `src/mining/` — 틀린 것만 골라 다시 배우기

모델이 헷갈린 클립만 사람에게 보여주고, 확인된 오답을 학습셋에 되먹이는 순환입니다.

`candidates.py`(헷갈린 클립 고르는 규칙) → `run.py`(예측을 훑어 **리뷰 대기열 생성**) →
`review.py`(사람 판정 받기) → `hnm.py`(판정 결과를 학습셋에 되먹이기).
설정은 `config.py` + `configs/mining/hnm.yaml`.

### `src/collect/` — 데이터 모으기

| 파일 | 하는 일 |
|---|---|
| `audioset.py` | AudioSet 라벨(mid)을 우리 클래스에 매핑해 10초 클립 매니페스트 생성 |
| `download.py` | yt-dlp로 유튜브에서 받아 ffmpeg으로 10초·16kHz 모노 wav로 자름 (삭제된 영상은 실패로 집계) |

### `src/cascade/` — 3층을 하나로 묶는 판정

| 파일 | 하는 일 |
|---|---|
| `decision.py` | **순수 판정 로직**. 게이트가 막으면 트리거는 아예 계산하지 않고, 말 갈래는 소리 게이트와 무관하게 동작 — 실제 기기 동작을 그대로 코드로. 모델을 import하지 않아서 테스트가 쉽습니다 |
| `pipeline.py` | 모델 배선: 트리거(CED-mini int8) · 텍스트(KoELECTRA int8 + 은어 어휘목록 = `HybridTextScorer`) · 받아쓰기(Moonshine 기본, whisper 선택). ASR 선정 근거 표가 주석에 있음. int8 엔진을 CPU 종류에 맞게 고름(애플 실리콘=qnnpack) |
| `server_stub.py` | 서버로 보낼 요청 형식을 고정 (실제 판정은 `src/app/judge.py`) |

임계값은 코드에 박지 않고 `artifacts/cascade_thresholds.json`에 저장합니다
(val로 맞춘 값 + 어떻게 맞췄는지 이력까지).

### `src/app/` — 실제로 돌아가는 앱

지금 노트북이 **재생 중인 소리**를 실시간으로 감시합니다. 마이크는 쓰지 않습니다.
돌려보는 방법은 [04-run.md](04-run.md#실제-앱으로-돌려보기--지금-재생-중인-소리를-감시-srcapp)에 있습니다.

| 파일 | 하는 일 |
|---|---|
| `sources.py` | 재생음 캡처: macOS(Core Audio taps) · 리눅스(PipeWire) · BlackHole 장치 · 파일. `python -m app.sources`로 캡처만 따로 진단 가능 |
| `engine.py` | 10초 창(2초씩 이동)으로 잘라 캐스케이드에 넣음. 링버퍼라 오래 켜둬도 메모리가 안 늘어남. 연속 발화 창들을 **한 사건(HarmEvent)으로 묶어** 서버에 한 번만 보냄 |
| `language_gate.py` | 선택형 한국어 라우터. DeepFilterNet 사전 강화 → 원본/강화 Silero VAD 합집합 → 원본 발화 Wiener 강화 → Whisper-tiny 한국어 LID. 한국어 누락을 줄이기 위해 애매한 판정과 오류는 기본적으로 통과 |
| `vad.py` | 받아쓰기 **환각 반복 탐지**(실출력으로 검증됨, 사용 중). 스펙트럼 말소리 게이트도 있지만 **기각 상태** — 실제 영화 대사를 0.00으로 매겨 ASR을 막았음. 기각 이유가 파일 맨 위에 기록돼 있음 |
| `escalate.py` | 사건을 서버로 전송(백그라운드 큐, 실패해도 로컬에 wav+jsonl 보관). `--upload-audio`면 파형까지 |
| `dashboard.py` | 브라우저 실시간 화면 (위험도·점수·받아쓰기·**사건 목록**) |
| `judge.py` | **3층 판정.** Qwen2.5-Omni-7B를 4-bit로 로드해 오디오+전사를 듣고 정도(%)·범주·근거 JSON을 냄. 파싱 실패 시 추측하지 않고 degree=None |
| `server.py` | 리눅스 서버 쪽 수신기. `--judge qwen-omni`면 위 판정기로 실제 판정, 없으면 기록만 |
| `main.py` | `python -m app.main` 진입점 |

### 실행 진입점

| 파일 | 명령 |
|---|---|
| `src/app/main.py` | `python -m app.main` (실시간 앱) |
| `src/app/server.py` | `python -m app.server` (서버 수신기) |
| `src/train.py` | `python -m train` |
| `src/evaluate.py` | `python -m evaluate` |
| `src/infer_stream.py` | `python -m infer_stream` |

---

## `.autorun/` — 채택 모델을 실제로 만든 스크립트

`src/`의 부품을 조립해 **논문에 쓸 숫자를 만든** 스크립트들입니다.
대부분 `data_dl/` 경로를 기본값으로 갖고 있어 인자 없이 돌아갑니다
(예외: `train_koelectra.py`는 오디오를 안 쓰고 HuggingFace 텍스트 데이터로 학습합니다).

| 파일 | 하는 일 | 결과 |
|---|---|---|
| `train_ced_vio.py` | CED-mini 폭력 파인튜닝 | **채택 트리거** |
| `quantize_ced.py` | 위 모델 int8 양자화 | **채택 아티팩트 10MB** |
| `train_koelectra.py` | KoELECTRA-small 텍스트 분류기 (`SLANG=1 ASR_REAL=1 ASR_AUG=1`이 채택 조합) | **채택 분류기** (fp32로 저장) |
| `eval_text_asr_noise.py` | 받아쓰기 오류를 넣어가며 텍스트 분류기 평가 | [02장](02-models.md) CER 표 |
| `asr_cer_eval.py` | Moonshine / Zipformer / Whisper 글자 오류율 측정 | Moonshine 채택 근거 |
| `train_beats_vio.py` | BEATs 폭력 파인튜닝 | 비교 기준 + 증류 선생 |
| `dump_beats_probs.py` | BEATs의 테스트셋 예측을 `.npz`로 저장 | 비교용 입력 |
| `compare_vio.py` | **심판.** 여러 모델을 부트스트랩 신뢰구간으로 비교 | [02장](02-models.md) 비교표 |
| `calibrate.py` | 온도 보정 — 모델이 말하는 "%"를 믿어도 되는지 (대상은 BEATs) | 보정 온도 T + ECE |
| `cascade_offline.py` | val로 임계값을 맞추고 test에서 캐스케이드 평가 | **`artifacts/cascade_thresholds.json`** (앱이 쓰는 동작점) |
| `eval_e2e_text.py` | 한국어 문장 → TTS → 노이즈 → **Moonshine 실출력** → 텍스트 분류기 | 말 갈래 실검증 ([05장](05-limits.md)) |
| `eval_profanity_slang.py` | 욕설·은어 40행이 ASR과 분류기를 **각각** 통과하는지 측정 | 은어가 약점임을 처음 밝힌 실험 |
| `compare_asr_harm.py` | ASR 후보를 CER이 아니라 **유해 단어 생존율**로 비교 | ASR 선정 근거 (CER로 뽑은 base-ko 기각) |
| `make_asr_corrupted_corpus.py` | 은어 문장 → TTS → 노이즈 → 실ASR → **진짜 오인식 학습 데이터** | `configs/text/asr_corrupted_corpus.jsonl` |

`compare_vio.py`가 가장 볼 만합니다. **"더 좋아 보인다"와 "더 좋다"를 가르는 코드**입니다.

---

## `distill/` — 1층 상시 게이트 만들기

| 파일 | 하는 일 |
|---|---|
| `dump_teacher_targets.py` | 선생(BEATs)이 전 클립에 매긴 확률을 저장 |
| `student_models.py` | 학생 CNN 정의. **채널 폭 튜플 하나**가 크기를 결정합니다 |
| `train_distill.py` | 학생이 선생을 따라 하도록 학습 (soft + cosine + hard). 세 가지 크기 후보(`_PRESET`의 s1/s2/s3 = 0.32M / 0.94M / 2.9M)도 여기 있습니다 |

---

## `scripts/` — 데이터·텍스트·통계 도구

| 파일 | 하는 일 |
|---|---|
| `fetch_data.sh` | GitHub Release에서 데이터 번들 받아 `data_dl/`에 풂 |
| `combined_data.py` | **매니페스트 → 학습/검증/평가 분할을 결정적으로 생성.** 재현성의 핵심 |
| `train_beats_finetune.py` | 원시 오디오 Dataset + 학습 유틸. 다른 스크립트들이 가져다 씀 |
| `train_text_head.py` | e5 + MLP 텍스트 분류기 학습 (KoELECTRA의 선생) |
| `gen_train_corpus.py` | 텍스트 학습 말뭉치 생성 |
| `gen_slang_corpus.py` | 은어 학습 말뭉치 생성 — 용어 4개는 **일부러 빼서** 일반화를 측정 |
| `gen_language_testset.py` | 텍스트 평가셋 생성 (한국어/영어) |
| `eval_bootstrap.py` | 부트스트랩 신뢰구간 + 대응 표본 유의성 검정 |
| `calibrate_threshold.py` | 실제 데이터에서 결정 문턱 보정 |

> ⚠️ `combined_data.py`와 `train_beats_finetune.py`는 이름만 보면 옛날 파일 같지만
> **소리 갈래 스크립트가 전부 이 둘을 import**합니다 — `train_ced_vio.py`, `quantize_ced.py`,
> `train_beats_vio.py`, `dump_beats_probs.py`, `calibrate.py`, `asr_cer_eval.py`,
> `distill/` 둘 다. 지우면 소리 파이프라인이 통째로 깨집니다.
> (텍스트 쪽 `train_koelectra.py` / `eval_text_asr_noise.py`와 `compare_vio.py`는 무관합니다.)

---

## `configs/` — 코드에 박지 않는 숫자들

| 파일 | 내용 |
|---|---|
| `data/classes_vio.yaml` | **지금 쓰는 라벨 체계** — 폭력 4종, 나머지는 전부 네거티브 샘플 |
| `data/classes.yaml` | 예전 23클래스 체계 (도박이 소리 갈래에 있던 시절) |
| `data/preprocess.yaml` | 로그-멜 설정 (샘플레이트, 홉, 멜 밴드 수 ...) |
| `train/train.yaml` | 학습 설정 (배치, 에폭, 학습률 ...) |
| `train/loss.yaml` | 손실 가중치 (focal γ, SupCon μ) |
| `model/harm_model.yaml` | 모델 구조 설정 |
| `data/audioset_labels.yaml` | AudioSet 527클래스 중 우리가 쓸 것들의 매핑 |
| `data/audioset_violence_topup.yaml` | 그중 폭력만 추려 더 모을 때 쓴 매핑 |
| `data/*_seed_*.yaml` | 수집한 유튜브 영상 목록 (ID·라벨·길이·검색어) — **데이터의 출처 기록** |
| `mining/hnm.yaml` | 하드 네거티브 마이닝 문턱·리뷰 예산·중단 조건 |
| `risk_policy/default.yaml` | 위험도 가중치·등급 문턱 |
| `text/harm_lexicon.yaml` | 한국어 유해 표현 목록 |
| `text/harm_prototypes.yaml` | 의미 기반 판정(`harm_semantic.py`)이 쓰는 기준 문장들 |
| `text/*.jsonl` | 텍스트 학습/평가 데이터 |
| `text/recording_key.json` | 실제 녹음 41문장의 정답 스크립트 |

한 가지 주의: **설정 파일이 전부 실제로 읽히는 건 아닙니다.**
`classes*.yaml`, `risk_policy/`, `mining/`, `text/`는 코드가 읽어서 씁니다.
반면 `preprocess.yaml` / `harm_model.yaml` / `loss.yaml` / `train.yaml` 네 개는 아직
**코드 안 dataclass 기본값의 사본**이고, hydra 연결은 남은 일입니다.
그래서 둘이 어긋나지 않는지 `tests/test_config_drift.py`가 감시합니다.

---

## `tools/` — 사람이 보는 도구

| 파일 | 하는 일 |
|---|---|
| `predict_app.py` | 클립을 올리면 확률·위험도·로그-멜·**어텐션**을 그려주는 뷰어 |
| `predict.py` | 명령줄에서 파일 하나 판정 |
| `annotator/app.py` | 헷갈린 클립을 사람이 판정하는 라벨링 UI |

`predict_app.py`는 현재 옛 BEATs 체크포인트 경로가 박혀 있습니다 — [04장](04-run.md) 참고.

---

## `tests/` — 333개 단위 테스트

`src/`의 폴더 구조를 그대로 따라갑니다 (`tests/preprocess/`, `tests/models/`, ...).

테스트를 **읽는 것도 공부가 됩니다.** "이 함수가 무엇을 보장해야 하는가"가
가장 짧게 적혀 있는 곳이기 때문입니다. 예를 들어:

- `tests/datasets/test_splits.py` — 같은 영상이 학습/평가에 갈라지지 않는지
- `tests/training/test_checkpoint.py` — 중단 후 재시작이 정말 이어지는지
- `tests/training/test_metrics.py` — 직접 만든 AP·recall@FPR이 손으로 계산한 값과 맞는지
- `tests/integration/test_pipeline_e2e.py` — 소리 넣어서 판정 나올 때까지 전 과정
- `tests/collect/test_download.py` — 영상이 삭제됐을 때 수집이 안 죽고 실패로 집계되는지

```bash
uv run pytest -q                          # 전부
uv run pytest tests/risk -v               # 한 폴더만
uv run pytest tests/training/test_metrics.py -v
```

---

## 여기 없는 것들

`main` 브랜치에는 **지금 쓰는 것만** 있습니다.
기각된 실험(BEATs 층 축소, PANNs 베이스라인 등), 연구 일지, 강의 자료는
`process` 브랜치에 그대로 있습니다.

```bash
git show process:process.md            # 연구 일지만 잠깐 보기 (브랜치 전환 없음)
git worktree add ../danger_process process   # 옆 폴더에 통째로 펼쳐 보기
```

> `git switch process`로 갈아타도 되지만, 지금 폴더에 받아둔 데이터(`data_dl/`)와
> 체크포인트가 있는 상태라면 **worktree로 옆에 펼치는 쪽이 안전합니다.**
> 다 봤으면 `git worktree remove ../danger_process`.
