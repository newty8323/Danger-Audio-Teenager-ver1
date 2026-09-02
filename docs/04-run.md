# 4. 직접 돌려보기
혹시나 싶어서 남겨둡니다만 굉장히 연산량이 많은 task입니다.\
**노트북에서 돌리지 마세요**\
무조건 **colab**이나, **GPU가 있는 컴퓨터**에서 돌리길 바랍니다.\
아마 E-room에 좋은 컴퓨터가 있을텐데, 사용 가능하다면 그걸 쓰던가 하는게 좋을거에요.\
이영준 선생님께 colab pro+를 여러분께 지원하는 것이 좋겠다고 말씀드린 상태이니, colab은 조금 기다려주길 바랍니다.

이 문서의 명령은 **전부 실제로 실행해서 확인한 것**입니다.
안 되는 것은 안 된다고 적어뒀습니다.

## 준비물

| 도구 | 왜 |
|---|---|
| [uv](https://docs.astral.sh/uv/) | 파이썬 3.11 환경과 의존성 관리 |
| `ffmpeg` | 오디오 디코딩 (PATH에 있어야 합니다) |
| [`gh`](https://cli.github.com/) | 데이터 번들이 GitHub Release에 있어서 필요 |

```bash
uv sync                        # 기본 의존성 + 테스트 도구
uv run pytest -q               # 332개 통과(1개는 건너뜀)하면 환경이 제대로 잡힌 것
```

> ⚠️ **`uv sync`만으로는 `.autorun/` 스크립트가 돌지 않습니다.**
> 무거운 라이브러리는 선택 그룹으로 빼놨습니다. 필요한 것만 받으세요.
>
> ```bash
> uv sync --group nlp            # transformers, sentence-transformers, pandas → 텍스트 갈래
> uv sync --group asr            # faster-whisper, jiwer, soundfile → 받아쓰기 평가
> uv sync --group annotator      # streamlit → 뷰어·라벨링 UI
> ```
>
> 그리고 그냥 `uv sync`를 다시 돌리면 **이미 깔린 선택 그룹이 지워집니다.**
> 한 번에 여러 그룹을 쓰려면 `uv sync --group nlp --group asr`처럼 같이 적으세요.

> ⚠️ 명령 앞에 **항상 `uv run`을 붙이세요.** 이 프로젝트는 `src/`를 패키지 루트로 설치하는데,
> 그 설정은 uv가 만든 가상환경 안에만 있습니다. 시스템 `python`으로는 `python -m train`도
> `.autorun/*.py`도 import 에러가 납니다.

## 데이터 받기

데이터는 이 레포의 GitHub Release에 있습니다. 레포에 초대돼 있어야 받을 수 있습니다.
**디스크 여유 ~11GB** 필요합니다 (5.4GB 다운로드 + 압축 해제본).

```bash
gh auth login                  # 처음 한 번만
bash scripts/fetch_data.sh     # data-v1 다운로드 + 압축 해제
```

풀리는 위치는 `data/`가 아니라 **`data_dl/`** 입니다. 스크립트들이 이 경로를 기본값으로 갖고 있습니다.

```
data_dl/release_download/  다운로드 원본 (~5.4GB) — 압축 푼 뒤엔 지워도 됩니다
data_dl/clips/             10초 원본 클립 8,648개
data_dl/features/          미리 계산해둔 로그-멜 .npy 8,409개
data_dl/manifests/         라벨 매니페스트 (violence_v2.jsonl, gambling.jsonl)
data_dl/artifacts/         평가 결과(probs_*.npz), 정규화 통계, 보정값
data_dl/asr/               ASR CER 측정 결과 + 청취 샘플
data_dl/weights/           BEATs 백본 (파인튜닝 입력)
ckpt_ced_mini_vio/         채택된 폭력 트리거 체크포인트
artifacts/koelectra_small_harm_asraug_slang/  채택된 텍스트 분류기 (은어+실ASR오류 학습)
```

`data_dl/release_download/`는 자동으로 지워지지 않습니다. 11GB 중 절반이 여기 있으니
압축 해제가 끝났으면 지우세요.

> ⚠️ 원본 클립은 YouTube·AudioSet에서 온 것입니다. **이 레포 밖으로 재배포하지 마세요.**
> 연구·수업 용도로만 씁니다.

---

## 바로 되는 것 — 채택 모델 재현

```bash
uv run python .autorun/compare_vio.py
```

**여기서 시작하는 걸 추천합니다.** 학습 없이 이 프로젝트의 핵심 결과를 직접 확인할 수 있습니다.
이미 계산된 예측(`probs_*.npz`)만 읽어서 부트스트랩으로 비교하므로 **GPU가 필요 없습니다.**
[02장](02-models.md)의 CED vs BEATs 표를 신뢰구간까지 다시 만들어 봅니다.

- 반드시 **레포 최상위 폴더에서** 실행하세요 (스크립트가 상대경로를 씁니다).
- 부트스트랩을 수천 번 돌리므로 **수십 초** 걸립니다.
- `data_dl/artifacts/`에 `probs_*.npz`가 없으면 안내 없이 죽습니다. `fetch_data.sh`를 먼저 돌리세요.

## 학습까지 돌려보기 (GPU 필요, 각각 몇 시간)

### 2층 소리 갈래 — 채택 트리거

```bash
uv run python .autorun/train_ced_vio.py     # → ckpt_ced_mini_vio/
uv run python .autorun/quantize_ced.py      # → int8 (CPU에서 실행됩니다)
```

`train_ced_vio.py`는 `EPOCHS`, `SEED` 환경변수를 읽습니다
(예: `EPOCHS=5 SEED=1 uv run python .autorun/train_ced_vio.py`).
스크립트마다 읽는 환경변수가 다르니 각 파일 맨 위 docstring을 보세요 —
`quantize_ced.py`는 `CKPT`/`LIMIT`, `asr_cer_eval.py`는 `N_CLEAN`/`SNRS` 같은 식입니다.

### 2층 말 갈래

```bash
SLANG=1 ASR_REAL=1 ASR_AUG=1 uv run python .autorun/train_koelectra.py
#                     → artifacts/koelectra_small_harm_asraug_slang/  (채택 모델)
uv run python .autorun/eval_text_asr_noise.py
uv run python .autorun/asr_cer_eval.py                 # 한국어 ASR 글자 오류율 측정
uv run --group nlp --group asr python .autorun/eval_profanity_slang.py   # 욕설·은어 측정
```

> ⚠️ **환경변수 세 개를 다 켜야 채택 모델이 나옵니다.** `ASR_AUG`는 자모 코럽션 증강,
> `SLANG`은 은어 코퍼스, `ASR_REAL`은 **진짜 받아쓰기 오류**(TTS→노이즈→Moonshine 실출력,
> `configs/text/asr_corrupted_corpus.jsonl`에 이미 들어 있음)입니다.
> 셋 다 빼면 깨끗한 문장에만 강한 모델([02장](02-models.md) 표의 맨 윗줄)이 나옵니다.

### 1층 게이트 증류

```bash
uv run python distill/dump_teacher_targets.py
uv run python distill/train_distill.py
```

> ⚠️ **이 둘은 데이터 번들만으로는 돌지 않습니다.**
> `dump_teacher_targets.py`가 선생 모델 체크포인트(`ckpt_beats_finetune_top4/best.ckpt`)를
> 찾는데, 그 파일은 번들에 없습니다. BEATs를 먼저 직접 학습시켜야 합니다
> (`.autorun/train_beats_vio.py`).

---

## 범용 파이프라인 직접 돌려보기

`src/` 아래의 일반 파이프라인은 경로를 **직접 지정**해야 합니다.
백본이 가벼운 `conv`라서 GPU 없이도 돌아갑니다. 구조를 익히는 용도로 좋습니다.

### 1) 소리 → 로그-멜 특징 만들기

```bash
uv run python -m preprocess.precompute \
    --manifest data_dl/manifests/violence_v2.jsonl \
    --audio-root data_dl/clips \
    --feature-root data/features \
    --stats-path artifacts/norm.npz \
    --output-manifest data/manifests/violence_v2.features.jsonl
```

### 2) 학습

```bash
uv run python -m train \
    --manifest data/manifests/violence_v2.features.jsonl \
    --feature-root data/features --stats artifacts/norm.npz \
    --resume auto
```

`--resume auto`는 마지막 체크포인트에서 이어서 시작합니다. 세션이 끊겨도 처음부터 안 합니다.

> ⚠️ **여기에 `--classes configs/data/classes_vio.yaml`을 붙이면 실패합니다.**
> 매니페스트에는 폭력 4종 말고도 혼동음 라벨(`door`, `clap`, `coin` …)이 들어 있는데,
> `classes_vio.yaml`에는 그 이름들이 없어서 매니페스트 검증이 "모르는 라벨"이라고 거부합니다.
> 붙이지 않으면 기본 23클래스 체계로 정상 동작합니다.
> 채택 스크립트(`.autorun/`)는 매니페스트에서 폭력 라벨만 걸러내는 함수를 따로 두어 이 문제를 피합니다.

> 백본은 `src/models/harm_model.py`의 `ModelConfig` 기본값(`conv`)으로 정해집니다.
> BEATs/CED로 바꾸려면 위의 `.autorun/` 스크립트를 쓰세요.

### 3) 평가

```bash
uv run python -m evaluate \
    --manifest data/manifests/violence_v2.features.jsonl \
    --feature-root data/features --stats artifacts/norm.npz \
    --ckpt artifacts/checkpoints/best.ckpt --split test \
    --out artifacts/eval.json
```

클래스별 AP/AUROC/오경보율별 recall, 매크로 mAP, 지연시간을 냅니다.

### 4) 스트리밍 위험도

먼저 위험도 산출기를 val로 맞춘 뒤, 오디오 파일을 흘려보냅니다.

```bash
uv run python -m risk.fit \
    --manifest data/manifests/violence_v2.features.jsonl \
    --feature-root data/features --stats artifacts/norm.npz \
    --ckpt artifacts/checkpoints/best.ckpt --split val --out artifacts/risk.json

uv run python -m infer_stream --audio clip.wav \
    --ckpt artifacts/checkpoints/best.ckpt --stats artifacts/norm.npz \
    --risk-params artifacts/risk.json --out artifacts/stream.jsonl
```

---

## 실제 앱으로 돌려보기 — 지금 재생 중인 소리를 감시 (`src/app/`)

여기까지는 "파일을 넣고 결과를 본다"였는데, 이건 **노트북이 지금 재생하는 소리를 실시간으로**
3층 캐스케이드에 흘려보내는 앱입니다. **마이크는 쓰지 않습니다** — 기기가 재생하는 소리만 봅니다
(그래야 학습 도메인과 일치하고, 배터리도 훨씬 덜 씁니다. 이유는 [1. 개요](01-overview.md) 참고).

학습이 필요 없어서 **GPU 없이 노트북에서 그냥 됩니다.** 모델만 받으면 됩니다(약 0.2GB):

```bash
bash scripts/fetch_data.sh --models     # 전체 5.4GB 대신 채택 체크포인트만
uv sync --group nlp
```

**맥북 (macOS 14.2 이상)** — 재생음 캡처 도구를 한 번 빌드해야 합니다:

```bash
git clone https://github.com/makeusabrew/audiotee.git && cd audiotee
swift build -c release
sudo cp .build/release/audiotee /usr/local/bin/
codesign --force --sign - /usr/local/bin/audiotee   # 서명이 없으면 권한 창이 아예 안 뜹니다
cd -                                                # 다시 레포 폴더로

uv run --group nlp python -m app.main
```

첫 윈도에서 macOS가 **오디오 녹음 권한**을 물어보면 허용하세요.
이 권한은 **마이크 권한과 별개 항목**이라, 마이크는 계속 꺼진 상태입니다.
그 다음 브라우저로 <http://127.0.0.1:8765> 를 열면 실시간 대시보드가 보입니다.

**리눅스**는 빌드할 게 없습니다(PipeWire 사용):

```bash
uv run --group nlp python -m app.main --source pipewire
```

**캡처 없이 파일로 흉내내기**(무엇이든 오디오 파일 하나로 시연):

```bash
uv run --group nlp python -m app.main --source file --file 아무거나.wav --realtime
```

### 3층까지: 서버가 실제로 "얼마나 유해한지" 판정합니다

의심 구간을 서버(리눅스, GPU 필요)로 넘기려면, 서버 쪽에서:

```bash
uv sync --group nlp --group server        # 4-bit 로딩용 (bitsandbytes 등)
uv run --group nlp --group server python -m app.server --host 0.0.0.0 --port 8770 \
    --judge qwen-omni
```

노트북 쪽은 `--server`에 **`--upload-audio`를 같이** 붙입니다:

```bash
uv run --group nlp python -m app.main --server http://<서버IP>:8770/ --upload-audio
```

`--upload-audio`가 없으면 서버는 전사 텍스트만 받아서 판정이 약해집니다.
**이 플래그가 캡처한 소리가 내 컴퓨터를 떠나는 순간**이라 일부러 기본값이 아닙니다.
(`--judge`를 빼면 예전처럼 기록만 하는 모드라 GPU 없는 컴퓨터에서도 시연이 됩니다.)

실측 (RTX 3060 12GB): Qwen2.5-Omni-7B가 4-bit로 **VRAM 6.3GB**, 로드 21초, **판정 6초/건**.
욕설 영화 장면 → **80% / abuse** + 한국어 근거 한 문장, 평범한 문장 → **0% / none**.

측정된 동작(리눅스, 실제 스피커 재생을 캡처):
소리 갈래는 **10초 창당 17~21ms**(CPU), 받아쓰기까지 도는 창은 100~300ms.
폭력 구간에서 에스컬레이션이 뜨고, 겹치는 창들은 **한 사건으로 묶여 한 번만** 서버로 갑니다.

> ⚠️ 판정 "품질"은 아직 위 2건으로만 확인했습니다. 정도(%)가 사람 판단과 얼마나
> 일치하는지는 라벨된 평가셋으로 재봐야 합니다 — [05장](05-limits.md)의 남은 일입니다.

설치 옵션·플래그·한계는 [`src/app/README.md`](../src/app/README.md)에 더 자세히 적어뒀습니다.

> ⚠️ 창이 10초라서 **장면이 시작된 뒤 최대 10초쯤 뒤에** 반응합니다.
> 임계값을 10초 창에 맞춰 정했기 때문인데, 더 빠르게 하려면 창을 줄이고
> 임계값을 다시 맞춰야 합니다(별도 실험).

---

## 클립 하나 눈으로 보기 (뷰어)

```bash
uv sync --group annotator
uv run streamlit run tools/predict_app.py
```

오디오를 올리면 클래스별 확률, 위험 등급, 로그-멜 그림,
그리고 **모델이 어느 시점에 주목했는지**(어텐션)를 함께 보여줍니다.

> ⚠️ 이 뷰어는 예전 BEATs 체크포인트 경로(`artifacts/ckpt_beats_v2/best.ckpt`)가
> 코드에 박혀 있고, 그 파일은 데이터 번들에 들어 있지 않습니다.
> 그대로 실행하면 안내 메시지만 뜹니다. 쓰려면 위 3)·4)로 체크포인트와
> `risk.json`을 직접 만든 뒤 `tools/predict_app.py:40-41`의 경로를 바꿔야 합니다.
> (채택 모델인 CED-mini를 물리는 건 아직 안 돼 있습니다 — 남은 숙제입니다.)

---

## 레포 구조

파일 하나하나가 뭘 하는지는 → **[06-files.md](06-files.md)** (읽는 순서까지 정리해뒀습니다)
코드를 줄 단위로 읽으려면 → **[07-code.md](07-code.md)**

```
src/         재사용되는 부품 (라이브러리) — src/cascade 판정, src/app 실시간 앱
.autorun/    그 부품으로 채택 모델을 만든 스크립트 (data_dl/ 기본 경로)
distill/     1층 게이트 증류
scripts/     데이터 조립, 텍스트 분류기, 통계 검정
configs/     라벨 체계·위험도 정책·학습 설정
tools/       뷰어 · 라벨링 UI
tests/       단위 테스트 333개 (332 통과 + 1 건너뜀)
```

## 규칙 두 가지

1. **설정과 시드가 실험을 정의합니다.** 같은 config + 같은 시드 = 같은 결과여야 합니다.
2. **정규화 통계·임계값·클래스 가중치는 코드에 박지 않습니다.** 전부 파일로 저장하고 버전을 붙입니다.
   숫자를 코드에 박으면 나중에 그 숫자가 어디서 왔는지 아무도 모르게 됩니다.
