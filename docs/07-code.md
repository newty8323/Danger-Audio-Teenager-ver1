# 7. 코드 상세 해설 — 파일 하나하나 읽기

이 문서는 [06장 파일 지도](06-files.md)의 확대판입니다.
지도가 "어디에 무엇이 있는지"라면, 이 문서는 **"그 안에 실제로 무슨 코드가 있는지"** 입니다.

레포에 있는 파이썬 파일 전부를 다룹니다. 길지만, 처음부터 끝까지 읽으라고 만든 문서는 아닙니다.
읽고 있는 파일이 있으면 여기서 그 파일을 찾아 옆에 놓고 보세요.

그리고 영상에서 이 내용을 굵게 다루긴 합니다. 영상을 잘 찍었다고 생각되진 않는데 영상을 보고 온다면 이해가 조금 더 쉬워질 것 같긴 합니다..\
내용이 되게 많습니다. 다 읽기보다 필요한 부분 발췌해서 읽는게 좋을 것 같아요.\
물론 다 읽고 모든 내용을 이해한다면 더 좋긴 합니다.

---

## 목차

1. [이 문서를 읽는 법](#이-문서를-읽는-법)
2. [읽기 전에 — 파이토치 최소한](#읽기-전에-파이토치-최소한)
3. [1. 데이터 장부 — 무엇을 유해로 볼 것인가를 코드로 적기](#1-데이터-장부-무엇을-유해로-볼-것인가를-코드로-적기)
4. [2. 소리가 텐서가 되기까지](#2-소리가-텐서가-되기까지)
5. [3. 모델 — 몸통, 주목, 머리](#3-모델-몸통-주목-머리)
6. [4. 무엇을 최소화할 것인가, 그리고 학습 루프](#4-무엇을-최소화할-것인가-그리고-학습-루프)
7. [5. 2층 트리거를 실제로 만든 스크립트](#5-2층-트리거를-실제로-만든-스크립트)
8. [6. 1층 상시 게이트 — 지식 증류](#6-1층-상시-게이트-지식-증류)
9. [7. 말 갈래 — 받아쓰고, 그 문장이 유해한지 판단하기](#7-말-갈래-받아쓰고-그-문장이-유해한지-판단하기)
10. [8. 확률을 판단으로 — 위험도, 심판, 그리고 사람이 개입하는 자리](#8-확률을-판단으로-위험도-심판-그리고-사람이-개입하는-자리)
11. [9. 명령줄 진입점과 나머지 도구들](#9-명령줄-진입점과-나머지-도구들)
12. [10. 합체 — 실시간 앱과 3층 판정](#10-합체-실시간-앱과-3층-판정)

---

## 이 문서를 읽는 법

[06장](06-files.md)이 **지도**라면, 이 문서는 그 지도를 따라 **직접 걷는 것**입니다.
파일 하나하나에 대해 세 가지를 적었습니다.

- **무엇을 하나** — 이 파일이 담당하는 일
- **왜 이렇게 짰나** — 다른 방법도 있었는데 왜 이걸 골랐는지
- **여기서 조심할 것** — 제가 실제로 틀렸거나, 틀리기 쉬운 지점

**코드 조각은 전부 실제 파일에서 그대로 가져온 것**입니다. 설명용으로 지어낸 코드는 없습니다.
각 조각 위에 `# 파일경로:줄번호`를 붙여뒀으니, 궁금하면 그 파일을 열어서 앞뒤를 더 보세요.
`...`는 중간을 생략했다는 뜻입니다.

순서는 파일 이름순이 아니라 **데이터가 흐르는 순서**입니다.

```
라벨 정하기 → 소리를 텐서로 → 모델 → 손실 → 학습 루프
   → 채택 모델 만들기 → 경량화 → 판정과 심판 → 합체(실시간 앱 + 3층)
```

한 번에 다 읽을 필요는 없습니다. 06장의 "처음 읽을 5개 파일"만 먼저 보고,
나머지는 필요할 때 이 문서에서 찾아 읽는 쪽을 권합니다.

---

## 읽기 전에 — 파이토치 최소한

딥러닝 강의에서 배운 개념이 코드에서 어떤 이름으로 나타나는지만 짚고 가겠습니다.
이것만 알면 나머지는 읽힙니다.

### ① 텐서 = 숫자 상자, 그리고 그 "모양"

파이토치의 모든 데이터는 **텐서**입니다. 리스트의 여러 겹 버전이라고 보면 됩니다.
중요한 건 값이 아니라 **모양(shape)** 입니다. 코드를 읽다 막히면
"지금 이 변수의 모양이 뭐지?"를 물으면 대개 풀립니다.

이 프로젝트에서 숫자 상자의 모양은 이렇게 변합니다.

```
10초 오디오 파일
  → (160000,)          16kHz × 10초 = 샘플 16만 개
  → (1, 128, 501)      로그-멜: 음높이 128칸 × 시간 501칸  ← "소리의 사진"
  → (B, 1, 128, 501)   배치 B개를 묶고 채널 1을 붙여 모델 입력으로
  → (B, T, D)          백본 통과: 시간 T칸마다 D차원 특징
  → (B, D)             어텐션 풀링: 시간축을 하나로 압축
  → (B, 4)             머리: 폭력 4종에 대한 점수
```

**마지막 `(B, 4)`가 답안지**입니다. 클립 하나당 숫자 4개.
이 4개가 각각 비명·타격·총성·욕설에 대한 점수입니다.

### ② `nn.Module` = 부품 하나

파이토치에서 모델 조각은 전부 `nn.Module`을 상속합니다. 규칙은 두 개뿐입니다.

- `__init__`에서 **필요한 재료(층)를 만들어 둡니다.**
- `forward`에서 **입력을 받아 출력을 만듭니다.**

`model(x)`라고 부르면 `forward(x)`가 실행됩니다.
이 프로젝트의 `HarmModel`도, 그 안의 어텐션 풀링도, 증류용 학생 CNN도 전부 이 틀입니다.

### ③ 학습 루프 = 다섯 줄

강의에서 배운 "예측 → 오차 → 미분 → 갱신"이 코드로는 이 다섯 줄입니다.

```python
out = model(x)              # 1. 예측
loss = criterion(out, y)    # 2. 얼마나 틀렸나
loss.backward()             # 3. 각 가중치를 어느 쪽으로 밀지 계산 (역전파)
optimizer.step()            # 4. 실제로 민다
optimizer.zero_grad()       # 5. 계산해둔 방향을 비운다 (안 비우면 계속 쌓임)
```

`src/training/trainer.py`의 학습 루프도 결국 이 다섯 줄이 중심입니다.
나머지 코드는 전부 **"끊겨도 이어지게", "메모리가 모자라도 돌게", "운 좋은 결과에 속지 않게"** 를 위한 장치입니다.
그 장치들이 실전 코드의 대부분을 차지한다는 것 자체가, 이 문서에서 배울 것 중 하나입니다.

### ④ 이 레포의 규칙 세 가지

코드를 읽다 보면 반복해서 마주칠 원칙입니다.

1. **숫자는 코드에 박지 않는다.** 클래스 목록, 위험도 가중치, 정규화 통계, 결정 문턱 —
   전부 파일로 빼서 저장합니다. 코드에 박으면 반년 뒤에 그 숫자가 어디서 왔는지 아무도 모릅니다.
2. **끊기는 건 정상이다.** 무료 GPU는 12시간이면 강제 종료됩니다.
   그래서 모든 학습 스크립트가 매 에폭 저장하고 `--resume auto`로 이어서 시작합니다.
3. **"좋아 보인다"와 "좋다"는 다르다.** 모델을 비교할 때는 반드시 신뢰구간을 붙입니다.
   이걸 담당하는 파일이 `.autorun/compare_vio.py`이고, 이 문서의 마지막에 나옵니다.

---

---

## 1. 데이터 장부 — 무엇을 유해로 볼 것인가를 코드로 적기

모델 코드를 보기 전에 저는 항상 데이터 쪽 파일을 먼저 봅니다. 신경망은 결국 "내가 준 정답표를 따라 하는 기계"라서, 정답표를 어떻게 정의했는지 모르면 모델 코드를 아무리 읽어도 그 모델이 무엇을 배웠는지 알 수 없기 때문입니다. 이 절의 다섯 파일은 순서대로 이렇게 답합니다. 유해를 어떤 이름으로 나눌 것인가(`classes_vio.yaml`) → 그 정의를 파이썬 객체로 바꾸기(`taxonomy.py`) → 클립 한 개를 한 줄로 적는 장부(`manifest.py`) → 장부를 train/val/test로 가르는 규칙(`splits.py`) → 전부 조립해 최종 학습 데이터를 만드는 지점(`combined_data.py`).

### `configs/data/classes_vio.yaml` — 클래스 목록이 왜 코드가 아니라 YAML에 있나

이 파일은 모델의 출력 노드가 무엇인지를 적은 목록입니다. 내용 자체는 매우 짧습니다.

```yaml
# configs/data/classes_vio.yaml:21-31
version: v2.0-vio

harm:
  vio:
    - vio_scream
    - vio_impact
    - vio_gunshot
    - vio_verbal

# Empty on purpose: non-violence clips are pure negatives, not explicit nodes.
confusable: []
```

폭력 계열 4개(비명 `vio_scream`, 타격음 `vio_impact`, 총성 `vio_gunshot`, 언어 폭력 `vio_verbal`)만 출력 노드입니다. 그리고 맨 위에 `version: v2.0-vio`가 붙어 있는데, 이게 이 파일의 핵심입니다.

클래스 목록을 파이썬 코드에 리스트로 박아두면 편하지만, 그러면 "이 체크포인트는 클래스가 몇 개였지?"를 나중에 알 수 없게 됩니다. 모델이 뱉는 확률 벡터의 3번 칸이 무슨 뜻인지는 오직 학습 당시의 클래스 순서로만 결정되는데, 코드를 한 번 수정하면 과거 순서가 사라지기 때문입니다. 그래서 클래스 목록은 **버전이 붙은 설정 파일**로 뺐고, 실제로 이 저장소에는 두 개의 taxonomy가 공존합니다.

옛날 것인 `configs/data/classes.yaml`이 23클래스짜리 v1.0입니다.

```yaml
# configs/data/classes.yaml:1-5
# Class taxonomy (spec §2). VERSIONED source of truth — changing this is a
# critical task (CLAUDE.md rule 1): output-node indices and risk weights depend
# on the exact order below ...
version: v1.0
```

v1.0에는 성적(`sex_*`) 3개, 폭력 4개, 도박(`gmb_*`) 2개에 더해 헷갈리기 쉬운 소리 14개(운동 중 신음 `exercise_grunt`, ASMR, 아기 울음 `baby_cry`, 박수 `clap`, 풍선 터짐 `balloon_pop` 등)가 들어 있습니다. 즉 "유해 세 종류를 한꺼번에 소리로 구분한다"는 초기 설계입니다. v2.0-vio가 왜 4개로 줄었는지는 파일 주석이 직접 설명합니다.

```yaml
# configs/data/classes_vio.yaml:3-11
# Rationale (user decision 2026-07-18): the on-device acoustic model is a pure
# VIOLENCE trigger. Non-violence harm is handled elsewhere in the cascade:
#   - gambling  -> TEXT branch (betting keywords / UI text; weak acoustic signal).
#                  gmb clips are kept in the data as PURE NEGATIVES (all-zero target),
#                  hardening the violence boundary (casino != violence).
#   - sexual    -> DEFERRED. Tags removed for now. Data collection is blocked by the
#                  ethics constraint (spec §6.4, no adult-content sourcing). ...
```

도박은 소리만으로 잡기엔 신호가 약합니다(슬롯머신 효과음과 아케이드 게임 효과음은 파형상 거의 같습니다). 그래서 도박 판정은 텍스트 분기로 옮겼습니다. 다만 모아둔 도박 클립을 버리지는 않고, **정답이 전부 0인 음성 예제(pure negative)** 로 그대로 씁니다. "카지노 소리는 폭력이 아니다"를 명시적으로 가르치면 폭력의 경계가 더 또렷해지기 때문입니다. 성적 콘텐츠는 데이터 수집 자체가 윤리 제약에 걸려 보류했고, 그 판단을 잊지 않으려고 주석으로 남겼습니다. `confusable: []`가 빈 리스트인 것도 같은 논리입니다.

```yaml
# configs/data/classes_vio.yaml:13-16
# Only the 4 violence classes are output nodes -> the backbone learns the easier
# "violence vs everything" representation ... confusable is intentionally EMPTY: every non-violence
# clip (former gmb + former confusables) becomes an all-zero negative example.
```

헷갈리는 소리를 별도 노드로 두지 않고 전부 "전부 0"으로 처리하면 문제가 "23개 중 어느 것인가"에서 "폭력이냐 아니냐"로 단순해집니다. 경량 모델 실험에서는 문제를 쉽게 만드는 게 곧 성능입니다. 조심할 것은 v1.0 파일을 **수정하면 안 된다**는 점입니다. 주석이 이유를 못 박아 뒀습니다.

```yaml
# configs/data/classes_vio.yaml:18-20
# v1.0 (configs/data/classes.yaml, 23 classes) is kept UNCHANGED so existing 23-class
# artifacts (probs_beats.npz etc.) stay interpretable. Changing taxonomy is a critical
# task (CLAUDE.md rule 1); this is a NEW versioned file, not an edit of v1.0.
```

과거에 저장한 23차원 확률 배열(`probs_beats.npz`)은 v1.0의 클래스 순서를 알아야만 해석됩니다. v1.0을 고치는 순간 그 파일들은 의미를 잃은 숫자 덩어리가 됩니다. taxonomy를 바꿀 때는 기존 파일을 수정하는 게 아니라 **새 버전 파일을 만듭니다.**

### `src/datasets/taxonomy.py` — YAML을 파이썬 객체로

YAML을 읽어 `Taxonomy` 객체 하나로 만들고, "이 클래스는 몇 번 노드인가", "이 라벨 리스트를 멀티핫 벡터로 바꿔줘" 같은 질문을 처리합니다. 모듈의 계약은 docstring에 그대로 적혀 있습니다.

```python
# src/datasets/taxonomy.py:3-5
Loaded from ``configs/data/classes.yaml`` (versioned source of truth). Output
node order is fixed: harm groups first (sex, vio, gmb), then confusables. Code
must never hardcode the class list — always go through :func:`load_taxonomy`.
```

앞 절의 원칙("클래스 목록을 하드코딩하지 마라")이 코드 주석으로 강제된 셈입니다. 구현의 핵심은 `frozen=True` 데이터클래스입니다.

```python
# src/datasets/taxonomy.py:22-33
@dataclass(frozen=True)
class Taxonomy:
    version: str
    harm_classes: tuple[str, ...]  # ordered, sex -> vio -> gmb
    confusable_classes: tuple[str, ...]
    # class name -> category ("sex" | "vio" | "gmb" for harm, "confusable" otherwise)
    categories: dict[str, str] = field(default_factory=dict)

    @property
    def all_classes(self) -> tuple[str, ...]:
        """Ordered class list; index == output-node index."""
        return self.harm_classes + self.confusable_classes
```

`frozen=True`는 "한 번 만들면 못 고친다"는 뜻이고 리스트 대신 튜플을 쓴 것도 같은 목적입니다. 학습 도중 누가 실수로 클래스 순서를 바꾸면 그 실험 전체가 조용히 오염되는데, 애초에 못 바꾸게 막아 둔 겁니다. 그리고 `all_classes`의 docstring 한 줄이 이 모듈에서 제일 중요합니다. **인덱스가 곧 출력 노드 번호**입니다. 모델 마지막 층이 내놓는 벡터의 0번 칸은 `all_classes[0]`입니다. 이 대응이 깨지면 모델은 멀쩡히 학습되는데 해석만 틀리는, 가장 찾기 어려운 버그가 생깁니다.

라벨을 벡터로 바꾸는 부분은 짧습니다.

```python
# src/datasets/taxonomy.py:68-73
    def encode(self, labels: list[str]) -> np.ndarray:
        """Multi-hot encode a label list into a (num_classes,) float32 vector."""
        vec = np.zeros(self.num_classes, dtype=np.float32)
        for label in labels:
            vec[self.index_of(label)] = 1.0
        return vec
```

0으로 채운 벡터에서 해당 라벨 자리만 1로 켭니다. 이게 **멀티핫(multi-hot)** 입니다. 강의에서 배우는 원핫은 1이 딱 하나지만, 여기서는 한 클립에 비명과 타격음이 동시에 있을 수 있어 1이 여러 개일 수 있습니다. 그리고 도박 클립처럼 라벨 리스트가 비면 벡터는 전부 0으로 남습니다 — 앞에서 말한 pure negative의 실제 구현이 이겁니다. 별도 코드가 필요 없습니다. YAML 로딩부에서는 마지막 검사를 눈여겨보세요.

```python
# src/datasets/taxonomy.py:94-95
    if len(all_names) != len(set(all_names)):
        raise ValueError("duplicate class names in taxonomy config")
```

같은 이름이 두 번 있으면 `index_of`가 앞쪽 인덱스만 돌려주므로, 학습은 되는데 뒤쪽 노드는 영원히 안 켜집니다. 그런 버그는 며칠씩 잡아먹기에 즉시 에러를 냅니다. 그리고 조심할 것은 기본 경로가 여전히 v1.0을 가리킨다는 점입니다.

```python
# src/datasets/taxonomy.py:16-17
# Repo-root-relative default config path.
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data" / "classes.yaml"
```

`load_taxonomy()`를 인자 없이 부르면 23클래스 v1.0이 로드됩니다. v2.0-vio를 쓰려면 경로를 명시적으로 넘겨야 합니다. 코드를 읽다 "왜 클래스가 23개지?" 싶으면 이 기본값을 먼저 의심하세요.

### `src/datasets/manifest.py` — 클립 한 개 = 한 줄

오디오 데이터셋의 "장부"를 정의합니다. 실제 오디오와 별개로, 각 클립이 어떤 원본의 몇 초 지점에서 잘렸고 라벨이 무엇인지를 적어 둔 목록이며 형식은 **jsonl**입니다.

```python
# src/datasets/manifest.py:1-5
"""Clip manifest schema and jsonl I/O (spec §3).

One JSON object per line. Fields mirror the spec manifest. Validation is against
the class taxonomy and the confidence/split enums so a malformed manifest fails
loudly at load time rather than silently mid-training.
"""
```

jsonl은 "JSON Lines"의 줄임말로, 파일 한 줄이 완결된 JSON 객체 하나입니다. 전체를 감싸는 대괄호가 없습니다. 이유는 실용적입니다. 파일이 몇 GB가 되어도 한 줄씩 읽으면 되고, 새 클립은 파일 끝에 한 줄만 덧붙이면 되며, `wc -l`로 개수를 바로 셀 수 있고, git diff도 사람이 읽을 수 있게 나옵니다. 큰 JSON 배열 하나였다면 한 줄만 바뀌어도 전체가 바뀐 것처럼 보였을 겁니다. 한 줄에 들어가는 필드는 이게 전부입니다.

```python
# src/datasets/manifest.py:21-34
@dataclass
class ClipRecord:
    clip_id: str
    source: str  # dataset/source name (audioset, fsd50k, youtube, in_the_wild, ...)
    source_id: str  # video/channel id — split disjointness is enforced on this
    start_sec: float
    duration: float
    labels: list[str]
    label_confidence: str  # one of CONFIDENCE_LEVELS
    split: str  # one of SPLITS
    annotator: str | None = None
    snr_est: float | None = None
    # Ambiguous clips are flagged and excluded from *training* (spec §3 weak-label rule).
    flagged: bool = False
```

`clip_id`는 고유 이름으로, 나중에 `data/beats_feats/{clip_id}.npy` 같은 특징 파일과 이 이름으로 연결됩니다. `source`는 출처 데이터셋(AudioSet, FSD50K, 유튜브 등), `source_id`는 **원본 영상/채널 ID**로 주석에 "split disjointness is enforced on this"라고 적혀 있습니다 — 다음 파일의 주인공입니다. `start_sec`/`duration`은 원본의 어느 구간을 잘랐는지라서, 이게 있으면 클립을 다시 만들 수 있습니다. `labels`는 앞의 `encode`에 그대로 들어가고, `label_confidence`는 `("verified", "pseudo", "weak")` 중 하나로 사람이 확인한 라벨과 모델이 붙인 라벨을 섞어 쓰되 구분해 둡니다. `flagged`는 애매한 클립 표시인데, 실제로 쓰는 곳은 여기 하나입니다.

```python
# src/datasets/manifest.py:124-128
def training_records(records: Iterable[ClipRecord]) -> Iterator[ClipRecord]:
    """Train-split records that are not flagged as ambiguous (spec §3)."""
    for r in records:
        if r.split == "train" and not r.flagged:
            yield r
```

애매한 클립을 지우지 않고 표시만 해 두는 게 포인트입니다. 지우면 왜 뺐는지 기록이 사라지고, 나중에 판단이 바뀌어도 되살릴 수 없습니다. 검증 함수는 값이 말이 되는지 하나씩 확인합니다.

```python
# src/datasets/manifest.py:60-73
    if not record.clip_id:
        err("empty clip_id")
    ...
    for label in record.labels:
        if label not in taxonomy.categories:
            err(f"unknown label {label!r}")
```

마지막 줄이 이 절 전체를 하나로 묶습니다. **taxonomy에 없는 라벨이 장부에 있으면 에러**입니다. YAML의 클래스 정의와 데이터 장부가 어긋날 수 없게 코드가 붙잡아 주는 겁니다.

조심할 것은 `from_json`이 모르는 필드를 조용히 버린다는 점입니다.

```python
# src/datasets/manifest.py:39-42
    @classmethod
    def from_json(cls, obj: dict) -> ClipRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in obj.items() if k in known})
```

스키마가 바뀌어도 옛 manifest를 읽을 수 있다는 장점이 있지만, 필드 이름에 오타를 내면 아무 경고 없이 사라집니다. 또 `load_manifest`는 에러 미리보기를 앞 10개만 보여주니, 에러가 200개여도 화면엔 10개만 나옵니다. 숫자를 꼭 보세요.

### `src/datasets/splits.py` — 이 절에서 제일 중요한 파일

클립들을 train/val/test로 나눕니다. 다만 무작위로 나누지 않습니다.

```python
# src/datasets/splits.py:1-11
"""Source-disjoint train/val/test splitting (spec §3).

Hard constraint: the same ``source_id`` (video/channel) never crosses splits,
so a model can't memorize a source in train and be scored on it in val/test.
Certain sources (in-the-wild broadcast clips) are test-only.
...
"""
```

데이터셋을 나눌 때 가장 흔한 방법은 클립 리스트를 섞어 앞 70%를 train, 다음 15%를 val로 쓰는 겁니다. 그리고 이 프로젝트에서 그렇게 하면 **결과가 통째로 거짓말이 됩니다.**

이유는 클립이 만들어진 방식에 있습니다. 유튜브 액션 영화 한 편에서 폭력 장면을 10초씩 잘라내면 클립 40개가 나오는데, 이 40개는 같은 마이크, 같은 배경음악, 같은 배우 목소리, 같은 인코딩 아티팩트를 공유합니다. 무작위로 섞으면 28개가 train, 6개가 test로 갑니다. 이제 모델이 test에서 높은 점수를 냅니다. 모델이 "비명의 음향적 특징"을 배운 걸까요? 아닐 수 있습니다. **"이 영상 특유의 배경음"을 외운 것**일 수 있습니다. train에서 그 배경음과 폭력이 같이 나오는 걸 28번 봤으니, test에서 같은 배경음이 들리면 폭력이라 찍으면 맞습니다. 이걸 데이터 누출(leakage)이라 합니다. 실험실 점수는 올라가고 실제 성능은 그대로인, 가장 위험한 자기기만입니다.

그래서 여기서는 개별 클립이 아니라 **`source_id`(원본 영상) 단위로 통째로** 나눕니다. 한 영상에서 나온 40개 클립은 전부 train이거나 전부 test입니다. 실제로 묶는 코드는 이렇습니다.

```python
# src/datasets/splits.py:58-77
    # Group clip counts by source_id, and remember which sources are test-only.
    counts: dict[str, int] = defaultdict(int)
    forced_test: set[str] = set()
    for r in records:
        counts[r.source_id] += 1
        if r.source in test_only:
            forced_test.add(r.source_id)

    assignment: dict[str, str] = {sid: "test" for sid in forced_test}

    assignable = {sid: n for sid, n in counts.items() if sid not in forced_test}
    total = sum(assignable.values())
    targets = dict(zip(SPLITS, (r * total for r in ratios), strict=True))
    ...
    # Deterministic order: largest groups first (better balance), seeded tie-break.
    rng = random.Random(seed)
    order = sorted(assignable, key=lambda sid: (-assignable[sid], rng.random(), sid))
```

`counts`는 영상별 클립 개수를 셉니다 — 나누는 단위가 클립이 아니라 영상임이 여기서 확정됩니다. 그리고 함수가 돌려주는 것도 클립별 배정이 아니라 `source_id -> split` 딕셔너리입니다. 클립 하나만 따로 다른 split으로 보내는 게 **불가능한 구조**로 짠 겁니다. `forced_test`는 특정 데이터셋 전체를 test 전용으로 묶는데, 기본값이 `DEFAULT_TEST_ONLY_SOURCES = frozenset({"in_the_wild"})`입니다. 실제 방송에서 딴 클립은 학습에 안 쓰고 평가에만 씁니다 — 훈련 분포와 가장 다른 데이터를 평가용으로 남기니, 스스로에게 어려운 시험을 내는 셈입니다. 배정 자체는 그리디입니다.

```python
# src/datasets/splits.py:79-86
    for sid in order:
        n = assignable[sid]
        # Assign to the split with the largest remaining deficit.
        split = max(SPLITS, key=lambda s: targets[s] - current[s])
        assignment[sid] = split
        current[split] += n

    return assignment
```

큰 영상부터 보면서 목표 대비 가장 모자란 split에 통째로 넣습니다. 큰 것부터 넣는 이유는 마지막에 큰 덩어리가 남으면 비율이 크게 어긋나기 때문입니다 — 큰 돌부터 항아리에 넣는 것과 같습니다. 정렬 키 `(-assignable[sid], rng.random(), sid)`는 클립 수 내림차순 → 시드 고정 난수 → 이름순의 3단계라서, 마지막 `sid` 덕분에 이 정렬은 **완전히 결정적**입니다. 같은 시드면 항상 같은 split이 나옵니다. 그리고 누출을 사후에 잡는 검사가 따로 있습니다.

```python
# src/datasets/splits.py:30-38
def assert_source_disjoint(records: Iterable[ClipRecord]) -> None:
    """Raise if any ``source_id`` appears in more than one split."""
    source_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        source_to_splits[r.source_id].add(r.split)
    leaks = {sid: sorted(s) for sid, s in source_to_splits.items() if len(s) > 1}
    if leaks:
        ...
        raise SplitError(f"{len(leaks)} source(s) span multiple splits: {preview}")
```


split을 만드는 코드와 split이 올바른지 검사하는 코드가 분리돼 있다는 게 중요합니다. 손으로 만진 manifest도 외부에서 받은 manifest도 이 함수 하나로 검증할 수 있습니다.

조심할 것은 docstring이 스스로 밝힌 한계입니다. "Full label stratification (keeping per-class balance across splits) is a future refinement; size-balancing is the honest current behavior." 지금은 **클립 개수만** 맞추지 클래스 비율은 안 맞춥니다. 총성 클립이 특정 영상 몇 개에만 몰려 있으면 그 영상들이 우연히 다 test로 가서 train에 총성이 거의 없어질 수도 있습니다. 코드가 이 약점을 숨기지 않고 주석으로 적어 둔 태도를 눈여겨보세요. 연구 코드에서 "지금 안 된 것"을 명시하는 건 되는 것을 자랑하는 것만큼 중요합니다.

### `scripts/combined_data.py` — 채택된 학습 스크립트가 전부 import하는 조립 지점

폭력 manifest와 도박 manifest를 읽어 최종 `(train, val, test)` 레코드 리스트를 만들고, 그것을 PyTorch가 먹을 수 있는 `Dataset`으로 감쌉니다. `scripts/` 아래에 있어 일회성 스크립트처럼 보이지만 **아닙니다.** 채택된 학습 스크립트들이 전부 이 파일을 import합니다. 프로젝트의 `CLAUDE.md`에도 "죽은 연구 코드처럼 보이지만 절대 지우지 말 것"이라고 적혀 있을 정도입니다.

```python
# scripts/combined_data.py:1-10
"""Deterministic builder for the violence+gambling combined frozen-BEATs dataset.

This is the *committed, reproducible* replacement for the earlier ad-hoc combined
run (whose split logic lived only in a scratch script). Given a fixed ``split_seed``
it reconstructs the exact train/val/test used to train ``ckpt_beats_combined``:
...
  - gambling: re-split per class *by source video* 70/15/15 with ``random.Random(seed)``
    so each gmb class has its own train/val/test (source-disjoint, no leakage).
"""
```

첫 문단이 이 파일의 존재 이유입니다. 원래 split 로직이 **커밋되지 않은 임시 스크립트에만 있었습니다.** 그러면 체크포인트가 남아 있어도 정확히 어떤 데이터로 학습됐는지 재현할 수 없습니다. 그 로직을 정식 파일로 옮겨 커밋한 게 이 파일입니다.

"같은 시드면 항상 같은 split"이 실제로 어떻게 보장되는지, 장치가 네 개입니다. 첫째, 도박 클립을 clip_id 기준으로 합칩니다.

```python
# scripts/combined_data.py:61-72
    # gambling: dedup to one multi-hot record per clip_id (union labels).
    gmb_raw = [r for r in read_manifest(GAMBLING) if exists_fn(r.clip_id)]
    by_clip: dict[str, object] = {}
    label_union: dict[str, list] = defaultdict(list)
    for r in gmb_raw:
        by_clip.setdefault(r.clip_id, r)
        for lbl in r.labels:
            if lbl not in label_union[r.clip_id]:
                label_union[r.clip_id].append(lbl)
    for cid, r in by_clip.items():
        r.labels = label_union[cid]
    gmb = list(by_clip.values())
```

같은 클립이 `gmb_machine`으로도 `gmb_table`로도 등록돼 있으면, 나누기 전에 라벨을 합쳐 **하나의 멀티핫 레코드**로 만듭니다. 안 하면 물리적으로 동일한 오디오가 두 줄로 존재해 한 줄은 train, 다른 줄은 val로 갈 수 있습니다. 실제로 그런 버그가 있었다고 docstring이 밝힙니다: "This closes a train<->val leak that the earlier per-(class,video) split had (a dual-class video landed in two splits)."

둘째, 시드 고정 난수와 정렬을 함께 씁니다.

```python
# scripts/combined_data.py:76-90
    classes_of_video: dict[str, set] = defaultdict(set)
    for r in gmb:
        classes_of_video[r.source_id].update(r.labels)
    rng = random.Random(split_seed)
    video_split: dict[str, str] = {}
    all_classes = sorted({c for cs in classes_of_video.values() for c in cs})
    for cls in all_classes:
        vids = sorted(v for v, cs in classes_of_video.items() if cls in cs)
        unassigned = [v for v in vids if v not in video_split]
        rng.shuffle(unassigned)
        n = len(unassigned)
        ntr = max(1, int(n * 0.7))
        nva = max(1, int(n * 0.15))
        for j, v in enumerate(unassigned):
            video_split[v] = "train" if j < ntr else ("val" if j < ntr + nva else "test")
```

여기서 `sorted()`가 두 번 나오는 게 결정성의 열쇠입니다. `random.Random(split_seed)`로 시드를 고정해도 셔플하기 **전의** 리스트 순서가 매번 다르면 결과는 매번 달라집니다. 파이썬 `set`의 순회 순서는 보장되지 않으므로, `all_classes`와 `vids` 둘 다 `sorted()`로 순서를 못 박은 다음 `rng.shuffle`을 부릅니다. 시드 고정 + 입력 순서 고정, 둘 다 있어야 재현됩니다. 시드만 고정하고 끝내는 게 초심자가 가장 자주 하는 실수입니다. `if v not in video_split`도 중요한데, 클래스를 하나씩 돌면서 이미 배정된 영상은 건너뜁니다. 한 영상이 두 클래스에 걸쳐 있어도 처음 배정된 split을 유지하므로 영상 단위 disjointness가 깨지지 않습니다.

셋째, 클립이 아니라 영상을 배정합니다. 결과가 `source_id -> split` 딕셔너리이고 클립의 split은 그걸 그대로 받아 적습니다 — `splits.py`와 정확히 같은 원칙입니다.

```python
# scripts/combined_data.py:91-92
    for r in gmb:
        r.split = video_split[r.source_id]
```

넷째, 마지막에 누출을 직접 확인합니다.

```python
# scripts/combined_data.py:99-103
    # guard: splits must be clip-disjoint (no leakage) — cheap and catches regressions.
    tr_ids, va_ids, te_ids = ({r.clip_id for r in s} for s in (tr, va, te))
    assert not (tr_ids & va_ids), f"train/val leak: {tr_ids & va_ids}"
    assert not (tr_ids & te_ids), f"train/test leak: {tr_ids & te_ids}"
    assert not (va_ids & te_ids), f"val/test leak: {va_ids & te_ids}"
```

집합 교집합 세 줄이면 끝나는 검사입니다. 앞에서 아무리 조심히 짰어도 나중에 누가 코드를 고치다 실수하면 여기서 즉시 터집니다. 이런 값싼 assert를 조립 지점에 박아두는 습관을 꼭 가져가세요. 시드를 학습 시드와 따로 두는 이유도 docstring이 설명합니다.

```python
# scripts/combined_data.py:12-14
... Keeping the split
seed separate from the training seed lets Phase 1 vary optimization/init noise while
holding the data split fixed — the standard N-seed protocol.
```

시드를 여러 개 바꿔가며 실험할 때 데이터 split까지 같이 바뀌면, 성능 차이가 초기화 운 때문인지 데이터 분할 운 때문인지 구분할 수 없습니다. **데이터 split은 고정하고 학습 시드만 흔든다** — 이게 표준 N-seed 프로토콜입니다. 실제 학습에 쓰이는 Dataset은 이렇게 짧습니다.

```python
# scripts/combined_data.py:107-120
class BeatsFeatureDataset(Dataset):
    """Serves (BEATs frame embeddings (T,768), multihot label) from disk."""

    def __init__(self, records, tax=None):
        self.records = records
        self.tax = tax or load_taxonomy()
    ...
    def __getitem__(self, i):
        r = self.records[i]
        x = torch.from_numpy(np.load(f"{FEAT}/{r.clip_id}.npy").astype(np.float32))
        return x, torch.from_numpy(r.multihot(self.tax))
```

`__getitem__`이 돌려주는 두 값이 곧 신경망의 입력과 정답입니다. 왼쪽은 미리 계산해 둔 BEATs 프레임 임베딩 `(T, 768)`, 오른쪽은 taxonomy가 만든 멀티핫 벡터입니다. YAML의 클래스 정의부터 시작한 이 절의 모든 파일이 결국 이 한 줄로 흘러 들어옵니다.

조심할 것은 `_has_feat` 필터입니다.

```python
# scripts/combined_data.py:42-43
def _has_feat(clip_id: str) -> bool:
    return os.path.exists(f"{FEAT}/{clip_id}.npy")
```

`data/beats_feats/`에 특징 파일이 없는 클립은 조용히 빠집니다. 특징 추출이 중간에 실패해도 학습은 그냥 돌아가고 대신 데이터가 줄어들 뿐이니, 결과가 이상하면 manifest 줄 수와 실제 학습에 들어간 레코드 수를 먼저 비교해 보세요. 또 `FEAT`, `VIOLENCE`, `GAMBLING`이 전부 상대 경로라 저장소 루트에서 실행해야 하고, `BeatsFeatureDataset`의 기본 taxonomy가 `load_taxonomy()` 즉 23클래스 v1.0이므로 v2.0-vio로 학습하려면 `tax` 인자를 반드시 넘겨야 합니다. 안 넘기면 라벨 벡터 차원이 조용히 23이 됩니다.

---

## 2. 소리가 텐서가 되기까지

모델에 들어가는 것은 "소리"가 아니라 숫자 덩어리, 즉 텐서입니다. mp3 파일 하나가 신경망이 먹을 수 있는 배열이 되기까지 어떤 단계를 밟는지를 저는 `src/preprocess/`와 `src/datasets/`에 나눠 두었습니다. 이 절에서는 아홉 개 파일을 순서대로 따라가면서 **텐서의 모양(shape)이 매 단계 어떻게 바뀌는지**를 계속 소리 내어 확인하겠습니다.

### 1. `src/preprocess/config.py` — 숫자를 한 군데에만 적는다

전처리에는 마법의 상수가 잔뜩 나옵니다. 16000, 1024, 320, 128… 이 숫자들이 파일마다 흩어져 있으면 하나만 고쳤을 때 나머지가 조용히 어긋나면서 "학습은 되는데 성능만 이상한" 최악의 버그가 생깁니다. 그래서 전부 한 곳에 모았습니다.

```python
# src/preprocess/config.py:12-33
@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 10.0
    ...
    # Loudness gate: drop clip if RMS < this (dBFS, full-scale ref = 1.0)
    rms_gate_dbfs: float = -45.0
    # STFT
    n_fft: int = 1024
    hop_length: int = 320
    ...
    # Mel filterbank
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 8000.0
    # log(mel + offset)
    log_offset: float = 1e-6
```

`frozen=True`는 "한 번 만들면 못 고치는 설정"이라는 뜻입니다. 코드 중간에서 실수로 `cfg.n_mels = 64`처럼 바꾸면 그 자리에서 에러가 납니다. 설정이 학습 도중 몰래 바뀌는 사고를 막는 안전장치입니다. 여기서 텐서 모양의 출발점이 정해집니다.

```python
# src/preprocess/config.py:35-42
    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def expected_frames(self) -> int:
        """Frame count for a full clip with center-padded STFT."""
        return self.clip_samples // self.hop_length + 1
```

숫자를 직접 넣어 봅시다. `clip_samples = 16000 × 10.0 = 160,000` — 10초 오디오는 정확히 **160,000개 샘플**, 모양은 `(160000,)`입니다. `expected_frames = 160000 // 320 + 1 = 500 + 1 = 501`. 320샘플(=20ms)씩 창을 밀며 스냅샷을 찍으니 500번 밀 수 있고, 주석의 "center-padded"가 말하는 앞쪽 프레임 하나를 더해 **501프레임**입니다. 조심할 것 하나 — docstring에 "Mirrored by `configs/data/preprocess.yaml` for hydra; keep the two in sync"라고 적혀 있습니다. 같은 값의 사본이 YAML에도 있으니 둘을 함께 고쳐야 합니다.

### 2. `src/preprocess/audio.py` — 어떤 파일이든 같은 파형으로

수집한 오디오는 wav, mp3, m4a, opus로 제각각이고 샘플레이트도 뒤죽박죽입니다. 파이썬 라이브러리로 처리하면 백엔드 설치 문제로 고생하기 쉬워서, 디코딩 전부를 외부 프로그램 `ffmpeg`에 맡겼습니다.

```python
# src/preprocess/audio.py:36-52
        "-i", path,
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "f32le",  # raw 32-bit float little-endian
    ...
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
```

`-ac 1`은 모노로 합치기, `-ar 16000`은 16kHz 리샘플, 명령 끝의 `-`는 "결과를 파일이 아니라 표준출력으로 뱉어라"입니다. 그 바이트 스트림을 `np.frombuffer`로 읽으면 `(N,)` 1차원 파형이 나오고, 값은 대략 −1에서 +1 사이입니다. 두 번째 역할은 소리 게이트입니다.

```python
# src/preprocess/audio.py:66-74
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


def passes_rms_gate(waveform: np.ndarray, threshold_dbfs: float = -45.0) -> bool:
    """True if the clip is loud enough to keep (RMS >= threshold)."""
    return rms_dbfs(waveform) >= threshold_dbfs
```

RMS는 파형을 제곱해 평균 내고 제곱근을 씌운 "평균적인 크기"입니다. dB로 바꿔 `-45.0` dBFS보다 조용하면 버립니다. 사실상 무음인 클립은 라벨이 무엇이든 그 근거가 되는 소리가 없고, 학습에 넣으면 모델이 "무음 = 폭력" 같은 규칙을 배우기 때문입니다. 조심할 것: 완전 무음이면 `log10(0)`이 되므로 `-inf`를 따로 반환합니다. 이 가드가 없으면 NaN이 조용히 퍼집니다.

### 3. `src/preprocess/logmel.py` — 파형이 그림이 되는 순간

이 파일이 이 절의 심장입니다. 1차원 파형을 **시간 × 주파수의 2차원 지도**로 바꿉니다. docstring이 전체 요약입니다: "STFT (n_fft=1024, hop=320, Hann) -> mel (128 bands, 50-8000 Hz) -> log(mel + 1e-6). Output shape (1, n_mels, T)."

```python
# src/preprocess/logmel.py:19-30
        self.mel = T.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            ...
            hop_length=self.cfg.hop_length,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
            power=2.0,
            center=True,
            ...
```

모양을 단계별로 따라가면,

1. 입력 `(160000,)` → 내부에서 채널 축을 붙여 `(1, 160000)`.
2. STFT: 1024샘플(64ms) 창을 320샘플씩 밀며 분석 → `(1, 513, 501)`. 주파수 축이 `n_fft/2 + 1 = 513`인 것은 실수 신호의 FFT가 절반만 독립이기 때문이고, 시간 축 501은 앞서 계산한 `expected_frames`와 정확히 같습니다.
3. 멜 필터뱅크: 513개 빈을 사람 귀 해상도에 맞춘 **128밴드**로 묶습니다. 50Hz 아래(저주파 잡음)와 8000Hz 위(16kHz의 물리적 상한)는 아예 보지 않습니다 → `(1, 128, 501)`.
4. 로그: 소리 세기는 몇 백만 배씩 차이 나므로 압축합니다.

```python
# src/preprocess/logmel.py:33-39
    @torch.no_grad()
    def __call__(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return log-mel of shape (1, n_mels, T) as float32 numpy."""
        wav = self._as_tensor(waveform)
        mel = self.mel(wav)  # (1, n_mels, T)
        logmel = torch.log(mel + self.cfg.log_offset)
        return logmel.squeeze(0).unsqueeze(0).to(torch.float32).cpu().numpy()
```

`log(mel + 1e-6)`의 `1e-6`이 왜 필요할까요. 무음 구간의 멜 값은 0이고 `log(0) = -inf`입니다. 아주 작은 수를 더해 두면 최악이어도 `log(1e-6) ≈ -13.8`에서 멈춥니다. 설정 파일의 `log_offset`이 그 값입니다. 조심할 것 둘. `@torch.no_grad()`는 전처리가 학습 대상이 아니니 기울기를 만들지 말라는 지시입니다. 그리고 `T.MelSpectrogram`을 `__init__`에서 한 번만 만드는 이유는 멜 필터뱅크 행렬 생성 비용이 크기 때문입니다 — 클립마다 새로 만들면 수만 개 파일을 도는 내내 낭비합니다.

### 4. `src/preprocess/normalize.py` — 여기서 실험이 망가진다

신경망은 입력이 0 근처에 고르게 퍼져 있을 때 잘 학습합니다. 그래서 멜 밴드마다 평균을 빼고 표준편차로 나눕니다. 문제는 **그 평균과 표준편차를 어떤 데이터로 재느냐**입니다. docstring에 답이 못 박혀 있습니다: "computed per mel bin over all time frames of the *train* split".

```python
# src/preprocess/normalize.py:62-72
        total += arr.sum(axis=1)
        total_sq += np.square(arr).sum(axis=1)
        count += arr.shape[1]
    ...
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(var)
```

수만 개 클립을 메모리에 다 올릴 수 없으니 합과 제곱합만 누적해 마지막에 `평균 = 합/개수`, `분산 = 제곱평균 − 평균제곱`으로 계산합니다. `np.maximum(..., 0.0)`은 부동소수점 오차로 분산이 음수가 되어 `sqrt`가 NaN을 뱉는 것을 막는 가드입니다.

**왜 반드시 train split만인가.** 전체 데이터(train + val + test)로 평균을 재면, 테스트 클립의 정보가 "평균값"이라는 형태로 학습 입력에 스며듭니다. 이걸 데이터 누수(leakage)라고 합니다. 모델은 실전에서 절대 볼 수 없는 미래 데이터의 통계를 미리 본 셈이 되고, 평가 점수는 실제보다 좋게 나옵니다. 그리고 이 오류는 **절대 에러를 내지 않습니다** — 조용히 점수만 부풀립니다. 그래서 통계 계산이 한 함수에 갇혀 있고, 그 함수에는 train 레코드만 흘려보내는 스트림만 들어갑니다.

계산한 통계는 코드에 하드코딩하지 않고 `np.savez`로 `.npz` 파일에 남깁니다(`normalize.py:31-36`). 버전이 붙는 파일이라 나중에 "이 체크포인트는 어떤 통계로 학습했지?"를 되짚을 수 있습니다. 적용은 브로드캐스팅 한 줄입니다.

```python
# src/preprocess/normalize.py:80-83
    if arr.ndim == 3:
        return (arr - mean[None, :, None]) / std[None, :, None]
    if arr.ndim == 2:
        return (arr - mean[:, None]) / std[:, None]
```

`(1, 128, 501)`에서 128 축만 맞춰 빼려면 `mean`을 `(1, 128, 1)`로 펼쳐야 하고, `None`이 그 축 추가 표기입니다. 모양은 그대로 `(1, 128, 501)`.

### 5. `src/preprocess/pipeline.py` — 클립 하나의 전체 여정

앞의 세 파일을 순서대로 엮은 것이 이 파일입니다. docstring 한 줄 요약: "raw file -> 16 kHz mono -> RMS gate -> fix length (10s) -> log-mel (1, 128, ~501)".

```python
# src/preprocess/pipeline.py:47-51
    waveform = load_audio(path, sample_rate=cfg.sample_rate, mono=cfg.mono)
    if not passes_rms_gate(waveform, cfg.rms_gate_dbfs):
        return None
    waveform = fix_length(waveform, cfg.clip_samples)
    return extractor(waveform)
```

너무 조용하면 예외 대신 `None`을 돌려줍니다. "에러가 아니라 정상적인 탈락"이라는 뜻이고, 부르는 쪽은 그 클립을 manifest에서 빼면 됩니다. `fix_length`가 왜 필요한지는 주석이 직접 설명합니다.

```python
# src/preprocess/pipeline.py:20-32
    """Pad (right, zeros) or truncate a waveform to exactly ``target_len`` samples.

    Real AudioSet segments can be shorter than the requested 10s (the source video
    ended early), which otherwise yields variable-T features that break batching
    and mixup. spec §4 defines a fixed 10s clip, so we enforce it here.
    """
    ...
    out[:n] = waveform
```

9.3초 클립이 섞이면 프레임 수가 501이 아니라 466이 되고, 그러면 여러 클립을 하나의 배치 텐서로 쌓을 수 없습니다. 짧으면 뒤에 0을 채우고 길면 자릅니다. 이 덕분에 이후 모든 텐서가 `(1, 128, 501)`로 고정됩니다. 그리고 눈여겨볼 설계가 하나 더 있습니다 — 정규화를 여기서 **하지 않습니다**. docstring이 이유를 밝힙니다: "features are stored unnormalized and normalized at load time via a saved `NormStats`". 정규화를 미리 구워 넣으면 통계를 바꿀 때마다 수만 개 `.npy`를 다시 만들어야 하지만, 나중에 적용하면 `.npz` 하나만 갈아 끼우면 됩니다.

### 6. `src/preprocess/precompute.py` + `paths.py` — 한 번 만들고 계속 재사용

로그멜 계산은 GPU 학습에 비하면 싸지만, 25 에폭을 돌면 같은 계산을 25번 반복하게 됩니다. 그래서 **한 번 계산해 `.npy`로 저장해 두고** 학습 때는 읽기만 합니다.

```python
# src/preprocess/precompute.py:74-93
    for r in records:
        src = path_resolver(r, audio_root)
        if not src.exists():
            result.missing_ids.append(r.clip_id)
            continue
        logmel = preprocess_clip(str(src), extractor, cfg)
        if logmel is None:  # RMS-gated (too quiet)
            result.dropped_ids.append(r.clip_id)
            continue
        np.save(feature_path(r.clip_id, feature_root), logmel)
        ...
    # Fit normalization stats on the training split only (spec §4).
    stats = _fit_train_stats(kept, feature_root)
    ...
    write_manifest(kept, output_manifest_path)
```

파일이 없으면 `missing`, 너무 조용하면 `dropped`, 통과하면 `.npy`로 저장하고 `kept`에 담습니다. 그리고 살아남은 것들로만 통계를 뽑고 **살아남은 것들만 담긴 새 manifest를 다시 씁니다**. 게이트에서 탈락한 클립이 manifest에 남아 있으면 학습 중에 없는 `.npy`를 읽으려다 죽기 때문입니다. 통계 스트림은 앞 절의 원칙을 코드로 강제합니다.

```python
# src/preprocess/precompute.py:51-52
    for r in training_records(kept):
        yield np.load(feature_path(r.clip_id, feature_root))
```

이 두 줄이 `_fit_train_stats`가 보는 전부입니다. `training_records`는 `manifest.py`에서 "split이 train이고 flagged가 아닌 것"만 걸러 내는 제너레이터입니다. val/test는 애초에 이 스트림에 들어올 수 없습니다. 경로 규칙은 두 줄짜리 함수인데 굳이 자기 모듈을 하나 차지합니다.

```python
# src/preprocess/paths.py:12-13
def feature_path(clip_id: str, feature_root: str | Path) -> Path:
    return Path(feature_root) / f"{clip_id}.npy"
```

이유는 docstring에 있습니다: "Kept dependency-free so both the writer (preprocess.precompute) and readers (datasets.dataset, mining, evaluate) agree on where a clip's `.npy` lives." **쓰는 쪽과 읽는 쪽이 반드시 같은 규칙을 써야 하기 때문**입니다. precompute는 `{clip_id}.npy`로 저장하는데 dataset은 `{clip_id}_logmel.npy`를 찾는다면, 에러 메시지는 "파일 없음"이지만 진짜 원인은 같은 규칙을 두 곳에 적었다는 것입니다. 한 함수에 가두면 그 실수 자체가 불가능해집니다.

### 7. `src/datasets/dataset.py` — `.npy` 한 개 + manifest 한 줄 = 샘플 한 개

PyTorch의 `Dataset`은 `__len__`과 `__getitem__` 두 개만 구현하면 되는 약속입니다. 여기서 i번째 샘플은 `(특징 텐서, 멀티핫 라벨)` 한 쌍입니다.

```python
# src/datasets/dataset.py:79-90
        logmel = np.load(feature_path(self.records[idx].clip_id, self.feature_root))
        logmel = logmel.astype(np.float32)
        if logmel.shape[-2] != self.norm_stats.n_mels:
            raise ValueError(...)
        if self.train and rng is not None:
            # Gain is an additive log-domain shift on the UNNORMALIZED log-mel
            ...
            logmel = augment.apply_gain(logmel, self.aug.gain_max_db, self.aug.gain_p, rng)
        return apply_norm(logmel, self.norm_stats)
```

manifest 한 줄에서 `clip_id`를 꺼내 → `feature_path`로 경로를 만들고 → `(1, 128, 501)` 배열을 읽고 → 밴드 수가 통계와 맞는지 확인하고 → 정규화합니다. 저 `ValueError`가 필요한 이유는, 128밴드 통계로 64밴드 특징을 정규화하면 브로드캐스팅이 실패하거나 더 나쁘게는 엉뚱하게 성공할 수 있기 때문입니다. 데이터셋 버전을 섞어 쓸 때 나는 전형적인 사고를 여기서 잡습니다. 라벨은 한 줄입니다 — `_label`(dataset.py:92-93)이 `self.records[idx].multihot(self.taxonomy)`를 돌려줍니다. `multihot`은 `["vio_gunshot", "vio_impact"]` 같은 문자열 리스트를 `(num_classes,)` 크기의 0/1 float32 벡터로 바꿉니다. 클래스마다 독립적으로 0 또는 1이니 **멀티라벨**입니다 — 한 클립이 폭력이면서 동시에 도박 소리일 수 있으므로 "하나만 고르기"인 softmax가 아닙니다. 학습/평가 모드 차이는 `__getitem__`에 그대로 드러납니다.

```python
# src/datasets/dataset.py:96-106
        if not self.train:
            feat = self._load_normalized(idx, rng=None)
            label = self._label(idx)
            return torch.from_numpy(feat), torch.from_numpy(label)

        rng = self._rng(idx)
        ...
        # mixup with a random partner (label union); exclude self to avoid a no-op mix
        if rng.random() < self.aug.mixup_p and len(self.records) > 1:
```

평가는 완전히 결정적입니다(같은 입력 → 항상 같은 출력). 그래야 어제 잰 점수와 오늘 잰 점수를 비교할 수 있습니다. 학습에만 난수가 등장하는데 그 난수조차 재현 가능합니다 — `_rng`(dataset.py:74-75)가 전역 난수 대신 `np.random.default_rng((self.seed, self._epoch, idx))`로 매번 새 생성기를 만들기 때문입니다. DataLoader가 워커를 여러 개 띄워도 몇 번째 워커가 집었든 결과가 같고, 실험을 다시 돌리면 똑같은 증강이 재현됩니다. 이제 **증강 기본값과 실제 학습 경로**를 봅시다. 이 파일의 기본 설정은 이렇습니다.

```python
# src/datasets/dataset.py:43-45
    # mixup (spec: Beta(.5,.5) label-union, p.5)
    mixup_alpha: float = 0.5
    mixup_p: float = 0.5
```

즉 `LogMelDataset`을 그냥 만들면 **mixup이 절반의 확률(`mixup_p = 0.5`)로 걸립니다.** 그런데 실제로 채택된 학습 스크립트는 증강을 통째로 끕니다.

```python
# scripts/train_beats_finetune.py:45-48
# Augmentation is OFF by default: the single-seed A/B (2026-07-18) showed it did not
# improve the target (violence recall@FPR1% regressed, gunshot -.184; mixup blurs sharp
# transients). Kept toggle-able (AUGMENT=1) for a later MUSAN/multi-seed revisit.
AUGMENT = os.environ.get("AUGMENT", "0") == "1"
```

```python
# scripts/train_beats_finetune.py:106-108
        wav, label = self._load(i)
        if not (self.train and AUGMENT):
            return torch.from_numpy(wav), torch.from_numpy(label)
```

환경변수 `AUGMENT`를 명시적으로 `1`로 주지 않는 한 증강은 건너뜁니다. 이유도 주석에 남아 있습니다 — A/B 실험에서 목표 지표가 좋아지지 않았고 mixup이 총성처럼 날카로운 소리를 뭉갠다는 것입니다. 코드를 지우지 않고 스위치로 남긴 것은 조건을 바꿔 다시 볼 여지를 두기 위해서입니다. **"파일에 기본값이 이렇게 적혀 있다"와 "실제 학습이 이걸 쓴다"는 다른 이야기**라는 점을, 남의 저장소를 읽을 때 항상 확인하세요.

### 8. `src/datasets/sampler.py` — 배치를 반반씩 섞는다

유해 소리는 전체에서 소수입니다. 무작위로 배치를 뽑으면 한 배치에 유해 클립이 한두 개뿐인 일이 흔하고, 그러면 모델은 "전부 안전"이라고만 답해도 손실이 낮아집니다. 그래서 배치를 직접 구성합니다.

```python
# src/datasets/sampler.py:100-107
        self.harm_indices = [i for i, r in enumerate(records) if _is_harm(r, taxonomy)]
        # "non-harm" = confusable and/or safe clips; both act as negatives (spec §5).
        self.other_indices = [i for i, r in enumerate(records) if not _is_harm(r, taxonomy)]
        ...
        self.n_harm_per_batch = batch_size // 2
        self.n_other_per_batch = batch_size - self.n_harm_per_batch
```

실제 동작을 정확히 말하면, "유해 라벨이 하나라도 있는가"로 두 무더기를 나누고 배치 크기의 **절반(정수 나눗셈)**을 유해 몫, 나머지를 비유해 몫으로 잡습니다. 배치가 짝수면 정확히 반반, 홀수면 — 예를 들어 9 — 유해 4 / 비유해 5로 비유해가 하나 더 많습니다. docstring이 "roughly 1:1"이라고 쓴 것이 이 때문입니다. 정확히 1:1이 아닙니다.

```python
# src/datasets/sampler.py:147-153
            if harm_empty:
                batch = other.take(self.batch_size)
            elif other_empty:
                batch = harm.take(self.batch_size)
            else:
                batch = harm.take(self.n_harm_per_batch) + other.take(self.n_other_per_batch)
            yield batch
```

한쪽이 비어 있으면 균형을 포기하고 나머지로 채웁니다. 데이터가 아직 한쪽뿐인 초기 단계에서도 크래시 없이 굴러가게 하려는 배려입니다. 에폭 길이는 `_num_batches`(sampler.py:127-134)가 정하는데, 주석 그대로 "Cover the larger pool once per epoch given its per-batch quota" — 각 무더기 크기를 자기 몫으로 나눈 값 중 큰 쪽을 올림합니다. 즉 **큰 쪽 무더기를 한 바퀴 다 도는 것**이 1 에폭입니다. 그동안 작은 쪽은 여러 바퀴 돌고, `_CyclicShuffler`가 한 바퀴 끝날 때마다 다시 섞습니다. 즉 소수 클래스는 한 에폭에 여러 번 등장합니다(오버샘플링). 배치 안 중복도 막는데, `take`의 docstring이 "a batch never gets the same clip twice — trivial SupCon positives"라고 밝힙니다(sampler.py:61-64). 같은 클립이 한 배치에 두 번 들어가면 대조 손실(SupCon)에서 "자기 자신과 자기 자신"이라는 공짜 정답 쌍이 생겨 학습 신호가 오염됩니다. 무더기가 배치 몫보다 작아 중복이 불가피하면 `warnings.warn`으로 크게 알립니다. 재현성 쪽 함정도 하나 있는데, 파이썬 `hash()`는 실행할 때마다 값이 달라지므로(`PYTHONHASHSEED`) 시드를 섞을 때 정수 연산만 씁니다(`_mix`, sampler.py:31-41).

### 9. `src/datasets/augment.py` — 한 파일에 반씩 다른 두 세계

이 파일은 **위아래로 성격이 완전히 다른 두 덩어리**입니다. 이걸 모르면 왜 gain이 두 개(`apply_gain`, `wav_gain`)나 있는지 이해가 안 됩니다. **위쪽 절반 — 로그멜(특징) 도메인.** 이미 계산된 `(1, 128, 501)` 위에서 동작합니다. SpecAugment, 시간 이동, 게인, mixup 네 가지입니다.

```python
# src/datasets/augment.py:23-33
# Waveform gain g (in dB) scales power by g**2, i.e. adds ln(g**2) to a
# natural-log log-mel: shift = (dB / 20) * ln(10) * 2 = dB * ln(10) / 10.
_GAIN_DB_TO_LN = math.log(10.0) / 10.0
...
    """Random gain in [-max_db, +max_db], applied as an additive log-domain shift.

    Must run on the *unnormalized* log-mel (models the raw loudness change).
    """
```

"볼륨을 키운다"는 파형에서는 곱셈이지만 로그 도메인에서는 덧셈이고, 그 환산 상수를 주석이 유도해 두었습니다. "unnormalized에서만 돌려야 한다"는 조건이 붙는 이유는 정규화 후에 상수를 더하면 그건 더 이상 볼륨 변화가 아니기 때문입니다. `dataset.py`가 게인 → 정규화 순서를 지킨 것이 이 조건 때문입니다. SpecAugment 쪽에도 같은 종류의 논리가 있습니다 — docstring이 "Applied *after* normalization, so `mask_value=0` masks to the per-bin mean"이라고 못 박습니다(augment.py:57-61). 정규화 **후**라서 0은 "무음"이 아니라 "그 밴드의 평균값"입니다. 순서가 뒤집히면 마스크의 의미가 달라집니다. mixup은 라벨 처리가 평범한 mixup과 다릅니다.

```python
# src/datasets/augment.py:96-98
    lam = float(rng.beta(alpha, alpha))
    feat = (lam * feat_a + (1.0 - lam) * feat_b).astype(np.float32)
    label = np.maximum(label_a, label_b).astype(np.float32)
```

특징은 비율대로 섞지만 라벨은 **합집합**(`np.maximum`)입니다. 총성과 도박 소리를 겹쳐 들으면 둘 다 들리는 것이지 "0.7만큼 총성"이 되지 않기 때문입니다. **아래쪽 절반 — 파형 도메인.** 파일 중간에 구분선과 함께 이렇게 명시돼 있습니다.

```python
# src/datasets/augment.py:103-108
# Waveform-domain augmentations (spec §4), for the raw-waveform BEATs fine-tune
# path. BEATs consumes a raw 16 kHz waveform and computes its own fbank inside
# the backbone, so these run on the waveform BEFORE the model. SpecAugment stays
# feature-domain (BEATs' internal fbank is not exposed) and is intentionally not
# reproduced here; MUSAN-noise mixing accepts an external noise clip when the
# corpus is available and otherwise falls back to Gaussian noise (a stand-in).
```

여기가 핵심입니다. **채택된 BEATs/CED 경로는 로그멜 `.npy`를 쓰지 않고 원본 파형 `(160000,)`을 그대로 모델에 넣습니다.** BEATs가 백본 안에서 스스로 fbank를 계산하기 때문입니다. 그래서 그 경로가 import 하는 것은 아래쪽 절반이고, 위쪽 로그멜 함수들은 쓰지 않습니다. 실제 호출부가 그대로 보여 줍니다.

```python
# scripts/train_beats_finetune.py:117-120
            wav, label = augment.wav_mixup(wav, label, wav_b, label_b, a.mixup_alpha, rng)
        wav = augment.wav_gain(wav, a.gain_max_db, a.gain_p, rng)
        wav = augment.wav_time_shift(wav, a.time_shift_max_sec, self.sr, a.time_shift_p, rng)
        wav = augment.wav_add_noise(wav, None, (a.noise_snr_min, a.noise_snr_max), a.noise_p, rng)
```

`wav_add_noise`의 두 번째 인자가 `None`인 것도 보세요. 주석대로 MUSAN 잡음 코퍼스가 아직 없어 가우시안 잡음으로 대체된다는 뜻입니다. 그리고 이 네 줄 전체가 `AUGMENT=1`일 때만 실행됩니다 — 그 스위치는 5장에서 다시 봅니다.

---

한 줄로 정리하면,

```
mp3/wav → ffmpeg → (160000,) float32
        → RMS 게이트(-45 dBFS 미만 탈락) → fix_length → (160000,)
        → STFT(n_fft=1024, hop=320) → (1, 513, 501)
        → 멜 128밴드(50-8000Hz) → (1, 128, 501)
        → log(x + 1e-6) → .npy로 캐시
        → (train 통계로) 정규화 → [증강] → 배치로 쌓기 → 모델
```

읽으면서 계속 물어야 할 질문은 둘뿐입니다. **"지금 이 배열의 모양이 뭐지?"**, 그리고 **"이 숫자는 train 데이터만 보고 나온 건가?"** 이 둘을 놓치지 않으면 전처리 코드에서 크게 헤맬 일은 없습니다.

---

## 3. 모델 — 몸통, 주목, 머리

여기서부터가 실제로 학습되는 부분입니다. 모델은 세 덩어리입니다. **몸통(backbone)**은 스펙트로그램을 시간 축 위의 특징 벡터 열로 바꾸고, **주목(MIL attention pooling)**은 그 시간 축을 하나로 접으면서 "어느 순간이 중요한지"를 스스로 정하고, **머리(head)**는 접힌 벡터에서 최종 점수를 뽑습니다. 조립도가 `harm_model.py`, 나머지는 부품입니다.
(그림으로 먼저 보고 싶으면 → [02장의 모델 구조도](02-models.md#모델-하나의-속은-어떻게-생겼나))

### `src/models/harm_model.py` — 조립도

**무엇을 하나.** 부품을 순서대로 이어 붙이고, 텐서 하나를 넣으면 dict 하나가 나오게 합니다. 파일 맨 위 docstring이 설계도 그 자체입니다.

```python
# src/models/harm_model.py:1-8
"""Harm-detection model assembly (spec §5).

    log-mel -> backbone -> frame embeddings -> MIL attention -> z
             -> classifier head (logits)  +  projection head (SupCon embedding)

``forward`` returns logits, attention weights (temporal localization), the
pooled clip embedding, and — unless disabled — the projection embedding used by
the contrastive loss (dropped at inference).
"""
```

하이퍼파라미터는 함수 인자로 흩뿌리지 않고 얼려둔 데이터클래스 하나에 모읍니다. `frozen=True`는 "한 번 만들면 못 고친다"는 뜻으로, 학습 중에 설정이 슬쩍 바뀌는 사고를 막습니다.

```python
# src/models/harm_model.py:23-30
@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "conv"
    backbone_out_dim: int = 256
    attn_dim: int = 128
    ...
    proj_dim: int = 256
```

```python
# src/models/harm_model.py:41-46
        dim = self.backbone.out_dim
        self.pool = MILAttentionPooling(dim, self.cfg.attn_dim)
        self.classifier = ClassifierHead(
            dim, num_classes, self.cfg.classifier_hidden, self.cfg.dropout
        )
        self.projection = ProjectionHead(dim, self.cfg.proj_dim)
```

**왜 이렇게 짰나.** 핵심은 `dim = self.backbone.out_dim`입니다. 뒤쪽 부품의 입력 차원을 숫자로 적지 않고 몸통에게 물어봅니다. 몸통을 conv(256차원)에서 BEATs(768차원)로 바꿔도 pooling과 head가 자동으로 따라옵니다. 몸통 갈아 끼우는 실험을 계속할 생각이었기 때문에 처음부터 이렇게 했습니다.

**`forward()`를 모양과 함께 한 줄씩.**

```python
# src/models/harm_model.py:48-60
    def forward(
        self, x: torch.Tensor, return_projection: bool = True
    ) -> dict[str, torch.Tensor]:
        frames = self.backbone(x)  # (B, T', D)
        z, attention = self.pool(frames)  # (B, D), (B, T')
        out = {
            "logits": self.classifier(z),  # (B, C)
            "attention": attention,
            "pooled": z,
        }
        if return_projection:
            out["embeddings"] = self.projection(z)  # (B, proj_dim)
        return out
```

- 입력 `x` = `(B, 1, F, T)`. B는 배치 크기, 1은 채널(흑백 이미지 한 장처럼), F는 mel bin 수, T는 시간 프레임 수.
- `frames` = `(B, T', D)`. 시간 축이 T→T'로 줄 수 있습니다(conv가 pooling을 하니까요). 여기까지가 "10초 클립 = T'개의 짧은 순간들"이라는 표현입니다.
- `z` = `(B, D)`, `attention` = `(B, T')`. 시간 축이 사라지고 클립 하나가 벡터 하나가 됩니다.
- `logits` = `(B, C)`(C는 클래스 수), `embeddings` = `(B, proj_dim)`.

**왜 dict으로 네 개를 돌려주나.** 소비자가 셋이고 각자 다른 것을 필요로 하기 때문입니다. ① `logits`는 **손실 함수**가 씁니다(focal-BCE가 정답과 비교해 학습 신호를 만듭니다). ② `embeddings`는 **SupCon 대조학습 손실**이 씁니다 — "같은 클래스는 가깝게, 다른 클래스는 멀게"는 분류 점수가 아니라 임베딩 공간 위에서 계산되므로 별도 출구가 필요합니다. ③ `attention`은 **사람**이 씁니다. 학습에는 전혀 안 쓰이고, 대신 "이 10초 중 몇 초쯤을 보고 위험하다고 했는지"를 그려 볼 수 있게 해 줍니다. ④ `pooled`(`z`)는 pooling 직후의 클립 임베딩으로 분석·후속 실험에서 재사용합니다. 추론 때는 SupCon이 없으니 `return_projection=False`로 계산을 건너뜁니다.

```python
# src/models/harm_model.py:74-79
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg_dict = ckpt.get("model_config")
        cfg = ModelConfig(**cfg_dict) if cfg_dict else ModelConfig()
        model = cls(num_classes, cfg)
        model.load_state_dict(ckpt["model"])
        return model.to(map_location)
```

가중치 파일만으로는 모델을 못 만듭니다. 가중치는 "몇 차원짜리 상자에 담을 숫자들"일 뿐이고 상자의 모양은 `ModelConfig`가 압니다. 그래서 config를 같이 저장해 두고, `from_checkpoint`에서 꺼내 같은 모양을 먼저 만든 뒤 숫자를 붓습니다.

```python
# src/models/harm_model.py:81-90
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # Force eval for BN/Dropout, but restore the prior mode so validating
        # mid-epoch doesn't silently leave the model in eval for the rest of training.
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self.forward(x, return_projection=False)["logits"])
        finally:
            self.train(was_training)
```

**여기서 조심할 것.** ① `weights_only=False`는 체크포인트 안의 임의 파이썬 객체를 복원합니다(docstring에 경고가 붙어 있습니다). 악의적인 `.pt`는 로드만으로 코드를 실행시킬 수 있으니 출처 불명 체크포인트를 이렇게 열지 마세요. ② `predict_proba`의 모드 복원은 제가 당하고 나서 쓴 방어 코드입니다. `self.eval()`만 부르고 끝내면 epoch 중간에 검증을 한 번 돌린 뒤부터 **학습이 계속 eval 모드로 진행됩니다** — BatchNorm은 통계를 갱신 안 하고 Dropout은 꺼진 채로요. 에러 없이 조용히 성능만 이상해집니다. `finally`인 것은 예외가 나도 반드시 복구시키기 위해서입니다. 그리고 sigmoid가 여기서 처음 등장한다는 점을 기억해 두세요.

### `src/models/backbones.py` — 몸통 진열장

**무엇을 하나.** "log-mel `(B, 1, F, T)` → 프레임 임베딩 `(B, T', D)`"라는 계약(파일 상단의 `FrameBackbone` Protocol)을 지키는 부품들을 모아 두고 이름으로 고르게 합니다.

```python
# src/models/backbones.py:104-121
_NOT_WIRED = {"panns", "cnn14", "ast"}


def build_backbone(name: str = "conv", **kwargs) -> nn.Module:
    if name == "conv":
        return ConvFrameBackbone(**kwargs)
    if name == "mfcc_bilstm":
        return MFCCBiLSTMBackbone(**kwargs)
    if name in ("passthrough", "beats"):
        # "beats" trains on precomputed frozen-BEATs frame embeddings (see
        # models.beats_extractor); the backbone itself is an identity here.
        return PassthroughBackbone(**kwargs)
    if name in _NOT_WIRED:
        raise NotImplementedError(...)
    raise ValueError(f"unknown backbone {name!r}")
```

실제 등록부는 이렇습니다.

| 이름 | 실제로 돌려주는 것 |
|---|---|
| `"conv"` | `ConvFrameBackbone` — 직접 만든 작은 CNN |
| `"mfcc_bilstm"` | `MFCCBiLSTMBackbone` — log-mel→MFCC→양방향 LSTM 베이스라인 |
| `"passthrough"` | `PassthroughBackbone` — 항등(identity) |
| `"beats"` | **`PassthroughBackbone` — 역시 항등. BEATs 모델이 아닙니다** |
| `"panns"`, `"cnn14"`, `"ast"` | 없음. `NotImplementedError` |
| 그 외 | `ValueError` |

**`"beats"`가 항등이라는 점을 분명히 하고 갑니다.** `ModelConfig(backbone="beats")`로 모델을 만들어도 그 안에 BEATs 트랜스포머는 한 조각도 없습니다. 파라미터 0개인 껍데기입니다.

```python
# src/models/backbones.py:97-101
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B, T, D); tolerate (B, 1, D, T) stored like a log-mel
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1).transpose(1, 2)
        return x.contiguous()
```

**왜 이래도 되나.** 이 경로에서는 BEATs를 미리 한 번만 돌려 `(T, 768)` 임베딩을 파일로 저장해 두고 그 파일을 데이터로 읽기 때문입니다. 학습 루프에 들어올 때 입력이 **이미** 프레임 임베딩이라 뽑아낼 게 없습니다(클래스 docstring: `there is nothing to extract`). 실제 BEATs를 돌리는 코드는 따로 있습니다 — 미리 뽑을 때 `beats_extractor.py`, 학습 중에 BEATs까지 같이 학습시킬 때 `beats_finetune.py`. 아래에서 다시 다룹니다.

기본 몸통인 CNN도 짧습니다.

```python
# src/models/backbones.py:50-53
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)  # (B, out_dim, F', T')
        h = h.mean(dim=2)  # average over frequency -> (B, out_dim, T')
        return h.transpose(1, 2).contiguous()  # (B, T', out_dim)
```

핵심은 `h.mean(dim=2)` 한 줄입니다. **주파수 축은 평균으로 없애고 시간 축은 남깁니다.** 시간을 남겨야 "언제 위험한 소리가 났는지"를 뒤에서 따질 수 있습니다. 여기서 시간까지 평균 내 버리면 다음에 나올 MIL attention이 할 일이 사라집니다.

**여기서 조심할 것.** `"beats"`라는 이름만 보고 "BEATs가 학습된다"고 읽으면 안 됩니다. `_NOT_WIRED`의 이름들은 오타가 아니라 의도적 미구현입니다 — 외부 사전학습 가중치와 어댑터가 필요해서 아직 연결하지 않았고, 조용히 잘못된 모델을 만드느니 명확히 에러를 내게 했습니다.

### `src/models/pooling.py` — 주목, 이 절의 핵심

**무엇을 하나.** `(B, T, D)` 프레임 열을 `(B, D)` 클립 벡터로 접습니다. 단순 평균이 아니라 **가중 평균**이고, 그 가중치를 모델이 스스로 학습합니다.

**왜 평균이면 안 되나.** 라벨은 클립 단위입니다. 10초 오디오 하나에 "폭력 있음" 라벨 하나가 붙습니다. 그런데 실제 위험한 소리는 그중 0.5초일 수 있고 나머지 9.5초는 평범한 배경음입니다. 프레임을 전부 단순 평균 내면 결정적인 0.5초가 9.5초에 희석돼 신호가 20분의 1로 묽어집니다. 이렇게 **가방(bag)에는 라벨이 있는데 가방 속 알갱이(instance)에는 라벨이 없는** 문제를 다중 인스턴스 학습(MIL)이라 부르고, MIL attention은 어느 알갱이가 중요한지를 학습으로 알아냅니다. 수식은 docstring에 그대로 있습니다: `a_t = softmax(u^T tanh(V h_t)) ;  z = Σ_t a_t h_t`.

파일 본문은 20줄이 전부입니다.

```python
# src/models/pooling.py:16-37
class MILAttentionPooling(nn.Module):
    def __init__(self, dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.V = nn.Linear(dim, attn_dim, bias=False)  # V h_t
        self.u = nn.Linear(attn_dim, 1, bias=False)  # u^T (.)

    def forward(
        self, frames: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """frames: (B, T, D), optional bool mask: (B, T) with True = keep.

        Returns (z: (B, D), attention: (B, T)).
        """
        scores = self.u(torch.tanh(self.V(frames))).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=1)  # (B, T)
        if mask is not None:
            # A fully-masked row makes softmax([-inf,...]) NaN; zero it out.
            attention = torch.nan_to_num(attention, nan=0.0)
        z = torch.bmm(attention.unsqueeze(1), frames).squeeze(1)  # (B, D)
        return z, attention
```

가중치를 만드는 것은 사실상 세 줄입니다.

**29행 `scores = ...`** — 프레임 하나 `h_t`(길이 D)를 `V`로 128차원으로 줄이고, `tanh`로 비선형을 한 번 먹인 뒤, `u`로 **숫자 하나**로 압축합니다. `(B,T,D) → (B,T,128) → (B,T,1) → squeeze → (B,T)`. 프레임마다 "얼마나 중요해 보이는가" 점수 하나씩입니다.

**32행 `torch.softmax(scores, dim=1)`** — `dim=1`이 **시간 축**입니다. 이 파일에서 제일 중요한 한 글자입니다. 시간 축에 softmax를 걸면 한 클립 안의 T개 가중치가 전부 양수가 되고 **합이 정확히 1**이 됩니다. 즉 "10초 안에서 주목을 어떻게 배분할지"를 정하는 확률분포입니다. 총량이 1로 고정돼 있으니 어떤 순간을 더 보려면 반드시 다른 순간을 덜 봐야 합니다. 그 경쟁이 학습을 만듭니다. (`dim=0`으로 잘못 쓰면 배치 안의 서로 다른 클립끼리 경쟁하게 됩니다. 조용히 틀리는 종류의 버그입니다.)

**36행 `torch.bmm(...)`** — 가중 합입니다. `bmm`은 배치 행렬곱이라 `(B,1,T) @ (B,T,D) = (B,1,D)`를 배치마다 한 번에 계산하고, `squeeze(1)`로 `(B, D)`가 됩니다. 만약 attention이 전부 `1/T`로 똑같다면 이 식은 정확히 단순 평균이 됩니다. **단순 평균은 MIL attention의 특수한 한 경우**이고, 모델은 필요할 때 거기서 벗어날 자유를 얻습니다.

**attention이 해석 가능성 측면에서 뜻하는 것.** `attention[b]`는 길이 T이고 합이 1입니다. 시간 축에 그대로 그래프로 그리면 모델이 "3.2초 근처를 집중해서 봤다"는 게 눈에 보입니다. 학습에 쓰이는 값이 아니라 부산물이지만, 유해성 판정은 결과만 던져서는 신뢰받기 어렵고 "이 구간 때문입니다"라고 말할 수 있어야 하므로 `harm_model`이 굳이 반환값에 담아 둡니다.

**여기서 조심할 것.** 마스크 처리입니다. 패딩 프레임은 `-inf` 점수를 줘서 softmax가 0을 배정하게 합니다. 그런데 한 행이 통째로 마스킹되면 `softmax([-inf, -inf, ...])`가 NaN이 되고, NaN은 역전파를 타고 모델 전체를 오염시킵니다. 그래서 `nan_to_num`으로 0으로 눌러 둡니다.

### `src/models/heads.py` — 머리 두 개

**무엇을 하나.** 클립 벡터 `z`에서 두 갈래 출력을 만듭니다.

```python
# src/models/heads.py:23-31
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits (B, C)
```

**분류 head는 raw logit을 내보냅니다. sigmoid가 없습니다.** 파일 docstring도 못 박아 둡니다 — `The classifier outputs raw logits (sigmoid is applied by the loss / at inference) for numerically stable focal-BCE.` 학교에서 "출력층에 시그모이드를 붙인다"고 배웠다면 이상해 보이겠지만 실전 코드는 대개 이렇습니다. **수치 안정성** 때문입니다. sigmoid를 먼저 계산하고 log를 씌우면 확률이 0이나 1에 아주 가까울 때 `log(0)` 근처에서 값이 폭발합니다. PyTorch의 `binary_cross_entropy_with_logits`는 sigmoid와 log를 하나의 식으로 합쳐 이 문제를 피하는데, 그러려면 **손실 함수에 logit을 통째로 넘겨야** 합니다. 그래서 sigmoid는 딱 필요한 곳에서만 붙습니다 — 학습 중엔 손실 함수 안에서, 추론 땐 앞에서 본 `predict_proba`에서. 출력이 C개이고 각각 독립적으로 sigmoid를 먹는다는 건 **다중 라벨** 설정이라는 뜻입니다. softmax로 하나만 고르는 게 아니라 "폭력이면서 동시에 도박"인 클립도 표현할 수 있습니다.

```python
# src/models/heads.py:34-40
class ProjectionHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, proj_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(z), dim=-1)  # unit-norm (B, proj_dim)
```

**projection head는 무엇에 쓰나.** 오직 SupCon 대조학습 손실용입니다. 대조학습은 임베딩끼리의 유사도를 다루는데, 코사인 유사도를 재려면 벡터 길이가 방해가 됩니다. `F.normalize`로 전부 길이 1로 만들어 단위 구 위에 올려 두면 내적이 곧 코사인 유사도가 됩니다. 분류 head를 그대로 쓰지 않고 갈래를 나눈 이유는 분류에 좋은 표현과 대조학습에 좋은 표현이 다르기 때문입니다. 대조학습의 요구를 이 얇은 층이 흡수해 주면 `z` 자체는 분류에 집중할 수 있습니다.

**여기서 조심할 것.** projection head는 추론에서 버려집니다(`dropped at inference`). 학습된 파라미터인데 추론에 안 쓰이는 게 버그가 아닙니다. 그리고 `ClassifierHead` 출력을 0.5와 비교해 판정하면 안 됩니다 — logit의 0.5는 확률 0.5가 아닙니다(logit 기준 경계는 0). 임계값 얘기가 나오면 항상 "이게 logit이야 확률이야?"부터 확인하세요.

### `beats_extractor.py` vs `beats_finetune.py` — 얼린 길과 녹인 길

같은 BEATs를 쓰지만 목적이 정반대인 두 파일입니다.

**얼린 길** — `src/models/beats_extractor.py`.

```python
# src/models/beats_extractor.py:22-41
class BEATsExtractor:
    out_dim = 768

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT, device: str = "cpu") -> None:
        ...
        self.model.predictor = None  # feature-extractor mode -> encoder frame outputs
        self.model.eval().to(device)

    @torch.no_grad()
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """(N,) or (B, N) float32 waveform in [-1, 1] @16kHz -> (B, T, 768) embeddings."""
        ...
        feats, _ = self.model.extract_features(wav.to(self.device))
        return feats.cpu().numpy()
```

세 가지를 눈여겨보세요. ① `nn.Module`이 **아닙니다** — 학습 그래프에 들어갈 일 없는 도구 클래스입니다. ② `@torch.no_grad()` + `.eval()` — 기울기를 아예 만들지 않습니다. BEATs는 완전히 얼어 있습니다. ③ `self.model.predictor = None` — BEATs 원본의 AudioSet 분류 head를 떼어냅니다. 저는 AudioSet 클래스 예측이 아니라 그 직전 단계의 **프레임 표현**이 필요하기 때문입니다. 또 BEATs는 자기 안에서 kaldi fbank를 직접 계산하므로 이 클래스는 log-mel이 아니라 **원본 파형**을 먹습니다. 전처리가 두 갈래인 이유입니다. 이렇게 뽑은 `(T, 768)`을 저장해 두면 학습 때는 `PassthroughBackbone`이 받아 MIL+head만 학습합니다. 노트북/MPS에서도 학습이 도는 이유가 이것입니다.

**녹인 길** — `src/models/beats_finetune.py`. 파일 docstring이 차이를 직접 설명합니다.

```python
# src/models/beats_finetune.py:3-6
The frozen-feature path (`PassthroughBackbone`) consumes precomputed BEATs embeddings and
cannot adapt the backbone. Fine-tuning needs the actual BEATs model in the graph, running
on raw 16 kHz waveforms with gradients through the TOP-k transformer blocks (strategy B:
unfreeze top blocks, keep the lower/patch layers frozen — cheap + stable on a T4).
```

`cannot adapt the backbone`이 핵심입니다. 미리 뽑아 둔 임베딩으로는 BEATs 자체를 이 프로젝트 데이터에 맞게 고칠 수 없습니다. 고치려면 BEATs를 학습 그래프에 넣고 매 스텝 원본 파형에서 다시 계산해야 하며, 훨씬 무겁고 GPU가 필요합니다. **어느 블록에 기울기를 흘릴지 정하는 코드**가 이 파일의 심장입니다.

```python
# src/models/beats_finetune.py:50-61
    def _set_trainable(self) -> None:
        for p in self.beats.parameters():
            p.requires_grad = False
        layers = self.beats.encoder.layers
        for layer in layers[len(layers) - self.unfreeze_top_k:]:
            for p in layer.parameters():
                p.requires_grad = True
        # the encoder's final layer_norm (post-block) also adapts, if present
        ln = getattr(self.beats.encoder, "layer_norm", None)
        if isinstance(ln, nn.Module):
            for p in ln.parameters():
                p.requires_grad = True
```

읽는 순서가 중요합니다. **먼저 전부 얼리고, 그다음 위쪽 k개만 녹입니다.** 기본값이 `unfreeze_top_k: int = 4`이니 12개 인코더 층 중 위 4개이고, 슬라이스 `layers[len(layers) - k:]`가 "뒤에서 k개"입니다. 왜 위쪽만이냐면, 아래층은 소리의 기본 질감 같은 일반적 특징이라 어느 과제에나 쓸 만하고 위층이 과제 특화 부분이기 때문입니다. 아래를 얼려 두면 학습 파라미터가 확 줄고(메모리·시간 절약), 적은 데이터로 학습할 때 흔한 파괴적 망각도 덜합니다. 클래스에 붙은 `manages_own_freezing = True  # opt out of Trainer's all-or-nothing stage freeze` 플래그는 Trainer에게 "이 몸통은 얼리고 녹이는 걸 스스로 관리하니 건드리지 마"라고 알리는 장치입니다.

```python
# src/models/beats_finetune.py:78-84
    model = HarmModel(num_classes, ModelConfig(backbone="passthrough", backbone_out_dim=768))
    if head_ckpt is not None:
        ck = torch.load(head_ckpt, map_location=map_location, weights_only=False)
        # passthrough backbone has no params -> only MIL/classifier/projection load
        model.load_state_dict(ck["model"], strict=True)
    model.backbone = BEATsRawBackbone(beats_ckpt, unfreeze_top_k=unfreeze_top_k, use_layers=use_layers)
    return model
```

순서가 영리합니다. **먼저 passthrough 모델을 만들어 얼린 길에서 학습해 둔 MIL+head 가중치를 붓고, 그다음 몸통만 진짜 BEATs로 갈아 끼웁니다.** passthrough는 파라미터가 0개라 `strict=True`로 로드해도 충돌이 없습니다. 덕분에 fine-tuning이 백지가 아니라 이미 동작하는 head 위에서 시작합니다.

```python
# src/models/beats_finetune.py:45-46
        if use_layers is not None:
            self.beats.encoder.layers = self.beats.encoder.layers[:use_layers]
```

`use_layers`는 별도의 실험 스위치로, 12개 층 중 앞의 k개만 남기고 잘라 버립니다. `unfreeze_top_k`가 "어디를 학습할까"라면 이건 "모델을 얼마나 얇게 만들까"입니다.

**여기서 조심할 것.** 두 스위치를 같이 쓸 때 순서입니다. 코드는 `use_layers`로 자른 **뒤에** `_set_trainable()`을 부르므로 `len(layers)`는 자른 후의 개수입니다. `use_layers=6, unfreeze_top_k=4`면 남은 6개 중 위 4개가 녹습니다. 원본 12개 기준이 아닙니다. 코드를 직접 읽지 않으면 알 수 없는 종류의 사실입니다.

### `src/models/beats/` — 가져다 쓴 코드

이 디렉터리는 제가 쓴 코드가 아닙니다. Microsoft의 `microsoft/unilm` 저장소에서 BEATs 구현(`BEATs.py`, `backbone.py`, `modules.py`)을 그대로 복사해 왔고, MIT 라이선스라 이렇게 포함할 수 있습니다. 출처와 저작권은 `NOTICE.txt`에 남겨 두었습니다. **한 글자도 고치지 않았습니다** — 고치면 원본 사전학습 가중치와 어긋날 위험이 있고 나중에 업스트림 수정을 따라가기도 어려워집니다. 이렇게 외부 코드를 저장소에 복사해 두는 방식을 vendoring이라 부릅니다. 남의 코드를 쓸 때 라이선스 확인과 출처 표기는 선택이 아니라 의무라는 점, 그리고 "내가 짠 코드"와 "가져온 코드"의 경계를 디렉터리로 분명히 나눠 두는 습관을 함께 봐 두면 좋겠습니다.

---

## 4. 무엇을 최소화할 것인가, 그리고 학습 루프

모델이 로짓을 뱉는 데까지 봤습니다. 그 로짓이 얼마나 틀렸는지를 숫자 하나로 만드는 게 **손실 함수**, 그 숫자를 줄이는 쪽으로
파라미터를 미는 과정이 **학습 루프**입니다. 손실 세 파일(`src/losses/`)과 학습 다섯 파일(`src/training/`)을 순서대로 읽겠습니다.

### `src/losses/focal.py`

**무엇을 하나** — 멀티라벨의 기본 손실인 BCE에 가중치를 곱한 focal loss입니다. 독스트링에 공식이 그대로 있습니다:
`FL = (1 - p_t)^gamma * BCE(logits, targets)`.

```python
# src/losses/focal.py:23-26
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)  # prob of the true class
    focal = (1.0 - p_t).clamp(min=0.0).pow(gamma) * bce
```

**왜 이렇게 짰나** — `p_t`는 "모델이 정답 쪽에 준 확률"입니다. 정답이 1이면 `p`, 0이면 `1 - p`가 되도록 한 줄로 합쳐 놓은 게
25번 줄이고, 그러면 `1 - p_t`가 곧 "얼마나 헷갈렸나"가 됩니다. `gamma` 기본값은 시그니처에 있듯 `2.0`입니다.

```python
# src/losses/focal.py:15-21
def focal_bce_with_logits(
    ...
    gamma: float = 2.0,
    alpha: torch.Tensor | float | None = None,
    ...
```

숫자를 넣어 보면 감이 옵니다. 이미 자신 있게 맞힌 샘플은 `p_t = 0.99` → `(0.01)^2 = 0.0001`로 손실이 1만분의 1이 되고,
헷갈리는 샘플은 `p_t = 0.5` → `0.25`로 4분의 1만 줄고, 자신 있게 틀린 샘플은 `p_t = 0.1` → `0.81`로 거의 그대로 남습니다.
즉 **쉬운 예제의 발언권을 깎아 어려운 예제가 그래디언트를 지배하게** 만드는 장치예요. 안전한 소리가 압도적으로 많은 이
데이터에서 평범한 BCE를 쓰면 "전부 안전"이라 외치는 모델도 손실이 꽤 낮게 나오는데, `gamma=2`가 그 지름길을 막습니다.
`gamma=0`이면 `(1-p_t)^0 = 1`이라 그냥 BCE로 돌아갑니다. `alpha`는 별개의 축입니다.

```python
# src/losses/focal.py:28-30
    if alpha is not None:
        a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal = a_t * focal
```

클래스별 가중치(스칼라 또는 클래스 수만큼의 텐서)를 `p_t`와 똑같은 요령으로 양성/음성 위치에 나눠 곱합니다. `gamma`가 "샘플
난이도"로 무게를 조절한다면 `alpha`는 "클래스 희소성"으로 조절하고, 둘은 곱해져 함께 작동해요.

**여기서 조심할 것** — 입력은 확률이 아니라 **로짓**입니다. 밖에서 시그모이드를 걸어 넣으면
`binary_cross_entropy_with_logits`를 쓴 이유(독스트링의 "numerical stability")가 사라집니다. `alpha`는 하드코딩이 아니라
호출부가 넘겨줍니다 — 클래스 가중치는 버전 관리되는 파일에서 읽는 값이지 코드에 박는 상수가 아니라는 규칙 때문입니다.

### `src/losses/supcon.py`

**무엇을 하나** — 대조 학습은 정답을 맞히라고 하는 대신 **같은 편의 임베딩은 가깝게, 다른 편은 멀게** 밀어 놓습니다. 공간이
정돈되면 그 위의 분류기가 일하기 쉬워지죠. 문제는 "같은 편"의 정의입니다. 라벨이 하나면 간단한데 여기선 한 클립에 여러
라벨이 붙으니, **자카드 유사도 0.5 이상**을 기준으로 씁니다.

```python
# src/losses/supcon.py:18-31
def jaccard_positive_mask(labels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return a (B, B) float mask of Jaccard-overlap positives, self excluded.
    ...
    """
    binary = (labels > 0).float()
    inter = binary @ binary.t()  # (B, B)
    card = binary.sum(dim=1)
    union = card.unsqueeze(0) + card.unsqueeze(1) - inter
    jacc = torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(inter))
    mask = (jacc >= threshold).float()
    mask = mask - torch.eye(labels.size(0), device=labels.device, dtype=mask.dtype)
    return mask.clamp(min=0.0)
```

**왜 이렇게 짰나** — 자카드는 `|교집합| / |합집합|`입니다. 멀티핫끼리 `binary @ binary.t()`를 하면 `(i, j)` 자리에 두 라벨
집합의 교집합 크기가 한 번에 나옵니다 — 이중 for문 없이 배치의 모든 쌍을 동시에 계산하는 트릭이에요. `{violence, scream}`
vs `{violence}` → 1/2 = 0.5로 경계값이라 `>=`에 걸려 통과, `{violence}` vs `{gambling}` → 0으로 탈락. 라벨이 없는 안전
클립끼리는 합집합이 0인데, 독스트링이 밝히듯 "pairs of empty label sets (safe clips) have union 0 and are treated as
non-positive" — `torch.where`가 그때 0을 주도록 명시해 "안전 클립끼리 전부 같은 편"이 되는 사고를 막았습니다. 마지막 두
줄은 단위행렬을 빼서 자기 자신을 양성에서 제외해요. 파일 독스트링이 밝히는 의도는 "this makes confusable pairs
(vio_impact vs chair_scrape) negatives automatically — no manual pair labeling"입니다. 헷갈리는 쌍(충격음 vs 의자 끄는
소리)을 사람이 "다른 편"이라 표시할 필요가 없어요. 라벨 집합이 다르면 자동으로 음성이 되고, 샘플러가 유해 클립과 헷갈리는
클립을 섞어 뽑으니 그 어려운 쌍이 매 배치에 들어옵니다. 손실 본체(`multilabel_supcon`)는 임베딩을 L2 정규화해 코사인
유사도를 구하고 `temperature`로 나눈 뒤 행마다 log-softmax를 취해 **양성 위치의 log-확률만 평균 내고 음수**를 붙입니다.

**여기서 조심할 것** — 배치 안에 양성 쌍이 없는 앵커는 `valid = pos_count > 0`으로 제외되고, 배치 전체에 없으면 손실은
0입니다. 에러가 아니라 "이번 스텝엔 대조 신호가 없음"이라는 정상 동작이에요. 그래서 SupCon은 **배치 구성에 민감**합니다.
배치가 작거나 샘플러가 편향되면 양성 쌍이 사라져 조용히 무력해지니, 이 함수만 봐서는 안 되고 샘플러와 같이 봐야 합니다.

### `src/losses/combined.py`

**무엇을 하나** — 두 손실을 가중합해 최종 목적함수를 만듭니다. 비율은 `mu`이고 기본값은 `0.2`입니다.

```python
# src/losses/combined.py:20-24
class LossConfig:
    gamma: float = 2.0
    mu: float = 0.2
    temperature: float = 0.1
    jaccard_threshold: float = 0.5
```

```python
# src/losses/combined.py:39-46
        focal = focal_bce_with_logits(logits, targets, self.cfg.gamma, self.alpha)
        if self.enable_supcon:
            supcon = multilabel_supcon(
                embeddings, targets, self.cfg.temperature, self.cfg.jaccard_threshold
            )
        else:
            supcon = torch.zeros((), device=logits.device, dtype=logits.dtype)
        total = focal + self.cfg.mu * supcon
```

**왜 이렇게 짰나** — `total = focal + 0.2 * supcon`. 주 목적은 어디까지나 분류(focal)이고 SupCon은 표현을 다듬는 보조 항으로
5분의 1 무게만 받습니다. 독스트링엔 탐색 범위까지 있습니다: `mu=0.2 (search 0.1-0.5)`. 그리고 평범해 보이는 불리언 하나가
커리큘럼 스위치입니다.

```python
# src/losses/combined.py:32-34
        self.alpha = alpha
        # Curriculum toggle: S1 (heads-only warmup) runs focal-BCE only (spec §6).
        self.enable_supcon = True
```

학습 초반(S1, 헤드만 학습하는 워밍업)에는 이 값을 `False`로 두어 **focal만** 씁니다. 시작 시점의 임베딩은 아직 아무 의미 없는
좌표인데 거기서 "같은 편끼리 모여라"를 강하게 밀면 엉뚱한 구조로 굳을 수 있거든요. 헤드가 자리를 잡은 뒤 백본을 푸는 S2에서 켭니다.

**여기서 조심할 것** — `forward`는 `(total, parts)` 튜플을 돌려주고 `parts`는 `.detach()`된 로깅용 사본이라 역전파에 관여하지
않습니다. `LossConfig`는 `frozen=True`라 학습 도중 손실 하이퍼파라미터가 몰래 바뀌는 일을 타입 레벨에서 막습니다.

### `src/training/config.py`

**무엇을 하나** — 시드, 배치, 학습률, 정밀도, 시간 가드까지 학습 관련 숫자를 dataclass 하나에 모읍니다. 그중 커리큘럼 정의가 핵심입니다.

```python
# src/training/config.py:17-20
DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("S1-heads", epochs=5, freeze_backbone=True, use_supcon=False),
    CurriculumStage("S2-full", epochs=45, freeze_backbone=False, use_supcon=True),
)
```

**왜 이렇게 짰나** — 앞 절의 토글이 여기서 **데이터로** 선언됩니다. 5에폭은 백본을 얼리고 SupCon을 끄고, 다음 45에폭은 백본을
풀고 SupCon을 켜요. 어느 에폭이 어느 단계인지는 `stage_for_epoch`가 누적 합으로 찾습니다. 이 파일에서 더 배울 건 주석의
태도입니다 — 설명이 아니라 **경고**로 쓰입니다. `grad_accum_steps: int = 2` 위에는 "grad-accum widens the BCE gradient
estimate (spec §6 "effective 64"); it does NOT widen the SupCon contrastive set"이 붙어 있고, 정밀도 쪽은 이렇습니다.

```python
# src/training/config.py:43-47
    # precision: AMP only on CUDA, fp32 on MPS/CPU (spec §6).
    # amp_dtype: "fp16" (default; needs GradScaler) or "bf16" (Ampere+, no scaler,
    # fp32-range so no overflow — the safe choice for BEATs, which NaNs in fp16).
    amp: bool = True
    amp_dtype: str = "fp16"
```

"배치 32에 누적 2단계니 실질 배치 64"라고 착각하기 딱 좋은 지점, 그리고 "fp16으로 돌렸더니 NaN이 나더라"라는 실제로 밟은
지뢰가 남아 있습니다. 코드가 뭘 하는지가 아니라 읽는 사람이 어디서 헛다리를 짚을지를 적어 둔, 좋은 주석입니다.

**여기서 조심할 것** — 첫 줄에 "Mirrored by configs/train/train.yaml"이라 적혀 있습니다. 같은 내용이 두 군데 있으니 하나만
고치면 어긋납니다. `frozen=True`라 `cfg.batch_size = 64` 같은 대입은 런타임 에러예요.

### `src/training/optim.py`

**무엇을 하나** — AdamW를 만들되 파라미터를 **그룹으로 쪼개 각기 다른 학습률**을 붙입니다.

```python
# src/training/optim.py:40-48
        for depth, layer in enumerate(layers):
            # last layer (closest to head) -> full lr_backbone; earlier -> decayed
            scale = cfg.layer_decay ** (n - 1 - depth)
            lr = cfg.lr_backbone * scale
            _add_param_groups(groups, layer.parameters(), lr, cfg.weight_decay)

    # Head groups: everything not in the backbone.
    head_params = [p for p in model.parameters() if id(p) not in backbone_params]
    _add_param_groups(groups, head_params, cfg.lr_heads, cfg.weight_decay)
```

**왜 이렇게 짰나** — `config.py`의 값과 합치면 헤드는 `lr_heads = 1e-4`, 백본은 `lr_backbone = 1e-5`로 **열 배** 차이입니다.
백본은 이미 대규모 오디오로 사전학습된 모듈이라 "소리에서 무엇을 들어야 하는지"를 상당히 알고 있고, 분류 헤드는 방금 랜덤
초기화된 빈 종이예요. 같은 학습률을 쓰면 학습 초반 헤드에서 나오는 크고 지저분한 그래디언트가 백본까지 흔들어 애써 배운
표현을 뭉갭니다(**catastrophic forgetting**). 그래서 백본은 작은 학습률로 살짝만 조정하고 헤드는 큰 학습률로 빨리 배우게
합니다. 같은 논리를 백본 내부에 한 번 더 적용한 게 `layer_decay = 0.9`로, 입력에 가까운 층일수록 `0.9^(n-1-depth)`가 작아져
학습률이 더 깎여요 — 초기 층은 소리의 기본 결을 잡는 범용 특징이라 건드릴 이유가 적으니까요. `_is_no_decay`가
`param.ndim <= 1`로 편향·정규화 계수의 weight decay를 면제하는 것도, 이들은 크기 자체가 의미라 0으로 끌면 손해라서입니다.

**여기서 조심할 것** — 백본을 `getattr(model, "backbone", None)`으로 찾습니다. 모델에서 그 속성 이름을 바꾸면 백본이 조용히
"헤드"로 분류돼 10배 큰 학습률을 맞아요. 에러 없이 성능만 나빠지는, 제일 위험한 버그입니다. 또 `trainer.py`가
`param_groups[-1]["lr"]`을 헤드 학습률로 읽으므로 헤드 그룹을 마지막에 넣는 이 순서에 의존해요. 스케줄러
(`build_scheduler`)는 5% 워밍업 후 코사인 감쇠인데, 이 람다는 각 그룹 기본 학습률에 **곱해지는 배율**이라 10배 비율은
유지된 채 전체가 같이 오르내립니다.

### `src/training/trainer.py` — 심장부

학습 루프, 정밀도 관리, 커리큘럼 적용, 조기 종료, 시간 가드, 체크포인트가 모두 이 파일에 있습니다.

**1) 정밀도(AMP) 결정 — 생성자에서 딱 한 번**

```python
# src/training/trainer.py:97-103
        self.amp_enabled = self.cfg.amp and self.device.type == "cuda"
        self._bf16 = self.amp_enabled and self.cfg.amp_dtype == "bf16"
        self._amp_dtype = torch.bfloat16 if self._bf16 else torch.float16
        # bf16 has fp32 range -> no loss scaling needed; only fp16 uses GradScaler.
        self.scaler = (torch.amp.GradScaler("cuda")
                       if (self.amp_enabled and not self._bf16) else _NoScaler())
        self._autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
```

dtype이 정확히 어떻게 정해지는지 줄 단위로 보면 — (1) AMP는 **CUDA에서만** 켜집니다. `cfg.amp`가 True여도 장치가 MPS(맥)나
CPU면 `amp_enabled`가 False가 되고 전부 fp32로 돌아요. (2) `_bf16`은 AMP가 켜져 있고 **동시에** `cfg.amp_dtype == "bf16"`일
때만 True — 설정 문자열이 오타면 조용히 fp16으로 떨어집니다. (3) `_amp_dtype`은 그 불리언 하나로 `torch.bfloat16` 또는
`torch.float16`. (4) **GradScaler는 fp16일 때만** 생깁니다. bf16은 fp32와 지수 범위가 같아 그래디언트가 언더플로로 사라질
걱정이 없거든요. 그 외 경로엔 `_NoScaler`(실제 GradScaler와 같은 메서드 이름을 가진, 아무것도 안 하는 객체)가 들어가서 루프
본문에 `if self.scaler is not None:` 같은 분기가 하나도 없습니다 — 널 오브젝트 패턴입니다.

**2) 커리큘럼 적용** — `combined.py`의 `enable_supcon`을 조작하는 주체가 여기 첫 줄입니다.

```python
# src/training/trainer.py:110-117
    def _apply_stage(self, stage: CurriculumStage) -> None:
        self.loss_fn.enable_supcon = stage.use_supcon
        backbone = getattr(self.model, "backbone", None)
        ...
        if backbone is not None and not getattr(backbone, "manages_own_freezing", False):
            for p in backbone.parameters():
                p.requires_grad_(not stage.freeze_backbone)
```

매 에폭 시작 시 단계를 조회해 SupCon을 켜거나 끄고 백본의 `requires_grad`를 설정하되, 백본이 `manages_own_freezing`
플래그를 들고 있으면 건드리지 않습니다 — 상위 몇 층만 푸는 자기만의 전략을 가진 백본을 위한 탈출구예요.

**3) 학습 루프 본체**

```python
# src/training/trainer.py:148-169
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.autocast(self._autocast_device, dtype=self._amp_dtype,
                                enabled=self.amp_enabled):
                out = self.model(x)
                loss, _ = self.loss_fn(out["logits"], out["embeddings"], y)
                loss = loss / accum
            self.scaler.scale(loss).backward()
            running_loss += loss.item() * accum  # undo the accum scaling for logging
            n_batches += 1
            pending += 1
            if pending == accum:
                self._optimizer_step()
                pending = 0
                opt_steps += 1
                if opt_steps % self.cfg.time_guard_check_steps == 0 and self._time_exceeded():
                    return False  # save & exit; this epoch is redone on resume
        if pending > 0:  # flush a partial accumulation window
            self._optimizer_step()
        self._last_train_loss = running_loss / max(1, n_batches)
        return True
```

**forward** `out = self.model(x)` — 모델은 `logits`와 `embeddings`가 든 dict를 돌려줍니다. focal이 로짓을, SupCon이 임베딩을
먹으니 둘 다 필요해요. 이어서 **loss**, **`loss = loss / accum`**, **backward**, 그리고 `pending`이 `accum`에 도달했을 때만 **step**.

**나눗셈이 왜 중요한가.** PyTorch는 `.backward()`를 부를 때마다 그래디언트를 덮어쓰지 않고 **더합니다**. 이 성질이 누적의
원리인데, 손실 함수가 이미 배치 내부에서 평균을 내므로 마이크로 배치 손실을 그냥 두 번 더하면 배치 64의 **평균**이 아니라
**2배**가 됩니다. 미리 `accum`으로 나눠 두면 합쳐졌을 때 정확히 평균이 돼요. 잊으면 실질 학습률이 `accum`배로 뛴 것과
같아집니다 — 손실이 발산하는데 원인을 못 찾는 전형적인 버그입니다. 반대로 로깅 값은 왜곡되면 안 되니
`running_loss += loss.item() * accum`으로 되돌려요. `if pending > 0:`도 놓치지 마세요. 배치 수가 `accum`의 배수가 아닐 때
마지막에 남는 마이크로 배치를 flush하지 않으면 그 계산이 그냥 버려집니다.

```python
# src/training/trainer.py:171-175
    def _optimizer_step(self) -> None:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
```

**순서가 전부입니다.** step → scaler 갱신 → **zero_grad** → scheduler.step. `zero_grad`를 여기서 부르기 때문에 다음 누적
윈도가 깨끗하게 시작해요. 매 배치마다 불렀다면 누적은 애초에 성립하지 않습니다. 스케줄러도 배치가 아니라 **옵티마이저 스텝
단위**로 전진하므로 `fit`은 총 스텝을 `max(1, math.ceil(len(train_loader) / self.cfg.grad_accum_steps))`로 셉니다. 파일 상단
독스트링은 오해를 미리 차단해요 — "gradient accumulation ... widens the *BCE* gradient estimate but does NOT widen the
contrastive set — SupCon only ever sees one physical (sampler) batch." 누적은 BCE 계열에는 배치를 넓힌 효과를 주지만
SupCon처럼 **배치 안의 쌍 관계**를 보는 손실에는 도움이 안 됩니다. 알고 넘어가는 한계와 모르고 지나친 버그는 다릅니다.

**4) 시간 가드 (Kaggle 11시간)**

```python
# src/training/trainer.py:129-130
    def _time_exceeded(self) -> bool:
        return (self.clock() - self._start_time) >= self.cfg.time_guard_hours * 3600.0
```

Kaggle 세션은 12시간이면 강제 종료됩니다. 그래서 11시간에 스스로 저장하고 나가요. 중요한 건 이 검사를 **에폭 중간에도**
한다는 점입니다 — 위 루프의 `opt_steps % time_guard_check_steps == 0`, 즉 50 옵티마이저 스텝마다 봅니다. 에폭 하나가 몇
시간 걸리면 에폭 끝에서만 봐서는 늦으니까요.

```python
# src/training/trainer.py:228-236
                # Interrupted mid-epoch: checkpoint marking this epoch as NOT done
                # (epoch-1) so resume redoes it from the start; then exit.
                state = TrainState(epoch=epoch - 1, best_metric=best,
                                   epochs_no_improve=no_improve)
                ...
                status = "time_guard"
                last_epoch = epoch - 1
                break
```

에폭 번호를 `epoch - 1`로 적어 "이 에폭은 완료되지 않았다"고 표시합니다. 재개하면 `start_epoch = st.epoch + 1`이 되어 그
에폭을 **처음부터 다시** 돌아요. 반쯤 학습된 애매한 상태로 이어 붙이는 것보다 깔끔합니다. 생성자가
`clock: Callable[[], float] = time.monotonic`으로 시계를 주입받는 것도 의도적이에요 — 테스트에서 가짜 시계를 넣으면
11시간을 실제로 기다리지 않고 가드를 검증할 수 있습니다.

**5) 조기 종료(early stopping)와 patience**

```python
# src/training/trainer.py:242-246
            improved = (not math.isnan(val_map)) and val_map > best + 1e-6
            if improved:
                best, no_improve = val_map, 0
            else:
                no_improve += 1
```

```python
# src/training/trainer.py:265-267
            if no_improve >= self.cfg.patience:
                status = "early_stop"
                break
```

매 에폭 검증 mAP를 재고 최고 기록을 갱신하면 `no_improve`를 0으로 리셋합니다. 갱신 실패가 `patience = 10`번 연속 쌓이면
멈춰요. 더 돌려도 나아지지 않는데 GPU 시간을 태울 이유가 없고, 계속 돌리면 과적합만 깊어지니까요. `val_map > best + 1e-6`의
작은 여유값은 부동소수점 오차 수준의 "개선"을 진짜 개선으로 세지 않으려는 처리입니다 — 없으면 `no_improve`가 영영 안 쌓여
조기 종료가 사실상 꺼질 수 있어요. 저장은 두 갈래로, `last.ckpt`는 매 에폭 덮어써 재개용으로 쓰고 `best.ckpt`는 기록 갱신
때만 씁니다. 마지막 에폭 모델이 최고라는 보장이 없으니까요.

**여기서 조심할 것** — `self.optimizer`와 `self.scheduler`는 `__init__`이 아니라 `fit` 안에서 만들어집니다. `fit` 없이
`train_one_epoch`만 부르면 `AttributeError`예요. 재개 순서도 `build_optimizer` → `build_scheduler` → `load_checkpoint`여야
상태를 부을 대상이 존재합니다.

### `src/training/checkpoint.py`

**무엇을 하나** — `--resume auto`가 진짜로 "이어서"가 되게 만드는 파일입니다.

```python
# src/training/checkpoint.py:61-76
    payload = {
        "model": model.state_dict(),
        ...
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "epochs_no_improve": state.epochs_no_improve,
        "rng": _rng_state(),
    }
    tmp = Path(path).with_suffix(".ckpt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic-ish: never leave a half-written last.ckpt
```

**왜 각각이 필요한가** — 빠뜨렸을 때 무슨 일이 나는지로 보면 빠릅니다. `model`이 없으면 그냥 처음부터죠. `optimizer`는
AdamW가 파라미터마다 들고 다니는 1차·2차 모멘트라, 버리면 재개 직후 모멘텀이 0에서 다시 쌓이며 수백 스텝 동안 학습이
휘청거립니다. `scheduler`는 학습률 스케줄의 현재 위치라 없으면 코사인 감쇠가 끝나가던 중 워밍업으로 되돌아가 잘 수렴하던
모델에 갑자기 큰 학습률을 때려요. `scaler`는 fp16 스케일 값(적응적으로 바뀝니다). `epoch`/`best_metric`/`epochs_no_improve`가
없으면 조기 종료 카운터가 리셋돼, 이미 10에폭째 정체 중이던 학습이 재개 후 다시 10에폭을 더 돕니다. 생략한 `...` 자리에는
`model_config`가 들어가는데, 추론 때 모델을 정확히 같은 모양으로 되살리려고 아키텍처 설정까지 함께 저장한 겁니다. 마지막으로
`rng`가 필요한 이유는 셔플링·증강·드롭아웃이 전부 난수에 의존하기 때문입니다.

```python
# src/training/checkpoint.py:30-36
def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
```

복원하지 않으면 재개한 학습은 "같은 시드로 처음부터 돌린 학습"과 다른 궤적을 그립니다. 재현성이 깨지면 나중에 결과가
이상할 때 원인을 추적할 수 없어요.

**여기서 조심할 것** — 저장이 임시 파일 → `replace` 2단계입니다. 큰 파일을 쓰는 도중 세션이 죽으면 반쯤 쓰인 `last.ckpt`가
남아 재개 때 터지는데, `tmp.replace(path)`는 원자적에 가까워 "온전한 예전 것" 아니면 "온전한 새 것"만 남깁니다(주석의
"atomic-ish"). 그리고 `torch.load(..., weights_only=False)`를 씁니다 — 옵티마이저 상태나 RNG 튜플처럼 순수 텐서가 아닌
객체 때문에 필요하지만, 그만큼 **신뢰할 수 있는 체크포인트만** 읽어야 합니다.

### `src/training/metrics.py`

**무엇을 하나** — 평가 지표를 **넘파이만으로** 구현합니다. 첫 줄이 명시해요: "Evaluation metrics (spec §9), numpy-only
(no sklearn dependency)." sklearn을 쓰면 세 줄로 끝날 일이지만, 학습은 Kaggle 커널처럼 최소한의 런타임에서 도니 지표 하나
때문에 무거운 의존성을 끌고 들어오고 싶지 않았습니다. 대신 "sklearn의 `average_precision_score`와 값이 일치한다"를
독스트링에 명시하고 테스트로 못 박는 쪽을 택했습니다.

**Average Precision을 말로 풀면** — 모델 점수 순으로 샘플을 줄 세운 뒤 위에서 한 칸씩 내려가며 "지금까지 건진 것 중 진짜
비율(정밀도)"과 "전체 정답 중 건진 비율(재현율)"을 기록합니다. 재현율이 한 칸 오를 때마다 그 시점의 정밀도를 곱해 더한 게
AP예요 — 쉽게 말해 **정답들을 얼마나 위쪽에 몰아 놨는가** 점수이고 완벽하면 1.0입니다. `macro_map`은 클래스별 AP를 평균
내되 양성이 하나도 없는 클래스는 뺍니다(AP가 `nan`이니까). 이 값이 `trainer.py`의 조기 종료 기준입니다.

**Recall@FPR을 말로 풀면** — 이 프로젝트에서 제일 중요한 지표입니다. "오탐 비율을 1% 이하로 묶어 두면서 진짜 유해 클립을
몇 %나 잡아내나"를 묻습니다. 안전한 소리를 유해하다고 잘못 부르면 사용자 경험이 망가지므로 오탐에 상한을 걸고, 그 제약
아래 탐지율을 보는 게 현실적인 평가예요.

```python
# src/training/metrics.py:68-98
def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Recall (TPR) at the operating point with FPR <= target_fpr (spec §9).
    ...
    """
    ...
    order = np.argsort(-y_score, kind="mergesort")  # stable, descending
    s_sorted = y_score[order]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)

    # Only score-group boundaries are achievable operating points: a real
    # threshold can't split tied scores, so evaluate FPR/TPR at the last index of
    # each tied run (roc_curve semantics). Without this, ties inflate recall.
    boundary = np.ones(len(s_sorted), dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]

    fpr = fp[boundary] / n_neg
    tpr = tp[boundary] / n_pos
    allowed = fpr <= target_fpr
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())
```

`boundary` 부분이 제일 배울 게 많습니다. 점수가 완전히 같은 샘플이 여럿일 때 실제 임계값으로는 그 안을 **쪼갤 수 없습니다**
— "0.7 이상"이면 0.7짜리 다섯 개가 다 같이 넘어오거나 다 같이 안 넘어오죠. 그런데 누적합을 그대로 쓰면 그 중간에서 자를 수
있는 것처럼 계산돼 달성 불가능한 재현율이 나옵니다. 그래서 동점 구간의 **마지막 인덱스만** 유효한 동작점으로 봐요. 주석의
"Without this, ties inflate recall"이 정확히 그 경고입니다. 논문엔 안 나오고 라이브러리 소스에나 숨어 있는 디테일인데,
직접 구현하면 반드시 밟게 됩니다.

**여기서 조심할 것** — `nan` 규약이 일관됩니다. 양성이 없으면 AP는 `nan`, 양성이나 음성이 없으면 AUROC/recall@FPR도 `nan`이고
macro 계열은 `nan`을 걸러 평균 내요. 0으로 때우지 않는 게 중요합니다 — "잴 수 없음"과 "0점"은 다른 얘기예요. 정렬은 전부
`kind="mergesort"`(안정 정렬)라 동점 순서가 실행마다 흔들리지 않아 결과가 재현됩니다. 그리고 `target_fpr`을 만족하는 동작점이
아예 없으면 `nan`이 아니라 `0.0`을 돌려줍니다 — "잴 수는 있는데 그 조건에선 아무것도 못 잡는다"는 뜻이라 의미가 다르니까요.

---

## 5. 2층 트리거를 실제로 만든 스크립트

이 저장소는 `src/`에 "여러 실험이 공유하는 라이브러리"(모델 정의, MIL 풀링, 손실 함수, Trainer, 전처리)를 두고, `.autorun/`에는 "그 라이브러리를 특정 방식으로 조립해서 한 번 돌린 실제 실험"을 둡니다. 그래서 `src/`의 파일은 혼자서는 아무것도 학습하지 않고, `.autorun/`의 파일은 거의 전부 `import`와 환경변수 읽는 줄로 시작합니다 — 실험 스크립트가 곧 그 실험의 설계도인 셈입니다. 그 사이에 낀 게 `scripts/train_beats_finetune.py`입니다. 원래는 BEATs 파인튜닝 실험 자체였는데 안에 있는 부품이 쓸모가 많아서, 나중 실험들이 전부 여기서 꺼내 쓰게 됐습니다. 이 파일부터 봅니다.

### 1. `scripts/train_beats_finetune.py` — 나머지 넷이 전부 import 하는 공용 부품

**무엇을 하나.** 이름은 "BEATs 파인튜닝 학습 스크립트"지만 실질적으로는 세 가지를 제공합니다: (1) wav 파일을 읽어 원본 파형 그대로 내주는 `RawAudioDataset`, (2) 클래스별 focal loss 가중치를 계산하는 `_class_alpha`, (3) 클립 파일이 실제로 존재하는지 확인하는 `_has_clip`. 소리 갈래 스크립트 9개가 여기서 무언가를 가져다 씁니다 — `.autorun/`의 다섯 개(`train_ced_vio`, `train_beats_vio`, `quantize_ced`, `dump_beats_probs`, `calibrate`, 그리고 `asr_cer_eval`), `distill/`의 둘, `scripts/eval_bootstrap.py`. 세 개를 다 가져가는 건 `train_beats_vio.py`와 `train_ced_vio.py`이고, 나머지는 `RawAudioDataset`과 `_has_clip`만 씁니다. 핵심은 Dataset입니다. 보통 오디오 모델은 log-mel 스펙트로그램을 미리 계산해 `.npy`로 저장해두고 그걸 읽지만, BEATs나 CED 같은 사전학습 모델은 자기만의 방식으로 mel을 계산합니다. 그래서 여기서는 **파형(waveform) 그대로** 넘겨줍니다.

```python
# scripts/train_beats_finetune.py:64-70
class RawAudioDataset(Dataset):
    """Serves (raw 16kHz waveform (N,), multihot) — BEATs computes its own fbank.

    In train mode, applies on-the-fly waveform augmentation (spec §4: gain, time-shift,
    additive noise, mixup). Eval mode is always clean. Reproducible via a per-(seed,
    epoch, idx) RNG, mirroring LogMelDataset.
    """
    ...
# scripts/train_beats_finetune.py:92-103
    def _load(self, i):
        r = self.records[i]
        try:
            wav = load_audio(f"{CLIP_DIR}/{r.clip_id}.wav", sample_rate=self.sr)
            wav = fix_length(wav, self.n).astype(np.float32)
        except Exception as e:
            # Corrupt/undecodable clip (e.g. a truncated download): don't crash a
            # multi-hour run — substitute silence with an all-negative label (safe).
            print(f"[warn] undecodable clip {r.clip_id}: {type(e).__name__} -> silence", flush=True)
            return (np.zeros(self.n, dtype=np.float32),
                    np.zeros(self.tax.num_classes, dtype=np.float32))
        return wav, r.multihot(self.tax)
```

**왜 이렇게 짰나.** `try/except`가 중요합니다. 데이터가 몇천 개인데 그중 하나가 다운로드 도중 잘려 있으면 파이썬은 그 지점에서 예외를 던지고 학습이 죽습니다. 세 시간 돌던 학습이 2879번째 클립 때문에 죽는 건 최악이죠. 그래서 못 읽는 클립은 **무음 + 전부 0 라벨(= 아무 유해 클래스도 아님)** 으로 대체합니다. 무음을 유해하다고 배우게 만들지 않는, 안전한 쪽으로 실패하는 설계입니다. 그리고 증강(augmentation) 쪽이 이 파일에서 가장 교육적인 부분입니다. 코드는 다 짜여 있는데 **기본값이 꺼져 있습니다.** 그리고 왜 껐는지가 바로 위 주석에 남아 있습니다.

```python
# scripts/train_beats_finetune.py:45-48
# Augmentation is OFF by default: the single-seed A/B (2026-07-18) showed it did not
# improve the target (violence recall@FPR1% regressed, gunshot -.184; mixup blurs sharp
# transients). Kept toggle-able (AUGMENT=1) for a later MUSAN/multi-seed revisit.
AUGMENT = os.environ.get("AUGMENT", "0") == "1"
```

즉 `AUGMENT`의 기본값은 문자열 `"0"`이고, `AUGMENT=1`을 줘야만 켜집니다. 교과서는 "데이터 증강은 항상 도움이 된다"고 하지만, 여기서는 A/B 비교를 해보니 목표 지표가 오히려 나빠졌습니다. 특히 mixup은 두 오디오를 섞는 기법인데, 총성처럼 **짧고 날카로운 순간(transient)** 이 신호의 전부인 클래스에서는 그 날카로움을 뭉개버립니다. 코드를 지우지 않고 스위치로 남겨둔 것도 의도적입니다 — 나중에 MUSAN 소음 데이터로 다시 시도할 여지를 남기면서, "시도했고 이런 이유로 껐다"는 기록도 되기 때문입니다. 증강은 켜졌을 때만, 그것도 학습 때만 실행됩니다.

```python
# scripts/train_beats_finetune.py:105-113
    def __getitem__(self, i):
        wav, label = self._load(i)
        if not (self.train and AUGMENT):
            return torch.from_numpy(wav), torch.from_numpy(label)

        rng, a = self._rng(i), self.aug
        # mixup with a random partner (label union); exclude self to avoid a no-op mix
        if rng.random() < a.mixup_p and len(self.records) > 1:
            ...
```

검증/테스트에서는 절대 증강하지 않습니다. 무작위로 변형한 데이터로 점수를 재면 그 점수를 재현할 수 없으니까요. 난수도 그냥 `random`을 쓰지 않고 (시드, 에폭, 인덱스) 세 값으로 생성기를 매번 새로 만듭니다(89-90줄). 같은 시드로 다시 돌리면 3에폭 47번 샘플에 걸린 증강이 **정확히 똑같이** 재현되고, DataLoader가 여러 워커 프로세스로 병렬 로딩해도 결과가 흔들리지 않습니다.

**여기서 조심할 것.** `AUGMENT`는 모듈 수준 전역변수라, `import`되는 순간 환경변수를 읽어 값이 고정됩니다. `.autorun/`의 스크립트에서 `RawAudioDataset`을 가져다 쓸 때도 이 값이 그대로 따라옵니다 — 즉 CED 실험도 증강 OFF 상태로 돌아갑니다. 이건 우연이 아니라 "BEATs 실험과 같은 레시피로 비교한다"는 설계이지만, 부모 모듈의 전역 상태가 자식 실험의 동작을 바꾼다는 건 읽을 때 놓치기 쉬운 구조입니다.

### 2. `.autorun/train_ced_vio.py` — 채택된 폭력 트리거 (이 절의 핵심)

**무엇을 하나.** 사전학습된 CED-mini 인코더를 HuggingFace에서 가져와, 이 프로젝트의 MIL 풀링 + 분류 헤드에 붙이고, 폭력 4클래스만 보는 태스크로 파인튜닝합니다. 파일 맨 위 docstring이 그대로 설계 요약입니다.

```python
# .autorun/train_ced_vio.py:1-13
"""CED-mini VIOLENCE-ONLY fine-tune — candidate to replace BEATs (90M) as the acoustic
trigger with a 9x smaller backbone (10M, AudioSet mAP 49.0 > BEATs 48.6). See model_light.md.

Same recipe/split as the BEATs experiments (top-4 blocks unfrozen, focal-BCE + SupCon,
batch 8 x accum 4, class-balanced sampler, violence-only taxonomy v2.0-vio, non-violence
clips = all-zero negatives). Heads are NEW-init (CED dim 256 != BEATs 768 -> no warm-start;
noted as an honest difference vs the BEATs baseline which warm-started from a 23-class head).
...
Test split identical to probs_beats*.npz -> comparable via .autorun/compare_vio.py.
Env: CED_ID (default mispeech/ced-mini), UNFREEZE_TOP_K, EPOCHS, SEED, CKPT_DIR, OUT_PROBS, WANDB_*.
"""
```

#### (a) 코드가 조용히 틀리는 경우 — 이 절에서 제일 중요한 6줄

사전학습 모델을 불러올 때 보통 `AutoModel.from_pretrained(...)` 한 줄이면 끝납니다. 그런데 이 모델에서는 그게 **틀립니다.** 그것도 에러 없이 틀립니다.

```python
# .autorun/train_ced_vio.py:63-69
    def __init__(self, model_id: str = "mispeech/ced-mini", unfreeze_top_k: int = 4):
        super().__init__()
        # NOTE: must load via ForAudioClassification and take .encoder — the checkpoint keys
        # are prefixed "encoder.", so bare AutoModel silently random-inits EVERYTHING.
        from transformers import AutoModelForAudioClassification
        full = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True)
        self.ced = full.encoder
```

무슨 일이 일어나는지 풀어보면 이렇습니다. 체크포인트 파일 안의 가중치 이름들은 `encoder.blocks.0.attn.qkv.weight` 같은 식으로 전부 `encoder.` 접두사가 붙어 있습니다. 그런데 `AutoModel`로 부르면 만들어지는 객체는 인코더 **그 자체**여서 파라미터 이름이 `blocks.0.attn.qkv.weight`입니다. 이름이 하나도 안 맞습니다. 이때 HuggingFace는 예외를 던지지 않습니다. "이 키들은 못 찾았습니다"라는 경고 로그를 한 번 찍고, 못 찾은 파라미터는 **랜덤 초기값 그대로** 둔 채 모델을 돌려줍니다. 그러면 손에 남는 건 "사전학습됐다고 믿고 있는, 실제로는 완전히 백지인 인코더"입니다. 학습은 정상적으로 돌아가고, 손실도 내려가고, 점수도 나옵니다 — 다만 사전학습의 이점이 전혀 없는, 훨씬 나쁜 점수가 나옵니다. 그리고 그 원인을 코드 어디에서도 찾을 수 없습니다. 해결은 `AutoModelForAudioClassification`으로 불러서 **바깥 껍데기까지 포함한 구조**를 만든 다음 거기서 `.encoder`만 꺼내는 것입니다. 그러면 이름이 `encoder.blocks.0....`로 맞아떨어져 가중치가 제대로 들어갑니다.

그리고 저 `# NOTE:` 주석은 지우면 안 되는 종류의 주석입니다. 저게 없으면 다음 사람이 "왜 굳이 분류 모델로 불러서 인코더만 빼지? 한 줄로 줄이자"라며 리팩터링해 버그를 되살립니다.

여기서 배울 것은 CED 모델의 특수 사정이 아닙니다. **머신러닝 코드는 조용히 틀린다**는 것입니다. 일반 프로그램은 틀리면 멈추거나 이상한 값을 뱉지만, 학습 코드는 틀린 채로 끝까지 잘 돌아가고 그럴듯한 숫자를 내놓습니다. 그래서 "돌아갔다"가 아니라 "의도한 게 실제로 로드/실행됐다"를 따로 확인해야 합니다. 이 스크립트의 확인 장치는 학습 시작 직후 백본 파라미터 수와 학습 대상 파라미터 수를 찍어보는 156-158줄입니다 — 숫자가 예상과 다르면 그 자리에서 알아챌 수 있습니다.

#### (b) 사전학습 인코더를 이 프로젝트 모델에 붙이는 방법

**왜 이렇게 짰나.** CED 인코더는 파형이 아니라 mel 스펙트로그램을 받습니다. 그런데 이 프로젝트의 Dataset은 파형을 주죠. 그래서 mel 변환을 **모델 안에 넣습니다.**

```python
# .autorun/train_ced_vio.py:71-95
        # identical to feature_extraction_ced.py defaults (verified against cached source)
        self.mel = AT.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=512,
                                     hop_length=160, n_mels=64, f_min=0, center=True)
        self.to_db = AT.AmplitudeToDB(top_db=120)
    ...
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # feature extraction is fixed
            feats = self.to_db(self.mel(waveform))          # (B, 64, T)
        out = self.ced(input_values=feats)                   # token sequence in .logits
        return out.logits.contiguous()                       # (B, N', 256)
```

세 가지를 보세요. 첫째, mel 파라미터가 "CED의 feature extractor 기본값과 동일"임을 **캐시된 원본 소스와 대조해 확인했다**고 주석이 못 박고 있습니다. 사전학습 모델은 학습 때 본 것과 다른 방식으로 만든 입력을 주면 성능이 무너집니다 — hop_length 하나만 달라도요. 둘째, mel 변환이 `torch` 연산이라 GPU 위에서 돌아가고, 학습 대상이 아니니 `torch.no_grad()`로 감쌉니다. 셋째, `.logits`라는 이름이지만 실제로 나오는 건 분류 점수가 아니라 **토큰 시퀀스** `(B, N', 256)`입니다 — 라이브러리가 붙인 이름에 속으면 안 되고, 주석이 그걸 명시해줍니다. 이 백본을 꽂는 부분은 두 줄입니다.

```python
# .autorun/train_ced_vio.py:152-155
    backbone = CEDRawBackbone(CED_ID, unfreeze_top_k=UNFREEZE_TOP_K)
    model = HarmModel(tax.num_classes, ModelConfig(backbone="passthrough",
                                                   backbone_out_dim=backbone.out_dim))
    model.backbone = backbone  # heads NEW-init at dim 256 (no 768-head warm-start possible)
```

`backbone="passthrough"`는 "백본 자리는 일단 비워둬"라는 뜻이고 다음 줄에서 실제 CED 백본을 대입합니다. 그 뒤의 MIL 어텐션 풀링과 헤드는 `HarmModel`이 만들어 둔 것을 그대로 씁니다 — 즉 **남의 인코더 + 내 풀링/헤드** 조합입니다. 주석이 솔직하게 적어둔 대로, 헤드는 새로 초기화됩니다. BEATs는 출력 차원이 768이고 CED는 256이라 이전 실험의 헤드를 물려받을 방법이 없기 때문입니다. 이건 비교의 공정성에 불리한 조건인데, 숨기지 않고 파일 맨 위 docstring에 "honest difference"라고 써둔 게 포인트입니다.

#### (c) 어디를 학습시키나

```python
# .autorun/train_ced_vio.py:78-86
    def _set_trainable(self):
        for p in self.ced.parameters():
            p.requires_grad = False
        blocks = self.ced.blocks
        for blk in blocks[len(blocks) - self.unfreeze_top_k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.ced.norm.parameters():
            p.requires_grad = True
```

일단 인코더 전체를 얼리고(`requires_grad = False`), **위쪽 4개 블록**과 마지막 norm만 다시 풀어줍니다. 왜 위쪽만이냐면, 트랜스포머의 아래쪽 층은 "소리의 일반적인 질감" 같은 범용 특징을 배우고 위쪽 층일수록 "무슨 소리인가"라는 과제-특화 표현을 배우기 때문입니다. 제가 가진 데이터는 사전학습 데이터에 비해 아주 작아, 전체를 풀면 아래층의 좋은 표현까지 소량 데이터에 맞춰 망가집니다(catastrophic forgetting). 몇 개를 풀지는 `UNFREEZE_TOP_K` 환경변수로 바꿉니다. `manages_own_freezing = True`(54-61줄 클래스 속성)도 함께 봐야 합니다. Trainer에는 커리큘럼 단계마다 백본을 얼렸다 녹였다 하는 로직이 있는데, 이 플래그가 "이 백본은 자기 동결을 스스로 관리하니 건드리지 마"라고 알려줍니다. 이게 없으면 Trainer가 4개만 풀어놓은 걸 전부 풀어버립니다.

#### (d) 이 스크립트가 읽는 환경변수

하드코딩된 경로가 거의 없습니다. 로컬에서도, Kaggle에서도, 다른 모델 ID로도 같은 파일이 돌아야 하기 때문입니다.

```python
# .autorun/train_ced_vio.py:42-51
CLIP_DIR = os.environ.get("CLIP_DIR", "data_dl/clips")
FEATURE_ROOT = os.environ.get("FEATURE_ROOT", "data_dl/features")
CED_ID = os.environ.get("CED_ID", "mispeech/ced-mini")
TAX_CFG = os.environ.get("TAXONOMY_CFG", str(_ROOT / "configs/data/classes_vio.yaml"))
TAG = os.environ.get("TAG", CED_ID.split("/")[-1])          # e.g. ced-mini
UNFREEZE_TOP_K = int(os.environ.get("UNFREEZE_TOP_K", "4"))
CKPT_DIR = os.environ.get("CKPT_DIR", f"./ckpt_{TAG.replace('-', '_')}_vio")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
SEED = int(os.environ.get("SEED", "42"))
OUT_PROBS = os.environ.get("OUT_PROBS", f"data_dl/artifacts/probs_{TAG.replace('-', '_')}.npz")
```

여기에 더해 `VIOLENCE_MANIFEST`, `GAMBLING_MANIFEST`(29-30줄), `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_GROUP` / `WANDB_RUN_ID`(132-139줄)를 읽고, `import`로 딸려 오는 `train_beats_finetune`의 `AUGMENT`, `CLASS_BALANCE`까지가 이 실험의 동작을 바꾸는 스위치 전부입니다. 모든 값에 기본값이 있으므로 아무것도 설정하지 않고 그냥 실행해도 채택된 설정 그대로 돌아갑니다 — 이게 재현 가능한 스크립트의 조건입니다.

#### (e) 중단·재개와 시간 가드

Kaggle 세션은 12시간이면 끊깁니다. 그래서 "끊기는 것"을 에러가 아니라 **정상 상황**으로 취급합니다.

```python
# .autorun/train_ced_vio.py:162-165, 185-189
    cfg = TrainConfig(device="auto", batch_size=8, grad_accum_steps=4, num_workers=2,
                      lr_heads=1e-4, lr_backbone=1e-5, layer_decay=1.0, warmup_pct=0.05, patience=8,
                      amp=use_bf16, amp_dtype="bf16", ckpt_dir=CKPT_DIR, seed=SEED,
                      curriculum=(CurriculumStage("finetune", EPOCHS, freeze_backbone=False, use_supcon=True),))
    ...
    res = trainer.fit(tl, vl, resume="auto")
    print(f"[{TAG}] status={res.status} best_val_mAP={res.best_metric:.3f}", flush=True)
    probs, labels = predict(trainer.model, tel, device)
    np.savez(OUT_PROBS, probs=probs, labels=labels)
```

**여기서 4장과 어긋나는 부분을 짚고 갑니다.** 4장에서 `Trainer`의 기본 커리큘럼(5에폭 백본 동결 → 45에폭 전체 학습)과 `layer_decay = 0.9`를 설명했는데, **채택 모델은 둘 다 쓰지 않습니다.** 위 코드가 `layer_decay=1.0`(층별 감쇠 끔)과 단일 스테이지(`CurriculumStage("finetune", EPOCHS, freeze_backbone=False, use_supcon=True)`)를 넘기고 있습니다. 즉 워밍업 없이 처음부터 백본을 열고 SupCon도 처음부터 켭니다. 라이브러리의 기본값이 곧 실제로 돌린 설정은 아니라는 뜻이고, **무엇을 돌렸는지 알려면 라이브러리가 아니라 이 스크립트를 봐야 합니다.**

`resume="auto"`는 `CKPT_DIR`에 이전 체크포인트가 있으면 이어서, 없으면 처음부터 시작하라는 뜻입니다. 시간 가드는 여기서 명시하지 않았는데, `TrainConfig`의 기본값이 `time_guard_hours: float = 11.0`(`src/training/config.py:51`)이라 11시간이 지나면 Trainer가 체크포인트를 저장하고 `status="time_guard"`로 빠져나옵니다. 다시 실행하면 그 지점부터 이어집니다. `amp_dtype="bf16"`도 그냥 고른 값이 아닙니다. 형제 스크립트에 이유가 적혀 있습니다.

```python
# scripts/train_beats_finetune.py:200-203
    # Precision: BEATs NaNs in fp16 (narrow range), so use bf16 on Ampere+ GPUs
    # (RTX 30xx/40xx, A100 — fp32 range, no overflow, ~2x faster than fp32) and fall
    # back to fp32 elsewhere (e.g. T4, which has no bf16). Never fp16 for this model.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
```

fp16은 표현 범위가 좁아 큰 값에서 오버플로가 나면 손실이 NaN이 되고 학습이 통째로 망가집니다. bf16은 fp32와 같은 지수 범위를 가져 그 사고가 안 나죠. 다만 오래된 GPU에는 bf16이 없으므로 `is_bf16_supported()`로 확인하고 없으면 fp32로 떨어집니다 — CED 스크립트 160줄도 똑같은 한 줄을 씁니다.

**여기서 조심할 것.** 저장하는 `.npz`는 다른 모델의 결과와 **같은 테스트 셋 순서**여야 의미가 있습니다. 그 보장 장치가 `_has_both`(98-99줄)로, `wav`뿐 아니라 `.npy` log-mel 특징의 존재까지 요구합니다. CED는 그 `.npy`를 쓰지도 않는데 말이죠. 이상해 보이지만 목적이 다릅니다 — BEATs 실험이 `.npy`가 있는 클립만 썼으니, 비교 대상인 CED도 **똑같은 필터**를 통과한 똑같은 클립 집합으로 평가해야 두 `.npz`의 행이 1:1로 대응하기 때문입니다.

### 3. `.autorun/quantize_ced.py` — 양자화는 정말 몇 줄이다

**무엇을 하나.** 학습이 끝난 CED-mini 체크포인트를 불러와 CPU에서 fp32로 한 번, int8로 양자화해서 또 한 번 추론하고, 크기·속도·정확도를 나란히 찍습니다. 두 결과 모두 `.npz`로 남깁니다. 핵심은 딱 한 줄입니다.

```python
# .autorun/quantize_ced.py:87-91
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    int8_mb = _size_mb(q)
    print(f"[quant-ced] int8 ({int8_mb:.0f} MB); int8 eval (CPU)…", flush=True)
    p8, _, dt8 = _infer(q, tel)
    np.savez("data_dl/artifacts/probs_ced_mini_int8.npz", probs=p8, labels=y)
    ...
# .autorun/quantize_ced.py:96-99
    print(f" size:  fp32 {fp32_mb:.0f} MB -> int8 {int8_mb:.0f} MB  ({fp32_mb/max(int8_mb,1e-9):.1f}x)")
    print(f" cpu:   fp32 {dt32:.0f}s -> int8 {dt8:.0f}s ({len(te)} clips)")
    print(f" any-vio AP:  fp32 {a32:.3f} -> int8 {a8:.3f}  (Δ {a8-a32:+.3f})")
    print(f" mean|Δprob|: {float(np.abs(p32-p8).mean()):.4f}")
```

**왜 이렇게 짰나.** "동적 양자화(dynamic PTQ)"는 이름만 거창합니다. 모델 안의 `nn.Linear` 레이어들의 **가중치를** 32비트 실수에서 8비트 정수로 바꿔 저장합니다. 가중치가 1/4 크기가 되니 모델 파일이 작아지고, 정수 연산이라 CPU에서 더 빠릅니다. "동적"인 이유는 **활성값(입력으로 들어오는 값)** 은 미리 정하지 않고 추론할 때마다 그 배치의 범위를 보고 즉석에서 스케일을 정하기 때문입니다. 그래서 별도의 보정(calibration) 데이터가 필요 없고, 학습을 다시 할 필요도 없습니다 — 그래서 한 줄입니다.

건드리지 **않는** 것도 알아야 합니다. `{torch.nn.Linear}`라고 명시했으니 Linear만 대상이고, Conv 레이어·LayerNorm·어텐션의 softmax·앞단의 mel 변환은 전부 fp32로 남습니다. 그래서 "int8로 바꿨는데 크기가 왜 1/4이 안 되지?"라는 결과가 나옵니다 — 정상입니다. 또 활성값을 실시간으로 재는 오버헤드가 있어 속도 향상도 이론치인 4배와는 다릅니다. 그래서 위 블록 뒷부분처럼 이 스크립트는 추측하지 않고 크기·시간·정확도를 **실제로 재서 출력합니다.** 특히 `mean|Δprob|`을 같이 찍는 게 좋은 습관입니다. AP 같은 종합 점수는 같아 보여도 개별 확률값은 크게 흔들릴 수 있는데, 임계값을 걸어 알림을 띄우는 시스템에서는 **개별 확률의 흔들림**이 실제 오동작을 만들기 때문입니다.

**여기서 조심할 것.** int8 연산은 CPU 백엔드에 따라 이름이 다릅니다. x86은 `fbgemm`, ARM(휴대폰, Apple Silicon)은 `qnnpack`이라, 하나를 하드코딩하면 다른 기계에서 실패합니다. 그래서 실행 시점에 골라 씁니다.

```python
# .autorun/quantize_ced.py:55-61
def _pick_quantized_engine() -> str:
    """fbgemm (x86) is preferred; Apple Silicon / ARM only ships qnnpack."""
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("fbgemm", "qnnpack"):
        if engine in supported:
            return engine
    raise RuntimeError(f"no int8 engine available; supported={supported}")
```

쓸 수 있는 게 하나도 없으면 조용히 넘어가지 않고 `RuntimeError`로 죽습니다 — 잘못된 값으로 계속하느니 즉시 멈추는 쪽입니다. fp32 추론도 **CPU에서** 하는 게 중요합니다. int8은 CPU에서만 돌아가므로, GPU fp32와 CPU int8을 비교하면 하드웨어 차이가 섞여 무엇 때문에 느려졌는지 알 수 없게 됩니다.

### 4. `.autorun/train_beats_vio.py` — 비교 대상이 된 BEATs 쪽 실험

**무엇을 하나.** CED와 똑같은 태스크(폭력 4클래스)를 BEATs 백본으로 학습합니다. CED 스크립트와 구조가 거의 판박이인데, 결정적으로 다른 게 두 가지 있습니다. 첫째, 23클래스 시절의 강한 헤드를 **이름 기준으로 옮겨 심습니다.**

```python
# .autorun/train_beats_vio.py:68-86
def _warmstart_head(num_classes, tax, device):
    """HarmModel(num_classes) warm-started from the 23-class head_ckpt: shape-matching
    params load directly; the final classifier layer (classifier.net.3) is remapped
    row-by-row by CLASS NAME (v1.0 index -> v2 row). Backbone swapped to BEATs after."""
    ...
    for k, v in new_sd.items():
        if k in (HEAD_W, HEAD_B):
            src = old[k]
            for j, name in enumerate(tax.all_classes):
                v[j] = src[old_tax.index_of(name)]
            remapped += 1
        elif k in old and old[k].shape == v.shape:
            new_sd[k] = old[k]; loaded += 1
    ...
```

**왜 이렇게 짰나.** 주목할 것은 `old_tax.index_of(name)`입니다. 23클래스 분류기의 몇 번째 행이 `vio_gunshot`이었는지를 **인덱스가 아니라 이름으로** 찾습니다. 인덱스로 옮기면 나중에 클래스 목록 순서가 한 칸만 바뀌어도 총성 가중치가 비명 자리에 들어가고, 역시 아무 에러 없이 학습이 잘 돌아갑니다 — 앞에서 본 "조용히 틀리는" 유형이 반복됩니다. 그래서 옮긴 개수를 세어 출력합니다(87-88줄): `loaded 몇 개, remapped 몇 개`가 예상과 다르면 바로 알아챌 수 있습니다. 둘째, 백본 깊이를 줄이는 스위치가 있습니다.

```python
# .autorun/train_beats_vio.py:47-49
_ul = os.environ.get("USE_LAYERS", "").strip()
USE_LAYERS = int(_ul) if _ul else None
TAG = f"L{USE_LAYERS}" if USE_LAYERS else "full"
```

`USE_LAYERS=6`으로 주면 12층 BEATs의 아래 6층만 쓰고, 결과물은 `probs_beats_vio_L6.npz`로 저장됩니다. `TAG`가 체크포인트 폴더와 결과 파일 이름에 자동으로 붙기 때문에, 환경변수만 바꿔 여러 번 돌려도 결과가 서로 덮어쓰이지 않습니다. `_strip`은 CED 쪽과 같은 코드인데, 여기 주석이 이유를 더 잘 설명합니다.

```python
# .autorun/train_beats_vio.py:61-65
def _strip(records, tax):
    """Keep only labels present in the (violence-only) taxonomy; others -> [] (all-zero
    negative). ClipRecord is a mutable dataclass, but replace() keeps it clean."""
    keep = set(tax.all_classes)
    return [replace(r, labels=[l for l in r.labels if l in keep]) for r in records]
```

도박 클립을 데이터에서 **빼는** 게 아니라 "아무것도 아닌 것(전부 0)"으로 바꿔 남깁니다. 그래야 모델이 "폭력 vs 그 외 전부"라는, 실제 배포에서 필요한 판단을 배웁니다. `replace()`를 쓴 건 원본 레코드를 제자리에서 고치지 않기 위해서입니다 — 같은 리스트를 다른 곳에서도 참조한다면 제자리 수정은 추적 불가능한 버그가 됩니다.

**여기서 조심할 것.** 이 파일에서 딱 두 줄, `BEATS_CKPT = os.environ["BEATS_CKPT"]`와 `HEAD_CKPT = os.environ["HEAD_CKPT"]`(43-44줄)만 `.get()`이 아닌 대괄호입니다. 기본값이 없으면 바로 `KeyError`로 죽습니다. 의도적입니다 — 사전학습 가중치 경로는 사람마다 다르고, 여기에 엉뚱한 기본값을 두면 "가중치 없이 학습됐는데 아무도 모르는" 사태가 납니다. **잘못된 값으로 조용히 계속하느니 시작하자마자 죽는 게 낫습니다.**

또 하나, `_val_metrics`(101-118줄)가 왜 따로 있는지도 봐두세요. docstring이 이유입니다: *"Taxonomy-invariant val metrics so wandb curves are COMPARABLE across runs (23-class vs 4-class, L6 vs L8 vs full)"*. 클래스 개수가 다른 실험끼리 `val_mAP`를 비교하는 건 의미가 없습니다(23개 평균과 4개 평균은 애초에 다른 수치니까요). 그래서 어떤 실험이든 똑같이 계산되는 "폭력 중 하나라도 맞췄나" 지표를 따로 만들어 로그에 남깁니다. 이 지표가 곧 배포에서 트리거가 울리는 기준이기도 합니다. CED 스크립트에도 같은 함수가 복사돼 있습니다.

### 5. `.autorun/dump_beats_probs.py` — 채점표를 파일로 남긴다

**무엇을 하나.** 35줄짜리, 이 절에서 가장 짧은 파일입니다. 이미 학습된 BEATs 체크포인트를 불러 테스트 셋을 추론하고, 예측 확률과 정답 라벨을 `.npz` 하나로 저장합니다. 그게 전부입니다.

```python
# .autorun/dump_beats_probs.py:25-34
tax = load_taxonomy()
_, _, te = CD.build_combined_records(exists_fn=_has_both)
loader = DataLoader(RawAudioDataset(te, tax, PreprocessConfig(), train=False), batch_size=8)
device = resolve_device("auto")
model = build_finetune_model(tax.num_classes, head_ckpt=None,
                             beats_ckpt=os.environ["BEATS_CKPT"], unfreeze_top_k=4)
model.load_state_dict(torch.load("ckpt_p2_full/best.ckpt", map_location="cpu", weights_only=False)["model"])
...
probs, labels = predict(model, loader, device)
np.savez("data_dl/artifacts/probs_beats.npz", probs=probs, labels=labels)
```

**왜 이렇게 짰나.** 여기가 이 저장소 실험 설계의 요령입니다. 모델 비교를 하려면 보통 "두 모델을 불러 나란히 돌리는 비교 스크립트"를 떠올리는데, 그러면 비교할 때마다 GPU를 잡고 추론을 다시 해야 하고 모델이 하나 늘 때마다 그 스크립트를 고쳐야 합니다. 대신 각 모델이 **자기 채점표를 파일로 남기게** 합니다. `.npz` 안에는 `probs`(각 클립에 대한 예측 확률)와 `labels`(정답)만 들어 있습니다 — 모델도, 가중치도, GPU도 필요 없는 그냥 숫자 배열 두 개입니다. 그러면 채점하는 쪽은 파일 목록만 들고 있으면 됩니다.

```python
# .autorun/compare_vio.py:21-29
MODELS = {  # label -> npz  (only those that exist are used)
    "full-fp32":          "data_dl/artifacts/probs_beats_fp32.npz",
    ...
    "CED-mini(10M,vio)":  "data_dl/artifacts/probs_ced_mini.npz",
    "CED-mini-int8(10MB)": "data_dl/artifacts/probs_ced_mini_int8.npz",
}
BASELINE = "full-fp32"  # student/int8 Δ vs teacher fp32
```

새 모델을 비교에 넣으려면 줄 하나 추가하면 끝이고, 통계 비교 — 부트스트랩 신뢰구간이나 짝지은 차이(paired Δ) — 는 확률 배열만 있으면 되므로 **GPU 없이 몇 초 만에, 몇 번이든 다시** 돌릴 수 있습니다. 이 절의 스크립트들이 하나같이 마지막에 예측을 `.npz`로 떨구는 이유가 여기 있습니다.

**여기서 조심할 것.** 이 방식이 성립하려면 모든 `.npz`의 **행 순서가 정확히 같아야** 합니다. 짝지은 비교는 "3번째 클립에 대해 모델 A와 모델 B가 각각 뭐라 했나"를 보는 것이라, 순서가 어긋나면 결과가 통째로 거짓말이 됩니다. 그래서 이 짧은 파일에도 주석이 붙어 있습니다.

```python
# .autorun/dump_beats_probs.py:20-23
# Require BOTH wav (BEATs input) AND log-mel feature, so the test set matches the baselines
# exactly (precompute drops ~237 near-silent clips per spec §4) -> identical clip order -> arrays align.
def _has_both(cid):
    return _has_clip(cid) and os.path.exists(f"{FEATURE_ROOT}/{cid}.npy")
```

앞서 CED 스크립트에서 봤던 그 `_has_both`와 같은 조건입니다. 여러 스크립트에 걸쳐 같은 필터가 반복되는 건 중복이 아니라 **불변식(invariant)** 입니다. 다만 그 불변식이 문서가 아니라 코드 복사로 유지된다는 건 이 저장소의 약점이기도 합니다 — 누군가 한 파일에서만 조건을 바꾸면 그날부터 모든 비교표가 조용히 틀리기 시작합니다. 마지막으로 `dump_beats_probs.py`에는 `main()`도 인자 파서도 없이 모듈 최상단에서 코드가 그냥 실행됩니다. 한 번 쓰고 결과 파일만 남기면 되는 스크립트라 그렇습니다. 실험 코드에서 이런 파일은 "지저분한" 게 아니라 목적에 맞는 것입니다. 다만 저 `"ckpt_p2_full/best.ckpt"`처럼 경로가 하드코딩된 지점은, 왜 다른 파일들은 전부 환경변수로 빼놨는지를 거꾸로 보여주는 대비이기도 합니다.

---

## 6. 1층 상시 게이트 — 지식 증류

24시간 내내 꺼지지 않고 돌아야 하는 모델은 무조건 작아야 하는데, 작은 모델을 정답 라벨(0/1)만 보고 처음부터 학습시키면 큰 모델만큼의 감을 절대 못 잡습니다. 그래서 저는 작은 모델에게 정답 대신 **큰 모델이 내놓은 확률값 자체를 베끼게** 했고, 이걸 지식 증류(knowledge distillation)라고 부릅니다.

여기서 큰 모델을 교사(teacher), 작은 모델을 학생(student)이라고 합니다.

---

### `distill/dump_teacher_targets.py` — 교사의 답안지를 한 번만 뽑아서 얼려 둡니다

**무엇을 하나.** 파일 맨 위 독스트링이 이 스크립트의 존재 이유를 그대로 설명합니다.

```python
# distill/dump_teacher_targets.py:1-7
"""Dump BEATs teacher soft targets for distillation. For every clip (train/val/test) runs
the fine-tuned full BEATs and saves the 4 violence LOGITS + 256-d projection embedding +
the hard violence label. Order matches combined_data records; clip_ids saved for keyed lookup.

Env: FULL_CKPT (default ckpt_beats_finetune_top4/best.ckpt), BEATS_CKPT, CLIP_DIR.
Out: distill/teacher_targets_{train,val,test}.npz  (clip_ids, vio_logits[N,4], emb[N,256], hard[N,4])
"""
```

즉 train/val/test **모든 클립**에 대해 파인튜닝이 끝난 큰 BEATs 모델을 한 번씩 돌리고, 그 결과를 `.npz` 파일 세 개로 디스크에 저장합니다.

실제로 저장되는 게 뭔지가 중요합니다. 확률만 저장하는 게 아니라 **네 가지**를 같이 저장합니다.

```python
# distill/dump_teacher_targets.py:37-52
@torch.no_grad()
def _dump(model, records, tax, vi, cfg_pp, device, name):
    loader = DataLoader(RawAudioDataset(records, tax, cfg_pp, train=False), batch_size=8, shuffle=False)
    logits, embs = [], []
    for x, _ in loader:
        out = model(x.to(device), return_projection=True)
        logits.append(out["logits"][:, vi].float().cpu().numpy())
        embs.append(out["embeddings"].float().cpu().numpy())
    logits = np.concatenate(logits); embs = np.concatenate(embs)
    hard = np.stack([tax.encode(r.labels)[vi] for r in records]).astype(np.float32)
    cids = np.array([r.clip_id for r in records])
    path = OUT / f"teacher_targets_{name}.npz"
    np.savez(path, clip_ids=cids, vio_logits=logits, emb=embs, hard=hard)
    ...
```

정리하면 한 클립당 이렇게 저장됩니다.

- `vio_logits` — 폭력 4개 클래스에 대한 교사의 **로짓**(시그모이드 통과 전 원본 점수) `[N, 4]`
- `emb` — 교사 내부의 256차원 **임베딩**(사영 벡터) `[N, 256]`
- `hard` — 사람이 붙인 원래 정답 라벨 `[N, 4]`
- `clip_ids` — 위 세 배열의 각 행이 어느 클립인지 알려주는 이름표

즉 **확률(로짓)과 임베딩을 둘 다** 저장합니다. 확률만 베끼는 게 아니라 "교사가 이 소리를 머릿속에서 어떤 벡터로 표현했는지"까지 베끼게 하려는 것이고, 그게 뒤에 나올 feature matching 항의 재료가 됩니다. 폭력 4개 클래스는 `VIO = ["vio_scream", "vio_impact", "vio_gunshot", "vio_verbal"]`(29행)로 고정됩니다.

**왜 이렇게 짰나.** 핵심은 **2단계 분리**입니다. 교사 모델은 무겁습니다. 만약 학생을 학습시키는 루프 안에서 매 배치마다 교사도 같이 돌린다면, 40 에폭을 돌 때 교사를 40번 다시 돌리는 셈입니다. 교사는 학습되지 않고 가중치가 고정되어 있으므로 같은 클립에 대한 답은 매번 똑같습니다. 똑같은 계산을 40번 하는 건 순수한 낭비입니다.

그래서 여기서 교사를 **딱 한 바퀴** 돌리고 결과를 디스크에 얼려 둡니다. 이후 학생 학습은 교사 비용을 두 번 다시 지불하지 않고, `.npz`에서 숫자를 읽기만 합니다.

교사를 고정 상태로 쓴다는 건 코드에도 드러납니다. `@torch.no_grad()`(그래디언트 안 만듦), `.eval()`(BatchNorm/Dropout 평가 모드), 그리고 학습 루프가 아예 없다는 점입니다.

```python
# distill/dump_teacher_targets.py:61-63
    model = build_finetune_model(tax.num_classes, head_ckpt=None, beats_ckpt=BEATS_CKPT, use_layers=None)
    model.load_state_dict(torch.load(FULL_CKPT, map_location="cpu", weights_only=False)["model"], strict=True)
    model.to(device).eval()
```

또 하나, 데이터를 고를 때 클립 오디오와 특징 파일이 **둘 다** 있는 것만 씁니다(`_has_both`, 33-34행). 이 함수를 `build_combined_records`에 넘겨(59행) 목록을 만드는데, 나중에 학생 학습 스크립트도 **똑같은 함수**를 씁니다. 그래야 두 스크립트가 보는 클립 집합이 정확히 일치합니다.

**여기서 조심할 것.** - **`clip_ids`를 같이 저장한다는 점**이 생각보다 중요합니다. 독스트링에는 "Order matches combined_data records"라고 적혀 있지만, 순서만 믿고 인덱스로 붙이면 나중에 매니페스트가 한 줄이라도 바뀌는 순간 전 클립의 정답이 한 칸씩 밀립니다. 이름표를 같이 저장해 두면 이름으로 찾아 붙일 수 있습니다.
- 여기서 저장하는 건 **확률이 아니라 로짓**입니다. 시그모이드를 통과하지 않은 원본 점수여야 나중에 온도(temperature)로 나눠서 부드럽게 만드는 조작이 가능합니다. 확률로 저장해 버리면 그 정보가 이미 뭉개진 뒤입니다.
- 증강(augmentation)을 끄고 뽑습니다(`train=False`). 교사의 답은 "깨끗한 원본 클립"에 대한 답이어야 하기 때문입니다.

---

### `distill/student_models.py` — 작은 학생 모델

**무엇을 하나.** 교사와 **똑같은 16 kHz 원본 파형**을 입력으로 받아서, 내부에서 스스로 log-mel을 계산하고, 256차원 임베딩과 폭력 4클래스 로짓을 내놓는 작은 CNN입니다.

```python
# distill/student_models.py:1-9
"""Tiny student models for BEATs distillation (violence trigger, always-on candidate).

Student takes the SAME raw 16 kHz waveform as the BEATs teacher (alignment), computes its
own log-mel internally, and emits a 256-d embedding (for feature distillation vs the
teacher's projection) + 4 violence logits. Target ~1-3M params so it can run continuously
on a low-power core / serve as the on-device gate.

Kept deliberately simple and self-contained in distill/ (no dependency on src/ model code).
"""
```

모델 본체는 이렇게 생겼습니다.

```python
# distill/student_models.py:40-63
class TinyMelCNN(nn.Module):
    """~1-2M param log-mel CNN. forward -> {"logits": (B, C), "embeddings": (B, emb_dim)}.

    emb_dim matches the teacher projection (256) so feature distillation needs no adapter.
    The loss itself is cosine distance, not MSE — see train_distill.py:155."""

    def __init__(self, num_classes: int = 4, n_mels: int = 64, widths=(32, 64, 128), emb_dim: int = 256):
        super().__init__()
        self.logmel = LogMel(n_mels=n_mels)
        self.bn_in = nn.BatchNorm2d(1)
        chans = [1, *widths]
        self.features = nn.Sequential(*[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(widths[-1], emb_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, wav: torch.Tensor, return_projection: bool = True) -> dict:
        x = self.bn_in(self.logmel(wav))
        x = self.features(x)
        x = self.pool(x).flatten(1)             # (B, widths[-1])
        emb = self.proj(x)                      # (B, emb_dim)
        out = {"logits": self.classifier(emb)}
        if return_projection:
            out["embeddings"] = emb
        return out
```

한 덩어리(`_block`, 33-37행)는 아주 교과서적입니다. 3x3 합성곱 두 번 + BatchNorm + ReLU, 그리고 마지막에 MaxPool로 크기를 절반으로 줄입니다.

**왜 이렇게 짰나.** 모델의 크기를 결정하는 손잡이는 사실상 **`widths` 튜플 하나**입니다. `widths=(32, 64, 128)`이면 블록 3개짜리 CNN이고, 채널 수가 1 → 32 → 64 → 128로 늘어납니다.

CNN에서 파라미터 수는 채널 폭이 지배합니다. 3x3 합성곱 하나의 가중치 개수는 대략 `3 x 3 x (입력 채널) x (출력 채널)`입니다. 즉 **입력·출력 채널의 곱**에 비례하므로, 모든 채널 폭을 2배로 키우면 파라미터는 약 4배가 됩니다. 커널 크기(3x3)나 블록 개수는 이에 비하면 손잡이가 훨씬 둔합니다. 그래서 크기 실험을 할 때 저는 `widths`만 바꿉니다.

`emb_dim=256`이 기본값인 이유는 순전히 **교사와 맞추기 위해서**입니다. 교사가 256차원 임베딩을 뱉으니 학생도 256차원이어야 두 벡터를 직접 비교할 수 있습니다.

`AdaptiveAvgPool2d(1)`은 주파수·시간 축을 통째로 평균 내서 `(B, widths[-1])` 벡터 하나로 만듭니다. 이렇게 하면 입력 오디오 길이가 조금 달라져도 뒤쪽 Linear 층의 크기가 바뀌지 않습니다. 파라미터 수를 세는 `num_params()` 헬퍼도 들어 있습니다(65-66행).

**여기서 조심할 것.** - `LogMel`은 학습되지 않는 고정 변환입니다(독스트링에 "Fixed, non-trainable"). 로그를 씌우기 전에 `clamp_min(1e-6)`으로 바닥을 깔아두는데(29행), 이게 없으면 무음 구간에서 `log(0) = -inf`가 나와 학습이 즉시 터집니다.
- 이 독스트링은 원래 `"feature distillation is a plain MSE"`라고 적혀 있었습니다. 실제 학습 스크립트는 MSE가 아니라 코사인 거리를 쓰는데도요. **이 문서를 쓰면서 발견해서 고쳤습니다.** 주석이 옛 설계 그대로 남아 코드와 어긋나는 일은 실제 저장소에서 아주 흔합니다. 손실 함수의 진실은 언제나 손실을 계산하는 파일(`train_distill.py`)에 있습니다.
- `forward`가 텐서가 아니라 **딕셔너리**를 반환한다는 점에 주의하세요. 교사 모델과 반환 형식(`{"logits": ..., "embeddings": ...}`)을 똑같이 맞춰 둬서, 학습 코드가 교사·학생을 같은 방식으로 다룰 수 있습니다.

---

### `distill/train_distill.py` — 세 개의 손실로 학생을 가르칩니다

**무엇을 하나.** 앞의 두 파일을 합칩니다. 디스크에 얼려 둔 교사의 답안지를 읽어서, `TinyMelCNN` 학생이 그것을 따라 하도록 학습시킵니다.

```python
# distill/train_distill.py:1-7
"""Distill BEATs -> TinyMelCNN student (violence trigger). Trains the student to match the
teacher's soft violence logits (dark knowledge) + projection embedding, with a small hard-label
term. Same combined_data split; teacher targets from dump_teacher_targets.py (keyed by clip_id).
Dumps test probs (same order as probs_beats_fp32.npz) for .autorun/compare_vio.py.

Env: TAG (default s1), EPOCHS, SEED, ALPHA/BETA/GAMMA/TEMP (loss weights), LR, BATCH, WANDB_* .
"""
```

**크기 후보 세 개**가 딕셔너리 하나로 정의되어 있습니다.

```python
# distill/train_distill.py:32-36
# size sweep presets: SIZE -> (conv widths, emb dim) ~ {s1:0.32M, s2:~0.9M, s3:~2.8M}.
# emb dim MUST equal the teacher projection dim (256) — feature distillation compares them directly.
_PRESET = {"s1": ((32, 64, 128), 256), "s2": ((56, 112, 224), 256), "s3": ((100, 200, 400), 256)}
SIZE = os.environ.get("SIZE", "s1")
WIDTHS, EMB = _PRESET.get(SIZE, _PRESET["s1"])
```

코드 주석에 적힌 파라미터 추정치는 **s1이 약 0.32M, s2가 약 0.9M, s3가 약 2.8M**입니다. 세 프리셋의 `widths`를 보면 폭이 대략 1.75배씩 커지는데, 앞에서 말한 "폭의 제곱" 규칙대로 파라미터는 약 3배씩 뜁니다(0.32 → 0.9 → 2.8). 임베딩 차원은 세 후보 모두 256으로 고정인데, 주석에 대문자로 강조된 대로 교사의 사영 차원과 반드시 같아야 하기 때문입니다.

손실 가중치들은 전부 환경변수로 조절 가능하되 기본값이 코드에 박혀 있습니다.

```python
# distill/train_distill.py:40-43
ALPHA = float(os.environ.get("ALPHA", "1.0"))   # soft
BETA = float(os.environ.get("BETA", "1.0"))     # feature
GAMMA = float(os.environ.get("GAMMA", "0.3"))   # hard
TEMP = float(os.environ.get("TEMP", "2.0"))
```

#### 핵심 — 세 항짜리 손실 함수

이 파일의 심장은 학습 루프 안의 이 여섯 줄입니다.

```python
# distill/train_distill.py:151-158
            out = student(wav, return_projection=True)
            soft_t = torch.sigmoid(tlog / TEMP)
            # soft: temperature-softened BCE (no T^2 — keep ~O(0.5) so the 3 terms are comparable).
            l_soft = F.binary_cross_entropy_with_logits(out["logits"] / TEMP, soft_t)
            # feat: cosine distance (O(0..2)) — MSE-over-256 was ~0.008 and drowned out.
            l_feat = (1.0 - F.cosine_similarity(out["embeddings"], temb, dim=1)).mean()
            l_hard = F.binary_cross_entropy_with_logits(out["logits"], hd)
            loss = ALPHA * l_soft + BETA * l_feat + GAMMA * l_hard
```

**1항 `l_soft` (가중치 ALPHA=1.0, 온도 TEMP=2.0) — 부드러운 정답 베끼기.**

교사의 로짓 `tlog`를 온도 2로 나눈 뒤 시그모이드를 씌워 `soft_t`를 만들고, 학생의 로짓도 똑같이 2로 나눠서 이진 크로스엔트로피를 겁니다.

왜 정답 라벨(0/1) 대신 이 확률을 베끼게 할까요? 정답 라벨은 "이건 폭력이다"라는 사실 하나만 알려줍니다. 반면 교사가 "폭력 0.7 / 기타 0.2"라고 말했다면, 그 안에는 **교사가 이 소리를 무엇과 헷갈렸는지**가 들어 있습니다. 어떤 비명이 0.9가 아니라 0.7이라는 건 "이건 애매한 비명이다", 다른 클래스에 0.2가 붙어 있다는 건 "이 둘은 소리가 비슷하다"는 뜻입니다. 이런 클래스 간 유사도 정보를 힌튼(Hinton)의 표현을 빌려 **dark knowledge**라고 부르고, 이 파일 독스트링에도 그 단어가 그대로 쓰여 있습니다. 0/1 라벨에는 이 정보가 전혀 없습니다.

온도 `TEMP`로 나누는 이유는 이 정보를 **더 잘 보이게 만들기** 위해서입니다. 로짓을 큰 수로 나누면 확률이 0.5 쪽으로 눌리면서, 원래 0.01 대 0.001처럼 거의 안 보이던 미세한 차이가 눈에 띄는 크기로 벌어집니다. 그 차이가 바로 학생이 배워야 할 부분입니다.

주석의 `no T^2`도 짚고 넘어갈 만합니다. 원 논문에서는 온도로 나누면 그래디언트가 `1/T^2`만큼 줄어드니 손실에 `T^2`을 곱해 보정하라고 하는데, 여기서는 일부러 곱하지 않았습니다. 이유가 주석에 있습니다 — 세 항의 크기를 비슷하게(`~O(0.5)`) 유지해서 가중치를 직관적으로 다루기 위해서입니다.

**2항 `l_feat` (가중치 BETA=1.0) — 임베딩 방향 맞추기.**

`F.cosine_similarity`는 두 벡터가 이루는 각도의 코사인값입니다. 완전히 같은 방향이면 1, 직각이면 0, 반대면 -1이므로 `1 - cos`은 0에서 2 사이의 거리가 됩니다.

여기서 중요한 건 코사인이 **방향만 보고 크기(길이)는 안 본다**는 점입니다. 학생의 임베딩이 교사 임베딩의 절반 길이여도, 방향만 같으면 이 손실은 0입니다. 이게 합리적인 이유는, 임베딩에서 의미를 담는 건 "어느 쪽을 가리키는가"이지 "얼마나 긴가"가 아니기 때문입니다. 크기까지 억지로 맞추라고 하면 학생은 의미와 무관한 스케일 맞추기에 용량을 낭비합니다.

MSE를 먼저 시도했다가 갈아탄 흔적이 155행 주석에 그대로 남아 있습니다 — `# feat: cosine distance (O(0..2)) — MSE-over-256 was ~0.008 and drowned out.` 256개 원소에 대해 제곱오차를 평균 내면 값이 0.008 수준으로 아주 작아지는데, 다른 두 항이 0.5 부근이니 합칠 때 사실상 무시됐다("drowned out")는 뜻입니다. 손실을 여러 개 더할 때는 **각 항의 크기 스케일을 맞춰 주는 게 가중치 못지않게 중요하다**는 실전 교훈입니다.

**3항 `l_hard` (가중치 GAMMA=0.3) — 진짜 정답도 조금은 봅니다.**

사람이 붙인 원래 라벨 `hd`에 대한 평범한 BCE입니다. 가중치가 0.3으로 다른 두 항보다 작습니다. 교사도 틀릴 수 있으므로 교사만 100% 따라 하면 교사의 실수까지 그대로 물려받습니다. 진짜 정답을 약하게 섞어 학생을 땅에 붙들어 두는 역할입니다.

#### 체크포인트를 무엇으로 고르나

매 에폭 끝에서 검증 성능을 재고, 가장 좋았던 시점의 가중치만 저장합니다. 어떤 지표로 고르는지가 주석과 함께 명시되어 있습니다.

```python
# distill/train_distill.py:162-171
        m, _, _ = _eval(student, vl, device)
        key = m["anyvio_ap"]   # select on val AP: R@FPR on a 74-positive val is too noisy
                               # (it saved an early undertrained ckpt for s3 -> false regression).
        if key > best + 1e-6:
            best, no_imp = key, 0; torch.save({"model": student.state_dict(), "tag": TAG}, CKPT)
        else:
            no_imp += 1
        ...
```

선택 지표는 **검증셋의 anyvio AP**(Average Precision, 임계값을 모든 값으로 훑으면서 정밀도-재현율 곡선 아래 면적을 잰 값)입니다. `_eval`은 AP 말고도 `recall@fpr1/5/10`을 같이 계산하지만(103-107행), 그건 **로그로만** 찍고 선택에는 쓰지 않습니다.

이 선택이 왜 중요한가가 주석에 실패 사례로 적혀 있습니다. 검증셋 양성이 74개뿐이라 "FPR 1%일 때의 재현율" 같은 지표는 임계값 근처 샘플 한두 개가 넘나드는 것만으로 값이 크게 튑니다. 노이즈가 큰 지표로 최고 에폭을 고르면 **운 좋게 튄 초기 에폭**이 저장되고, 그 덜 학습된 체크포인트로 최종 평가를 하니 s3(가장 큰 모델)가 오히려 나빠 보이는 가짜 퇴보("false regression")가 발생했습니다. AP는 전체 순위를 다 보는 지표라 훨씬 덜 흔들립니다.

학습이 끝나면 마지막 에폭 모델이 아니라 저장해 둔 최고 체크포인트를 **다시 불러온 뒤**(`student.load_state_dict(torch.load(CKPT, ...)["model"])`, 178행) 테스트셋 확률을 뽑아 `.npz`로 저장합니다. 이 한 줄이 빠지는 실수가 의외로 흔합니다.

**여기서 조심할 것.** - **교사 답안지를 붙이는 방식.** `_load_targets`가 `.npz`를 열어 배열 3개와 `clip_id -> 행 번호` 맵을 만들고(60-63행), `DistillDataset.__getitem__`이 그 맵으로 해당 행을 찾습니다. 순서에 기대지 않고 이름으로 찾는다는 점이 안전장치입니다.
- **메모리 함정 두 개**가 주석으로 박제되어 있습니다. 하나는 `.npz`를 매번 인덱싱하면 캐시가 없어 반복 압축 해제가 일어나 OOM으로 죽었다는 것(56-59행), 다른 하나는 numpy 배열을 정수 인덱싱하면 **뷰(view)**가 반환되어 전체 배열이 메모리에 붙잡힌다는 것입니다. 그래서 `__getitem__`에서 `.copy()`를 명시적으로 부릅니다.

  ```python
  # distill/train_distill.py:81-85
          # .copy() detaches the row from the big shared array so torch.from_numpy / the
          # collated batch never pins the full (N x D) array (esp. across DataLoader workers).
          return (wav, torch.from_numpy(self.tl[j].copy()),
                  torch.from_numpy(self.te[j].copy()),
                  torch.from_numpy(self.hd[j].copy()))
  ```

- **증강을 쓰지 않습니다.** `DistillDataset` 독스트링에 이유가 있습니다 — "No aug in v1 (teacher target is for the clean clip)". 교사의 답은 깨끗한 원본에 대한 답인데 학생 입력만 흔들어 버리면, 학생은 "흔들린 소리에 대해 깨끗한 소리의 답을 내라"는 잘못된 문제를 풀게 됩니다.
- **`RawAudioDataset`을 23클래스 taxonomy로 만들면서 그 라벨은 버립니다.** 같은 독스트링에 "23-class tax just for its loader; its label is ignored"라고 적혀 있고, `__getitem__`에서 `wav, _ = self.base[i]`로 두 번째 반환값을 버리는 게 보입니다(79행). 기존 로더를 재활용하되 라벨은 교사 파일에서 온 것을 쓴다는 뜻입니다.
- **조기 종료(early stopping).** 8 에폭 연속으로 검증 AP가 개선되지 않으면 멈춥니다(175-176행). 최대 40 에폭이지만 대개 그 전에 끝납니다.

---

## 7. 말 갈래 — 받아쓰고, 그 문장이 유해한지 판단하기

소리 갈래가 "무슨 소리인가"를 듣는다면, 말 갈래는 "무슨 말을 했는가"를 읽습니다. 비명이나
총소리는 파형에 흔적을 남기지만 "밤길 조심해라"는 남기지 않습니다. 그래서 음성을 글자로
받아쓴 다음(ASR), 그 문장을 분류합니다.

이 갈래는 한 번에 완성된 게 아니라 네 단계로 진화했습니다.

1. **어휘 목록** (`harm_text.py`) — 위험 단어 목록에 문자열이 들어 있는지 봅니다.
2. **오타 복구** (`fuzzy_lexicon.py`) — ASR이 단어를 뭉개도 자모 거리로 되살립니다.
3. **의미 유사도** (`harm_semantic.py`) — 문장을 벡터로 바꿔 "비슷한 뜻"으로 잡습니다.
4. **학습된 분류기** (`harm_learned.py`, `train_koelectra.py`) — 데이터로 경계선을 학습합니다.

**지금 실제로 쓰는 건 4번입니다.** 서버 쪽은 e5+MLP 헤드, 기기 위에서 도는 건 KoELECTRA-small
입니다. 1~3번은 지우지 않고 남겨 뒀습니다. 새 모델이 정말 나아졌는지 비교하려면 비교 대상이
있어야 하니까요. 연구에서 "옛날 방법"은 폐기물이 아니라 눈금자입니다.

### `src/text/asr.py` — 음성을 글자로

**무엇을 하나.** Whisper로 파형(또는 파일 경로)을 문자열로 바꿉니다. 파일 전체가 37줄입니다.

```python
# src/text/asr.py:35-37
    m = _get_model(model)
    src = np.asarray(audio, dtype=np.float32) if isinstance(audio, np.ndarray) else audio
    return m.transcribe(src, language=language, fp16=False, initial_prompt=prompt)["text"].strip()
```

**왜 이렇게 짰나.** 장치가 둘입니다. 하나는 모델 캐시 — `_get_model`이 한 번 올린 모델을
`_MODELS` 딕셔너리에서 재사용합니다. Whisper 로딩은 수 초라서 클립마다 새로 부르면 그게 실행
시간의 전부가 됩니다. 다른 하나는 `initial_prompt`로, 독스트링에 이유가 있습니다:
`passing the harm lexicon markedly improves Korean recall of terms like "제삿날"/"필로폰"
without a larger model`. Whisper는 "앞에 이런 말이 나왔다"고 알려 주면 그쪽으로 기울어
받아씁니다. 모델을 키우지 않고 인식률을 올리는 트릭입니다.

**여기서 조심할 것.** `import whisper`가 함수 안에 있습니다(`# lazy: heavy import`). 무겁거나
선택적인 의존성은 함수 안에서 import하는 게 이 저장소 전체의 규칙입니다 — 말 갈래를 안 쓰는
사람이 저장소를 열었을 때 죽으면 안 되니까요.

### `src/text/harm_text.py` — 어휘 목록 (가장 단순한 출발점)

**무엇을 하나.** YAML에 적힌 위험 단어를 문장에서 찾고, 걸린 단어의 가중치를 더해 0~1
위험도를 냅니다.

```python
# src/text/harm_text.py:48-59
    low = (text or "").lower()  # case-insensitive (English); Korean unaffected
    ...
        hits = [term["t"] for term in spec["terms"] if term["t"].lower() in low]
        if not hits:
            continue
        weights = {term["t"]: float(term["w"]) for term in spec["terms"]}
        raw = min(1.0, sum(weights[t] for t in hits))
        cats[cat] = round(raw * float(spec.get("weight", 1.0)), 4)
        matched[cat] = hits
    text_risk = max(cats.values(), default=0.0)
```

**왜 이렇게 짰나.** 읽어 보면 정말 별게 없습니다. `term["t"].lower() in low` — 파이썬 부분
문자열 검사 한 줄이 이 방법의 전부입니다. 걸린 단어의 가중치 `w`를 더해 1로 자르고, 카테고리
가중치를 곱하고, 최댓값을 문장 위험도로 씁니다. 투명하고 결정적이라서, 왜 위험하다고 했느냐고
물으면 `matched`에 걸린 단어를 그대로 보여 줄 수 있습니다. 신경망은 이걸 못 합니다.

**여기서 조심할 것 — 이 방법의 태생적 한계.** 목록에 없는 단어는 존재하지 않는 것과 같습니다.
"네가 한 짓 절대 안 잊는다"는 명백한 협박이지만 위험 단어가 없어서 위험도 0이고, 반대로
"이 영화 죽인다"는 위험도가 튑니다. 목록을 늘려도 새 표현은 계속 생깁니다. 이 한계가 이
프로젝트의 목표를 정했습니다. **저는 특정 단어를 거르는 필터가 아니라, 유해한 발화 일반을
분류하는 모델을 만들려는 겁니다.** 어휘 목록은 그 목표에 도달 못 하는 출발점이자, 나중 모델이
얼마나 나아졌는지 재는 기준선입니다.

### `src/text/fuzzy_lexicon.py` — ASR이 뭉갠 단어 되살리기 (한국어 전용)

**무엇을 하나.** 실험해 보니 Whisper가 도메인 명사를 한두 자모씩 틀리게 받아썼습니다. 파일
첫머리에 실제 사례가 있습니다: 필로폰→필루폰, 코카인→코케인, 판돈→판똥. 이러면 앞의 부분
문자열 검사는 전부 놓칩니다. 그래서 한글을 **자모**로 쪼개 비교합니다. 한글 완성형 글자는
유니코드에 규칙적으로 배치돼 있어서(`가`=0xAC00부터 초성 19 × 중성 21 × 종성 28 = 11172자)
산수만으로 분해됩니다.

```python
# src/text/fuzzy_lexicon.py:30-36
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            i = c - 0xAC00
            out.extend((("C", i // 588), ("V", (i % 588) // 28), ("T", i % 28)))
        else:
            out.append(ch)
    return out
```

588 = 21×28이므로 `i // 588`이 초성, `(i % 588) // 28`이 중성, `i % 28`이 종성입니다.
"필로폰"은 글자 3개지만 자모로는 9개가 됩니다. **글자 단위로 보면 필로폰과 필루폰은 3분의 1이
다르지만, 자모 단위로 보면 9개 중 1개만 다릅니다.** 이게 핵심 아이디어이고, 그래서 거리 문턱을
아주 좁게 잡을 수 있습니다.

```python
# src/text/fuzzy_lexicon.py:23
MAX_JAMO_DIST = 1   # only 1 jamo error (d2 coincidentally matched real text everywhere)
```

```python
# src/text/fuzzy_lexicon.py:84-91
                    sj = _decompose(span)
                    # require the leading consonant to match: ASR errors preserve the
                    # initial sound, so this kills coincidental matches ("제로인"!=헤로인).
                    if not sj or sj[0] != term_cho:
                        continue
                    d = _edit_distance(sj, tj)
                    if d <= max_dist and (best is None or d < best.distance):
                        best = FuzzyHit(cat, term, span, d)
```

`_edit_distance`는 편집 거리(Levenshtein)입니다 — 한쪽을 다른 쪽으로 바꾸는 데 필요한
삽입/삭제/치환의 최소 횟수. 자모 리스트끼리 이 거리를 재서 1 이하면 같은 단어로 봅니다.

**왜 이렇게 짰나.** 퍼지 매칭은 잘못 쓰면 아무 문장이나 걸리는 재앙이 됩니다. 그래서 일부러
좁게 만들었습니다. (가) 대상을 손으로 고른 마약·도박 명사로 제한, (나) 3글자 미만 제외
(`if len(term) < 3`) — 짧으면 흔한 한국어와 충돌, (다) 초성 일치 강제 — ASR은 첫소리는
웬만하면 맞게 듣습니다. 이 제약은 추측이 아니라 실측에서 나왔습니다. 주석에 "실루엣"이
룰렛으로, "가카가"가 바카라로 잘못 걸려서 그 단어들을 뺐다고 남아 있습니다.

**여기서 조심할 것.** 정확히 일치하면 이 층은 일부러 아무것도 안 합니다
(`if span == term: best = None`). 정확 일치는 어휘 목록의 몫이고 여기는 "거의 맞은 것"만
담당합니다. 역할을 나눠 놔야 어느 층이 몇 건을 잡았는지 셀 수 있습니다.

### `src/text/harm_semantic.py` — 단어 대신 의미로

**무엇을 하나.** 문장을 다국어 임베딩 모델(multilingual-e5)로 벡터화하고, 카테고리별 "대표
문장(프로토타입)" 벡터와 코사인 유사도를 재서 가까운 쪽으로 판정합니다. 단어가 하나도 안
겹쳐도 뜻이 비슷하면 가까이 갑니다.

```python
# src/text/harm_semantic.py:98-102
        top = max(harmful, key=harmful.get) if harmful else None
        margin = (harmful[top] - safe_sim) if top is not None else -1.0
        # margin -> risk: sigmoid centered on the decision boundary DELTA.
        risk = 1.0 / (1.0 + math.exp(-self.k * (margin - self.delta)))
        flagged = margin > self.delta
```

**왜 이렇게 짰나.** 주석에 따르면 e5 코사인 값은 0.75~0.92에 몰려 있어서 "0.8을 넘으면 위험"
같은 절대 기준이 의미가 없습니다. 대신 **가장 가까운 유해 프로토타입과 가장 가까운 안전
프로토타입의 차이(margin)** 를 씁니다. 어느 쪽에 더 가까운가, 이 상대 비교만이 신호입니다.
`K = 40.0`이 시그모이드 기울기, `DELTA = 0.02`가 위험도 0.5가 되는 결정 경계입니다.

**여기서 조심할 것.** `"passage: "`, `"query: "` 접두사가 붙는 게 보일 겁니다. e5 계열은 학습할
때 그렇게 배웠기 때문에 빼먹으면 성능이 떨어집니다. 모델 카드를 꼭 읽으세요.

### `src/text/harm_learned.py` — 학습된 분류기 (현재 서버 쪽 본체)

**무엇을 하나.** 손으로 고른 프로토타입과의 거리 대신, 데이터로 **학습한** 작은 MLP가
판정합니다. e5 인코더는 얼려 두고(frozen) 그 위에 얹은 머리만 학습했습니다. 체크포인트 경로는
파일 위쪽에 상수로 박혀 있고(`_HEAD_PATH = ... / "artifacts" / "text_head.pt"`), 그 파일을
만드는 게 아래에서 볼 `scripts/train_text_head.py`입니다.

```python
# src/text/harm_learned.py:50-58
        ckpt = torch.load(self.head_path, map_location="cpu", weights_only=False)
        self._cats = ckpt["cats"]
        head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(self._cats)))
        # state_dict keys are "net.0.*"/"net.3.*" from the training Head wrapper
        head.load_state_dict({k.replace("net.", ""): v for k, v in ckpt["state"].items()})
        head.eval()
        self._head = head
        self._model = SentenceTransformer(self.model_name)
```

**왜 이렇게 짰나.** 체크포인트에 가중치(`state`)와 클래스 이름(`cats`)이 같이 들어 있어서,
클래스 순서를 하드코딩하지 않고 파일에서 읽습니다. `k.replace("net.", "")`도 눈여겨보세요.
학습 스크립트는 MLP를 `Head` 클래스의 `self.net`에 담아 저장된 키가 `net.0.weight`인데, 여기선
`nn.Sequential`을 바로 만들어 `0.weight`여야 합니다. 학습 코드와 추론 코드의 구조가 다를 때 흔히
겪는 일입니다. 위험도 정의는 `risk = 1.0 - p.get(SAFE, 0.0)` 한 줄입니다.

**여기서 조심할 것.** `available()`이 그냥 `self.head_path.exists()`입니다. 체크포인트가 없으면
조용히 이전 단계로 내려갑니다. 편리하지만 위험합니다 — 파일이 없는 줄 모른 채 "학습 모델
성능"을 재고 있을 수 있으니, 평가할 때는 어느 모드로 돌았는지(`mode` 필드) 꼭 확인하세요.

### `src/text/harm_toxicity.py` + `src/text/harm_combined.py` — 외부 모델과 합치기

**무엇을 하나.** `harm_toxicity.py`는 공개된 한국어 혐오표현 탐지 모델을 그대로 가져다 씁니다.
직접 만든 분류기는 결국 키워드에 묶여 관용구마다 패치를 붙여야 했는데, 실제 댓글 수만 건으로
학습된 모델은 "이 영화 죽인다 최고"를 그냥 무해하다고 읽습니다.

```python
# src/text/harm_toxicity.py:20-21
# labels that mean "not harmful" across the candidate models
_BENIGN = {"none", "clean", "neutral", "정상", "not_hate", "l0", "label_0"}
```

모델마다 "무해" 라벨 이름이 제각각이라 전부 나열해 놓고 그중 하나면 무해로 칩니다. 남의 모델을
앙상블할 때 피할 수 없는 잡일입니다. 단, 도박·마약은 여기 안 맡깁니다 — 그건 "혐오"가 아니라
"주제"라서 어휘 목록 담당입니다. `harm_combined.py`는 이 모든 층을 하나로 합치는데, 가장
중요한 결정이 이겁니다.

```python
# src/text/harm_combined.py:83-88
    if mode == "learned":
        # the learned head already handles idioms/stance/implicit (trained on real negatives)
        # with a 0.1% real FP — so it is the base risk. Lexicon is kept for explanation only;
        # letting it raise risk would re-introduce the keyword false positives.
        risk = sem_risk
        top = sem.top_category
```

학습 모델이 있으면 **어휘 목록은 위험도를 올리지 못합니다.** 설명용으로만 남습니다. 목록이
위험도를 올리게 두면 애써 없앤 키워드 오탐이 다시 들어오기 때문입니다. 프로토타입
모드(fallback)에서는 반대로 의미 분류기가 "확실히 안전"이라 하면 어휘 적중을 무효화합니다
(`VETO_CEILING = 0.30`). 단 **협박만은 절대 무효화하지 않습니다**(`NO_VETO_CATEGORY = "threat"`).
안전 시스템에서 협박을 놓치는 게 최악의 오류라서, 가끔 "영화 죽인다"를 잘못 잡더라도 잡는
쪽으로 기웁니다. 성능이 아니라 가치판단이고, 코드에 명시적으로 적어 둬야 하는 결정입니다.

### `scripts/train_text_head.py` — e5 + MLP 헤드 학습 (교사 모델)

**무엇을 하나.** `artifacts/text_head.pt`를 만듭니다. 앞에서 본 그 MLP —
`self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.3), nn.Linear(h, n))`
(`scripts/train_text_head.py:100`) — 만 학습하고 e5는 얼립니다.

**왜 이렇게 짰나 — 세 가지 설계.** 첫째, **얼린 인코더 + 작은 머리**. 전이학습의 표준
레시피입니다. e5는 이미 문장의 뜻을 잘 담는 벡터를 만들 줄 아니, 저는 그 벡터 공간에서
유해/무해를 가르는 경계선만 배우면 됩니다. 학습 파라미터가 몇십만 개뿐이라 GPU 없이 CPU로
끝납니다. 둘째, **진짜 부정 예시 섞기** — 합성 데이터만 쓰면 실제 문장에서 오탐이 터지므로
실제 한국어 댓글 데이터셋(kor_unsmile)의 무해 문장 3000개를 안전 라벨로 넣습니다.

```python
# scripts/train_text_head.py:163-169
    for s in toxic:
        if vpat.search(s):
            train_labels.append(CAT_IDX["threat"]); train_texts.append(s); n_threat += 1
        elif strong.search(s):
            train_labels.append(CAT_IDX["abuse"]); train_texts.append(s); n_abuse += 1
        else:
            n_skip += 1  # ambiguous casual profanity -> not a training positive
```

마지막 `else`가 흥미롭습니다. 애매한 가벼운 욕설은 **아예 학습에서 뺍니다.** 주석에 따르면
그걸 유해 예시로 넣었더니 평범한 구어체 한국어와 겹쳐 오탐이 31%까지 치솟았습니다. 애매한
데이터를 억지로 라벨링하는 것보다 버리는 게 나을 때가 있습니다. 셋째, **클래스 가중 손실** —
안전 문장이 압도적으로 많아 그냥 학습하면 전부 "안전"이라 답합니다.

```python
# scripts/train_text_head.py:181-182
    counts = np.bincount(y, minlength=len(CATS)).astype(np.float32)
    w = torch.from_numpy(counts.sum() / (len(CATS) * np.maximum(counts, 1))).float()
```

이 `w`가 `nn.CrossEntropyLoss(weight=w)`로 들어갑니다. 가중치 = 전체수 / (클래스수 × 그 클래스
개수)라서 드문 클래스일수록 큰 가중치를 받습니다.

**여기서 조심할 것 — 누수(leakage) 차단.** 평가에 쓸 문장이 학습에 섞이면 점수가 거짓말이
됩니다. 그래서 학습 셋을 만들기 전에 평가 파일의 문장을 미리 모아 둡니다.

```python
# scripts/train_text_head.py:116-121
    # eval texts to exclude from training (avoid leakage)
    eval_texts = set()
    for f in ["configs/text/harm_semantic_eval.jsonl", "configs/text/harm_language_testset.jsonl"]:
        for line in (_ROOT / f).read_text().splitlines():
            if line.strip():
                eval_texts.add(json.loads(line)["text"].replace(" ", ""))
```

`.replace(" ", "")`로 띄어쓰기를 지우고 비교하는 게 포인트입니다. 띄어쓰기만 다른 같은 문장이
통과해 버리는 걸 막습니다.

### `.autorun/train_koelectra.py` — 기기 위에서 도는 채택 모델 (이 절의 핵심)

**무엇을 하나.** e5+MLP는 인코더만 278M이라 휴대기기에 못 올립니다. 그래서 KoELECTRA-small-v3
(14M)을 통째로 미세조정합니다. 학습 데이터 레시피는 교사 모델과 **똑같이** 맞춰 놨습니다 —
그래야 "작아서 못 하는 것"과 "데이터가 달라서 못 하는 것"을 구분할 수 있습니다.

**왜 이렇게 짰나 — ASR 노이즈 증강.** 이 분류기가 실제로 받는 입력은 사람이 쓴 깔끔한 문장이 아니라
**ASR이 받아쓴, 오타투성이 문장**입니다. 그런데 학습은 깔끔한 문장으로 했습니다. 학습 입력과
배포 입력이 다른 겁니다. 그래서 학습 데이터를 일부러 망가뜨려 복사본을 하나 더 만듭니다.

```python
# .autorun/train_koelectra.py:119-132
    if os.environ.get("ASR_AUG") == "1":
        # ASR-noise augmentation: duplicate corpus with jamo corruption at CER~U(5,40)%
        # (same corruptor as the eval, but train uses its own rng -> no leakage of eval seeds)
        sys.path.insert(0, str(_ROOT / ".autorun"))
        import importlib.util as _il
        _s = _il.spec_from_file_location("ev", _ROOT / ".autorun/eval_text_asr_noise.py")
        _ev = _il.module_from_spec(_s); _s.loader.exec_module(_ev)
        rng_a = np.random.default_rng(SEED + 999)
        aug = [_ev.corrupt(t, rng_a.uniform(0.05, 0.4), rng_a) for t in texts]
        texts = texts + aug
        y = np.concatenate([y, y])
        global OUT
        OUT = _ROOT / f"artifacts/koelectra_small_harm_asraug{extra_tag}"
        print(f"[koelectra] ASR_AUG on -> {len(texts)} samples, OUT={OUT.name}", flush=True)
```

문장을 망가뜨리는 `corrupt`는 평가 스크립트의 것을 그대로 가져다 씁니다.

```python
# .autorun/eval_text_asr_noise.py:53-77
def corrupt(text, cer, rng):
    """Apply jamo substitutions / syllable deletions / spacing errors at ~CER rate."""
    out = []
    for ch in text:
        if "가" <= ch <= "힣" and rng.random() < cer:
            op = rng.random()
            if op < 0.25:
                continue                      # deletion
            c, j, g = _decomp(ch)
            if op < 0.75:                     # initial/vowel confusion
                if rng.random() < 0.5 and c in NEAR_CHO:
                    c = NEAR_CHO[c][rng.integers(len(NEAR_CHO[c]))]
                elif j in NEAR_JUNG:
                    j = NEAR_JUNG[j][rng.integers(len(NEAR_JUNG[j]))]
            else:                             # final-consonant drop
                g = " "
            try:
                out.append(_comp(c, j, g))
            except ValueError:
                out.append(ch)
        elif ch == " " and rng.random() < cer:
            continue                          # spacing error (ASR joins words)
        else:
            out.append(ch)
    return "".join(out)
```

한 글자씩 보며 `cer` 확률로 셋 중 하나를 합니다: 25%는 글자 삭제, 50%는 초성/중성을 **비슷한
소리로** 교체, 25%는 받침 떨어뜨리기. 띄어쓰기도 같은 확률로 지웁니다(ASR이 단어를 붙여 쓰는
실수). "비슷한 소리"의 정의는 위쪽 표에 있습니다.

```python
# .autorun/eval_text_asr_noise.py:39-41
# phonetically-near substitution pools (coarse ASR-style confusions)
NEAR_CHO = {"ㄱ": "ㅋㄲ", "ㄷ": "ㅌㄸ", "ㅂ": "ㅍㅃ", "ㅈ": "ㅊㅉ", "ㅅ": "ㅆ", "ㄴ": "ㅁㄹ", "ㅁ": "ㄴㅂ", "ㄹ": "ㄴ"}
NEAR_JUNG = {"ㅏ": "ㅑㅓ", "ㅓ": "ㅏㅗ", "ㅗ": "ㅜㅓ", "ㅜ": "ㅗㅡ", "ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅡ": "ㅜㅣ", "ㅣ": "ㅢㅔ"}
```

아무렇게나 망가뜨리는 게 아닙니다. ㄱ은 ㅋ이나 ㄲ으로, ㅏ는 ㅑ나 ㅓ로 — 실제 ASR이 헷갈리는
방향으로만 망가뜨립니다. 그래야 `fuzzy_lexicon.py`에서 관찰한 실제 오류(필로폰→필루폰)와 같은
종류의 노이즈가 됩니다.

**이 절에서 제일 중요한 관찰이 여기입니다. 학습 입력을 실제 배포 입력에 맞췄더니,
이 평가 조건에서는 작은 모델이 큰 모델을 이겼습니다.** 14M짜리 KoELECTRA는 언어 자체는
e5보다 훨씬 못 압니다. 하지만 e5는 깨끗한 문장만 보고 자랐고 KoELECTRA는 깨진 문장을 보고
자랐습니다. CER 20% 구간에서 recall@FPR15%가 .572 대 .803이었습니다([02장](02-models.md)).

다만 이건 **법칙이 아니라 한 번의 측정 결과**입니다. 오류를 넣은 방식이 이 스크립트가 흉내낸
자모 치환이고, 실제 Moonshine 출력으로 끝까지 확인한 실험은 아직 없습니다(→ [05장](05-limits.md) ③).
그래도 방향은 분명합니다 — 모델을 키우기 전에 **학습 분포와 배포 분포가 같은지**부터
확인하는 편이 싸고 빠릅니다.

**산출물이 어디로 가나.**

```python
# .autorun/train_koelectra.py:113-116
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    (OUT / "cats.json").write_text(json.dumps(CATS))
    print(f"[koelectra] saved -> {OUT}", flush=True)
```

증강 없이 돌리면 `artifacts/koelectra_small_harm/`, `ASR_AUG=1`이면 `_asraug`,
여기에 `SLANG=1`(은어 말뭉치)과 `ASR_REAL=1`(TTS→노이즈→Moonshine **실출력** 오류 말뭉치)까지
켜면 `_asraug_slang` — **채택된 건 이 마지막 조합**입니다. 은어와 실오류를 왜 넣게 됐는지,
그 데이터가 어떻게 만들어지는지는 [10장](#10-합체-실시간-앱과-3층-판정)에서 다룹니다.
이 스크립트는 int8 양자화까지 하지는 않습니다 — 텍스트 모델의 int8은 파일로 저장하는 대신
**로드할 때** 적용합니다(`src/cascade/pipeline.py`의 `TextScorer(int8=True)`, 10장 참고).
fp32 저장본 56MB가 로드 시점에 28MB 동적 양자화 모델이 됩니다.

**여기서 조심할 것.** `build_training_set`의 독스트링에 `kept in sync manually`라고 적혀
있습니다. 교사와 학생의 데이터 레시피가 복사·붙여넣기로 유지된다는 뜻입니다. 한쪽만 고치면
"공정한 비교"가 조용히 깨집니다.

### `.autorun/eval_text_asr_noise.py` — 여러 CER 수준에서 평가하기

**무엇을 하나.** 학생(KoELECTRA)과 교사(e5+MLP)를 깨끗한 문장과 깨진 문장 여러 조건에서
비교합니다. 여기서 **CER**은 Character Error Rate, 글자 단위 오류율입니다 — 받아쓴 결과가 정답
대비 몇 %의 글자를 틀렸는가. CER 20%면 다섯 글자에 한 글자꼴로 틀린 겁니다. 실제 CER은 상황마다
다르니(조용한 방 vs 시끄러운 길거리) 하나의 숫자가 아니라 곡선을 봅니다.

```python
# .autorun/eval_text_asr_noise.py:155-158
    for cer in CERS:
        r2 = np.random.default_rng(SEED + int(cer * 100))
        conds[f"cer{int(cer*100)}"] = ([corrupt(t, cer, r2) for t in pos],
                                       [corrupt(t, cer, r2) for t in neg])
```

기본값 5, 20, 40%는 임의로 고른 게 아닙니다. 독스트링에 `MEASURED CER levels {5, 20, 40}%
(asr_cer_eval.py results: clean 5.6 / SNR10 20.7 / SNR5 38.9)`라고 적혀 있듯, 다음 스크립트로
**실제로 측정한** 값에서 가져왔습니다. 조용할 때 5%, 시끄러울 때 40%.

**측정 지표.** 정확도가 아니라 "오탐률 15%에서의 재현율"을 씁니다.

```python
# .autorun/eval_text_asr_noise.py:145-147
def recall_at_fpr(pos_s, neg_s, fpr=FPR):
    thr = np.sort(neg_s)[::-1][max(0, int(np.floor(fpr * len(neg_s))) - 1)]
    return float((pos_s >= thr).mean()), float(thr)
```

무해 문장 점수를 내림차순 정렬해 상위 15% 지점을 문턱값으로 잡고, 그 문턱을 넘는 유해 문장의
비율을 셉니다. 이 층은 1차 트리거라 "오탐 15%까지는 허용할 테니 놓치는 걸 최소화하라"는
설계입니다. 조건마다 문턱을 다시 잡는 것도 중요합니다 — 실제 시스템도 ASR 출력 텍스트로
보정할 테니까요.

**여기서 조심할 것.** 독스트링이 한계를 정직하게 적어 놨습니다: 합성 노이즈는 실제 ASR 오류
분포가 아니라 근사이고, 영어 문장은 아예 뺐습니다(기기 위 ASR이 한국어 전용이라서). 자기
실험의 한계를 코드에 적어 두는 습관은 그 자체로 좋은 연구 태도입니다.

### `.autorun/asr_cer_eval.py` — CER을 실제로 측정하기

**무엇을 하나.** 기기에 올릴 만한 한국어 ASR 후보(Moonshine-tiny-ko 27M, sherpa-onnx
zipformer int8, 참고용 상한선 faster-whisper large-v3)의 CER을 잽니다. CER은 **편집 거리를 정답
길이로 나눈** 값입니다.

```python
# .autorun/asr_cer_eval.py:39-50
def norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    return _punct.sub("", t)
...
def cer(refs, hyps):
    import jiwer
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]  # skip empty refs
    ...
    return jiwer.cer([p[0] for p in pairs], [p[1] or "-" for p in pairs])
```

`jiwer.cer`가 (삽입 + 삭제 + 치환) / 정답 글자 수를 계산해 줍니다. 그 전 정규화가 필수입니다:
유니코드 NFC 정규화(같은 한글이 두 가지로 표현될 수 있어서), 문장부호 제거
(`_punct = re.compile(r"[^가-힣a-zA-Z0-9]")`), 그리고 **띄어쓰기 전부 제거**. 마지막이 한국어
CER의 관례입니다 — 한국어 띄어쓰기는 사람도 자주 틀리는데 그걸 오류로 세면 모델을 부당하게
깎게 됩니다.

**왜 잡음을 섞나.** 깨끗한 음성으로만 재면 배포 환경과 무관한 숫자가 나옵니다. 그래서 공개
데이터셋(Zeroth-Korean 테스트셋, 정답 전사가 있음)에 **이 프로젝트의 잡음 클립**을 여러 SNR로
섞습니다.

```python
# .autorun/asr_cer_eval.py:103-108
    ps = np.mean(speech ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    mixed = speech + scale * noise
    peak = np.abs(mixed).max()
    return (mixed / peak * 0.95).astype(np.float32) if peak > 1 else mixed.astype(np.float32)
```

SNR(신호 대 잡음비)은 데시벨 단위입니다. `scale` 줄이 핵심으로, 음성 파워 `ps`와 잡음 파워
`pn`을 재서 "잡음을 얼마나 키워야 목표 SNR이 되는가"를 풉니다. SNR 10dB면 음성 에너지가 잡음의
10배, 0dB면 둘이 같습니다. 마지막 두 줄은 클리핑 방지입니다. 잡음은 아무거나 쓰지 않습니다 —
`load_noise_pool`의 독스트링이 `Noise clips from OUR dataset: violence + confusables (the
deployment soundscape)`이고 마지막 줄이 `pool = [r for r in tr if "vio_verbal" not in r.labels]`
입니다. 이 시스템이 돌 상황은 폭력·도박 소리가 깔린 상황이니 그 소리를 잡음으로 써야 현실적인
시험이 되고, `vio_verbal`(말소리가 든 클립)은 일부러 뺍니다 — 잡음에 사람 말이 섞이면 ASR이
그걸 받아써서 정답 전사가 틀린 게 되니까요.

**여기서 조심할 것.** Moonshine 클래스에 함정 하나가 주석으로 기록돼 있습니다.

```python
# .autorun/asr_cer_eval.py:128-131
            # generous cap — the 6.5 tok/s English convention TRUNCATES Korean (verified:
            # sentences cut mid-way). Korean BPE needs far more tokens per second.
            max_new = max(32, int(len(wav) / SR * 20))
            ids = self.m.generate(**inp, max_new_tokens=max_new)
```

모델 문서의 "초당 6.5토큰" 기본값을 그대로 썼더니 한국어 문장이 중간에서 잘렸습니다. 한국어는
같은 시간에 훨씬 많은 토큰을 씁니다. 이걸 못 잡았으면 Moonshine의 CER이 실제보다 나쁘게 나와
멀쩡한 모델을 잘못 탈락시켰을 겁니다. **성능 숫자가 이상하게 나쁘면 모델을 의심하기 전에 자기
평가 코드를 먼저 의심하세요.**

---

## 8. 확률을 판단으로 — 위험도, 심판, 그리고 사람이 개입하는 자리

여기까지 오면 모델은 오디오 한 조각에 대해 클래스별 확률 벡터를 내놓습니다. 그런데 서비스가 실제로 해야 하는 말은 "vio_scream 0.71"이 아니라 "지금 위험합니다"입니다. 확률을 판단으로 바꾸는 층이 `src/risk/`이고, 그 판단이 진짜 나아졌는지 재는 층이 `.autorun/compare_vio.py`입니다. 그리고 판단이 틀렸을 때 사람이 끼어드는 통로가 `src/mining/`과 `tools/`입니다. 이 절은 "모델 바깥"의 코드만 모은 절이라고 보시면 됩니다.

### `src/risk/policy.py` + `configs/risk_policy/default.yaml` — 무엇이 위험한지는 코드가 아니라 데이터

**무엇을 하나.** "총성은 얼마나 위험한가", "몇 점부터 경고인가"를 정의합니다. 그런데 이 값들은 파이썬 코드 어디에도 없습니다. 전부 yaml 한 장에 있습니다.

```yaml
# configs/risk_policy/default.yaml:1-24
# Risk policy (spec §8, §1 Task B/C). VERSIONED — changing weights/thresholds is a
# critical task (CLAUDE.md rule 1). Weights live here, never in code.
version: v1.0

# Per-harm-class weights w_i (spec §8). Confusable/safe classes are implicitly 0.
weights:
  sex_moan: 1.0
  sex_breathing: 0.9
  sex_ambient: 0.6
  vio_gunshot: 1.0
  vio_scream: 0.9
  ...
  gmb_table: 0.7

# Risk levels: safe R<tau_warn | warn [tau_warn, tau_block) | block R>=tau_block
# (defaults are the FPR-1% operating point, spec §1 Task C).
tau_warn: 0.4
tau_block: 0.7

# Streaming (Task C)
ema_lambda: 0.3                # EMA weight on the newest window's risk
consecutive_warns_to_block: 3  # 3 warns in a row escalate to block (spec §8)
```

**왜 이렇게 짰나.** 이게 이 절에서 제가 가장 강조하고 싶은 설계입니다. "무엇을 위험으로 볼 것인가"는 **정책**이지 **모델**이 아닙니다. 정책은 자주 바뀝니다 — 학교용과 성인 서비스용 기준이 다르고, 규제가 바뀌면 또 달라집니다. 이 가중치가 코드에 하드코딩돼 있었다면 기준을 바꿀 때마다 코드를 고쳐야 하고, 최악의 경우 "재학습해야 하나?"라는 질문이 나옵니다. yaml로 빼두면 답이 명확합니다. **기준을 바꿔도 모델은 그대로**입니다.

정책을 읽는 쪽은 `frozen=True` dataclass 하나이고, `__post_init__`에서 `0 <= tau_warn <= tau_block <= 1`을 검사합니다(`src/risk/policy.py:35-39`) — 경고 임계값이 차단 임계값보다 큰 yaml은 로드 단계에서 죽습니다. 가중치를 모델 출력 순서에 맞춰 펴는 `weight_vector()`도 방어적입니다.

```python
# src/risk/policy.py:64-70
    """Full (num_classes,) weight vector aligned to taxonomy order; 0 where unset.

    Confusable/safe classes get weight 0, so they never contribute to risk.
    """
    unknown = validate_weights(policy, taxonomy)
    if unknown:
        raise ValueError(f"risk weights reference non-harm classes: {unknown}")
```

yaml에 없는 클래스(혼동 클래스, 안전 소리)는 자동으로 가중치 0이라, 박수 소리가 아무리 확실하게 잡혀도 위험도에 기여하지 못합니다. 반대로 유해 클래스가 아닌 이름을 적으면 `ValueError`로 죽습니다 — 오타로 정책이 조용히 무력화되는 걸 막습니다.

**여기서 조심할 것 (알려진 헐거운 부분).** 이 yaml은 유해 클래스 **9개**에 가중치를 줍니다: 성적 3개, 폭력 4개, 도박 2개. 23클래스 택소노미 v1.0 시절의 정책입니다. 그런데 최종 채택된 온디바이스 음향 모델은 `configs/data/classes_vio.yaml`(v2.0-vio)을 쓰고 출력 노드가 **폭력 4개뿐**입니다. 즉 `sex_*`, `gmb_*` 가중치 5개는 채택 모델에서 쓰이지 않습니다(성적은 윤리 게이트로 보류, 도박은 텍스트 분기로 이관). 정책 파일을 v2.0-vio에 맞춰 새 버전으로 내는 작업이 아직 남아 있습니다. `weight_vector`를 v2.0 택소노미로 부르면 `sex_moan`이 "택소노미에 없는 클래스"로 걸려 예외가 납니다 — 버그가 아니라 일부러 시끄럽게 실패하는 설계이지만, 알고 있어야 당황하지 않습니다.

### `src/risk/scorer.py` — 확률 벡터를 숫자 하나로

**무엇을 하나.** 확률 `p`와 가중치 `w`로 0~1 위험도 `R` 하나를 만듭니다. 공식은 docstring에 그대로 있습니다.

```python
# src/risk/scorer.py:1-8
"""Risk scorer — Task B (spec §1, §8).

    R = sigmoid(a * max_i(w_i p_i) + b * sum_i(w_i p_i) + c)

The two features (weighted max and weighted sum of harm probabilities) are
combined by a post-hoc logistic regression fit on the val split; the fitted
(a, b, c) are a versioned artifact. Weights come from the risk policy.
"""
```

**왜 이렇게 짰나.** 특징이 딱 둘입니다. **가중 최대값**은 "가장 위험한 소리 하나가 얼마나 확실한가", **가중 합**은 "위험한 소리가 여러 개 겹쳤는가". 총성 하나만 확실해도 위험하고, 비명+타격+고함이 애매하게 동시에 있어도 위험합니다. 이 둘을 섞는 비율은 감으로 정하지 않고 val 셋에서 로지스틱 회귀로 학습합니다 — 파라미터가 셋(a, b, c)뿐이라 과적합할 여지가 거의 없습니다. 가장 눈여겨볼 곳은 `score()`의 방어 코드입니다.

```python
# src/risk/scorer.py:72-83
    def score(self, probs: np.ndarray, require_fitted: bool = True) -> np.ndarray | float:
        """Risk score(s) in (0, 1) for (N, C) or (C,) probabilities.

        Raises unless the coefficients have been fit/loaded: an unfitted scorer
        (a=1,b=0,c=0) gives R=sigmoid(max_i w_i p_i) >= 0.5 for ALL inputs, so it
        silently over-flags. Pass ``require_fitted=False`` only for raw/testing use.
        """
        if require_fitted and not self.fitted:
            raise RuntimeError("RiskScorer is not fitted; call fit() or load_params() first")
        ...
```

계수를 안 맞춘 상태(a=1, b=0, c=0)면 `sigmoid(양수)`라서 모든 입력이 0.5 이상이 됩니다. 임계값이 0.4니까 **모든 클립이 최소 경고**로 뜨는데, 크래시가 아니라 눈치채기 어렵습니다. 조용히 틀리는 대신 시끄럽게 죽게 만든 게 이 세 줄입니다.

**여기서 조심할 것.** `load_params`는 저장된 `policy_version`이 현재 정책과 다르면 거부합니다(`src/risk/scorer.py:107-113`) — v1.0으로 맞춘 계수를 v1.1 가중치에 쓰면 숫자는 나오지만 의미가 없기 때문입니다.

### `src/risk/stream.py` — 시간 위의 판단, 히스테리시스

**무엇을 하나.** 5초 간격으로 흘러 들어오는 위험도를 받아 시간축 위의 등급(safe / warn / block)을 냅니다.

**왜 이렇게 짰나.** 스트리밍에서 제일 흔한 실패가 **깜빡임**입니다. 문이 쾅 닫히는 소리 한 번에 화면이 빨개졌다가 다음 창에서 초록으로 돌아오면 사용자는 경고를 아예 믿지 않게 됩니다. 그래서 EMA로 부드럽게 만들고, 경고가 **연속으로** 쌓여야만 차단으로 올립니다.

```python
# src/risk/stream.py:45-70
    def update(self, raw_score: float) -> StreamState:
        lam = self.policy.ema_lambda
        self._ema = raw_score if self._ema is None else lam * raw_score + (1.0 - lam) * self._ema

        base = risk_level(self._ema, self.policy)

        # Track consecutive warns (only an uninterrupted run of base==warn counts).
        if base == WARN:
            self._consecutive_warns += 1
        else:
            self._consecutive_warns = 0

        level = base
        if base == BLOCK or self._consecutive_warns >= self.policy.consecutive_warns_to_block:
            level = BLOCK

        stride = self.policy.stride_densified_s if level == WARN else self.policy.stride_default_s

        return StreamState(
            raw=float(raw_score),
            smoothed=float(self._ema),
            level=level,
            ...
        )
```

- λ가 0.3이니 새 창의 영향력은 30%, 쌓인 값이 70%입니다. 시끄러운 프레임 하나가 0.9를 찍어도 EMA는 훨씬 덜 움직입니다. **한 프레임이 등급을 뒤집지 못합니다.**
- 등급 판정은 원점수가 아니라 **`self._ema`로** 합니다. `raw_score`로 하면 스무딩이 아무 의미가 없습니다.
- `consecutive_warns`는 warn이 한 번이라도 끊기면 0으로 리셋됩니다. "경고 3번 누적"이 아니라 "3번 **연속**"입니다.
- 경고 상태면 다음 창 간격을 5초 → 2.5초로 좁힙니다. 평소엔 연산을 아끼고 의심스러울 때만 촘촘히 봅니다.

**여기서 조심할 것.** `base_level`(임계값만 적용)과 `level`(승격까지 적용)을 둘 다 반환하는 이유가 있습니다 — 승격 때문에 block이 된 건지 원래부터 block이었는지 구분이 안 되면 디버깅이 불가능합니다. 그리고 이 클래스는 **모델을 전혀 모릅니다.** 이미 계산된 점수 하나만 받으므로 모델 없이 단위 테스트가 가능합니다.

### `src/risk/fit.py` — 계수를 실제로 구하는 CLI

**무엇을 하나.** val 스플릿을 모델로 돌려 확률을 얻고, "유해 라벨이 하나라도 있으면 1"을 정답으로 삼아 `RiskScorer`를 학습시켜 json으로 저장합니다.

```python
# src/risk/fit.py:62-67
    probs, labels = predict(model, loader, device)
    harm_idx = list(taxonomy.harm_indices)
    targets = (labels[:, harm_idx].max(axis=1) > 0).astype(np.float64)

    scorer = RiskScorer.from_policy(policy, taxonomy).fit(probs, targets)
    scorer.save_params(args.out)
```

**왜 이렇게 짰나.** 이게 없으면 파이프라인에 구멍이 납니다. 스트리밍 추론은 `--risk-params` json을 요구하는데 그걸 만들어주는 사람이 없었습니다. docstring 첫 줄이 그 얘기입니다 — "Closes the pipeline gap between training and streaming."

**여기서 조심할 것.** 반드시 **val**로 맞춥니다. train으로 맞추면 모델이 이미 외운 클립 위에서 계수를 정하게 되고, test로 맞추면 test가 오염됩니다. 계수 셋짜리 작은 회귀여도 원칙은 같습니다.

### `.autorun/compare_vio.py` — 이 절의 심판

이 파일이 이 절에서 가장 중요합니다. 나머지가 전부 "무언가를 만드는" 코드라면, 이건 **만든 게 진짜 나은지 판정하는** 코드입니다.

**무엇을 하나.** 여러 모델(교사 fp32, int8 양자화, 얕은 학생들, CED-mini)이 같은 테스트셋에서 뽑아둔 확률 덤프를 읽어, 고정 FPR에서의 재현율을 신뢰구간과 함께 출력하고, 기준 모델과의 **차이**에도 신뢰구간을 붙입니다.

```python
# .autorun/compare_vio.py:13-31
VIO = ["vio_scream", "vio_impact", "vio_gunshot", "vio_verbal"]
V1 = "configs/data/classes.yaml"
FPRS = [0.01, 0.05, 0.10]
NBOOT = 2000
rng = np.random.default_rng(0)
...
MODELS = {  # label -> npz  (only those that exist are used)
    "full-fp32":          "data_dl/artifacts/probs_beats_fp32.npz",
    ...
    "CED-mini-int8(10MB)": "data_dl/artifacts/probs_ced_mini_int8.npz",
}
BASELINE = "full-fp32"  # student/int8 Δ vs teacher fp32
```

평가하는 동작점은 **FPR 1%, 5%, 10%** 세 개, 부트스트랩 반복은 **2000회**, 시드는 0으로 고정입니다(같은 명령은 같은 숫자를 내야 하니까요). 모델마다 출력 클래스 수가 다른 문제(v2.0-vio는 4열, v1.0은 23열)는 로드 단계에서 흡수합니다 — 4열이면 그대로, 23열이면 폭력 4개 열만 골라 **최댓값**을 "폭력 트리거 점수"로 씁니다(`load_anyvio`, 34-39행).

#### 부트스트랩이 뭔가

테스트셋 908개(이 프로젝트의 실제 크기입니다)에서 잰 재현율이 0.82라고 합시다. 이 값은 "이 908개"에서 나온 것입니다. 다른 908개였다면 0.79였을 수도, 0.85였을 수도 있습니다. **그 흔들림의 폭**을 알아야 두 모델을 비교할 수 있습니다.

새 테스트셋을 계속 모을 수는 없으니 가진 것으로 흉내를 냅니다. **가진 908개에서 복원추출(같은 클립이 두 번 뽑혀도 됨)로 908개를 다시 뽑아** 가짜 테스트셋을 만들고 거기서 재현율을 다시 잽니다. 이걸 `NBOOT = 2000`번 반복하면 값이 2000개 모이고, 그 분포의 2.5%~97.5% 지점을 자른 것이 **95% 신뢰구간**입니다.

```python
# .autorun/compare_vio.py:79-93
for name, (s, y) in data.items():
    line = [f"[{name:26s}] AP={ap(s,y):.3f}"]
    for fpr in FPRS:
        thr = thr_at_fpr(s, y, fpr); r = recall_at(s, y, thr)
        boot = []
        idx = np.arange(len(y))
        for _ in range(NBOOT):
            bi = rng.choice(idx, size=len(idx), replace=True)
            sb, yb = s[bi], y[bi]
            if yb.sum() == 0 or (yb == 0).sum() == 0:
                continue
            boot.append(recall_at(sb, yb, thr_at_fpr(sb, yb, fpr)))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        line.append(f"@{int(fpr*100)}%={r:.3f}[{lo:.2f},{hi:.2f}]")
    print("  " + "  ".join(line))
```

`rng.choice(idx, size=len(idx), replace=True)`가 복원추출 그 자체입니다. `replace=True`가 핵심이고, 이게 없으면 원래 셋을 섞기만 한 것이라 아무 정보도 안 생깁니다. 리샘플 안에서 **임계값도 다시 계산**한다는 점(`thr_at_fpr(sb, yb, fpr)`)도 중요합니다 — 임계값 자체가 데이터에서 추정한 값이라 그 불확실성까지 구간에 넣어야 정직합니다.

#### 짝지어(paired) 비교 — 여기가 진짜 핵심

모델 A의 구간이 [0.78, 0.86], 모델 B가 [0.81, 0.89]라고 합시다. 겹칩니다. 그럼 차이가 없는 걸까요? **그렇게 결론 내면 안 됩니다.** 두 모델은 **같은 클립들**을 봤기 때문입니다. 어려운 클립이 많이 뽑힌 리샘플에서는 두 모델이 **동시에** 내려갑니다. 각자의 구간은 "테스트셋이 통째로 흔들리는" 폭까지 포함해 넓게 나오지만, 알고 싶은 건 그게 아니라 **차이**입니다. 그래서 리샘플마다 **같은 인덱스**로 두 모델을 동시에 재고 그 자리에서 뺍니다.

```python
# .autorun/compare_vio.py:98-117
    for name, (s, y) in data.items():
        if name == BASELINE:
            continue
        cells = []
        idx = np.arange(len(y))
        for fpr in FPRS:
            d_obs = recall_at(s, y, thr_at_fpr(s, y, fpr)) - recall_at(sb0, yb0, thr_at_fpr(sb0, yb0, fpr))
            boot = []
            for _ in range(NBOOT):
                bi = rng.choice(idx, size=len(idx), replace=True)
                yb = y[bi]
                if yb.sum() == 0 or (yb == 0).sum() == 0:
                    continue
                d = recall_at(s[bi], yb, thr_at_fpr(s[bi], yb, fpr)) - \
                    recall_at(sb0[bi], yb, thr_at_fpr(sb0[bi], yb, fpr))
                boot.append(d)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"
            cells.append(f"@{int(fpr*100)}%:{d_obs:+.3f}[{lo:+.2f},{hi:+.2f}]{sig}")
        print(f"  [{name:26s}] " + "  ".join(cells))
```

`bi`를 한 번 뽑아 `s[bi]`(비교 모델)와 `sb0[bi]`(기준 모델)에 **똑같이** 적용하고, 두 점수의 **차이 하나**를 `boot`에 넣습니다. 공통으로 흔들리는 부분이 뺄셈에서 상쇄되므로 짝지은 구간은 각자의 구간보다 훨씬 좁습니다.

#### "유의(SIG)"가 여기서 뜻하는 것

`sig = "SIG" if (lo > 0 or hi < 0) else "ns"` — 딱 한 가지 뜻입니다. **차이의 95% 신뢰구간이 0을 포함하지 않는다.** 구간이 전부 양수면 확실히 낫고, 전부 음수면 확실히 나쁩니다. 0을 걸치면 `ns` — "차이가 없다"가 아니라 **"이 데이터로는 방향을 말할 수 없다"**입니다. 이 구분을 흐리는 순간 통계는 장식이 됩니다.

**여기서 조심할 것.** 짝지은 비교는 두 모델이 **정확히 같은 테스트셋**을 봤을 때만 성립합니다. 그래서 먼저 검사합니다.

```python
# .autorun/compare_vio.py:72-74
# sanity: same test set / same violence positives across models
ys = [tuple(y.tolist()) for _, y in data.values()]
same = all(y == ys[0] for y in ys)
```

`same`이 False면 짝지은 Δ 블록 자체를 건너뜁니다(`if BASELINE in data and same:`). 출력 헤더의 "in-sample thr"도 정직한 고백입니다 — 임계값을 별도 홀드아웃이 아니라 평가셋에서 잡았다는 뜻이고, 그만큼 절대 수치는 낙관적일 수 있습니다. 그래서 이 스크립트의 결론은 절대 수치가 아니라 **모델 간 차이**에 있습니다.

한 문장으로 줄이면 이렇습니다. **`compare_vio.py`는 "더 좋아 보인다"와 "더 좋다"를 갈라놓는 파일입니다.**

### `scripts/eval_bootstrap.py` — 같은 논리를 클래스별로

**무엇을 하나.** 모델 A와 B를 **동일한** 테스트셋에서 돌려 유해 클래스마다 AP / AUROC / 고정 FPR 재현율을 재고, 그 차이에 부트스트랩 신뢰구간을 붙입니다. docstring이 목적을 대놓고 적어놨습니다 — `Answers "are the reported improvements real or noise?" — the core brutal question.`

리샘플 인덱스를 미리 한 번만 만들어(`idx_boot`) 모든 지표가 공유합니다.

```python
# scripts/eval_bootstrap.py:89-102
        for mname, mfn in metrics.items():
            a = mfn(yt, pA[:, c]); b = mfn(yt, pB[:, c])
            dboot = []
            for bi in idx_boot:
                ytb = yt[bi]
                ...
                dboot.append(mfn(ytb, pB[bi, c]) - mfn(ytb, pA[bi, c]))
            ...
            sig = "SIG" if (lo > 0 or hi < 0) else "ns"  # 95% CI excludes 0
```

같은 `bi`, 같은 정답 `ytb`, 두 모델 — `compare_vio.py`와 완전히 같은 논리입니다. 여기에 `P(B>A)`(부트스트랩 중 B가 이긴 비율)를 하나 더 찍습니다. 지표는 AP, AUROC, `R@FPR5%`, `R@FPR10%`, `R@FPR1%`이고 `N_BOOT`은 환경변수로 받되 기본 2000입니다.

**여기서 조심할 것.** docstring이 스스로를 깎아내립니다 — "recall@FPR1%는 임계값이 음성 9개 정도로 정해져서 무의미하므로 주 지표에서 제외, 참고용으로만 출력." 샘플이 적으면 신뢰구간이 아무리 예뻐도 지표 자체가 흔들립니다. 통계 도구를 쓴다고 데이터가 늘어나지는 않습니다.

### `.autorun/calibrate.py` — "70%"가 정말 70%인지

**무엇을 하나.** 모델이 뱉는 확률이 믿을 만한 숫자인지 검사하고, 온도 스케일링으로 교정합니다.

**왜 이렇게 짰나.** 신경망은 대체로 과신합니다. "0.9"라고 말한 예측 100개 중 실제로 맞는 게 70개면 그 0.9는 거짓말입니다. 위험도 정책이 임계값 0.4/0.7을 쓰는 이상 이건 실질적 문제입니다. 교정 방법은 놀랄 만큼 단순합니다 — **로짓을 상수 T로 나누기**, 파라미터 딱 하나입니다.

```python
# .autorun/calibrate.py:69-82
    # fit temperature T on val (minimize BCE-with-logits NLL)
    ...
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    bce = torch.nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = bce(lv / logT.exp(), yv); loss.backward(); return loss
    opt.step(closure)
    T = float(logT.exp())
    ...
    print(f"\nfitted temperature T = {T:.3f}  (T>1 = model was over-confident)")
```

`logT`를 학습하고 `exp`를 취하는 건 T가 항상 양수가 되게 하는 흔한 요령입니다. 지표는 ECE로, 예측 확률을 10구간으로 나눠 "평균 확신"과 "실제 정답률"의 차이를 가중 평균합니다(38-49행).

**여기서 조심할 것.** T는 **val**에서 맞추고 **test**에서 평가합니다 — 같은 셋에서 하면 당연히 좋아 보입니다. 그리고 T는 확률의 **크기**만 바꾸고 **순위**는 안 바꿉니다. AUROC 같은 순위 기반 지표는 교정해도 그대로입니다. 교정은 성능을 올리는 게 아니라 **숫자를 정직하게** 만드는 작업입니다.

### `src/mining/` — 사람이 개입하는 루프

모델이 틀리는 방식은 대개 편향돼 있습니다. 박수를 총성으로, 응원 함성을 비명으로 오인합니다. 이걸 무작정 데이터를 더 모아 고치려 하면 비효율적입니다. **모델이 헷갈리는 것만 골라 사람에게 보여주고, 그 판단을 학습 데이터로 되먹이는** 게 하드 네거티브 마이닝(HNM)입니다. 코드를 순서대로 읽으면 루프가 그대로 드러납니다. **1단계 `candidates.py` — 무엇을 보여줄지 고르는 규칙.**

```python
# src/mining/candidates.py:86-97
    for i, clip in enumerate(pool):
        harm_probs = probs[i, harm_idx]
        j = int(np.argmax(harm_probs))
        top_prob = float(harm_probs[j])
        ...
        if top_prob >= config.fp_prob_threshold:
            reason, priority = FALSE_POSITIVE, top_prob
        elif config.uncertain_low <= top_prob < config.uncertain_high:
            reason, priority = UNCERTAIN, 1.0 - 2.0 * abs(top_prob - 0.5)
        else:
            continue
```

두 종류만 뽑습니다. **확신에 찬 오탐 후보**(유해 확률 ≥ 임계값, 기본 0.6)와 **애매한 것**(0.4~0.6). 애매한 쪽 우선순위 `1.0 - 2.0 * abs(top_prob - 0.5)`는 0.5에 가까울수록 1에 가까워집니다 — 가장 헷갈리는 것부터 보여줍니다. 정렬은 오탐 우선이고 `top_k`(기본 500)로 자릅니다(113-115행). 사람의 시간이 유한 자원이라는 걸 코드에 박아둔 것입니다.

**2단계 `run.py` — 라벨 없는 풀을 채점해 리뷰 큐 파일을 만듭니다.** 풀은 라벨이 없는 게 정상이라 남은 라벨을 아예 지웁니다: `# The pool is unlabeled by design; drop any labels so a stray/out-of-taxonomy label can't KeyError`(57-60행).

**3단계 `review.py` — 사람의 판단을 받습니다.** 이 파일이 `tools/`가 아니라 `src/`에 있는 이유가 docstring에 있습니다: "Kept in `src` (not `tools/`) so it is unit-tested; the Streamlit app is a thin wrapper over this." UI에 로직을 섞으면 테스트가 불가능해집니다. 판정은 네 가지(false_positive / positive / reject / skip)이고, 모순된 라벨은 그 자리에서 막습니다.

```python
# src/mining/review.py:66-73
    def _validate_label(self, action: str, label: str) -> None:
        if label not in self.taxonomy.categories:
            raise ValueError(f"unknown label {label!r}")
        is_harm = self.taxonomy.is_harm(label)
        if action == FALSE_POSITIVE and is_harm:
            raise ValueError(f"false_positive needs a confusable label, got harm {label!r}")
        if action == POSITIVE and not is_harm:
            raise ValueError(f"positive needs a harm label, got confusable {label!r}")
```

"오탐이었다"면서 유해 클래스를 다는 건 모순이니까요. `save_decisions` / `load_decisions`도 있어서 500개를 한 번에 다 볼 필요 없이 중단하고 이어서 할 수 있습니다.

**4단계 `hnm.py` — 판단을 학습 데이터로 되먹입니다.** 사람이 확인한 클립은 `label_confidence="verified"`, `split="train"`으로 새 `ClipRecord`가 됩니다(44-55행). 자동 수집된 약한 라벨과 사람이 확인한 라벨을 구분해 두는 게 나중에 아주 중요해집니다. 언제 멈출지도 코드에 박혀 있습니다.

```python
# src/mining/hnm.py:95-101
    if n >= config.max_iterations:
        return True
    if n >= 2:
        improvement = fpr_history[-2] - fpr_history[-1]  # reduction in FPR
        if improvement < config.min_fpr_improvement:
            return True
    return False
```

반복 예산(기본 3회)을 다 썼거나 FPR 개선폭이 기준(기본 0.005) 아래로 떨어지면 멈춥니다. "언제 그만둘지"를 미리 정해두는 건 연구에서 정말 중요한 습관입니다 — 안 정해두면 좋아 보이는 결과가 나올 때까지 계속 돌리게 됩니다.

**여기서 조심할 것.** `fp_distribution()`은 오탐 후보를 예측 클래스별로 세어줍니다. "모델이 어느 클래스에서 주로 헛것을 보는가"를 알려주는 진단표이고, 다음 반복에서 어떤 혼동 클래스를 더 모을지 정하는 근거가 됩니다.

**`config.py`가 이 순환의 숫자들을 쥐고 있습니다.** 오탐으로 볼 확률 문턱, 한 번에 사람에게 보여줄 개수, 그리고 언제 멈출지입니다.

```python
# src/mining/config.py:13-20
@dataclass(frozen=True)
class MiningConfig:
    fp_prob_threshold: float = 0.6
    uncertain_low: float = 0.4
    uncertain_high: float = 0.6
    top_k: int = 500
    max_iterations: int = 3
    min_fpr_improvement: float = 0.005
```

`frozen=True`라 만든 뒤에는 못 바꿉니다. 실험 도중에 설정이 슬쩍 바뀌면 재현이 안 되기 때문입니다. `max_iterations`와 `min_fpr_improvement`가 같이 있는 것도 봐두세요 — **"세 번까지만, 그리고 나아지는 폭이 0.5%p 미만이면 그만"** 이라는 중단 조건입니다. 사람 손이 들어가는 순환은 멈추는 조건을 정해두지 않으면 끝없이 돌게 됩니다.

### `src/collect/` — 데이터가 들어오는 입구

**`audioset.py`**는 AudioSet 세그먼트 CSV(유튜브 ID, 시작/끝 초, mid 목록)를 이 프로젝트의 택소노미로 매핑해 매니페스트를 만듭니다. mid는 `/m/032s66` 같은 불투명한 문자열이라 오타가 나면 클래스 하나가 조용히 사라집니다. 그래서 존재 검사 위에 사람 눈 검사를 하나 더 얹었습니다 — `describe_label_map`의 docstring: `Existence validation can't tell a real-but-misassigned mid from a correct one; printing the ontology name lets a human catch e.g. clap -> "Applause".`(95-99행)

**`download.py`**는 yt-dlp로 스트림 주소를 얻고 ffmpeg로 10초를 잘라 16kHz 모노 wav로 저장합니다. 핵심은 **실패를 정상으로 취급**한다는 점입니다. AudioSet은 유튜브 링크 모음이라 시간이 지나면 상당수가 삭제/비공개/지역차단됩니다.

```python
# src/collect/download.py:24-28
class DownloadStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"  # already on disk
    UNAVAILABLE = "unavailable"  # yt-dlp could not resolve (gone/private/geo)
    FAILED = "failed"  # ffmpeg cut failed
```

클립 하나 실패로 전체가 죽으면 수천 개 다운로드는 불가능하니 상태만 집계하고 계속 갑니다. 그리고 ffmpeg가 실패하면 `out_path.unlink(missing_ok=True)  # don't leave a truncated file`(81-83행)로 반쪽 파일을 지웁니다 — 이게 없으면 다음 실행에서 "이미 있음"으로 건너뛰어 깨진 wav가 학습 데이터에 섞입니다.

### `tools/` — 만져볼 수 있는 형태

**`tools/predict.py`**는 파일 하나를 넣으면 SAFE/WARN/BLOCK을 찍어주는 CLI입니다. 긴 파일은 10초 창을 밀면서 **가장 나쁜 창**을 대표로 삼습니다.

```python
# tools/predict.py:67-68
    results = StreamRiskInference(tax, scorer, policy).run(wav, predict_window, clip_id="clip")
    worst = max(results, key=lambda r: _ORDER[r.risk_level] * 10 + r.risk_score)
```

정렬 키가 `등급 * 10 + 점수`인 게 재밌습니다 — 등급이 먼저고 같은 등급 안에서만 점수로 비교합니다. `--asr`를 주면 음성을 전사해 텍스트 분기까지 태우고 최종 판정은 `max(음향, 텍스트)`입니다. 소리는 평온한데 말이 위험한 경우를 잡기 위해서입니다.

**`tools/predict_app.py`**는 같은 일을 Streamlit UI로 합니다. 스펙트로그램, 클래스별 확률 막대, MIL 어텐션(언제 그 소리가 났는지)까지 그려줍니다. **여기서 조심할 것** — 이 파일은 체크포인트 경로를 상수로 박아뒀습니다.

```python
# tools/predict_app.py:40-41
CKPT = "artifacts/ckpt_beats_v2/best.ckpt"
RISK = "artifacts/risk_beats_v2.json"
```

`tools/predict_app.py:40`의 **`artifacts/ckpt_beats_v2/best.ckpt`**는 예전 BEATs 실험의 체크포인트입니다. `.gitignore`가 `ckpt_*/`를 제외하고 있어서 저장소에도, 배포되는 데이터 번들에도 들어 있지 않습니다. 그래서 이 앱을 그냥 실행하면 "모델/리스크 아티팩트가 없습니다"에서 멈춥니다(108-110행). `tools/predict.py`는 `--ckpt` / `--risk`로 덮어쓸 수 있지만 `predict_app.py`는 상수라 코드를 고쳐야 합니다. 채택된 CED-mini 폭력 모델로 갈아끼우는 작업이 남아 있습니다.

**`tools/annotator/app.py`**는 3단계에서 말한 리뷰 UI입니다. 파형·스펙트로그램·플레이어를 띄우고 버튼 하나로 판정을 기록하는데, 로직은 전부 `mining.review.ReviewSession`에 있고 이 파일은 **렌더링만** 합니다. 121줄밖에 안 되는 이유입니다.

### `tests/` — 테스트를 명세로 읽기

이 저장소에는 테스트 파일이 42개, 테스트 333개가 있습니다. **처음 보는 코드베이스는 테스트부터 읽는 편이 빠릅니다.** docstring은 낡을 수 있지만 테스트는 낡으면 빨갛게 터집니다. 즉 테스트는 **항상 사실인 문서**입니다. 읽어볼 만한 것 넷을 고른다면:

- **`tests/risk/test_stream.py`** — 이 절에서 설명한 히스테리시스가 그대로 문장이 돼 있습니다. `test_three_consecutive_warns_escalate_to_block`, `test_safe_interrupts_consecutive_warns` 같은 이름만 읽어도 규칙을 알 수 있고, 계산까지 주석에 있습니다: `s3 = t.update(0.1)  # 0.3*0.1+0.7*0.5=0.38 -> safe, resets`. EMA를 손으로 검산해보기 딱 좋습니다.
- **`tests/risk/test_policy.py`** — `test_shipped_policy_loads_and_matches_spec_weights`는 실제 배포되는 yaml 값을 하드코딩해 검증합니다. 누가 정책을 몰래 바꾸면 터집니다. "정책 변경은 중대한 작업"이라는 규칙을 코드로 강제한 셈입니다.
- **`tests/mining/test_review.py`** — 사람의 판단을 받는 상태 기계가 어떻게 동작해야 하는지(잘못된 라벨 거부, skip 후 재방문, 저장/재개)가 전부 들어 있습니다.
- **`tests/test_config_drift.py`** — yaml과 dataclass 기본값이 어긋나는 걸 잡습니다. "설정이 두 군데 있으면 언젠가 달라진다"는 경험칙에 대한 방어이고, 이 발상 자체가 배울 점입니다.

돌리는 법은 간단합니다. `pyproject.toml`에 `pythonpath = ["src"]`가 있어서 경로 설정 없이 바로 됩니다.

```bash
uv run pytest                          # 전체
uv run pytest tests/risk -v            # 위험도 모듈만, 테스트 이름까지 출력
uv run pytest tests/risk/test_stream.py::test_ema_smoothing   # 하나만
uv run pytest -k "stream or policy"    # 이름으로 골라서
```

먼저 테스트를 읽고 "이 함수는 이렇게 동작하겠구나"를 예상한 뒤 구현을 열어보세요. 반대 순서로 읽는 것보다 훨씬 빨리 이해됩니다.

---

## 9. 명령줄 진입점과 나머지 도구들

지금까지 본 파일들은 대부분 "부품"이라 그 자체로는 아무것도 실행하지 않습니다. 이 절의
파일들은 그 부품을 조립해 **터미널에서 실제로 실행되는 명령**으로 만드는 층입니다.

### `src/train.py`

**무엇을 하나.** 학습 진입점입니다. 받는 인자는 여섯 개뿐입니다.

```python
# src/train.py:63-71
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the harm-detection model.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--classes", default=None, help="taxonomy yaml (default: configs)")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--resume", default="auto", choices=["auto", "none"])
    return p
```

`--manifest`, `--feature-root`, `--stats`는 필수, `--classes`와 `--ckpt-dir`은 기본값 `None`,
`--resume`은 기본값 `"auto"`이고 `auto`/`none` 중에서만 고를 수 있습니다. 학습률·배치 크기·에폭
수 같은 하이퍼파라미터는 **여기 없습니다.** 그건 전부 `TrainConfig`에 있고, CLI는 "데이터가
어디 있는지"만 받습니다. 부품을 조립하는 부분은 네 줄이 전부입니다.

```python
# src/train.py:86-90
    model = HarmModel(num_classes, ModelConfig())
    loss_fn = CombinedLoss(LossConfig())
    trainer = Trainer(model, loss_fn, cfg)

    result = trainer.fit(train_loader, val_loader, resume=args.resume)
```

**왜 이렇게 짰나.** 로직을 CLI에 두지 않으면 테스트가 쉬워집니다. `main(argv)`가 리스트를 받게
되어 있어 테스트에서 `main(["--manifest", ...])`처럼 직접 부를 수 있습니다.

**여기서 조심할 것.** 이 스크립트가 쓰는 백본은 `ModelConfig()`의 기본값, 즉
`backbone: str = "conv"`(`src/models/harm_model.py:25`) — **직접 만든 CNN 백본**입니다. 최종
채택한 BEATs·CED-mini는 이 CLI로 학습하지 **않고** `.autorun/train_beats_vio.py`,
`.autorun/train_ced_vio.py`(텍스트는 `.autorun/train_koelectra.py`)로 돌립니다. `src/train.py`는
"일반 파이프라인이 끝까지 돈다"를 보여 주는 기준선이자 개발용 경로입니다.

### `src/evaluate.py`

**무엇을 하나.** 체크포인트를 불러와 한 split 전체를 예측하고, spec §9 목표치를 통과했는지
`PASS`/`FAIL`로 찍습니다(목표는 `TARGET_MACRO_MAP = 0.70`, `TARGET_HARM_AUROC = 0.90`,
`TARGET_RECALL_AT_FPR = 0.80`, `TARGET_FPR = 0.01`로 상수 선언).

```python
# src/evaluate.py:180-190
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint against §9 targets.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature-root", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--classes", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None, help="write the JSON report here")
    return p
```

필수는 `--manifest`, `--feature-root`, `--stats`, `--ckpt`. `--split` 기본 `"test"`,
`--batch-size`는 정수 기본 `32`, `--out`을 안 주면 화면에만 출력합니다. 조립부는 이렇습니다.

```python
# src/evaluate.py:204-207
    model = HarmModel.from_checkpoint(args.ckpt, taxonomy.num_classes, map_location=device)

    probs, labels = predict(model, loader, device)
    report = harm_report(probs, labels, taxonomy)
```

**왜 이렇게 짰나.** 점수 계산의 핵심인 `harm_report()`는 numpy만 쓰는 순수 함수라 모델 없이
가짜 확률 배열로 단위 테스트할 수 있습니다(*"the CLI wires a checkpoint + data around it"*).

**여기서 조심할 것.** 어떤 클래스에 양성 샘플이 없으면 AP가 `NaN`이 되는데, JSON에는 `NaN`이
없어서 그대로 저장하면 `jq`나 wandb가 파일을 못 읽습니다. `_json_safe()`가 저장 직전에
유한하지 않은 실수를 전부 `None`으로 바꿉니다.

### `src/infer_stream.py`

**무엇을 하나.** 여기가 "실시간처럼" 도는 부분입니다. 10초 클립 하나를 분류하는 게 아니라,
**긴 오디오 위를 창(window)이 미끄러지듯 지나가며** 구간마다 위험도를 내고, 그 위험도가 시간에
따라 어떻게 변하는지를 만들어 냅니다.

```python
# src/infer_stream.py:102-118
        """Slide over ``waveform``; stride adapts to each window's level."""
        self.tracker.reset()
        results: list[StreamResult] = []
        pos = 0
        idx = 0
        n = len(waveform)
        while pos + self.window_n <= n:
            window = waveform[pos:pos + self.window_n]
            probs = np.asarray(predict(window), dtype=np.float64).reshape(-1)
            raw = float(self.scorer.score(probs))
            state = self.tracker.update(raw)
            results.append(
                _build_result(clip_id, idx, pos / self.sample_rate, probs, self.taxonomy, state)
            )
            pos += max(1, int(round(state.stride_s * self.sample_rate)))
            idx += 1
        return results
```

`pos`는 지금 창의 시작 위치(샘플 단위)입니다. `waveform[pos:pos+window_n]`로 10초를 잘라 모델에
넣고, 확률을 위험 점수로 바꾸고, 트래커가 누적해 등급을 정한 뒤 `pos`를 다음 창 시작점으로
밉니다. 이 이동 거리(stride)가 고정이 아니라 **직전 창의 판정에 따라 달라진다**는 게
핵심입니다 — *"Slides a 10s window over audio (default 5s stride, densified to 2.5s while in
`warn`)"*. 경고 상태에서는 2.5초씩만 움직여 더 촘촘히 봅니다. 사람이 수상한 소리에 귀를
기울이는 것과 같은 발상입니다.

CLI 인자는 `--audio`, `--ckpt`, `--stats`, `--risk-params`(넷 다 필수, 마지막은 도움말이
`"fitted risk (a,b,c) json"`), 그리고 기본값이 모두 `None`인 `--classes`, `--policy`,
`--out`(`"write per-window results as JSONL"`)입니다. 위험 점수 함수의 계수 `(a, b, c)`가
코드에 박혀 있지 않고 파일로 들어온다는 점을 눈여겨보세요.

**왜 이렇게 짰나.** `run()`이 모델을 직접 부르지 않고 `predict`라는 **함수를 인자로 받습니다**
(`Predict = Callable[[np.ndarray], np.ndarray]`). 덕분에 테스트에서는 가짜 predict로 창 이동
로직만 검증하고, 실행할 때는 `make_model_predictor()`가 전처리 + 모델을 묶어 줍니다.

**여기서 조심할 것.** 결과의 `risk_level`은 `risk_score`만 보고 정해지지 않습니다.

```python
# src/infer_stream.py:43-46
    Note: ``risk_level`` reflects streaming escalation (spec §8), so it can be
    ``block`` while ``risk_score`` is still in the warn band (e.g. after 3
    consecutive warns). Don't assume ``risk_level == threshold(risk_score)``.
```

또 오디오가 10초보다 짧으면 창이 하나도 안 만들어져 결과가 빈 리스트가 됩니다.

### `scripts/fetch_data.sh`

**무엇을 하나.** 실험 데이터 묶음을 통째로 내려받아 푸는 셸 스크립트입니다. 저장소를 처음
클론했다면 **가장 먼저 실행하는 명령**입니다.

```bash
# scripts/fetch_data.sh:24-38
TAG="${1:-data-v1}"
REPO="soysaucecrab/Danger-Audio-Teenager"
DL="data_dl/release_download"
...
gh release download "$TAG" -R "$REPO" -D "$DL" --skip-existing
...
cat "$DL"/clips.tar.part*    | tar -xf -    # -> data_dl/clips/
cat "$DL"/features.tar.part* | tar -xf -    # -> data_dl/features/
tar -xzf "$DL"/meta.tar.gz                  # -> data_dl/{manifests,artifacts,asr}
tar -xf  "$DL"/ckpt_final.tar               # -> ckpt_ced_mini_vio/ + artifacts/koelectra_*
mkdir -p data_dl/weights
cp -f "$DL"/BEATs_iter3_plus_AS2M.pt data_dl/weights/
```

내려받는 곳은 비공개 저장소 `soysaucecrab/Danger-Audio-Teenager`의 GitHub 릴리스, 쓰는 도구는
**GitHub CLI(`gh`)** 입니다. 기본 태그는 `data-v1`이고 인자로 바꿀 수 있습니다
(`bash scripts/fetch_data.sh [tag]`). 파일이 커서 `clips.tar`와 `features.tar`는 여러
조각(`.part*`)으로 나뉘어 있고, `cat`으로 이어붙여 `tar`에 파이프로 흘려 넣습니다.

```bash
# scripts/fetch_data.sh:12-20
# Restored layout:
#   data_dl/clips/*.wav            raw 10s audio clips (8,648)
#   data_dl/features/*.npy         precomputed log-mel features (8,409)
#   data_dl/manifests/*.jsonl      label manifests (v2.0-vio taxonomy)
#   data_dl/artifacts/*.npz        eval outputs (probs_*, norm stats, calibration)
#   data_dl/asr/                   ASR CER results + listen samples
#   data_dl/weights/BEATs_iter3_plus_AS2M.pt   BEATs backbone (fine-tune input)
#   ckpt_ced_mini_vio/best.ckpt    adopted violence trigger (CED-mini)
#   artifacts/koelectra_small_harm*/           adopted text classifier (KoELECTRA-small)
```

**여기서 조심할 것.** 전제 조건이 둘입니다. (1) `gh auth login`으로 저장소 접근 권한이 있는
계정에 로그인돼 있어야 하고, (2) 디스크 여유가 **약 11GB**(5.4GB 다운로드 + 압축 푼 사본)
필요합니다. 끝나면 `data_dl/release_download`를 지워 5.4GB를 회수하라고 알려 줍니다. 맨 위
`set -euo pipefail`이 없으면 다운로드가 실패해도 압축 해제로 넘어가 이상한 상태가 되고,
`cd "$(dirname "$0")/.."`는 어디서 실행하든 저장소 루트로 이동시킵니다.

### `scripts/gen_train_corpus.py`

**무엇을 하나.** 텍스트 분류기(말의 내용으로 유해성을 보는 쪽)의 **학습 문장을 코드로 만들어
냅니다.** 카테고리별(`threat`, `sexual`, `gambling`, `drug` + `safe`)로 직접 쓴 문장
목록(clause bank)에서 둘을 뽑아 접속사로 이어 붙여 수를 불립니다.

```python
# scripts/gen_train_corpus.py:157-168
    for cat, base in {**BANK, "safe": SAFE}.items():
        target = args.per_category if cat != "safe" else max(900, args.per_category // 2)
        ...
        while sum(1 for r in rows if r["label"] == cat) < target and tries < target * 40:
            a, b = rng.sample(base, 2)
            if _lang(a) != _lang(b):          # keep each combined example monolingual
                continue
            conn = rng.choice(CONNECT[_lang(a)])
            emit(f"{a}{conn}{b}", cat)
```

목표는 카테고리당 `--per-category`(기본값 `1800`)문장, 시드(`SEED = 991`)가 고정이라 몇 번을
돌려도 결과가 같습니다. 한국어와 영어가 섞이지 않게 언어가 다르면 건너뜁니다.

**왜 이렇게 짰나.** 사람이 수천 문장을 직접 쓸 수 없고 유해 발화 데이터는 공개된 것을 그대로
쓰기도 어려워서, "문장 은행 + 조합"으로 규모를 만듭니다. `SAFE` 목록에는 "배고파 죽겠다" 같은
죽-관용구를 일부러 넣었습니다(주석: `죽-idioms — extremely common benign Korean; must NOT read
as threats`). "죽겠다"만 보고 협박으로 분류하면 한국어에서 오탐이 폭발하기 때문입니다.

**여기서 조심할 것.** 이건 저 혼자 쓴 문장에서 출발한 **합성 데이터**이고, docstring이 그
한계를 스스로 밝힙니다.

```python
# scripts/gen_train_corpus.py:8-9
Honest limit: still synthetic (author distribution), so diversity is bounded; real benign
negatives (kor_unsmile clean) are added at train time, and eval stays on real held-out data.
```

평가셋과 겹치는 문장은 생성 단계에서 거릅니다(`emit()`이 공백을 지운 키로 비교해
`eval_texts`에 있으면 버립니다).

### `scripts/gen_language_testset.py`

**무엇을 하나.** 같은 방식으로 이번엔 **평가셋**을 만듭니다. `(카테고리, 종류, 언어)` 조합마다
`PER_CELL = 22`개씩 채우고 시드는 `SEED = 20260712`입니다.

```python
# scripts/gen_language_testset.py:27-28
# Each entry: complete grammatical clauses.
# Kind: explicit/implicit (harm), plain/tricky/stance (safe).
```

`kind` 축이 핵심입니다. 유해 문장은 `explicit`(대놓고)과 `implicit`(에둘러)로, 안전 문장은
`plain`(평범), `tricky`(단어만 보면 위험해 보이는 것), `stance`(마약·도박을 **반대하는** 발언)로
나뉩니다. `stance`가 있어야 "마약은 절대 하면 안 되는 거야" 같은 문장을 마약 홍보로
오분류하는지 확인할 수 있습니다.

**여기서 조심할 것 (가장 중요).** 학습 문장도, 평가 문장도 스크립트가 만듭니다. 두 파일의
문장 은행에는 겹치는 표현이 있습니다 — "너 오늘 내 손에 죽는다"는 양쪽 `threat` 목록에 다
있습니다. 완전히 같은 문장은 걸러지지만 **문체와 표현 습관이 같은 사람에게서 나왔다는 사실**은
걸러지지 않아서, 이 평가셋 점수는 실제보다 후하게 나올 수 있습니다.

```python
# scripts/gen_language_testset.py:10-11
Honest limit: still synthetic (my authoring), so it probes generalization within my
distribution; real transcripts remain the final validation.
```

그래서 최종 검증은 항상 실제 녹음·실제 코퍼스로 합니다 — 다음 파일이 그 작업입니다.

### `scripts/calibrate_threshold.py`

**무엇을 하나.** **결정 임계값(decision threshold)** 을 정합니다. 분류 모델은 "위험/안전"을 바로
뱉지 않고 0~1 점수를 냅니다. 그 점수가 얼마 이상일 때 "위험"으로 볼지 정하는 숫자가
임계값입니다. 0.5가 당연해 보이지만 전혀 그렇지 않습니다 — 낮추면 놓치는 건 줄지만 멀쩡한
문장을 위험이라 부르는 오탐이 늘고, 높이면 그 반대입니다.

이 스크립트는 후보(`THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]`)를 훑으며 세
가지를 동시에 봅니다. (1) 실제 한국어 코퍼스(`kor_unsmile`의 clean 문장) 오탐률, (2) 실제 음성
녹음 전사본의 정밀도/재현율/F1, (3) 합성 held-out 평가셋의 P/R/F1. 그리고 추천값을 규칙으로
뽑아 `artifacts/calibration.json`에 남깁니다.

```python
# scripts/calibrate_threshold.py:75-78
    # recommend: lowest threshold whose real FP <= 2%, else the one closest to 2%
    ok = [t for t in table if t["real_corpus_fp_rate"] <= 0.02]
    rec = (min(ok, key=lambda t: t["threshold"]) if ok
           else min(table, key=lambda t: abs(t["real_corpus_fp_rate"] - 0.02)))
```

오탐률을 먼저 제약으로 걸고, 그 안에서 가장 낮은(= 재현율이 가장 높은) 임계값을 고릅니다.

**왜 검증 데이터로 맞춰야 하나.** 임계값도 데이터에서 학습되는 값입니다. 테스트셋 위에서 여러
값을 시험해 제일 좋은 걸 고르면 그 테스트 점수는 더 이상 "처음 보는 데이터의 성능"이 아니라
테스트셋에 맞춰 튜닝한 결과라 실제보다 좋게 나옵니다. 그래서 임계값은 검증/held-out
데이터에서 정하고, 정한 뒤에는 **고정한 채로** 테스트셋에 한 번만 적용해야 합니다.

**여기서 조심할 것.** 위쪽 import 순서가 특이합니다.

```python
# scripts/calibrate_threshold.py:20-24
import pandas as pd  # noqa: E402  (import HF stack before src to dodge src/datasets clash)
from huggingface_hub import hf_hub_download  # noqa: E402

sys.path.insert(0, str(_ROOT / "src"))
from text.harm_combined import score_text_all  # noqa: E402
```

이 저장소의 `src/datasets/`는 HuggingFace의 `datasets` 패키지와 이름이 같습니다. 그래서
`sys.path`에 `src`를 넣기 **전에** HF 쪽을 먼저 import해야 충돌이 안 납니다. `# noqa: E402`는
"import가 파일 맨 위에 있지 않다"는 ruff 경고를 여기서만 끄는 표시입니다.

### 그 밖의 파일들

**`__init__.py` 파일들.** `src/collect/`, `src/datasets/`, `src/losses/`, `src/mining/`,
`src/models/`, `src/preprocess/`, `src/risk/`, `src/text/`, `src/training/`,
`src/models/beats/`에 하나씩 있고, 전부 5~37줄로 짧습니다. 하는 일은 한 줄 설명 docstring과
하위 모듈 이름 재수출 두 가지뿐입니다.

```python
# src/preprocess/__init__.py:1-4
"""Audio preprocessing: raw clip -> normalized log-mel spectrogram (spec §4)."""

from preprocess.audio import load_audio, passes_rms_gate, rms_dbfs
from preprocess.config import PreprocessConfig
...
```

이러면 `from preprocess import LogMelExtractor`처럼 짧게 쓸 수 있고, 이어지는 `__all__`은 "이
패키지의 공개 API는 여기까지"라는 선언입니다. 로직이 거의 없는 게 정상입니다 — `__init__.py`에
실제 코드를 넣으면 import만으로 무거운 작업이 돌고 순환 import가 생기기 쉽습니다.

**`src/models/beats/`.** 3장에서 다룬 vendoring 폴더입니다. 라이선스 고지는 여기 있습니다.

```
# src/models/beats/NOTICE.txt:1-2
BEATs (MIT License) vendored from https://github.com/microsoft/unilm/tree/master/beats
Copyright (c) Microsoft Corporation.
```

**`pyproject.toml` / `uv.lock`.** `pyproject.toml`은 의존성과 도구 설정을 한 파일에 모은 표준
파일이고, 실제로 들어 있는 내용은 이렇습니다.

- 파이썬 버전을 `>=3.11,<3.12`로 못 박고, 기본 의존성은 `torch>=2.2`, `torchaudio>=2.2`,
  `numpy>=1.26`, `hydra-core>=1.3`, `omegaconf>=2.3` 다섯 개뿐입니다.
- 나머지는 선택 그룹입니다: `dev`(pytest, ruff), `annotator`(streamlit), `asr`(openai-whisper),
  `nlp`(transformers, sentence-transformers 등), `kaggle`(kaggle, wandb).
  `uv sync --group annotator`처럼 필요한 것만 깝니다. 오디오만 만질 때 무거운 NLP 스택을 안
  깔아도 되게 하려는 구성입니다.
- pytest 설정은 `pythonpath = ["src"]`, `testpaths = ["tests"]`,
  `addopts = "--import-mode=importlib"` — 테스트는 `tests/`에서만 찾고 `src`를 import 경로에 넣습니다.
- src 레이아웃 설정. 이것이 `python -m train`이 되는 이유입니다.

```toml
# pyproject.toml:43-45
[tool.setuptools]
package-dir = { "" = "src" }
py-modules = ["train", "evaluate", "infer_stream"]
```

- ruff(린터) 설정: 한 줄 최대 100자, 대상 py311, 규칙은 `E`(pycodestyle), `F`(pyflakes),
  `I`(import 정렬), `W`, `UP`(구식 문법 현대화), `B`(버그 유발 패턴). vendoring한 BEATs는 통째로
  제외합니다.

```toml
# pyproject.toml:55-62
[tool.ruff]
line-length = 100
target-version = "py311"
# vendored third-party (MIT, microsoft/unilm) — not our style
extend-exclude = ["src/models/beats"]

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
```

- 파일별 예외도 있습니다. 앞에서 본 긴 한국어 문장 목록 때문에 `scripts/gen_train_corpus.py`는
  `E501`(줄 길이)을, `scripts/train_text_head.py`는 `E501`과 `E702`를 면제받습니다.

`uv.lock`은 사람이 쓰는 파일이 아닙니다. `uv`가 위 의존성 범위를 실제 버전으로 전부 풀어
1800줄 넘게 기록한 잠금 파일이라, 다른 컴퓨터에서도 **완전히 같은 버전 조합**이 설치됩니다.
직접 편집하지 말고 `uv` 명령으로만 갱신하면 됩니다.

---

## 10. 합체 — 실시간 앱과 3층 판정

여기까지의 코드는 전부 "부품을 만들고 재는" 코드였습니다. 이 장의 코드는 그 부품들을
**한 줄로 이어서 실제로 돌리는** 코드입니다. 노트북이 지금 재생하는 소리를 붙잡아 →
10초 창으로 잘라 → 소리·말 두 갈래로 판정하고 → 의심 구간을 서버로 보내면 → 서버의
큰 모델이 "얼마나, 왜 유해한지"를 답합니다. [04장](04-run.md)에서 직접 돌려본 그 경로의
안쪽입니다.

이 장의 파일들은 하루 만에 완성된 게 아니라, **실제로 돌려보다 깨진 자리를 고친 흔적**이
그대로 남아 있습니다. 어디가 왜 깨졌는지가 이 장의 절반입니다.

### `src/cascade/decision.py` — 판정 규칙만 따로 떼어낸 이유

**무엇을 하나.** "게이트가 막으면 트리거를 깨우지 않는다", "음향 또는 텍스트가 임계값을
넘으면 서버로 보낸다" — 이 규칙을 **모델 없이** 순수 함수로 적어둔 파일입니다.

```python
# src/cascade/decision.py:73-79
    gate_passed = (not gate_enabled) or gate_score >= thr.gate
    a = acoustic_prob if gate_passed else None
    reasons = []
    if a is not None and a >= thr.acoustic:
        reasons.append("acoustic")
    if text_prob is not None and text_prob >= thr.text:
        reasons.append("text")
```

**왜 이렇게 짰나.** 두 가지 이유입니다. 첫째, 규칙이 모델과 섞여 있으면 규칙만 테스트할
방법이 없습니다. 이 파일은 torch를 import하지 않아서 테스트가 1초 안에 끝납니다
(`tests/cascade/test_decision.py`). 둘째, 74행의 `a = ... else None`이 중요합니다 —
게이트가 막은 창의 음향 확률은 "낮은 값"이 아니라 **"계산되지 않음"**입니다. 실제 기기에서는
트리거가 아예 안 돌기 때문입니다. 오프라인 평가가 배포와 같은 결과를 내려면 이 구분이
코드에 있어야 합니다.

임계값(`Thresholds`)은 코드에 박지 않고 `artifacts/cascade_thresholds.json`에서 읽습니다.
val 분할로 적합한 값과 "어떻게 적합했는지" 이력(meta)까지 같이 저장됩니다 — 6장에서 본
"숫자는 전부 버전 있는 파일로" 규칙의 연장입니다.

### `src/cascade/pipeline.py` — 부품 배선, 그리고 ASR을 갈아치운 기록

**무엇을 하나.** 채택 모델들을 실제로 로드하는 곳입니다. `load_trigger()`(CED-mini + int8),
`TextScorer`(KoELECTRA + int8), `make_asr()`(받아쓰기), 그리고 이 셋을 합친
`CascadePipeline`. int8은 파일이 아니라 **로드 시점에** 적용합니다.

**여기서 조심할 것 — int8 엔진은 CPU 종류를 탑니다.** 처음엔 `fbgemm`(인텔/AMD 전용)을
하드코딩했다가 맥북(애플 실리콘)에서 터질 뻔했습니다. 지금은 가용 엔진을 보고 고릅니다
(`_set_quant_engine`, 32행) — ARM에서는 `qnnpack`이 잡힙니다. 정확도는 엔진과 무관하고
속도만 다릅니다.

**왜 이렇게 짰나 — ASR 선정 기준을 바꾼 이야기.** 이 파일에서 가장 읽을 만한 건 코드가
아니라 217행부터의 주석입니다. 원래 ASR은 7장의 CER 실측으로 골랐고, CER 기준으로는
moonshine-**base**-ko(61.5M)가 압도적이었습니다(clean 2.8% vs tiny 6.0%). 그런데 욕설·은어
발화 40개를 넣어보니 base가 **15%의 발화에서 전사 대신 "audiotext"라는 문자열을 출력**했습니다.
"야 이 씨발놈아 당장 나와"가 통째로 사라졌고, tiny는 멀쩡히 받아썼습니다. 낭독체 CER은
이런 실패를 아예 볼 수 없는 지표였던 겁니다.

그래서 지표를 "유해 단어가 전사에 살아남는가"로 바꿔 다시 쟀습니다
(`.autorun/compare_asr_harm.py`). 결과 표가 주석에 그대로 있습니다:

```python
# src/cascade/pipeline.py:222-226 (주석)
#   model                        recall       profanity    slang        artifact  CPU RTF
#   moonshine-tiny-ko  (27M)     .53 / .40    .889 / .611  .23 / .23    0.00      0.04
#   moonshine-base-ko  (61.5M)   .43 / .43    .611 / .667  .27 / .23    0.15      0.05
#   whisper-large-v3-turbo int8  .55 / .60    .778 / .722  .36 / .50    0.00      1.28  (too slow)
#   whisper-small int8 (244M)    .65 / .68    .833 / .833  .50 / .55    0.00      0.32
```

whisper-small이 전 지표 우세지만 int8로도 ~250MB라 폰 예산(10~90MB)을 넘어서 옵션으로만
남겼습니다(`--asr-model whisper-small`). 기본값은 moonshine-tiny입니다 — 욕설 생존율 .889는
**완벽한 전사를 줬을 때의 상한과 같은 값**이라, 욕설에 관한 한 27M으로 잃는 게 없습니다.
은어(.23)는 어떤 예산 내 ASR도 못 잡는데, 모델이 그 단어 자체를 몰라서입니다. 그건 ASR
크기가 아니라 텍스트 쪽에서 풀 문제였고, 그 해법이 바로 아래 `HybridTextScorer`입니다.

### `HybridTextScorer` — 분류기가 모르는 단어는 목록이 잡는다

**무엇을 하나.** 텍스트 점수를 `max(분류기, 어휘목록 위험도)`로 냅니다. 분류기는 문맥을
읽지만 학습 말뭉치에 없던 은어("사다리", "먹튀", "일탈계")에는 깜깜했고 — 완벽한 전사를
줘도 recall이 절반 수준이었습니다 — 어휘목록(`configs/text/harm_lexicon.yaml`)은 문맥을
모르지만 단어는 확실히 압니다. 서로의 빈 곳을 max로 채웁니다.

```python
# src/cascade/pipeline.py:209-212
    def score(self, texts: list[str]) -> np.ndarray:
        clf = self.clf.score(texts)
        lex = np.array([self.lexicon_score(t) for t in texts], dtype=np.float32)
        return np.maximum(clf, lex)
```

**여기서 조심할 것 — 테스트가 잡아준 오탐 두 개.** 어휘목록에 은어를 넣을 때 모호한
짧은 단어는 **구절로만** 등록했습니다("떨 한번", "조건 만남"). 그런데도 테스트를 쓰다가
두 개가 걸렸습니다. "사다리 타"는 "사다리 타고 올라가서 전구 갈았어"(진짜 사다리)에
발화해서 삭제했고, 2음절 퍼지 매칭("야짤")은 실제 문장 300개 중 4개에서 "야빨"에
오매칭돼 **원저자의 경고("짧은 항목은 흔한 한국어와 충돌") 그대로**라 되돌렸습니다.
대신 ASR이 실제로 만든 오인식("예짤", "목키")을 정확 항목으로 넣었습니다 — 퍼지는
위험하지만 관찰된 문자열은 안전합니다.

측정 결과: 실제 정상 문장 935개 기준 어휘목록 오탐은 300개 샘플 중 1건(그마저
"죽여버리지"가 든 문장이라 올바른 발화)이고, 임계값 상승은 0.575→0.580으로 무시할
수준이었습니다. 은어 recall은 완벽 전사에서 .682→.955로 올랐습니다. 이 어휘목록의
빈 곳을 근본적으로 메운 재학습(`SLANG=1 ASR_REAL=1`)은 7장 끝에서 설명했습니다 —
학습용 실오류 말뭉치를 만드는 `.autorun/make_asr_corrupted_corpus.py`에는 한 가지
필터가 있는데, 유해어가 **흔한 일상어로 바뀐** 행("떨"→"또")은 버립니다. 그걸 배우면
"또 한번 할래"가 마약 문장이 되기 때문입니다.

### `src/app/engine.py` — 링버퍼, 겹치는 창, 그리고 "사건"

**무엇을 하나.** 오디오 프레임을 계속 받아 10초 창(2초 간격)으로 잘라 캐스케이드에
넣습니다. 이 프로젝트에서 컴퓨터를 먹통으로 만들었던 게 메모리 버그였기 때문에([03장](03-training.md)),
이 파일은 메모리 상한이 설계의 중심입니다.

```python
# src/app/engine.py:186-196
        while self._samples_seen - self._next_window_at >= self.window_n:
            ...
            self._next_window_at += self.hop_n
            # drop audio no longer reachable by any future window (bounded memory)
            keep_from = self._next_window_at - self._consumed
            if keep_from > 0:
                self._buf = self._buf[keep_from:].copy()
                self._consumed += keep_from
```

버퍼는 "앞으로 어떤 창도 다시 안 볼 구간"을 즉시 버립니다. 몇 시간을 켜둬도 버퍼는
창 하나 남짓입니다. `.copy()`가 붙은 이유는 6장 증류에서 배운 그 교훈입니다 —
슬라이스는 뷰라서, 복사하지 않으면 큰 배열 전체가 메모리에 붙잡혀 있습니다.

**여기서 조심할 것 ① — 같은 말을 두 번 받아쓰던 버그.** 창이 8초씩 겹치는데 매번 창
전체를 ASR에 넣었더니, 같은 문장이 화면에 두 번씩 떴습니다. 지금은 직전 ASR 호출이
커버하지 않은 새 구간만 잘라 넣습니다(`_maybe_text`, 295행 부근). ASR 비용도 같이
줄었습니다.

**여기서 조심할 것 ② — 사건 묶기의 함정.** 한 장면이 창 3~5개에 걸치니 서버에 같은
사건이 3~5번 신고됐습니다. 그래서 연속 발화 창들을 `HarmEvent` 하나로 묶어 **끝났을 때
한 번만** 보냅니다. 함정은 "끝났다"의 판정 간격이었습니다 — 텍스트 갈래는 ASR이 돈
창에서만 발화할 수 있고 ASR은 6초에 한 번만 도니, 간격 임계값이 6초 이하면 **한 사건이
ASR 호출마다 쪼개집니다**(실측에서 6s/12s/18s로 3건이 됐습니다). 그래서 간격을 고정값이
아니라 `ASR 주기 + 홉 2개`로 유도합니다(130행). 사건이 아무리 길어도 오디오는 **점수가
가장 높았던 창 하나만** 보관합니다 — 증거로는 그거면 충분하고, 메모리는 유한해야 하니까요.

### `src/app/vad.py` — 성공한 필터 하나, 실패한 필터 하나

이 파일은 **절반이 묘비**입니다. 파일 맨 위 docstring에 그렇게 적혀 있습니다.

살아있는 쪽은 `is_degenerate`입니다. 관중 소음 구간에서 Moonshine이 "와! 와! 와! …"를
지어내고(환각) 분류기가 그걸 0.908로 채점해 오탐이 났습니다. 이 함수는 같은 토큰/구절이
비정상적으로 반복되는 전사를 잡아 **점수화만 막습니다**(화면 표시는 유지).

```python
# src/app/vad.py:83-88
def _has_repeated_phrase(toks: list[str], span: int = 4, times: int = 3,
                         single_token_times: int = 5) -> bool:
    """Detect an n-gram (n<=span) repeated in a row — the classic decode loop.

    A SINGLE word repeated three times is normal emphatic speech (observed on real audio:
    "거짓의 깨 깨 깨" from a movie line, which this filter wrongly discarded), ...
```

처음엔 "3회 반복 = 환각"이었는데, 실제 영화 클립에서 배우가 "깨 깨 깨"라고 강조하는
정상 대사를 버리는 걸 발견하고 **한 단어 반복만 5회 이상**으로 완화했습니다. 구절 반복은
3회 유지입니다.

죽은 쪽은 `speech_score`입니다. 같은 환각을 ASR **전에** 막아보려고 스펙트럼 지표
(음성 대역 비율 × 비평탄도 × 음절 변조)로 말소리 게이트를 만들었는데, 합성 테스트는
통과했지만 실제 오디오로 재보니 **영화 대사를 0.00, 총성을 0.96**으로 매겼습니다.
욕설 장면의 받아쓰기를 통째로 막는, 오탐 잡으려다 미탐 만드는 필터였던 겁니다. 지금은
기본값 `speech_min=0.0`으로 꺼져 있고, 코드는 기각 이유와 측정값을 주석으로 달아
남겨뒀습니다. **합성 데이터로 검증한 필터는 실제 데이터에서 다시 재기 전에는 믿으면
안 된다** — 이 프로젝트에서 세 번째로 배운 같은 교훈입니다.

### `src/app/escalate.py` — 전송은 절대 캡처를 막으면 안 된다

**무엇을 하나.** 사건을 서버로 POST합니다. 핵심 설계는 "서버가 죽어도, 느려도, 없어도
**오디오 캡처는 멈추지 않는다**"입니다. 전송은 별도 스레드가 큐에서 꺼내 하고, 로컬
jsonl+wav 기록은 서버와 무관하게 항상 남습니다.

큐가 가득 차면 **가장 오래된 것을 버립니다**(149행) — 최신 증거가 더 가치 있다는
판단입니다. 버린 개수는 세어둡니다. 조용히 사라지는 것과 "3건 버렸음"은 다르니까요.

**여기서 조심할 것 — 종료할 때 두 번 데였습니다.** 첫째, 3층 판정이 사건당 ~6초 걸리는데
POST 타임아웃이 5초였습니다. **서버가 이미 처리한 요청을 클라이언트가 실패로 버리는**
어이없는 조합이라 30초로 늘렸습니다. 둘째, 종료 시 큐를 비우는 `drain()`을 넣었는데
처음엔 `qsize()`만 봤습니다. `queue.get()`은 POST를 **시작하기 전에** 큐를 비우므로,
마지막 한 건이 전송 중인데도 큐는 비어 보입니다. 그래서 전송 중 카운터를 따로 셉니다:

```python
# src/app/escalate.py:158-160
            payload = self._q.get()
            self._inflight += 1                 # counted separately: get() already emptied the
            try:                                # queue, so qsize() alone would look "done"
```

오디오 파형을 서버로 보내는 건(`--upload-audio`) 일부러 옵트인입니다. 3층이 오디오
모델이라 들려줘야 제 성능이 나오지만, **그 순간이 캡처한 소리가 기기를 떠나는 시점**이라
기본값으로 켜지 않았습니다. 서버 로그에도 base64 본문은 안 남기고 크기만 적습니다.

### `src/app/judge.py` — 3층: 큰 모델에게 "몇 %냐"고 묻기

**무엇을 하나.** Qwen2.5-Omni-7B에게 사건의 오디오와 (오류 많은) 온디바이스 전사를 주고
JSON 하나를 요구합니다: 정도 0~100, 범주, 근거 한 문장, 확신 여부. 프롬프트가 곧 코드라던
8장의 약속이 여기서 실물이 됩니다(35행 `PROMPT`).

**왜 이렇게 짰나 ① — Thinker만 올립니다.** Omni 체크포인트는 절반이 "답을 음성으로
말하는" Talker+보코더입니다. 서버는 JSON만 돌려주면 되니 그 절반을 로드하지 않습니다.
덕분에 7B가 4-bit로 **VRAM 6.3GB**에 들어가 3060(12GB)에서 돕니다 — 이 카드에서 돌릴
수 있는 최대 크기입니다. 로드 시 transformers가 `talker.*` 텐서 73줄을 "UNEXPECTED"라고
쏟아내는데, 그건 "체크포인트에 있는데 안 씀"이라 정상입니다. 위험한 건 반대 방향
(MISSING = 랜덤 초기화)이고, 그건 0건임을 확인했습니다 — 5장의 CED 사고(bare `AutoModel`이
전부 랜덤 초기화)를 겪은 뒤로 이 확인은 습관이 됐습니다.

**왜 이렇게 짰나 ② — 숫자를 지어내지 않습니다.** 모델이 JSON 대신 수다를 떨거나 900%
같은 값을 내면, 근사치로 뭉개지 않고 `degree=None` + 원문을 돌려줍니다:

```python
# src/app/judge.py:64-65
    if isinstance(d, (int, float)) and 0 <= float(d) <= 100:
        out["degree_percent"] = int(round(float(d)))
```

범위 밖이면 이 if를 통과하지 못하고 None으로 남습니다. "혼란스러운 모델의 답을 그럴듯한
숫자로 세탁하지 않는다"가 규칙입니다(`tests/app/test_judge.py`에 명세로 적혀 있습니다).

**여기서 조심할 것 — GPU를 잡고 있는 유령.** 셸을 죽여도 그 밑의 파이썬이 살아남아
VRAM을 계속 점유할 수 있습니다. 그 상태로 다시 로드하면 transformers는 "Some modules are
dispatched on the CPU or the disk"라는, 원인을 전혀 알려주지 않는 에러를 냅니다. 그래서
로드 전에 여유 VRAM을 확인하고, 부족하면 필요량과 함께 유령 프로세스를 찾는 `nvidia-smi`
명령까지 안내합니다(`check_vram`, 101행).

실검증은 아직 2건입니다 — 욕설 영화 장면 80%/abuse, 정상 문장 0%/none. 방향은 맞지만
**정도(%)의 품질은 라벨된 세트로 재기 전까지 미검증**입니다([05장](05-limits.md) 1순위).

### `src/app/server.py`, `sources.py`, `main.py`, `dashboard.py` — 나머지 배선

**`server.py`** 는 stdlib `http.server`로 짠 수신기입니다. `--judge qwen-omni`면 위 판정기를
물리고, 없으면 기록만 합니다(GPU 없는 컴퓨터에서도 시연되도록). 두 가지가 볼 만합니다:
페이로드 형태를 사건/창 **둘 다** 받습니다 — 전송 형식을 바꿔도 옛 클라이언트가 계속
동작해야 하니까요. 그리고 판정기가 죽어도 수신기는 안 죽습니다(82행 `except`) — 에러를
응답에 담아 돌려주고 다음 요청을 받습니다.

**`sources.py`** 는 캡처 백엔드 추상화입니다(맥 Core Audio taps / 리눅스 PipeWire /
BlackHole / 파일). 여기서 배운 것 하나 — 맥에서 "점수가 0.223에 고정되고 전사가 안 뜬다"는
증상의 원인이 모델이 아니라 **무음 캡처**(권한 미허용, 에어팟 등)였습니다. 무음은 항상
같은 점수를 내니 모델이 멈춘 것처럼 보입니다. 그래서 `python -m app.sources`로 모델 없이
캡처만 진단하는 probe를 넣었고, 앱도 무음 3창이 연속되면 점검 목록을 출력합니다.
증상("모델이 이상해요")과 원인(소리가 안 들어와요)이 멀리 떨어진 버그일수록, 원인을
분리해서 재는 도구가 필요합니다.

**`main.py`** 는 이걸 다 묶는 CLI이고, 시작 전에 모델 파일 존재를 한 번에 검사합니다
(`_preflight`) — 없으면 torch의 traceback 대신 "fetch_data.sh --models를 돌리세요" 한
줄이 나옵니다. **`dashboard.py`** 는 프레임워크 없이 stdlib로만 짠 로컬 웹 UI입니다.
127.0.0.1에만 바인딩합니다 — 지금 재생 중인 소리의 전사가 흐르는 화면이라, 같은 와이파이의
다른 사람이 보면 안 됩니다.

### 이 장의 테스트 — 깨졌던 자리마다 하나씩

이 장의 테스트들은 전부 "실제로 깨졌던 자리"의 회귀 방지입니다. 명세로 읽기 좋은 셋:

- **`tests/app/test_events.py`** — `test_consecutive_escalations_become_one_event`(4번 발화
  → 1건 전송), `test_default_gap_exceeds_the_asr_duty_cycle`(위의 6초 함정),
  `test_only_the_peak_window_audio_is_kept`(30창짜리 사건도 오디오는 창 하나).
- **`tests/app/test_judge.py`** — `test_out_of_range_degree_is_rejected_not_clamped`
  ("900%를 100%로 뭉개면 모델의 혼란을 세탁하는 것"), `test_drain_waits_for_an_inflight_post`.
- **`tests/cascade/test_hybrid_text.py`** — 은어는 잡고(`사다리 한 판 돌리자`) 일상어는
  놓아주는(`사다리 타고 올라가서 전구 갈았어`) 경계가 문장 목록으로 적혀 있습니다.
  이 테스트를 쓰다가 오탐 두 개를 실제로 잡았습니다.

---

여기 없는 것들 — 기각된 실험, 연구 일지, 영상 자료는 `process` 브랜치에 있습니다.

```bash
git show process:process.md                  # 연구 일지
git worktree add ../danger_process process   # 전체 기록 펼치기
```
