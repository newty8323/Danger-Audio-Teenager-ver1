# 소리로 유해 콘텐츠 판별하기

> **최종 결과 브랜치입니다.** 연구의 핵심 결론과 현재 구현만 정리합니다. 세부 실험 일지와 기각한 방법은 [`research-process`](https://github.com/newty8323/Danger-Audio-Teenager-ver1/tree/research-process) 브랜치에서 확인할 수 있습니다.

| 자료 | 내용 |
| --- | --- |
| [최종 논문](docs/final/Danger-Audio-Teenager-Ver1_최종연구논문_2026-09-02.docx) | 제출용 DOCX 원고 |
| [논문 요약](docs/final/final-paper.md) | GitHub에서 바로 읽는 최종 연구 내용 |
| [Android 실행 안내](docs/final/android-run.md) | ONNX 기준 앱 빌드·설치 방법 |
| [APK Release](https://github.com/newty8323/Danger-Audio-Teenager-ver1/releases/tag/v0.1.0) | 대용량 APK 다운로드 |

**화면을 보지 않고, 소리만 듣고, 지금 재생 중인 콘텐츠가 청소년에게 유해한지 판단하는 연구입니다.**

폰에서 실제로 돌아가야 하기 때문에 "정확한 모델 하나"를 만드는 게 아니라,
**작고 항상 도는 모델 → 중간 모델 → 큰 서버 모델**의 3층으로 나눠서
필요한 순간에만 큰 모델을 깨우는 구조를 씁니다.

```
1층 상시 게이트 (0.32M)  →  2층 트리거 (10MB / 28MB)  →  3층 서버 (Qwen2.5-Omni)
"소리가 나긴 했나?"          "유해가 의심되나?"            "얼마나, 왜 유해한가?"
   24시간 계속                  가끔                         아주 가끔
```

## 문서 (이 순서로 읽는 것을 권장합니다.)

| | 내용 |
|---|---|
| [1. 개요](docs/01-overview.md) | 무슨 문제를 왜 소리로 푸는가, 3층 구조 |
| [2. 모델](docs/02-models.md) | 각 층에 뭘 왜 골랐나 — 측정 숫자와 그 해석 |
| [3. 학습](docs/03-training.md) | 실제로 학습시키며 깨진 것들과 해결 |
| [4. 실행](docs/04-run.md) | 환경 설정, 데이터 받기, 직접 돌려보기 |
| [5. 한계](docs/05-limits.md) | **아직 안 되는 것들**  |
| [6. 파일 지도](docs/06-files.md) | 어떤 파일이 뭘 하는지, 어디부터 읽을지 |
| [7. 코드 해설](docs/07-code.md) | 파일별 상세 해설 — 실제 코드를 인용해 줄 단위로 |
| [8. 한국어 음성 게이트](docs/08-korean-language-gate.md) | DeepFilterNet → 이중 Silero VAD → Whisper-tiny LID 통합 구조와 실측 |
| [9. macOS 실행](docs/09-macos-runtime.md) | 재생음 캡처와 통합 모델 실행·배포 절차 |
| [10. 100MB 이하 통합 ASR](docs/10-mobile-asr.md) | 일반 발화·영화·노래·무발화 음악을 함께 학습하는 Whisper-tiny 증류 실험 |

처음이면 [1 → 2 → 5]만 읽어도 프로젝트 전체가 잡힙니다.
직접 돌려볼 거라면 [4](docs/04-run.md), 코드를 읽을 거면 [6](docs/06-files.md) → [7](docs/07-code.md)로 가세요.
검증을 여러번 하긴 했는데, 내용이 조금 다를수도 있긴 합니다. 혹시 이상한게 있거나 궁금한게 있으면 바로 연락바랍니다.


## 빨리 시작하기

**여기서부터는 코드 실행과 관련된 내용입니다. 관련 지식이 없다면 uv와 git을 공부하고 돌아오면 좋을 것 같아요.**
```bash
uv sync                      # 환경 구성
uv run pytest -q             # 잘 깔렸는지 확인 (332개 통과)
bash scripts/fetch_data.sh   # 데이터 받기 (gh 필요, 디스크 ~11GB)
uv run python .autorun/compare_vio.py   # 학습 없이 핵심 결과 재현 (GPU 불필요)
```

## 지금까지 나온 결과 요약

| 무엇 | 결과 |
|---|---|
| 폭력음 트리거 | BEATs 364MB → **CED-mini int8 10MB** (36배 작은데 성능은 동등~우세) |
| 텍스트 분류기 | KoELECTRA-small **28MB** — **진짜 받아쓰기 오류 + 은어**로 학습해 훨씬 큰 모델을 이김 (은어 recall .23→.96) |
| 상시 게이트 | 증류 **0.32M** — 크기를 9배 키워도 점수가 안 오름 → **천장은 모델이 아니라 데이터** |
| int8 양자화 | 용량 2~3.8배 감소, 폭력 트리거 성능 손실 **통계적으로 0** |
| 실시간 앱 + 3층 판정 | 재생음 감시 앱(`src/app/`)이 의심 구간을 서버로 보내면 **Qwen-Omni 7B(4-bit)가 정도(%)·근거 판정** — [4장](docs/04-run.md)에서 직접 돌려볼 수 있음 |

숫자를 읽는 법과 배경은 [2. 모델](docs/02-models.md)에 있습니다.

## 윤리 · 데이터 취급

- 공개 출처에서 **폭력음과, 그와 헷갈리기 쉬운 소리**를 수집했습니다.
- 성인물 오디오는 수집하지 않았습니다. 다루려면 기관 승인이 필요하고,
  그 경우에도 **원본 오디오를 저장하지 않고 특징(feature)만** 저장해야 합니다.
- 데이터셋의 원본 클립은 YouTube·AudioSet 출처입니다. **이 레포 밖으로 재배포 금지.** (상업용으로 사용하면 안되는데, 연구용으로 사용해도 되는지도 의문이라, 한번 찾아볼 필요가 있긴 합니다.)

## 개발 이력

이 브랜치(`main`)에는 **현재 채택된 파이프라인만** 있습니다.
기각된 실험, 중간 시행착오, 연구 로그, 영상 자료는 `process` 브랜치에 그대로 남아 있습니다.

```bash
git show process:process.md                  # 연구 일지만 보기 (브랜치 전환 없음)
git worktree add ../danger_process process   # 전체 기록을 옆 폴더에 펼치기
```
