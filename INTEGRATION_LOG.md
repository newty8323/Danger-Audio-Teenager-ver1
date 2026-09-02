# 한국어 음성 게이트 통합 기록

이 파일은 `soysaucecrab/Danger-Audio-Teenager` 실시간 앱에 외부 실험 파이프라인을 합친 과정을
시간순으로 기록한다. 항목은 이전 내용을 수정하지 않고 아래에 추가한다.

## 2026-08-23 19:02 KST — 비공개 저장소와 기준 구조 확인

- 로그인된 GitHub 세션으로 `soysaucecrab/Danger-Audio-Teenager` 접근 권한을 확인했다.
- `main` 기준 ZIP SHA-256:
  `0BCE805C3C8B3B221183BCFDCCB19B17A9FEF98E4F8F331807DE8D8BDD2F302F`.
- `process` 기준 ZIP SHA-256:
  `B3A58AF291B99E8A70E1396D2CC1CF37B4B84EB8BEB55956FC2BD846CE5D44E2`.
- `spec.md`, `CLAUDE.md`, `process.md`, `src/app/engine.py`, `src/app/vad.py`를 검토했다.
- 기존 스펙트럼 `speech_score`가 실제 영화 대사를 놓친다는 기각 기록을 확인했다. 대체 게이트로
  학습된 Silero VAD가 필요하다는 기존 결론과 이번 통합 방향이 일치한다.
- 결정: 원본 CED-mini 음향 경로는 건드리지 않고, ASR 직전에만 강화 → 이중 VAD → 한국어 LID를
  넣는다. 한국어 누락 비용이 크므로 애매한 판정과 오류는 fail-open으로 처리한다.

## 2026-08-23 — 구현

- `src/app/language_gate.py` 추가:
  DeepFilterNet CLI/Wiener 보강, 독립형 Silero TorchScript VAD, Whisper-tiny encoder LID,
  원본·강화 VAD 합집합, 원본·강화 LID 최대값 결합, 재현율 우선 시간 정책을 구현했다.
- `src/app/engine.py` 연결:
  ASR 듀티사이클 조각을 언어 게이트에 통과시키고, 선택된 강화 음성만 기존 ASR로 보낸다.
  선택/억제/fail-open 횟수와 상세 메타데이터를 결과와 통계에 기록한다.
- `src/app/main.py` 연결:
  `--language-gate`, 모델 경로, 장치, VAD 임계값, strict 모드 CLI를 추가했다.
- `tests/app/test_language_gate.py` 추가:
  강화본에서만 검출된 음성 복구, 애매한 발화 보존, 연속 비한국어 억제, VAD 실패 시 보존,
  엔진→ASR 라우팅, 오류 fail-open을 검사한다.

## 2026-08-23 — 검증

- 수정 파일 4개의 AST/바이트코드 컴파일 통과.
- 새 테스트 6개와 영향 범위 회귀 테스트 29개, 합계 35개를 Python 3.12 실험 환경에서 통과.
  현재 PC에 프로젝트 지정 버전인 Python 3.11과 pytest가 없어 표준 `uv run pytest` 전체 332개는
  아직 실행하지 않았다.
- 실제 아티팩트 스모크 테스트 통과:
  Silero VAD 2.27MB + Whisper-tiny LID 16.48MB + DeepFilterNet 26.91MB.
- 미나리 한국어 음성 2.848초 결과:
  원본/강화/합집합 VAD 2.824초, 출력 2.824초, 그룹 1개 유지, 한국어 확률
  원본 0.568580 / 강화 0.444089, fail-open 없이 통과.
- 강화 확인 지표:
  DeepFilterNet 550.4ms, RMS -9.9387dB, 원본-강화 상관계수 0.630458, 클리핑 0%.
- ASR 전사는 이번 검증에서 실행하지 않았다. 기존 ASR 호출 지점까지의 연결만 가짜 ASR로
  회귀 검사했다.

## 2026-08-23 — 자체 검토에서 2단 강화 경로 복원

- 첫 통합 초안은 DeepFilterNet 출력 그룹을 LID/ASR에 직접 전달했다. 기존 확정 실험을 다시
  대조해 이것이 원래 구조와 다름을 발견했다.
- 원래 구조대로 수정: DeepFilterNet은 사전 VAD 타임스탬프 탐색에만 사용하고, 합집합
  타임스탬프로 **원본** 발화를 추출한 뒤 Wiener 강화본을 LID/ASR에 전달한다.
- 한국어 재현율 보호를 위해 LID는 원본과 Wiener 강화본을 모두 평가하고 큰 확률을 사용한다.
- 수정 후 새 테스트 7개 통과. 실제 스모크 결과는 원본/Wiener 한국어 확률
  0.568580/0.974427로 한국어 확정 통과했다. 출력은 2.824초로 동일했다.
- 강화 실측: DeepFilterNet 561.2ms, RMS -9.9387dB, 상관계수 0.630458, 클리핑 0%;
  Wiener RMS -4.1175dB, 상관계수 0.959531, 클리핑 0%.

## 2026-08-23 — 최종 영향 범위 검사

- 최종 코드 기준 언어 게이트 7개 + 엔진 14개 + 기존 VAD 11개 + ASR 조각 처리 4개,
  합계 **36개 통과**.
- 독립형 Silero 구현을 공식 `get_speech_timestamps`와 실제 영화 10초 구간에서 비교했다.
  두 구현 모두 `(27008, 65664)`, `(66944, 99456)`, `(115072, 132736)`을 반환했다.
- 수정 Python 파일의 바이트코드 컴파일과 100자 줄 제한 검사를 통과했다.
- 표준 프로젝트 전체 테스트는 Python 3.11 + dev 의존성 환경을 만든 뒤 다시 실행해야 한다.

## 2026-08-23 — macOS 배포 경로 준비

- macOS에는 Windows x86-64 DeepFilterNet CLI를 배포하지 않는다. `--deepfilter-exe`를 생략해
  Wiener 강화로 사전 VAD와 선택 발화 강화를 수행하는 경로를 채택했다.
- `docs/09-macos-runtime.md`에 macOS 재생음 캡처, 모델 수집, 실행, 용량, 검증 범위를 기록했다.
- `scripts/fetch_language_gate_models.sh`는 새 저장소 릴리스에서 Silero VAD와 Whisper-tiny LID
  체크포인트를 받아 `artifacts/language_gate/`에 배치한다. 기존 CED-mini·KoELECTRA는 원 연구
  저장소의 `data-v1` 모델 릴리스로부터 기존 `fetch_data.sh --models`가 받는다.
