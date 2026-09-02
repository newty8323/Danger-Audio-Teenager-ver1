# 연구 과정 기록

이 브랜치는 최종 결과만 보여 주는 `final-results`와 달리, **어떤 문제가 발견되었고 어떤 근거로 다음 설계를 선택했는지**를 기록한다. 원음·개인정보·대형 모델 가중치는 올리지 않는다.

## 상세 연구 발전판

[`Danger-Audio-Teenager-Ver1_연구발전_상세기록_2026-09-03.docx`](Danger-Audio-Teenager-Ver1_연구발전_상세기록_2026-09-03.docx)는 초기 세 층 구상부터 다음 중간 구조와 최종 Android ONNX 기준선까지를 하나의 인과 흐름으로 설명한다.

- 작은 CNN 기준선과 source_id 분할 수정
- BEATs·CED-mini·증류 게이트 비교 및 1층 CNN의 앱 경로 기각
- Moonshine→KoELECTRA 초기 언어 구조와 실제 ASR 오류 재학습
- DeepFilterNet·Silero VAD·Whisper LID를 사용한 발화·한국어 게이트 버전
- Demucs 보컬 분리와 Whisper Base를 사용한 정확도 우선 Ver1
- Q5/Q6·Demucs 양자화 후보의 품질 회귀와 선택적 양자화 결정
- Qwen 사건 서버, ONNX 변환 실패·수정, 8초/30초 Whisper 실기기 비교

논문대회 제출용으로 압축한 최종 정리본은 `final-results` 브랜치에 두고, 이전 구조와 기각 실험을 포함한 이 문서는 이 브랜치에서 관리한다.

## 핵심 연구 흐름

| 단계 | 관찰·실험 | 핵심 결과 | 다음 결정 |
| --- | --- | --- | --- |
| 음향 기준선 | 작은 CNN과 BEATs 비교 | 폭력 mAP 0.228 → 0.746 | 사전학습 표현을 사용 |
| 데이터 검증 | source_id 분할·누수 수정 | 같은 원본이 split에 섞이면 점수가 과장됨 | source 단위 split 고정 |
| 음향 경량화 | BEATs 축소·int8·증류·CED 비교 | CED-mini int8 10 MB, AP 0.834 | 2층 음향 트리거 채택 |
| 언어 경로 | ASR 오류·비속어·Moonshine 비교 | 평균 CER만으로는 유해어 보존을 보장하지 않음 | 오류 강건 KoELECTRA 재학습 |
| 영화 입력 | 반복 전사·대사 누락 분석 | 단순 반복 삭제는 실제 발화를 지울 위험 | 원음 음향 / 분리 음성 언어 경로 분리 |
| 사건·서버 | 겹친 창과 Qwen 연결 | HarmEvent·opt-in 전송·사설망 health 확인 | 3층 품질 평가는 별도 과제 |
| Android ONNX | Demucs·Whisper Base 실기기 기준 실행 | 새 4초 2.20초, RTF 0.550 | 실시간 가능성 확인, 장시간 측정 필요 |

## 검증에서 중요했던 실패

- **처음부터 학습한 작은 CNN:** 데이터가 적은 약한 라벨 음향에서는 정밀 트리거가 되지 못했다.
- **무분별한 증강:** 평균 지표가 올라도 총성 재현율이 내려갈 수 있어 기본 설정에서 제외했다.
- **Moonshine-base:** 평균 CER은 더 낮았지만 유해어 보존에서 인공 산출물이 나타나 최종 채택하지 않았다.
- **반복 문자열 삭제:** 반복되는 실제 위협 발화를 없앨 수 있어 핵심 해결책으로 사용하지 않았다.
- **일반 QDQ INT8 Demucs:** FP32 가중치를 제거하지 못해 파일이 작아지지 않았다. 현재 Android 기준선은 FP32 Demucs다.

## 재현 시 주의할 점

1. 논문 수치는 약한 라벨·개발 시험의 수치이며, 독립 강한 라벨 자료의 일반화 성능이 아니다.
2. Android RTF는 한 실기기 기준의 기능 측정이다. 배터리·온도·장시간 지연을 의미하지 않는다.
3. `/health` 성공은 Qwen 서버 연결 확인일 뿐, Qwen의 유해 정도·근거가 정확하다는 평가가 아니다.
4. Android 경로와 데스크톱 경로는 아직 golden audio 기준 단계별 동등성 검증이 끝나지 않았다.

## 관련 코드

- `src/app/engine.py`
- `src/app/ver1_audio.py`
- `src/app/server.py`
- `android-onnx-benchmark/`
- `docs/final/` (on `final-results` branch)
