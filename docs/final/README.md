# 최종 결과

이 폴더는 논문 제출과 시연에 필요한 **최종 결론만** 담는다. 날짜별 로그·여러 후보 모델의 세부 비교는 [`research-process`](https://github.com/newty8323/Danger-Audio-Teenager-ver1/tree/research-process) 브랜치로 분리했다.

- [최종 논문 DOCX](Danger-Audio-Teenager-Ver1_최종연구논문_2026-09-02.docx)
- [GitHub용 논문 요약](final-paper.md)
- [Android ONNX 실행 안내](android-run.md)

## 최종 채택 조합

`원음 → CED-mini → Demucs vocal 분리 → Whisper Base ONNX → KoELECTRA → 의심 구간만 Qwen2.5-Omni`

Android에서 정확도를 우선한 기준 조합은 FP32 Demucs와 Whisper Base 원본 ONNX를 사용한다. 실기기 한 번의 기준 시험에서 새 4초 입력을 약 2.20초에 처리해 RTF 0.550을 기록했다. 이는 실시간 가능성을 보이는 기능 측정이며, 장시간 발열·배터리·다양한 기기에서의 성능을 보장하는 수치는 아니다.

또한 이 시스템은 법률상 청소년유해매체물을 확정하는 판정기가 아니다. 음향·언어의 위험 신호를 선별하고, 맥락 판단이 필요한 구간을 서버로 전달하는 연구용 보조 시스템이다.
