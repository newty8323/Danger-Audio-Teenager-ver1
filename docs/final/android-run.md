# Android ONNX 앱 실행 안내

## 준비

- Android Studio 최신 안정판
- Android 10 이상 실기기 권장
- USB 디버깅을 허용한 기기
- 약 1 GB 이상의 여유 저장 공간

APK는 저장소에 직접 넣지 않고 [GitHub Release](https://github.com/newty8323/Danger-Audio-Teenager-ver1/releases/tag/v0.1.0)로 제공한다. APK가 매우 크기 때문이다.

## Android Studio에서 빌드

1. 이 저장소의 `final-results` 브랜치를 받는다.
2. Android Studio에서 `android-onnx-benchmark` 폴더를 연다.
3. `gradle.properties`에 아래 설정이 있는지 확인한다.

```properties
modelProfile=baseline
```

4. 상단 기기 선택 목록에서 연결한 Android 기기를 고른다.
5. 실행 버튼을 누른다.

`baseline`은 정확도를 우선한 기본 조합으로, FP32 Demucs와 Whisper Base ONNX를 포함한다. `demucs-fp16`은 비교용 경량 프로필이며 최종 기준값과는 다르다.

## 앱 동작

앱은 Android 10 이상에서 미디어 재생 캡처를 사용한다. 재생 앱이 캡처를 허용하지 않거나 DRM으로 보호된 경우 오디오를 얻지 못할 수 있다. 수집된 4초 구간은 CED, Demucs, Whisper, KoELECTRA를 거친 뒤 의심 구간만 선택적으로 Qwen 서버에 전달한다.

서버는 선택 사항이다. 서버 주소가 비어 있으면 온디바이스 1·2층 판단만 확인할 수 있다. 서버 사용 시 같은 네트워크에서 접근 가능한 URL과 명시적 전송 동의가 필요하다.

## 해석 시 주의

- 이 앱은 연구·시연용이며 법률상 유해매체물 판정기가 아니다.
- APK 크기와 실시간성은 정확도를 우선한 기준 조합의 결과다.
- 실기기 RTF 0.550은 한 번의 짧은 측정이므로, 장시간 배터리·발열 평가는 별도로 수행해야 한다.
