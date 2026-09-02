# 참고문헌

이 프로젝트의 설계 결정에 **실제로 근거가 된** 논문·데이터셋·라이브러리만 모았습니다.
읽어본 것 전부가 아니라 "이 문헌 때문에 이렇게 정했다"고 말할 수 있는 것만 올립니다.
각 항목의 마지막 문장이 **이 프로젝트에서의 쓸모**입니다.

어느 결정이 어느 문헌에서 왔는지는 [docs/02-models.md](docs/02-models.md)와 코드 주석의 `[key]` 표기로 이어집니다.

- [panns] Kong et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks. IEEE/ACM TASLP. — Weak-label AudioSet pretraining; CNN14 = our baseline backbone.
- [beats] Chen et al. (2022). BEATs: Audio Pre-Training with Acoustic Tokenizers. arXiv:2212.09058. — AudioSet SOTA-level; primary backbone, frame-level embeddings for MIL pooling.
- [audioset] Gemmeke et al. (2017). Audio Set. ICASSP. — 527-class ontology; source of vio_* and most confusable-class data; weak-label setting motivates MIL.
- [supcon] Khosla et al. (2020). Supervised Contrastive Learning. NeurIPS. — Label-driven automatic pos/neg pairs; basis of our no-manual-pair contrastive design (multi-label variant: Jaccard ≥ 0.5 positives).
- [mil-attn] Ilse et al. (2018). Attention-Based Deep Multiple Instance Learning. ICML. — Attention pooling for weak labels; attention weights reused for temporal localization.
- [specaug] Park et al. (2019). SpecAugment. Interspeech. — Time/freq masking params in §4.
- [clap] Wu et al. (2023). Large-Scale Contrastive Language-Audio Pretraining. ICASSP. — Text-audio similarity for pseudo-labeling; human review limited to 0.3–0.7 confidence band.
- [scream-gunshot] Valenzise et al. (2007). Scream and Gunshot Detection for Audio-Surveillance. IEEE AVSS. — Precision 93% @ FRR 5%, SNR 10dB; motivates SNR-sweep robustness eval.
- [porn-radon] Kim & Kim (2011). Pornographic Content Extraction w/ Radon Transform Audio Features. CBMI. — Earliest obscene-sound detection; large temporal variation as discriminative cue.
- [porn-lstm] Wazir et al. (2019). Acoustic Pornography Recognition Using RNN. IEEE ICSIPA. — MFCC+LSTM, Acc 86.5% — sanity-check target for sex_* baseline.
- [porn-nn] What Did I Just Hear? Detecting Pornographic Sounds in Adult Videos. (2022). ACM Audio Mostly. — Sound composition analysis of adult audio (moans/breathing/ambient); informs sex_* subclass split; segment- vs audio-level dual detection.
- [porn-swin] Porn Streamer Audio Recognition Based on DL and Random Forest. (2023). Applied Intelligence. — LMS+MFCC+GFCC complementary features + parallel Swin (DPFTNet); Transformer trend in this domain.

## On-device / distillation / surveillance-AED (lit review 2026-07-19, deep-research verified)
- [efficientat] Schmid, Koutini & Widmer (2023). Efficient Large-Scale Audio Tagging via Transformer-to-CNN Knowledge Distillation. ICASSP. arXiv:2211.04772. — Closest analog to our BEATs→CNN distillation. KD adds ~.057 mAP (MN-Baseline .401→.458); mAP monotonic in width, tiny ~1M sits well below. Near-teacher ONLY at AudioSet-2M scale → confirms our small-student plateau on 8.6k is expected; more unlabeled distillation data (v2) is the literature-implied lever. Code public.
- [dymn] Schmid et al. (2024). Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio Models. IEEE/ACM TASLP. arXiv:2310.15648. — DyMN-L 49.0 mAP beats single-model BEATs at <5% MACs (but ~40M, AudioSet-2M). Better student architecture than our plain CNN — candidate for student upgrade.
- [mivia-ae] Foggia et al. (2015). Reliable Detection of Audio Events in Highly Noisy Environments. Pattern Recognition Letters. — MIVIA Audio Events: 6,000 events (scream/gunshot/glass) × 6 SNRs. Canonical surveillance-AED benchmark matching our violence classes → data source + baseline.
- [sesa] Spadini (2019). Sound Events for Surveillance Applications (SESA). arXiv:1910.12369. — gunshot/explosion/siren + "casual" catch-all with deliberate confusables (fireworks, thunder). Purpose-built confusable/false-alarm eval, mirrors our confusable-node design.
- [aren] Greco et al. (2020). AReN: A Deep Learning Approach for Sound Event Recognition. IEEE TETCI. — 99.62% MIVIA / 91.43% SESA — BUT accuracy on small SNR-controlled sets, NOT stream recall/FPR. Caveat: our recall@FPR is a harder eval, not directly comparable.
- [kws-cascade] Gruenstein et al. (2017). A Cascade Architecture for Keyword Spotting on Mobile Devices. arXiv:1712.03603. — Tiny DSP 1st stage → larger 2nd stage; combined FRR ≈ 2nd-stage-alone (non-additive) IF both trained on same data with similar FRR. Validates our gate→trigger cascade + gives the design rule. Gate <1mA, wakes few times/hour. **우리 데이터에서 확인(2026-07-30)**: 캐스케이드 recall 0.657 vs 트리거 단독 0.662. 다만 우리가 어긴 전제 — 1층은 *싸야* 한다: 0.32M CNN 게이트가 int8 트리거와 CPU 시간이 같았다.
- [mlperf-tiny] Banbury et al. (2021). MLPerf Tiny Benchmark. NeurIPS Datasets & Benchmarks. arXiv:2106.07597. — DS-CNN 38.6K-param KWS; int8 "de-facto edge format, little accuracy impact", PTQ+calibration permitted / retraining prohibited. Confirms our int8-lossless finding.
- [submw-kws] Cerutti et al. (2022). Sub-mW Keyword Spotting on an MCU. IEEE. arXiv:2201.03386. — analog binary feature + BNN; can trade ~2% accuracy for ~71× energy. Quantifies the recall/energy tradeoff for the always-on gate.
- [ced] Dinkel et al. (2023). CED: Consistent Ensemble Distillation for Audio Tagging. ICASSP. arXiv:2308.11957. — ViT students distilled from a 5-model ensemble. CED-mini 10M = 49.0 AudioSet mAP > BEATs 90M (48.6) at 1/9 params; CED-tiny 5.5M 48.1, small 22M 49.6, base 86M 50.0. HF `mispeech/ced-*`. → candidate to REPLACE BEATs as the acoustic trigger (9× smaller, ≈quality) and/or a better teacher.
- [micro-acdnet] Mohaimenuzzaman et al. (2023). Environmental Sound Classification on the Edge (ACDNet / Micro-ACDNet). Pattern Recognition. — Micro-ACDNet 0.131M params, 0.50MB, 14.82M FLOPs, ~96% ESC. MCU-scale raw-audio CNN → candidate for the always-on gate (smaller than our 0.32M).
- [moonshine] Moonshine team (2025). Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices. arXiv:2509.02523. — monolingual 27M edge ASR incl. Korean; −48% error vs Whisper-tiny, beats Whisper-small (9×), matches/beats Whisper-medium (28×). On-device Korean ASR for the text branch.
- [sherpa-onnx] k2-fsa. sherpa-onnx (on-device ASR runtime). — runs Zipformer/Whisper/Paraformer/SenseVoice/Moonshine int8 on Android/iOS/MCU. Korean streaming Zipformer ~60MB, ~160ms mobile latency. Deployment runtime for on-device Korean ASR.
- [ezwhisper] ENERZAi (2025). Low-bit Korean Whisper (EZWhisper). — 1.58-bit QAT Whisper-small ~70MB, Korean CER 6.45% (< Whisper-large-v3 11.13%), trained on ~50K h Korean. High-accuracy commercial on-device Korean ASR option.

## Text-harm classification (language branch)
- [e5] Wang et al. (2024). Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672. — multilingual sentence encoder; used FROZEN as the text-harm backbone (KO+EN in one space).
- [linear-probe] Peters, Ruder & Smith (2019). To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks. RepL4NLP@ACL. — frozen encoder + lightweight head (probing) is competitive & cheap; justifies our e5+MLP head over full fine-tune.
- [eda] Wei & Zou (2019). EDA: Easy Data Augmentation Techniques for Text Classification. EMNLP-IJCNLP. — cheap synthetic augmentation (combination/edits) for low-resource text; basis for our clause-combination corpus generator.
- [unsmile] Smilegate AI (2022). Korean UnSmile Dataset (kor_unsmile). — real, messy Korean benign/hate comments; used as REAL negatives (train) + real-world false-positive eval (valid). Exposed 5.5% FP hidden by synthetic-only tests.
- [kmhas] Lee et al. (2022). K-MHaS: Korean Multi-label Hate Speech Dataset. COLING. — ~109k; scale reference for how large Korean harmful-text corpora are. Also benchmarks KoELECTRA(14M) as best/2nd-best of 6 metrics vs ~99M-avg models — small-model competitiveness evidence for the on-device classifier.
- [koelectra] Park (monologg) (2020-). KoELECTRA: Pretrained ELECTRA Model for Korean. GitHub. — small-v3 = 14M params, NSMC ~90 (≤1pt below base); primary candidate for the on-device text-harm trigger (int8 ~14MB fits the trigger tier).
- [kcelectra] Lee (Beomi) (2021-). KcELECTRA: Korean Comments ELECTRA. GitHub. — pretrained on noisy user comments; NSMC 91.71 / BEEP F1 69.91 (SOTA-class). Comparison arm for ASR-noise robustness of the on-device classifier.
- [aihub-harm] AI Hub (2023). 유해표현 검출 AI모델 학습용 데이터 (dataset 71833). — labeled Korean harmful-expression corpus; fine-tuning data candidate for the on-device text-harm classifier.
- [beep] Moon, Cho & Lee (2020). BEEP! Korean Corpus for Toxic Speech Detection. SocialNLP@ACL. — ~9.4k Korean hate-speech corpus; scale/label-scheme context.
- [focal] Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV. — class-imbalance handling; we use class-weighted cross-entropy (related) for the safe-dominated training mix.
- [whisper] Radford et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. ICML. — Whisper ASR; frozen speech→text front-end. Real speech >> synthetic TTS for Korean (our end-to-end finding).
- [mms] Pratap et al. (2024). Scaling Speech Technology to 1,000+ Languages (MMS). JMLR. — per-language VITS TTS checkpoints (`facebook/mms-tts-kor`, 36M, 16 kHz). Input MUST be uroman-romanized (the tokenizer has no Hangul); used to synthesize Korean speech for the end-to-end language-branch validation.
- [uroman] Hermjakob, May & Knight (2018). Out-of-the-box Universal Romanization Tool. ACL demo. — script-agnostic romanizer; the documented MMS-TTS front-end (Hangul → Latin) in our e2e text eval.
- [coreaudio-taps] Apple (2024). Capturing system audio with Core Audio taps. Developer Documentation (macOS 14.2+). — per-process/system audio capture without the microphone; its permission (`NSAudioCaptureUsageDescription`) is a TCC category separate from mic access, and the binary must be code-signed or no prompt appears. Basis of the macOS client's playback-only capture (`src/app/sources.py`).

## Mobile ASR small-set sources (2026-08-27)
- [zeroth] Zeroth-Korean, OpenSLR SLR40 (CC BY 4.0). — Human-read Korean with reference transcripts; source of the clean general-speech rows and speech tracks for source-disjoint movie-like mixtures.
- [csd] Choi et al. (2020). Children’s Song Dataset for Singing Voice Research, CSD 1.1 (CC BY-NC-SA 4.0). — Korean singing WAV, lyrics, and syllable timings; source of word-boundary-aligned song evaluation clips.
- [esc50] Piczak (2015). ESC: Dataset for Environmental Sound Classification, ESC-50 (CC BY-NC 3.0). — Five-second labeled non-speech recordings; source of the false-transcript arm and movie-like background effects.
