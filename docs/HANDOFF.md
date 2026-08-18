# HANDOFF — 다음 세션 이어가기 (2026-08-18)

> 컨텍스트가 차서 세션을 넘김. 이 문서 + 자동 메모리로 바로 이어갈 수 있게 정리.

## 한 줄 상황
konro_ignite 오염(빨간 LED 타이머) 제거 후 **깨끗한 데이터로 Phase B 학습(v8 15회)이 진행 중** (대략 3/15, baseline 구간). 끝나면 v11 미러링(15회) → 평가(CI) → 실제화재 검증 → ablation 재검증.

**실제화재 검증 준비는 학습과 병행해 완료(2026-08-18)**: BEST 경로 수정(ablation·realfire → v8_baseline_s1), 구간 주석 도구 추가, real_fire.json shots 채움(5개, simulation 애니오염 배제), smoke_frames 를 seochorobotics 에 공유+바로가기·Colab 접근 확인(59 mp4). → **realfire 는 학습 완주 후 실행만 남음.** 상세는 docs/TIMELINE.md.

## 프로젝트
- GitHub **K-H-MOON/kitchen-fire-noise-poc**. 목표: 급식실 배경 + 실제 유류화재 불꽃을 **합성**해 학습 → ① 화재 인식하나 ② 노이즈에 강건한가(+극복).
- 실행: **Colab Pro+ 계정 seochorobotics**(fire_frames 저장), 원본 데이터는 **blessmoonkh** 소유(smoke_frames·조리 데이터 영상). fire_frames는 seochorobotics→blessmoonkh 공유됨.

## 지금까지 (요약)
- ✅ 사이트 분할 · 배경 2,881 · 불꽃 매트 · 합성 · (구)학습 · (구)ablation · Phase A 미리보기
  → **단, (구) 산출물은 konro 오염 데이터 기반이라 재생성/재검증 대상**
- ✅ **오염 발견·수정**: konro_ignite(火災再現実験動画 こんろ火災編, ih_debris와 같은 札幌市消防科学研究所 시리즈)의 **빨간 LED 타이머 "00:27:05"** 가 스프라이트 31개 전부에 오려 들어감. 색 제거(green-floor) 시도 → konro 불꽃 자체가 빨강이라 불꽃까지 파괴 → **konro 탈락**. 상세는 docs/TIMELINE.md.
- ✅ 깨끗한 재생성: **train 불꽃 4**(tempura01·reproduce·low_oil·dirty_pan) · **test 불꽃 2**(clean_pan·grease_prev). synth = train 2278 / val 294 / test 309, 검산 통과, 오염 없음 확인.
- ▶ **Phase B v8 학습 진행 중** (colab_phaseB_train.py, L4 GPU). baseline_s2까지 저장(val mAP50 0.98), baseline_s3 진행. 회당 ~1시간.

## 실험 설계 (확정 — 재론 불필요)
- **Level 2**: v8 15회 + v11 15회 = **30회 미러링** (MODEL 한 줄만 교체). **LONO 보류**(30회 결과 보고 결정).
- **A→B 15회** = baseline×5(노이즈 없음) → modelA×5(**노이즈 9개 전부**) → modelB×5(**화질 6개만**, held-out = steam·grayscale·random_erasing).
- **5 seed ±95%CI**. 오프라인 증강, 학습 데이터 2×(baseline은 clean 2배로 공정성).
- 노이즈 9종(noise_lib.py): gaussian·jpeg·motion_blur·defocus·low_light·steam·contrast·grayscale·random_erasing.

## 다음 할 일 (순서)
1. **v8 15회 완주** — resumable. 끊기면 셀 재실행(완료분 skip). `runs_phaseB/`에 저장.
2. **v11 15회**: `!sed -i "s/yolov8s.pt/yolo11s.pt/" /content/kitchen-fire-noise-poc/scripts/colab_phaseB_train.py` 후 재실행. (증강셋은 아키텍처 무관, 재사용)
3. **평가**: `colab_phaseB_eval.py` → config별 seed평균±CI 곡선(flame_rate·fp_rate, held-out 표시). 증분이라 모델 쌓이면 언제든.
4. **실제화재 검증**: 준비 완료(shots 5개 채움·simulation 배제·BEST=v8_baseline_s1·smoke_frames 접근 확인). **학습 완주 후 `colab_realfire_test.py` 실행만** 하면 됨(seochorobotics, GPU). 경계는 시트 근사라 원하면 웹 플레이어로 재조정 가능.
5. **ablation 재검증**: 깨끗한 baseline 모델로 `colab_ablation.py` 재실행. **BEST 경로는 v8_baseline_s1 로 이미 수정됨**(BEST_MODEL 변수). 학습 완주 후 실행.
6. **결과 채우기**: docs/TIMELINE.md "남은 것" 자리 → 최종 결론 문서(README, kitchen-fire-poc 형식).

## Colab 재개 (새 런타임마다)
```python
from google.colab import drive; drive.mount('/content/drive')   # seochorobotics 계정
```
```bash
!rm -rf /content/kitchen-fire-noise-poc && git clone https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
```
```python
%run /content/kitchen-fire-noise-poc/scripts/colab_phaseB_train.py   # resumable
```

## 주의 / 함정
- **Claude Drive MCP**: fire_frames 폴더 목록·메타데이터는 보지만 **이미지 픽셀은 못 봄**(다운로드 base64 과대, read_file_content OCR 빈결과). 이미지 확인은 **사용자 육안** 또는 **Colab 진단 셀**로.
- **Colab /content 는 런타임마다 초기화** → 매번 clone. **runs_phaseB 는 스크립트가 안 지움** → 오염/재생성 시에만 `shutil.rmtree`로 수동 삭제.
- flame_matte·synth·aug셋은 스크립트가 시작 시 rmtree로 자동 재생성.
- Drive 소파일 대량쓰기 느림 + "Google Drive 오류/저장 못함" 토스트 = **노트북 파일 저장 얘기, 결과와 무관(무해)**.
- 파일명 한글 → NFC 정규화로 매칭(스크립트에 반영됨).

## 핵심 수치
- (구·오염 데이터) test mAP50 0.763 · 이미지단위 86% · ablation 배경오탐 0/116 통과 → **깨끗한 데이터로 재확인 대상**.
- (신·clean baseline) val mAP50 ~0.98.

## 작업 방식 (사용자 선호)
옵션은 **trade-off와 함께 제시, 사용자가 결정**. "왜"를 설명. **과장 금지**. TIMELINE에 **탈락·실패까지 정직히** 기록. 정확성=seed/CI, 완결성=A→B·실제화재·v11.

## 스크립트 지도 (scripts/)
- 데이터: videos.json · assign_split.py→split.json · colab_extract_frames.py
- 불꽃: flames.json · assign_flame_split.py→flame_split.json · colab_extract_flames.py
- 합성: colab_synth.py
- 학습: colab_train.py(단일) · **colab_phaseB_train.py**(15회 오케스트레이션)
- 검증: colab_gradcam.py(판정불가) · **colab_ablation.py**(결정적)
- 노이즈: **noise_lib.py**(단일소스) · colab_noise_curve.py(Phase A 미리보기) · **colab_phaseB_eval.py**(CI 집계)
- 실제화재: real_fire.json(shots 대기) · **colab_realfire_annotate.py**(구간 주석용 컨택트 시트, CPU) · colab_realfire_test.py
- 문서: PREREGISTER.md · TIMELINE.md · HANDOFF.md(이 파일)
