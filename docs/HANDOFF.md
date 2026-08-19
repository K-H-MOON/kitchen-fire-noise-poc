# HANDOFF — 다음 세션 이어가기 (2026-08-19)

> 컨텍스트가 차서 세션을 넘김. 이 문서 + 자동 메모리로 바로 이어갈 수 있게 정리.

## 한 줄 상황
**Phase B 학습 30회 전부 완료** (v8 15 + v11 15, `runs_phaseB/`에 저장). **지금 `colab_phaseB_eval.py`(노이즈 곡선+CI) 실행 중.** 끝나면 결과 읽기 → 실제화재 검증 → ablation 재검증 → 결론 문서.

## 프로젝트
- GitHub **K-H-MOON/kitchen-fire-noise-poc**. 목표: 급식실 배경 + 실제 유류화재 불꽃을 **합성**해 학습 → ① 화재 인식하나 ② 노이즈에 강건한가(+극복·일반화) ③ 아키텍처에 견고한가.
- 실행: **Colab Pro+ 계정 seochorobotics**(fire_frames·runs_phaseB 저장, L4 GPU). 원본 영상은 **blessmoonkh**(smoke_frames). smoke_frames는 seochorobotics에 공유+바로가기됨(확인: 59 mp4 접근 OK).

## 학습 완료 (30회) — 전부 Drive에 저장
- 경로: `fire_frames/runs_phaseB/{tag}_{cfg}_s{seed}/best.pt`
  - tag = **v8**(yolov8s) · **v11**(yolo11s)
  - cfg = **baseline**(노이즈 증강 없음) · **modelA**(노이즈 9종 전부) · **modelB**(화질 6종만, held-out 3)
  - seed = 1~5
- = 2아키텍처 × 3config × 5seed = **30 best.pt**. 모두 저장 확인.
- **공정성 실측**: 모든 런 train=**4556장**(2278×2). baseline은 clean 2배로 config간 데이터량 동일 → 증강 효과만 분리.
- **증강셋 재사용**: v11이 v8의 `train_aug_modelA/B`를 그대로 씀(아키텍처 무관) → v8 vs v11 사과 대 사과.
- **clean val mAP50 ~0.97~0.99** (전부 건강). **단 이건 노이즈 강건성과 무관** — 진짜 결과는 eval이 test에 노이즈를 입혀야 나옴.
- 학습 로그 무해 알림: Slow image access(Drive I/O 경고, 무해) · EarlyStopping 2건(patience=20, 정상) · epoch1 콜드스타트 몇 seed(즉시 회복).

## 노이즈 9종 (`noise_lib.py`, 강도 0→5)
- 화질계 6: gaussian·jpeg·motion_blur·defocus·low_light·contrast → modelB 학습군
- held-out 3: **steam·grayscale·random_erasing** → modelB는 안 배움(일반화 시험용). modelA는 9종 전부 배움.

## 다음 할 일 (순서)
1. **eval 결과 읽기** (`colab_phaseB_eval.py`, 진행 중). runs_phaseB의 30개 자동 발견 → config별 5seed 평균±95%CI 곡선(flame_rate·fp_rate, 노이즈별·강도별). **볼 것 4가지**:
   - baseline 저하 곡선(Phase A) — 노이즈↑ 시 얼마나 떨어지나
   - modelA — 9종 전부 배웠으니 덜 떨어져야(회복 상한)
   - modelB **held-out 3종**에서 baseline보다 높으면 → **일반화 성공**(modelB의 핵심 질문)
   - **v8 곡선 ≈ v11 곡선** 인가 → 같으면 "결론이 아키텍처에 견고"(Level 2)
2. **실제화재 검증** (`colab_realfire_test.py`, seochorobotics GPU). 준비 완료 — 5개 영상 shots 채움(불꽃 ~406장), BEST=`v8_baseline_s1`, smoke_frames 접근 확인됨. **실행만** 하면 `real_flame_rate`(불 있음→검출, 높아야)·`real_fp_rate`(불 없음→검출, 낮아야). ※ simulation은 애니 오염으로 원칙 배제됨.
3. **ablation 재검증** (`colab_ablation.py`). BEST=`v8_baseline_s1`로 이미 수정됨(`BEST_MODEL` 변수). 실행만. → 불꽃 제거 시 검출 사라지는지(배경 지름길 안 쓰는지) 재확인.
4. **결과 채우기**: docs/TIMELINE.md "남은 것" + README "결과 채우기" 자리 → kitchen-fire-poc 형식 최종 결론.

## Colab 재개 (새 런타임마다 · seochorobotics · GPU)
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
!rm -rf /content/kitchen-fire-noise-poc && git clone https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
```
```python
%run /content/kitchen-fire-noise-poc/scripts/colab_phaseB_eval.py     # 또는 colab_realfire_test.py / colab_ablation.py
```
- **학습을 더 돌릴 일은 없음**(30회 완료). 재실행해도 전부 skip.
- v8/v11 미러링 참고(이미 완료): `colab_phaseB_train.py`의 `ARCH`를 yolov8s↔yolo11s로 바꾸는 방식이었음. **sed 셀을 %run 앞에** 둬야 반영됨(빠뜨리면 v8로 돎).

## 주의 / 함정
- **Colab /content는 런타임마다 초기화** → 매번 clone. runs_phaseB는 Drive에 있어 영속(학습 resumable). eval/realfire/ablation은 증분이라 언제든.
- **realfire·ablation의 BEST 경로**: 구 `runs/fire_s`(konro 오염)에서 `runs_phaseB/v8_baseline_s1`로 고침(커밋됨). (구 경로는 colab_noise_curve.py·gradcam.py에 남아있으나 이 둘은 대체됨/폐기 — 재실행 안 함.)
- **Claude Drive MCP**: 파일 목록·메타데이터는 보지만 **이미지 픽셀·영상은 못 봄**. 확인은 사용자 육안 또는 Colab 셀.
- 파일명 한글 → NFC 정규화로 매칭(스크립트 반영됨).
- Slow image access / ultralytics 업데이트 안내 = 무해.

## 핵심 수치·결정
- 배경 2,881(train 2278/val 294/test 309) · synth train 2278·val 294·test 309(음성/하드네거 포함) · 학습 train **4556**(2×).
- 불꽃 소재: train 4(tempura01·reproduce·low_oil·dirty_pan)·test 2(clean_pan·grease_prev). **konro_ignite는 빨간 LED 오염으로 탈락**.
- 실제화재 검증셋: 5개(jikken_douga·tokyo_bousai·kanetsu·grease_spread·hisomu), simulation 배제.
- (구·오염 데이터) test mAP50 0.763·이미지단위 86%·ablation 0/116 — **깨끗한 데이터로 재확인 대상**(위 2·3번).

## 작업 방식 (사용자 선호)
옵션은 **trade-off와 함께 제시, 사용자가 결정**. "왜"를 설명. **과장 금지**. TIMELINE에 **탈락·실패까지 정직히** 기록. clean val을 성능으로 오독하지 말 것 — 판정은 eval 곡선·CI로. 정확성=seed/CI, 완결성=A→B·실제화재·v11.

## 스크립트 지도 (scripts/)
- 데이터: videos.json · assign_split.py→split.json · colab_extract_frames.py
- 불꽃: flames.json · assign_flame_split.py→flame_split.json · colab_extract_flames.py
- 합성: colab_synth.py
- 학습: colab_train.py(단일) · **colab_phaseB_train.py**(30회 오케스트레이션, ARCH로 v8/v11)
- 검증: colab_gradcam.py(판정불가·폐기) · **colab_ablation.py**(결정적, BEST 고침)
- 노이즈: **noise_lib.py**(단일소스) · colab_noise_curve.py(Phase A 미리보기·대체됨) · **colab_phaseB_eval.py**(CI 집계, 지금 실행 중)
- 실제화재: **real_fire.json**(5개 채움) · **colab_realfire_annotate.py**(구간 주석 시트, CPU) · **colab_realfire_test.py**
- 문서: PREREGISTER.md · TIMELINE.md · HANDOFF.md(이 파일)
