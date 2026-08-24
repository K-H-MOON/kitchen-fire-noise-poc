# HANDOFF — 다음 세션 이어가기 (2026-08-24 갱신)

> **v1·v2·v3 모두 실제 전이 약함(불꽃·표현 레버 아님) → 병목 = 데이터/도메인.** 회의 후 = **실제 데이터로 공략** → **실험 A 완료: 실데이터 학습이 놓침을 없앰 확인**(같은 Indoor test recall **0.235→0.985**). 단 같은 도메인·프레임분할 누수라 절대값 낙관적; 견고한 건 방향(§AFTER_meeting §5).
> 완료: v2 A/B(음성)·v2 precision/recall·v3 DINOv3 프로브(음성 근접)·realfire 오염 발견·Indoor 깨끗한 첫 측정(recall 0.20)·**실험 A(합성 0.235 / 실 0.985 / 혼합 0.985)**·미팅 문서 일습.

## ▶ 새 세션 첫 작업 (2026-08-24 실험 A 완료 후)
1. ~~실험 A 학습 결과 받기~~ ✅ · ~~결과 표 채우기~~ ✅(AFTER_meeting §5·아래 §실험 A) · ~~판정~~ ✅(② ≫ ①: 놓침 병목 실데이터로 해소 · ③≈②: 합성 무기여).
2. **판정의 절대값 검증 (핵심 다음)**: 0.985는 같은 도메인+프레임분할 누수로 부풀었을 가능성 큼. → **(a) 씬/영상 단위 분할로 재측정**(누수 통제) · **(b) 도메인 이동 평가** = real_only/mixed best.pt를 **급식실 근접 realfire(정리분)**·유류화재에 돌려 전이 확인.
3. **다음 후보**: ① 급식실 근접 검증(정리한 realfire, 위 2b) ② New_sample(야외·JSON) 변환해 실데이터 확대 ③ (합성 큰 불꽃 추가는 실험 A가 "합성 무기여"를 보였으니 **우선순위 낮음**).
4. 모델: `runs_if/real_only/weights/best.pt`·`runs_if/mixed/weights/best.pt`(Drive) · 결과 `indoorfire_eval/indoorfire_train.json` · **RUN='evalonly'로 재학습 없이 재측정 가능**.
- 문서 지도: 미팅 이후 전체 = `docs/AFTER_meeting.md` · 수집사양/놓침진단 = `docs/DATA_collection_spec.md` · 미팅용(v1~v3) = `docs/SUMMARY_meeting.md`.

## ⚠️ 중대 발견 — realfire 평가셋 오염 (2026-08-24)
error-analysis(`colab_realfire_erroranalysis.py`) 영상별 시트 육안 결과:
- **jikken·kanetsu·hisomu = 편집 소방 PSA 영상**(자막·PIP·만화 불·타이틀 카드) → "fire" 라벨인데 실제 불 없거나 그래픽에 가려짐 → 정당한 무검출이 miss로 집계.
- **grease "nofire" 오라벨**(실제 불 포함) → 정확 검출이 FP로 집계. + 야외 화재(주방 아님).
- **"밝은 장면 놓침"은 완전한 착오** — 그 "밝음"은 불이 아니라 파란 타이틀 카드·분홍 만화 배경·흰 자막의 밝기였음. → 육안 감사 없이 이 지표만 봤으면 "밝은 장면 화재"라는 엉뚱한 방향으로 데이터 모을 뻔(error-analysis가 잡아냄).
- **함의: realfire recall(0.31·0.27…) 절대값은 오염·과소평가 → 신뢰 불가.** 상대 비교(C3≈C0·DINOv3 미미)는 비교적 견고(같은 셋 공유). 가장 깨끗한 grease 54% 검출이 방증.
- **미확정 부차 신호**: 큰/벽 화재 과소검출(kanetsu·hisomu) 가능성 — 오염과 뒤섞여 확정 못 함.
- **→ #1 우선순위 = 깨끗한 realfire 평가셋 재큐레이션/확보** (편집·오라벨 제외, 실촬영 주방·유류). 상세 `docs/DATA_collection_spec.md §9`.

## 깨끗한 실데이터 첫 측정 (2026-08-24)
확보 데이터 zip: Drive `Indoor Fire Smoke.zip`(5000·YOLO·fire/smoke 2클래스·실내 일반화재) · `New_sample.zip`(750·AI-Hub JSON·야외, 후순위·변환필요).
**Indoor Fire Smoke로 기존 v8_C0_s1 재측정(`colab_indoorfire_eval.py`, fire=class0):** **recall 0.201 · precision 0.846 · fpr 0.040** (fire 2624장).
- **낮은 recall이 진짜임을 오염 없이 확인**(PSA 탓 아님, 오히려 더 낮음) · **precision 높고 fpr 낮음 → 문제는 놓침(recall)** 확정.
- 경계: 이 셋은 일반 실내 화재(주방/유류 아님)라 "전이 약함+도메인 상이" 혼재. PSA(0.27~0.31)보다 낮은 건 PSA가 그나마 주방/유류라서로 해석. 방향(recall↓·precision↑)은 확정, 절대값은 도메인 의존.
- **다음 실험(핵심): 실제 화재를 train에 넣으면 recall이 오르나** — Indoor Fire Smoke train으로 학습 → 그 test로 평가(누수 차단). 이게 "실데이터 투입=도메인갭 축소" 직접 검증.

### 실험 A — 합성 vs 실 vs 혼합 (완료, 2026-08-24)
- **목적**: 실데이터 학습이 recall(놓침)을 올리나.
- **데이터**: Indoor Fire Smoke를 **fire-only(1클래스)**로 재구성(smoke 제거, 합성과 공정 비교). **70/15/15 = train 3500 / valid 750 / test 750(fire 392·nofire 358).**
- **라벨 육안 감사 통과**: nofire=연기·수증기·가습기 미스트·향연기(불꽃 없음, 좋은 하드네거) · fire=박스가 실제 불꽃 위. grease류 체계적 오라벨 없음 → precision·fpr 신뢰 가능.
- **3조건 · 같은 test**: ① 합성-only(기존 v8_C0_s1) · ② 실-only(Indoor train) · ③ 혼합(합성 synth_C0 + Indoor train). **val=Indoor valid(실제 val → "합성 val 손해" 해소).**
- **지표**: frame-level recall/precision/fpr. 스크립트 `colab_indoorfire_train.py`(EPOCHS 60 cap·patience 15·학습곡선 자동표시).
- **경계**: 일반 실내 화재(주방/유류 아님) → "실데이터가 실화재 recall 올리나"는 답하나 급식실 성능은 별개. Roboflow 랜덤분할 약한 누수 가능.
- **결과 (같은 Indoor test · frame-level · 각 60ep)**:

  | 조건 | recall | precision | fpr | 검출(fire) |
  |---|---:|---:|---:|---:|
  | ① 합성-only | 0.235 | 0.868 | 0.039 | 92/392 |
  | ② 실-only | **0.985** | **0.990** | 0.011 | 386/392 |
  | ③ 혼합 | **0.985** | 0.980 | 0.022 | 386/392 |

  - **판정**: ② ≫ ① → **놓침(recall) 병목은 실데이터로 해소**(0.235→0.985, 놓침 300→6). ③≈② (혼합이 precision·fpr 미세하게 나쁨) → **합성 무기여**(v1~v3 정합).
  - **⚠️ 절대값 경계**: 같은 도메인 test + 프레임 랜덤분할 누수 → 0.985 부풀었을 가능성 큼. **방향만 견고.** 다음 = 씬/영상 분할 재측정 · 급식실 근접 도메인이동 평가(§새 세션 첫 작업 2).
  - json `indoorfire_eval/indoorfire_train.json` · 모델 `runs_if/(real_only|mixed)/weights/best.pt`.

## 회의 후 신규 방향 메모 (2026-08-24)

**전이 낮음의 기여 요인 (주→부):**
- **주**: 학습 데이터가 **합성**(합성↔실제 분포 간극) — ablation/v2/v3로 확인.
- **부**: 모델 선택·하이퍼파라미터를 **합성 val로** 튜닝 → 실제에 최적 아닌 체크포인트(§5.7에서 인지). "합성 val 손해"는 독립 원인이라기보다 **학습 전 과정이 합성에 정렬**된 것의 일부.
- **부**: 라벨/평가 기준 차이 등.

**"실제 val 확보" (팀 피드백 '분할 재점검'의 올바른 방향):**
- 의미 = 실제 화재 데이터를 모아 일부를 **val(모델 선택 기준)**로, (선택) **train에도** 섞기.
  - (a) val로만: 학습은 합성, **고를 때만 실제 기준** → "합성 val 손해" 직접 해소.
  - (b) train에도: **도메인 갭 직접 축소**(라벨링 비용·양↑).
- 필요조건: val/test 겹치지 않게 나눌 **충분한 독립 장면**(realfire 5개론 불가) · 도메인 일치+하드네거(§5.1) · **val 튜닝 시 test 불가침**.
- 트레이드오프: 실제를 val/train에 쓰면 "순수 sim2real 전이 측정"은 약해지나 **실전 성능↑**(신규 방향은 실전 성능 지향).
- ⚠️ **val=test는 누수 → 금지.**

**실화재가 급식실이어야 하나 — 아니오(도메인 스펙트럼):** 이상=급식실 유류화재(자료 부재) · **양호(현실적 대체)=일반 주방/튀김유 화재**(현 realfire 5편 부류) · 약함=산불·건물 등 일반 불(피함, §5.1). 두 축=①불 종류(유류) ②장면(주방); 배포(급식실 조리)에 가까울수록 대표성↑.

**회의 피드백 적용도(요약):** 🟢 error-analysis(놓침/헛불 원인 분석)·train/val loss 곡선·프레임 중복 정리 / 🟡 생성형 AI(gpt-image·nano-banana→Seedance 영상, **장면·temporal** 겨냥) — 평가는 반드시 실제로·generative artifact 주의·급식실 도메인 근접 확인 / 🟡 Roboflow·Label Studio(§5.1 검증용) / 🟠 합성 불꽃 퀄만↑·edge필터·baseline 교체(우리 결과와 상충, 소규모 검증부터) / 🔴 val=test(누수). 근거: 이 대화·pre-reg v2 §10·§11.

## (이전) v2/v3 요약
> **v2 Phase 1 완료. 결과 = C3 ≈ C0 (값진 음성 — 불꽃 현실성/다양성은 실제전이 병목이 아님).**
> ✅ ① ablation 가드 완료(통과·지름길 아님) · ✅ ② 결과 문서화 완료(pre-reg §10 + README v2 결과 절).
> **✅ ③ 방향 결정: DINOv3(표현 축). Phase 0 프로브 완료 → 음성 근접(진짜지만 작고 비확장).**

## ✅ 이번 세션(2026-08-22) 완료
- **ablation 가드 통과** — v8_C3_s1(합성 test 173양성·136음성): flame_rate 0.942 · bg_fp_rate 0.000 · 평균conf 불꽃0.739/배경0.000 · 음성오탐 0/136. → C3 모델은 불꽃을 필요로 함(지름길 아님). 음성이 "C3 망가짐"이 아니라 "진짜 전이 안 됨"임을 확정.
- **문서화 완료** — `docs/PREREGISTER_v2.md §10` 결과 채움 + 상단 상태줄 갱신 · `README.md` "v2 결과" 절 신설(SVG 링크 포함) + 상단 로드맵 v2 줄 갱신. **아직 커밋 안 함 — 커밋/푸시 필요.**

## 한 줄 상황
v2(실사 아틀라스 + 발광 합성)로 **C0(컷아웃) vs C3(발광) 단일변수 A/B** 완료.
realfire 영상단위: **C0 0.237±0.213 · C3 0.223±0.188 · delta −0.014 → C3 ≈ C0.**
불꽃을 발광·다양하게 해도 실제 화재 전이가 안 올랐다 → **가설 기각(pre-reg §7 음성).**
이전 kitchen-fire-poc(3D 아틀라스·셰이더로도 미해결)와 정합.

## v2 Phase 1 결과 (핵심 수치)
- realfire 영상단위(5영상 · 각 5seed 평균 · ±군집CI): **C0 0.237±0.213 · C3 0.223±0.188.**
- 헛불: C0 0.227±0.305 · C3 0.156±0.260 (C3 약간 낮으나 CI 큼).
- 영상별(C0/C3): jikken 0.40/0.20 · tokyo 0.32/0.41 · kanetsu 0.10/0.18 · grease 0.36/0.32 · **hisomu 0.01/0.01**.
  → **일관된 방향 없음(2↑ 2↓ 1tie)**. hisomu(거대 불꽃) 둘 다 ~0 = 사각(불꽃 종류 아닌 별개 문제).
- **경계:** 영상 5개뿐 → CI 거대. "큰 효과 없음"은 확실, **작은 효과는 못 배제** → realfire 확장으로 확인 필요.
- 실측 그림: `docs/img/v2_result_realfire.svg`. json: Drive `fire_frames/v2_eval/v2_eval_v8.json`(id 1AYlLBF2CVG5tlDOX4TL3VkAiOki8ATi2).

## 완료된 것 (v2)
- **S0** flamelib 390장·8소스(v01·02·03·05·06·07·09·10)·전부 RGBA (kitchen-fire-poc repo `assets/flamelib`). **0-overlap 통과**(스톡화염 ↔ realfire 튀김유화재).
- **S1** `colab_synth.py` C0~C3 MODE 스택 + 가시성 게이트. 근거 pre-reg §5.5·§5.6·§5.7.
- **S3** `synth_C0`·`synth_C3`(Drive). C0/C3 카운트 완전 동일(단일변수). train 1385양성/335하드네거/558음성.
- **Phase 0** 컷아웃 vs 발광 육안 통과.
- **Phase 1** 학습 10모델(v8 · C0·C3×5seed · EPOCHS=60 · `runs_phaseB/v8_{C0,C3}_s1..5`) + realfire 평가.

## 스크립트 (scripts/)
- `colab_synth.py`(C0~C3+게이트) · `colab_v2_train.py`(10모델·resumable) · `colab_v2_eval.py`(realfire 영상단위±CI·판정)
- `colab_v2_probe.py`(epoch 측정→60·버림) · `colab_v2_ablation.py`(지름길 가드) · `colab_overlap_check.py`(0-overlap)
- 로컬 생성기(scratchpad, 미커밋): `gen_v2_result_svg.py`(eval json → v2_result_realfire.svg) · `gen_svgs.py`(노이즈 강건성 SVG)

## 결정 기록 (docs/PREREGISTER_v2.md)
- §5.5 라벨박스=**알파bbox 고정** · 블렌딩=**스크린+코어블룸** (밝은 급식실서 가산은 클리핑)
- §5.6 아틀라스 분할 **train 348(6소스) / test 42(v05·v09 held-out)** — 이미지 분할과 매칭
- §5.7 **EPOCHS 60** (프로브 측정: best55·수렴52·val ep40 후 평평 → 80은 과함)

## ★ 다음 세션 첫 작업
0. **커밋/푸시** — 이번 세션 문서 변경(pre-reg §10 · README v2 결과 절 · 이 HANDOFF) 아직 미커밋.
1. ~~ablation 가드~~ ✅ 완료(통과) · 2. ~~결과 문서화~~ ✅ 완료 (위 "이번 세션 완료" 참고).
3. **결정됨 (2026-08-22): ③ DINOv3 방향 · Phase 0 프로브 먼저.** 10h GPU + 미팅 제약 → 전면 탐지기 재구축(A) 대신 **고정특징 전이 프로브**로 "표현 축이 답인가"만 싸게 단독 검증. 설계·성공기준 = **`docs/PREREGISTER_v3.md`**(결과 전 확정). 팀 제안 근거 = pre-reg v2 §11.

### v3 Phase 0 프로브 실행 (Colab · L4)
```python
%run /content/kitchen-fire-noise-poc/scripts/colab_v3_probe.py
# 선택 env: PROBE_BACKBONE='dino,yolo_synth,yolo_coco,resnet' · DINO_HUB='facebookresearch/dinov2/dinov2_vitb14'(강제) · MAX_PER_CLASS=1500
```
- 비교: DINOv3(폴백 DINOv2) vs YOLO(합성 v8_C0_s1) vs YOLO(COCO) vs ResNet50, 전부 고정특징+선형/kNN.
- 주지표 realfire AUROC(영상단위±CI). 판정 GO(Δ≥+0.10) / NO-GO(|Δ|<0.05) / 애매 — pre-reg v3 §6.
- 산출: Drive `fire_frames/v3_probe/v3_probe.json` + `v3_probe.png`(미팅 막대그림).
- **함정:** DINOv3 가중치는 **라이선스 게이트(fbaipublicfiles 403)** — torch.hub 익명 다운로드 불가. **진짜 DINOv3는 HF 인증 경로로만**: ① https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m 에서 로그인 후 **라이선스 동의**, ② https://huggingface.co/settings/tokens 에서 read 토큰 생성, ③ Colab에서 실행 전 `import os; os.environ['HF_TOKEN']='hf_...'`(토큰은 본인만 — 공유 금지), ④ `%run colab_v3_probe.py`. 토큰 없으면 자동 DINOv2 폴백(로그 used_dino 표기). 프로브는 학습 없음(고정) → GPU 몇 시간, 10h 여유.
- **진짜 DINOv3 base·large 실행 완료**(HF 인증·register 4). vitb16: realfire 0.660±0.348·kNN 0.660·Δ+0.037. **vitl16: 0.676±0.287·kNN 0.649·Δ+0.052.** **스케일 3.5배→ realfire +0.016뿐(비확장).** sanity 둘 다 <0.90(0.896·0.882) → **공식 보류.**
- **실질 결론: 표현 축은 "진짜지만 작고 비확장" 신호 → 전이 병목 레버 아님.** DINOv3만 kNN~0.65로 우연 초과(진짜 신호, DINOv2는 0.500 가짜)나, 두 스케일 모두 GO(+0.10) 근처도 못 감. → **DINOv3 탐지기 착수 근거 약함.** 상세 pre-reg v3 §8·MEETING_v3_probe.md.
- **다음 = ③ 씬·도메인·temporal 피벗 권장.** (v2 불꽃 아님 + v3 표현도 레버 아님 → 병목은 씬/도메인갭/시간축). 표현 축 종료 권장(재방문 여지만), realfire 확장은 GO 미달이라 우선순위 낮음.
- **DINOv3 승인 대기 상태**: HF 라이선스 동의·요청 제출됨, **Meta 수동 승인 대기**(즉시 아님). 상태 확인 https://huggingface.co/settings/gated-repos → Accepted 되면 토큰 넣고 재실행.
- **미팅 브리프**: `docs/MEETING_v3_probe.md`(현재까지 정직 요약·결정지점).

### (B) realfire 확장으로 검정력↑ (DINOv3 승인과 무관·지금 가능)
N=5 거대 CI가 프로브의 진짜 약점 → 실제 화재 이미지를 추가해 좁힌다(학습 0, 프로브에 그대로 얹힘).
1. Roboflow 등에서 **유류/주방 화재 이미지(fire) + 불 없는 주방 이미지(nofire)** 확보(외부 다운로드·본인 계정).
2. Drive에 배치: `REALFIRE_EXTRA/fire/<source>/*.jpg` · `REALFIRE_EXTRA/nofire/<source>/*.jpg` (source 하위폴더 = 독립 씬=군집 CI 단위, 여러 개일수록 좋음).
3. 실행: `import os; os.environ['REALFIRE_EXTRA']='/content/drive/MyDrive/fire_frames/realfire_extra'` 후 `%run colab_v3_probe.py`.
- 착시 차단(pre-reg §2·§4) 유지: 추가분도 **학습에 안 쓴 실제 불꽃·장면**이어야 함. Roboflow를 **검증에만** 쓰고 학습 소스로 직접 넣지 말 것(triage §11).
- GO면 남은 시간에 (A) 고정백본 탐지 헤드 착수(보너스). 경계: 프로브=분류 proxy지 최종 탐지 아님·5영상 CI 큼.

## 미완 / loose end
- **Phase 2(v11 미러·성분분해 C1·C2)는 음성이라 기본 안 열림** — 방향 전환이 우선.
- ~~pre-reg §10·README v2 결과 절 미작성~~ ✅ · ~~ablation 미실행~~ ✅.
- **문서 변경 커밋/푸시 남음** (2026-08-22 세션 산출).

## Colab 재개 (seochorobotics · L4 GPU · 매 런타임 clone)
```bash
!rm -rf /content/kitchen-fire-noise-poc && git clone https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
!rm -rf /content/kitchen-fire-poc && git clone https://github.com/K-H-MOON/kitchen-fire-poc.git /content/kitchen-fire-poc   # 아틀라스
```
- 학습 10모델: Drive `fire_frames/runs_phaseB/v8_C0_s1..s5 · v8_C3_s1..s5` (best.pt 저장됨).
- realfire 영상: Drive `smoke_frames` (`real_fire.json` src_dir). 5영상 fire/nofire.

## 함정
- **프로브**(`colab_v2_probe.py` · `/content/probe` · 측정용·버림)와 **본 학습**(`colab_v2_train.py` · `runs_phaseB` · 10모델) **혼동 주의** — 둘 다 v8_C0_s1로 시작함.
- Drive 커넥터로 json/csv 텍스트는 읽힘(이미지 픽셀은 못 봄 → 시트는 채팅 첨부).
- 작업 방식: 옵션+trade-off·사용자 결정·**과장 금지·주장 경계 명시**. clean/합성분포 성적을 전이로 오독 금지.
