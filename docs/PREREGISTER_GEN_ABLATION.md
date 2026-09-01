# 사전 등록 — 생성형 합성(gen) 혼합비 ablation

> **한 줄**: 누수 통제된 실 화재 학습셋에 **생성형 합성(gen_synth)** 을 **가산**으로 섞을 때
> (real-only · 1:1 · 1:3 · synth-only), 배포 대표 실화재 hold-out(`oilfire_realtest`)에서
> **recall(장면단위) + fpr_급식실** 이 좋아지는가, 어느 비율에서 좋아지는가.
> **결과를 보기 전에** 읽는 규칙·한계를 여기 못 박는다(1회차 원칙: 못 읽게 되는 걸 미리 막는다).

작성 2026-08-27 · 대상 데이터 = 기존 생성형 합성(NB Pro·Codex·FLUX 혼합, Roboflow 박스) ·
근거 비평 = "판별기/Grad-CAM/hold-out mAP/박스규칙" 4종 검증 배터리 중 **③ hold-out** 를
우리 데이터에 맞게 번역한 것.

---

## 0. 왜 "mAP"가 아니라 recall + fpr 인가 (데이터 현실)

비평 ③은 **boxed 실화재 hold-out에서 mAP** 를 전제한다(NIST FCD 류). 그러나 우리 배포 대표
hold-out `oilfire_realtest` 는 **박스가 없는 frame-level 셋**이다(`colab_realtest_eval.py` 가
`fire/·nofire_kitchen/·nofire_presrc/` 에서 검출 여부만 잰다 — 메모리: "박스 아님(frame-level 프록시)").

→ **우리 데이터에서 ③의 정직한 번역은 mAP 가 아니라 매칭 conf 에서의 recall(장면단위) + fpr_급식실**
이며, 이는 마침 실제 배포 결정 지표다(메모리: fpr_급식실 = 배포 지표). mAP 를 원하면 Indoor
grouped **val**(boxed)에서 arm별로 공짜로 나오지만, 그건 real-train 과 동일 도메인이라 **배포
조리불 전이를 못 재는 부차 지표**로만 본다.

- **주 지표**: `oilfire_realtest/fire` 장면단위 recall(평균±std, N=장면수) · `nofire_kitchen` fpr(=fpr_급식실) · `nofire_presrc` fpr
- **부차 지표**: Indoor grouped `valid` box mAP50 / mAP50-95 (arm별, 도메인-내, 해석 주의)

---

## 1. 질문 · 가설

- **질문**: 내 실 화재 학습셋에 gen 합성을 **더하면**(가산) 배포 실화재 탐지가 좋아지나? 어느 비율에서?
- **H1(희망)**: 가산 arm(1:1 또는 1:3)이 real-only 대비 **매칭 recall 에서 fpr_급식실 을 낮춘다**(또는 매칭 fpr 에서 recall 을 올린다), 시드 노이즈를 넘는 폭으로.
- **H0(귀무)**: 가산 arm 이 real-only 와 시드 노이즈 안에서 구별 안 됨(무기여).
- **H2(해로움)**: 가산 arm 이 real-only 보다 나쁨(매칭 recall 에서 fpr 상승).

과거 맥락(단정 아님): 커리큘럼 2gencurr(gen 사전학습→real 파인튜닝)은 fpr_급식실 을 **악화**시켰고
sc14 만 이득이었음. 이 ablation 은 커리큘럼이 아니라 **직접 혼합**이라 다른 실험이며, 어느 쪽도 미리 단정하지 않는다([[no-premature-conclusions]]).

---

## 2. 설계 (가산 · 3시드)

고정(모든 arm 공통): 아키텍처 yolov8s(`yolov8s.pt`) · epochs 60 · imgsz 640 · patience 15 ·
val = 공유 실 `if_fire_grouped/valid`(부차 mAP 비교 가능) · 평가 hold-out = `oilfire_realtest`(고정).

| arm | 학습 train 구성 | 총 이미지(가산) |
|---|---|---|
| **synth-only** | gen 합성 전부 | G |
| **real-only** | 실 train 전부(R) | R |
| **1:1** | 실 train R + gen 표본 R | 2R |
| **1:3** | 실 train R + gen 표본 3R | 4R |

- **R** = `if_fire_grouped/train` 이미지 수(양성+음성 포함, 전체를 "1"로). **G** = gen 라벨 이미지 수.
- gen 표본은 **고정 표본 시드(1234)로 한 번 뽑아** 3개 학습 시드에 **동일 데이터**로 재사용(데이터 변동원 제거, 학습 시드만 변수).
- **3R > G 면 복원추출(중복 허용)** 로 채우고 로그에 중복 수 명시(중복 합성 = 약한 신호).
- 학습 시드 = **0, 1, 2**(`YOLO.train(seed=s, deterministic=True)`), arm×시드 = 4×3 = **12 학습**.

데이터 출처(모두 세션-로컬 재구성, Drive 미변경):
- 실 pool: `Indoor Fire Smoke.zip` → **dHash(Hamming≤6) 근접중복 그룹 → 그룹단위 70/15/15 재분할**(누수 통제, `split_audit` 로직 그대로 포팅) → `/content/if_fire_grouped`. fire-only(nc=1), 빈 라벨=음성.
- 합성 pool: `colab_gen_build.py`(`TRAIN=0`) → `/content/gen_synth`. Roboflow 박스, 클래스 전부 0(fire).
- **누수 점검**: 실 split 은 test(=Indoor test)와 그룹 분리됨. 평가 hold-out `oilfire_realtest` 는 실 학습셋과 **출처가 다른 별개 셋**(누수 아님). gen 은 test 와 무관.

---

## 3. 결정 규칙 (결과 보기 전 확정)

conf 를 쓸어(sweep) 각 모델의 (장면 recall, fpr_급식실) 곡선을 만든다. 비교는 **고정 conf 가 아니라 매칭 운영점**에서 한다(메모리: 고정 conf 비교는 conf-matched 아님).

- **읽기 ① 매칭 recall**: 기준 recall\* = real-only 3시드 평균의 conf 0.25 장면 recall. 각 arm/시드에서 recall\* 를 내는 conf 를 찾아 그 지점 **fpr_급식실** 을 읽음. **낮을수록 우수**.
- **읽기 ② 매칭 fpr**: 기준 fpr\* = real-only conf 0.25 fpr_급식실. 각 arm 에서 fpr\* 지점의 **장면 recall**. **높을수록 우수**.

**판정(사전 커밋)**:
- **gen 도움됨(확정)**: 가산 arm(1:1 또는 1:3)이 **읽기 ①에서 mean fpr_급식실 이 real-only 보다 낮되 그 폭이 시드 pooled std 를 초과**하고, **읽기 ②에서도 같은 방향**(매칭 fpr 에서 recall 높음).
- **무기여**: 가산 arm 이 두 읽기 모두에서 real-only 의 ±(시드 std) 안.
- **해로움**: 가산 arm 이 시드 노이즈를 넘어 나쁨.
- **synth-only**: 맥락용. real 에서 recall≈0 이면 "gen 단독은 전이 안 됨"(혼합은 별개). 낮게 나와도 혼합 판정과 분리해 읽음.
- **sc14(역대 최난 장면)**: 별도 보고하되 **N 작음 → 약한/탐색적 신호**, 확정 근거로 쓰지 않음(2gencurr 때도 sc14 이득은 취약했음).

**신뢰 비대칭 선언**: fpr_급식실 악화는 N 큼(견고) · 특정 장면 이득은 N 작음(취약). 하방 증거가 상방보다 견고하다는 걸 미리 인정한다.

---

## 4. 한계 · 교란 (미리 나열 — 사후 변명 금지)

1. **사이즈 교란(가산의 흠)**: arm 마다 총 이미지 수가 다름 → 이득이 "합성 덕"인지 "그냥 양이 늘어서"인지 못 가름. **가산이 이기면 총량-고정 후속으로 분리**(이번엔 안 함).
2. **frame-level hold-out**: 위치정확도 무시(관대) · mAP 아님. box mAP 는 도메인-내 Indoor val 에서만(부차).
3. **장면 N 작음**: CI 큼 · 단일 장면 스윙 가능 → 장면단위 std 항상 보고.
4. **복원추출**: 1:3 에서 3R>G 면 중복 → 유효 다양성 < 명목 장수. 로그에 중복 수 명시.
5. **gen 셋 특정성**: NB/Codex/FLUX 이 **특정 혼합** → 결론은 "이 gen 셋"에 SCOPE, "생성형 합성 일반"으로 확대 금지.
6. **도메인 이질**: real(Indoor, 일반 화재) · gen(생성 조리유불) · test(실 조리유불 프레임)가 서로 다른 도메인 → 혼합 효과와 도메인 정합이 섞임.
7. **3시드는 최소**: std 추정 자체가 노이즈. 유망하면 5시드로 승격.

---

## 5. 무엇이 H1 을 반증하나

가산 arm 이 **읽기 ①에서 real-only 대비 fpr_급식실 을 시드 std 넘게 낮추지 못하면** H1 기각(무기여 또는 해로움). 좋은 숫자가 안 나와도 **자료·기준을 바꿔 다시 돌리지 않고** 못 읽는/안 되는 이유를 적는다(1회차 원칙).

---

## 6. 사전 커밋 상수

```
아키텍처   yolov8s.pt
epochs     60      imgsz 640      patience 15      deterministic True
학습 시드   0, 1, 2
표본 시드   1234 (gen 부표본, 전 학습시드 공유)
split      dHash Hamming≤6 · 그룹 70/15/15 · split 시드 0 (split_audit 동일)
conf sweep 0.05 … 0.85 step 0.05  (+ 매칭점 보간)
평가        oilfire_realtest/{fire, nofire_kitchen, nofire_presrc}
출력        {FIRE}/indoorfire_eval/gen_ablation_result.json (신규 파일) ·
           모델 {FIRE}/runs_genabl/<arm>_s<seed>/ (신규 네임스페이스, 기존 미덮음)
```
