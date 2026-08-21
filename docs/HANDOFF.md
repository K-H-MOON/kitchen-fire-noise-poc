# HANDOFF — 다음 세션 이어가기 (2026-08-21 갱신)

> **v2 Phase 1 완료. 결과 = C3 ≈ C0 (값진 음성 — 불꽃 현실성/다양성은 실제전이 병목이 아님).**
> 다음 = ① ablation 가드(음성 성격 확정) → ② 결과 문서화 → ③ realfire 확장 or 방향 전환.

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
1. **ablation 가드 (아직 안 돌림)** — v8_C3_s1이 **합성 분포 안에선 여전히 불을 보는지** 확인(음성이 "C3 망가짐"이 아니라 "진짜 전이 안 됨"임을 확정).
   ```python
   import os; os.environ['ABLATION_MODEL']='v8_C3_s1'
   %run /content/kitchen-fire-noise-poc/scripts/colab_v2_ablation.py   # 비교: v8_C0_s1
   ```
2. **결과 문서화 (아직 안 함)** — pre-reg §10 결과 채움 + README v2 결과 절(음성 정직 기록) + `v2_result_realfire.svg` 링크.
3. **택1:** ② **realfire 확장**(Roboflow 유류화재 추가 · 학습0 · n↑ CI↓ · ≈ 확정) / ③ **방향 전환**(배경·씬 도메인갭 · temporal · 표현/백본 **DINOv3** — 팀 제안, 음성이라 이제 더 유력).

## 미완 / loose end
- **Phase 2(v11 미러·성분분해 C1·C2)는 음성이라 기본 안 열림** — 방향 전환이 우선.
- pre-reg §10·README v2 결과 절 미작성.
- ablation 미실행.

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
