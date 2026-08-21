# HANDOFF — 다음 세션 이어가기 (2026-08-20 갱신)

> v1 PoC 완료·문서화 끝. **방향 결정됨** (README 상단 배너): v2=실사 아틀라스(병행) · v3=시뮬레이션(전환).
> **다음 작업 = v2 파이프라인 설계** (불꽃 다양성 + 합성 퀄리티 올리는 방안 → 파이프라인이 어떻게 바뀌나).

## 한 줄 상황
v1 결론: 합성 파이프라인이 **지름길 아닌 진짜 불꽃 검출기**(ablation 통과)·**노이즈 강건성 성립**·**v8≈v11**이나,
**실제 불 전이 약함**(realfire **recall 0.31**·precision 0.80). 병목 = **합성 불꽃의 현실성·다양성**(납작한 컷아웃 + 소재 4종).
→ **v2(실사 아틀라스+발광 합성)로 병목 공략**, v3(시뮬레이션)는 시뮬레이터 구축 동안 병행.

## v1 핵심 수치 (팀 질문 답에도 씀)
- 데이터: 배경 2,881(train 2,278/val 294/test 309, 사이트 단위) · 학습입력 train **4,556**(2× 증강) · **불꽃 소재 train 4종/test 2종** · realfire 검증셋 5영상(불꽃 286 + 발화전 244 프레임).
- 지표: 합성 test mAP@0.5 ≈0.97 · **realfire recall 0.31**(88/286, 하한)·**precision ≈0.80**(88/110)·발화전 오탐 0.09(22/244). → **낮은 건 recall(놓침)이지 precision 아님.**
- 모델: YOLOv8s(COCO-pretrained, 단일클래스 fire) + YOLO11s 미러(v8≈v11). 검증 = v8_baseline_s1.

## 로드맵 (README 최상단 배너에 정식 기재)
- **v1 (완료)** — 실제 주방+합성 불꽃 학습 → 실제 화재 검증. 결과 0.31.
- **v2 (기존 계획·병행)** — kitchen-fire-poc **실사 화염 아틀라스 + 발광 합성**으로 병목 공략. v1 하네스 재사용 → **~3~5일**(v1보다 짧음). 상세 `docs/PREREGISTER_v2.md`.
- **v3 (전환)** — 시뮬레이션 주방화재 학습 → 합성 검증 (sim→합성→실제 커리큘럼). 시뮬레이터 구축에 시간 필요 → 구축 동안 v2 병행.

## ★ 다음 세션 첫 작업 — v2 파이프라인 설계
사용자가 "불꽃 다양성 + 합성 퀄리티를 올리면 파이프라인이 어떻게 되나"를 물음. 답해줄 v2 파이프라인:
1. **S0 소재 확보**: 아틀라스는 **이미 확보** — kitchen-fire-poc repo `assets/flamelib`(680 WebP). Colab에서 `git clone` → `FLAME_ATLAS=/content/kitchen-fire-poc/assets/flamelib`. **분리 점검**: flamelib 원본 스톡화염 ↔ realfire 5영상 0 overlap 확인.
2. **S1 합성 개선**(핵심): `colab_synth.py`(v1 컴포지터)에 **가산/스크린 블렌딩 · 엣지 페더 · 색보정 · 코어 블룸 · 가짜 조명 스필**(배경 글로우) 추가. + 불꽃 외형 증강(색/밝기/스케일 지터).
3. **S2 Phase 0**(값쌈, 재학습 없음): 몇 장 합성 → `colab_flame_compare.py`로 v1 컷아웃과 육안 비교. 별로면 S1로 되돌림.
4. **S3 합성 데이터**: C0(=v1)·C3(=풀 v2) synth 생성. 배경·분할 고정.
5. **Phase 1 핵심 A/B**: `colab_phaseB_train.py`로 C0 vs C3(baseline·5seed=10모델) → `colab_realfire_test.py`(주지표)·`colab_ablation.py`(가드)·`colab_phaseB_eval.py`(노이즈 가드). **게이트: C3 realfire ≫ C0 → 진행 / ≈ → 병목은 불꽃 아님, 방향전환.**
조건 C0~C3 정의·성공기준·해석트리 = `docs/PREREGISTER_v2.md`.

## 이번 세션 추가 스크립트 (scripts/)
- `colab_flame_compare.py` — 3단 비교(A 우리합성 · B 실제불 자동검출 · C 아틀라스). `FLAME_ATLAS`=flamelib. 문서 삽화 `docs/img/flame_compare.jpg`.
- `colab_noise_examples.py` — 노이즈 9종 그리드(한글 라벨·설명, NanumGothic 자동설치). `docs/img/noise_grid.jpg`.
- (기존 v1: `colab_phaseB_train/eval.py`, `colab_ablation.py`, `colab_realfire_test.py`, `colab_synth.py`, `noise_lib.py`, `real_fire.json`.)

## 저장소 문서 상태 (전부 커밋·푸시됨, main)
- `README.md`: 상단 **방향 전환 배너(v1/v2/v3)** · 결과 요약(성과/한계/**병목 관찰+3단 삽화**/외적타당성/요약) · 노이즈 9종 그리드 삽화 · **v2 상세** 절.
- `docs/TIMELINE.md`: v1 전 과정 + 결과 + 최종 결론.
- `docs/PREREGISTER_v2.md`: v2 사전등록(착시 차단·C0~C3·성공기준·Phase). 아틀라스 flamelib 회수 반영됨.
- `docs/img/`: flame_compare.jpg(3단), noise_grid.jpg.

## 미완 / loose end
- **팀 공유용 아티팩트** — 팀원이 물은 "데이터규모·지표·모델" 3질문 답을 슬라이드로 만드는 중이었음. `scratchpad/fire_recall_qa.html` 작성됨(Q1 데이터/Q2 recall·precision/Q3 모델 표 + 병목 결론 + flame_compare 이미지 자리). **아직 이미지 임베드·발행 안 함.** 원하면 다음 세션에서 base64 임베드 후 Artifact로 발행. (내용은 위 'v1 핵심 수치'와 동일.)

## Colab 재개 (seochorobotics · GPU · 매 런타임 clone)
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
!rm -rf /content/kitchen-fire-noise-poc && git clone https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
!rm -rf /content/kitchen-fire-poc && git clone https://github.com/K-H-MOON/kitchen-fire-poc.git /content/kitchen-fire-poc   # v2 아틀라스(flamelib)
```

## 함정
- **Drive `fire_project`(kitchen-fire-poc 데이터) 비어 있음** — 아틀라스는 **repo `assets/flamelib`에 살아있음**(그걸 씀). Drive 아님.
- **절차적 셰이더(v3) 코드는 repo에 없음**(에셋 전용) — Drive `fire_test1.ipynb`에 있을 수도 / v3 때 새로 작성 가능.
- Colab `/content` 런타임마다 초기화 → 매번 clone. runs_phaseB 등 산출물은 Drive에 영속.
- **Claude Drive MCP는 이미지 픽셀 못 봄** — 시트·삽화는 사용자가 채팅에 첨부해야 Claude가 봄. (로컬 파일은 Read로 봄.)
- 파일명 한글/일본어 → NFC 정규화.

## 작업 방식 (사용자 선호)
옵션은 trade-off와 함께 제시·사용자 결정·**과장 금지·한계 정직**. **모든 답변에서 주장의 경계 명시.** 판정은 seed/CI로, 착시(같은 텍스처)·의사반복(프레임 상관) 경계. clean을 성능으로 오독 금지.
