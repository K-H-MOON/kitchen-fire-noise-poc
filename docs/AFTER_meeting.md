# 미팅 이후 과정 요약 (2026-08-24~, 진행 중)

> [미팅용 요약(SUMMARY_meeting.md)](SUMMARY_meeting.md)이 v1~v3까지라면, 이 문서는 **회의 후 새 방향의 진행 기록**이다. 결론: v1~v3로 **병목 = 데이터/도메인(합성↔실제 간극)**으로 좁혀졌고, 회의 후 **실제 화재 데이터**로 이를 공략 중.

## 회의에서 정해진 방향
- v1(불꽃 현실성)·v2·v3(표현/DINOv3) 모두 실제 전이 못 올림 → **병목은 모델이 아니라 데이터/도메인.**
- → **실제 화재 데이터**를 확보해 학습/평가에 투입(합성만으로는 진짜 불을 못 배움).

## 단계별 진행

**1. "무엇을 놓치나" 진단 (error-analysis)**
기존 모델을 실제 화재에 돌려 놓침 패턴 분석. → 그 과정에서 **평가셋 오염 발견**: 기존 realfire 5편 중 3편(jikken·kanetsu·hisomu)이 **편집 소방 홍보(PSA) 영상**(자막·PIP·만화 불)이라 "불 라벨인데 실제 불 없음" → 놓침 과다 집계. **기존 realfire recall(0.31 등)은 신뢰 불가로 판명.**

**2. 실제 데이터 확보·검증**
- **Indoor Fire Smoke** (5000장·YOLO·fire/smoke, 실촬영 실내 화재) — 주 대상.
- New_sample (750장·AI-Hub·야외) — 후순위.
- **육안 감사 통과**: Indoor은 실촬영이고 라벨 신뢰할 만함(nofire=연기·미스트 등 하드네거, fire=박스가 실제 불꽃). 단 **주방/유류 아닌 일반 실내 화재**(도메인 부분 일치).

**3. 깨끗한 첫 측정 (오염 없는 자로)**
기존 합성 모델을 Indoor Fire Smoke에 측정 → **recall 0.201 · precision 0.846 · fpr 0.040.**
- **낮은 recall은 진짜** (PSA 오염 탓 아님). **precision 높고 헛불 적음 → 문제는 놓침(recall).** 확정.

**4. 놓침 원인 진단**
놓친 이미지 육안(`_miss.jpg`): **크고 화면을 채우는 불·다양한 장면을 놓침.** → 우리 합성이 **작은 불꽃 스프라이트(화면 15~45%)만** 얹어서 모델이 "작은 국소 불꽃"에 과적합 → **크고 다양한 실제 불을 못 알아봄.** (작은 뚜렷한 불은 잘 잡아 precision 높음.) **고칠 수 있는 병목.**

**5. 실험 A — 합성 vs 실 vs 혼합 (진행 중)**
"**실제 화재를 학습에 넣으면 놓침이 줄어드나?**" 직접 검증.
- 데이터: Indoor Fire Smoke fire-only(1클래스), 70/15/15 = train 3500 / valid 750 / **test 750(fire 392)**.
- 3조건 · **같은 실제 test**: ① 합성-only(기존) · ② 실-only · ③ 혼합(합성+실). val=실제(Indoor valid).
- 지표: recall/precision/fpr. → **결과 (학습 후 채움).**

## 결과 (채워질 예정)
| 조건 | recall | precision | fpr |
|---|---|---|---|
| 1. 합성-only | — | — | — |
| 2. 실-only | — | — | — |
| 3. 혼합 | — | — | — |

- 읽는 법: 2·3 ≫ 1 → 실데이터가 놓침을 줄임 / 3 vs 2 → 합성 기여 여부.

## 정직한 경계
- Indoor Fire Smoke는 **일반 실내 화재(주방/유류 아님)** → "실데이터가 실화재 recall 올리나"는 답하나 **급식실 성능은 여전히 별개**(급식실 실화재 자료 부재).
- Roboflow 랜덤분할 → 인접 프레임 약한 누수 가능.
- 기존 realfire(PSA)는 오염 → 절대 수치 신뢰 불가(상대 비교는 견고).

## 관련 문서·스크립트
- 수집 사양·놓침 진단: [DATA_collection_spec.md](DATA_collection_spec.md)
- 진행 로그·실험 A 설계: [HANDOFF.md](HANDOFF.md)
- 스크립트: `colab_realfire_erroranalysis.py` · `colab_indoorfire_eval.py` · `colab_indoorfire_train.py`
