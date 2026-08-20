# HANDOFF — 상태 (2026-08-20 · 완료)

> Phase B 30회 학습 + eval + 실제화재 + ablation 전부 완료. 결론까지 문서화됨.
> **최종 수치·결론**: `docs/TIMELINE.md` 의 `결과`·`최종 결론`. 요약: `README.md` 의 `결과 요약`.

## 한 줄 상황
**PoC 완료.** 합성에서 노이즈 강건성 기제 성립·아키텍처 견고(v8≈v11)·배경 지름길 아님을
확인했으나, **합성 불꽃의 다양성이 좁아 실제 불 전이는 약함**(0.31, 하한). 노이즈 강건성 증강도
실제 전이를 못 도움(오히려 헛불 증가).

## 무엇을 돌렸나 (전부 완료 · 산출물은 Drive `fire_frames/`)
- **학습 30회**: `runs_phaseB/{v8,v11}_{baseline,modelA,modelB}_s{1..5}/best.pt` (2아키 × 3config × 5seed)
- **eval**: `phaseB_eval/phaseB_agg.json` + `v8/v11_flame.png`·`_fp.png` (config별 5seed 평균±95%CI 곡선)
- **ablation**: v8_baseline_s1 — 불꽃 0.808 ≫ 배경 0.005, 음성 오탐 0/116 (**콘솔 출력만, json 없음**)
- **realfire baseline**: `realfire/realfire.json` — 0.308 / 0.090. 시트 `_realfire.jpg`(놓친 불꽃·헛불)
- **realfire modelA**: `realfire_v8_modelA_s1/` — 0.322 / 0.164

## 재실행 (필요 시 · seochorobotics 계정 · GPU · 새 런타임마다 clone)
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
!rm -rf /content/kitchen-fire-noise-poc && git clone https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
```
```python
%run /content/kitchen-fire-noise-poc/scripts/colab_phaseB_eval.py       # eval (배치, 캐시라 재실행 즉시 skip)
%run /content/kitchen-fire-noise-poc/scripts/colab_ablation.py          # ablation (콘솔 출력)
%run /content/kitchen-fire-noise-poc/scripts/colab_realfire_test.py     # realfire baseline
# modelA 등 다른 모델: os.environ['REALFIRE_MODEL']='v8_modelA_s1' 후 위 realfire 실행
```
- 학습은 더 돌릴 일 없음(30회 완료).

## 남은 것 (선택 · 결론엔 불필요)
- **realfire 경계 조이기**: `real_fire.json` 의 fire_shots 에서 연기 가림·경계 프레임 제외 → 0.31 상향 재측정.
- **향후 개선 방향**: 실제 불 전이를 높이려면 **불꽃 소재 다양성 확대**가 병목(현재 학습 소재 4종).
- **LONO**(노이즈별 leave-one-out): 미실시 — 결론이 충분히 나와 생략.

## 이번 세션의 스크립트 변경
- `colab_phaseB_eval.py`: 한장씩 predict → **배치 추론**(결과 불변) + `PHASEB_VERIFY` 등가검증 모드.
- `colab_realfire_test.py`: 시트 stale-경로 버그 수정 + **놓친 불꽃 위주**로 재구성 + `REALFIRE_MODEL` env(모델별 출력 분리).

## 함정 / 참고
- Colab `/content` 는 런타임마다 초기화 → 매번 clone. Drive 산출물은 영속.
- **Claude Drive MCP 는 이미지 픽셀 못 봄** — 시트(`_realfire.jpg`)는 사용자가 채팅에 첨부해야 보임.
- 파일명 한글/일본어 → NFC 정규화로 매칭(스크립트 반영됨).

## 작업 방식 (사용자 선호)
옵션은 trade-off 와 함께 제시·사용자가 결정·**과장 금지·한계 정직**. **모든 답변에서 주장이 덮는
범위와 안 덮는 경계를 명시**(2026-08-20 명시 요청). 판정은 seed/CI 로, clean val 을 성능으로 오독 금지.
