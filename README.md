# 합성 화재 인식 + 노이즈 강건성 PoC

실제 급식실 조리 영상에 **불꽃을 합성**해 화재 이미지를 만들고, 검출 모델이 화재를
인식하는지 — 그리고 그 이미지에 **노이즈를 얹었을 때도** 인식하는지를 판별한다.

> 이전 저장소 [kitchen-fire-poc](https://github.com/K-H-MOON/kitchen-fire-poc) 와
> **같은 원본 영상을 쓰지만 독립 프로젝트**다. 그쪽의 가중치·튜닝값을 재사용하지
> 않고, 원본만 새로 가져와 처음부터 분할한다. (데이터 누수 없음)

## 두 실험

1. **합성 화재 인식** — 주방 배경 + 합성 불꽃(양성) / 배경·하드네거티브(음성)로
   학습 → 미학습 주방에서 화재를 인식하는가.
2. **노이즈 강건성** — 노이즈 강도를 올리며 인식률 저하 곡선을 그리고(Phase A),
   떨어지면 노이즈 증강으로 극복해 곡선을 되올린다(Phase B).

## 실행 환경

- 데이터·영상: **구글 드라이브**
- 처리·학습: **Google Colab** (Drive 마운트)
- 저장소에는 스크립트·분할 매니페스트·문서만. 영상·프레임·가중치는 올리지 않음.

## 파이프라인

| 단계 | 스크립트 | 상태 |
|------|------|:---:|
| 0. 영상 인벤토리 | `scripts/videos.json` (28영상·13사이트) | ✅ |
| 1. 사이트 분할 (사전 등록) | `scripts/assign_split.py` → `split.json` | ✅ |
| 2. 배경 프레임 추출 | `scripts/colab_extract_frames.py` | ✅ (2,881장) |
| 3a. 불꽃 소재 인벤토리 | `scripts/flames.json` | ✅ |
| 3b. 불꽃 풀 배정 | `scripts/assign_flame_split.py` → `flame_split.json` | ✅ |
| 3c. 불꽃 매트 추출 | `scripts/colab_extract_flames.py` | ✅ (konro 오염 제외) |
| 4. 불꽃 합성 + 자동 박스 | `scripts/colab_synth.py` | ✅ (2,278/294/309) |
| 5. 학습 (YOLO) | `scripts/colab_train.py` · `colab_phaseB_train.py` | ▶ v8 진행 중 |
| 5.6 불꽃 제거 검증 — 불꽃을 지우면 검출도 사라지나 (배경이 아닌 **불 자체**를 보는지·핵심) | `scripts/colab_ablation.py` | ◐ 깨끗한 데이터로 재검증 대기 |
| 6. 노이즈 저하 곡선 (Phase A) | `scripts/colab_noise_curve.py` | ◐ 미리보기 완료(재확인 대상) |
| 6b. 노이즈 증강 극복 (Phase B) | `scripts/colab_phaseB_train.py` · `colab_phaseB_eval.py` | ▶ 진행 중 |
| 7. 실제 화재 검증 — 합성이 실전 불에 전이되나 (정직성) | `scripts/colab_realfire_test.py` | ◐ 준비완료·학습 후 실행 |

> **노이즈 9종** (`noise_lib.py`, 강도 0→5): 화질계 6 — gaussian(가우시안)·jpeg(압축)·motion_blur(모션블러)·defocus(초점흐림)·low_light(저조도)·contrast(대비); 그 외 3 — steam(수증기)·grayscale(흑백)·random_erasing(무작위 가림). **6번**은 9종 전부에 저하 곡선을 그리고, **6b**는 modelA(9종 전부 증강)·modelB(화질 6종만 증강 + steam·grayscale·random_erasing을 held-out으로 일반화 검증)로 나눔.

> **Level 2 (아키텍처 감도)**: 위 5·5.6·6·6b·7을 YOLO8s로 한 바퀴 → 동일 프로토콜을 YOLO11s로 미러링(각 15회) → 결론이 아키텍처에 견고한지 CI로 교차검증. (LONO는 결과 보고 조건부.)

확정한 설계·근거는 **[docs/PREREGISTER.md](docs/PREREGISTER.md)** 에 있음.

## 분할 (사이트 단위)

| 세트 | 사이트 | CCTV |
|------|------|:---:|
| train | 금정초·남일고·논현중·부산체고·숭곡중·영동중·울산현대차·원촌중·진선여고 | 0 |
| val | 로봇고·인화여중 | 0 |
| test | 개원중·내곡중 | 2 (배포에 가장 가까움) |

같은 주방은 한 세트에만 사용 — 배경·조명을 외워서 생기는 누수를 막음.
