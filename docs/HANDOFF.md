# HANDOFF — 다음 세션 이어가기 (2026-08-25 갱신)

## ▶▶▶ 새 세션 즉시 작업 (2026-08-25 · 발표회 데모 영상 제작, 진행 중)

**프로젝트 목표 = 실배포 아님, 발표회 시연**("이런 모델 만들었고 성능 이 정도"). 핵심 조사(④⑤AB+교차주방)는 완결·문서화됨 → 지금 남은 건 **데모 영상 + 결과그림 + 서사**.

**어디까지 왔나(데모)**: 화재영상 후보 검수 완료 → **Kitchen Grease Fire Safety.mp4(1920×1080) 선정**, 구간 **③90–100s(발연→발화 전환·서사 1순위)·②70–80s(안정구도)**, 길이 **10초 기준**. 데모 스크립트 `colab_demo_video.py` 완성(박스+시간축 2-of-3 경보 오버레이 MP4 + `_strip.jpg` 몽타주 자동출력). **미완 = 데모 MP4 실행·확정**(런타임 끊겨 중단됨) → ③②둘다 뽑아 `_strip.jpg` 육안비교 → 최종 클립 확정 → 조리(조용함) 영상 짝지어 완성.

**즉시 재현 레시피 (새 런타임 가정)**:
```python
# (1) 재clone (oilfire_raw 재구성 불필요 — 데모는 Drive 루트 영상 직접 읽음)
!rm -rf /content/kitchen-fire-noise-poc && git clone -q https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
# (2) 데모 MP4 — Kitchen Grease 두 구간 (drive.mount 는 스크립트가 함)
import os
os.environ['DEMO_VIDEO']='/content/drive/MyDrive/Kitchen Grease Fire Safety.mp4'   # seochorobotics 계정
os.environ['MODEL']='real_only_grouped_ck'; os.environ['FPS']='12'; os.environ['DEMO_LABEL']='real fire'
os.environ['START']='90'; os.environ['END']='100'; os.environ['OUT_MP4']='/content/demo_grease_90_100.mp4'
%run /content/kitchen-fire-noise-poc/scripts/colab_demo_video.py
os.environ['START']='70'; os.environ['END']='80'; os.environ['OUT_MP4']='/content/demo_grease_70_80.mp4'
%run /content/kitchen-fire-noise-poc/scripts/colab_demo_video.py
# (3) _strip.jpg(첨부용) + MP4(발표용) 다운로드 → strip 육안비교로 구간 확정
```
- **경보 로직**: 최근 PERSIST_SEC(1.0s) 중 PERSIST_FRAC(0.5) 이상 검출 시 ALARM(= 2-of-3 정신). `FPS/PERSIST_SEC/FRAC`로 민감도 조정.
- **화재영상은 학습 무관**(2ck=Indoor+조리네거 학습, 이 영상 미학습) → 데모 정직. **비-급식실**(소방데모)이라 "실화재 검출 시연(불꽃종류 무관)"으로 프레이밍.
- **조리(조용함) 영상**: 한국급식실 CCTV(`조리 데이터 영상`). seen=조용(~0.2/분)·unseen=헛불(1.465/분) → seen만 vs seen+unseen 대비 정할 것.

**핵심 함정(데모)**: 영상은 **seochorobotics 계정**(Colab 마운트 계정)의 MyDrive 루트에 있음(blessmoonkh=Claude 커넥터와 별개). MP4는 채팅 첨부/AI관찰 불가 → **`_strip.jpg` 몽타주로 확인**. 파일명 후보: `Kitchen Grease Fire Safety.mp4`·`How to Prevent & Douse a Kitchen Fire _ Deep-Frying.mp4`(둘 다 1080p)·`2 東京防災 天ぷら油火災実験.mp4`·`天ぷら油火災が発生するまで！（訓練）.mp4`·`天ぷら油火災シミュレーション.mp4`(720p·일본자막).

**교차-주방 결론(④⑤ 이후 추가 검증, README/STATUS 반영됨)**: 28영상=13개 학교 급식실. 사이트단위 leave-out+이벤트level(헛경보/분) → **본 급식실 0.206/분 vs 안 본 급식실 1.465/분(~7×, 같은 모델 2cks). = per-site 보정 작동·범용 one-model 미해결.** 배포결론 정직화 완료. 팀 보고 = `docs/STATUS.md`(README 상단 링크).

**base YOLO 팀원질문 답(기록)**: 순정 yolov8s=COCO 80클래스에 fire 없음 → 화재 recall **0**/precision N/A(구조적). baseline=합성-only(v8_C0_s1) 0.237. 단계 순정0→합성0.237→실0.899. 합성 무기여=혼합(합성+실)=실(랜덤분할 둘다 0.985·그룹 미측정).

---

## ✅ A안 ④→⑤ 완료 (2026-08-25) — 배포후보 2ck @ conf 0.25

**한 줄**: 실데이터로 recall 해결(0.81/0.85) → 급식실 조리 헛불(fpr)이 유일 병목 → **⑤ 급식실 조리 하드네거 재학습으로 fpr 10×↓(held-out 0.302→0.032), recall −3점(약한 흐린불에 집중)** → **배포후보 = `real_only_grouped_ck`(2ck) @ conf 0.25**(recall_sc 0.815 · fpr_급식실 0.032, ⑤ 전엔 불가능했던 "recall>0.80 & fpr<0.10" 운영점 달성).

**A안 전체 궤적**:
1. **④ 측정**: frame-level 실화재 test(양성=사람지정 불구간·음성=급식실 조리CCTV+발화전) 빌드→2g 재측정. 초측정 recall 0.710/fpr_급식실 0.260.
2. **오염 발견·교정**: 셀2b(`colab_inspect_firetest_pos.py`) 양성 전수검증 → 교육/lab 영상 ±4s 근사가 비화염 삼킴(sc05 IHこんろ 78%=감자·텍스트카드·sc15 NIST ignition 50%=예열·sc02 잔해·sc08 빈프레임). 교정 RANGES(아래)+`FIRE_CAP=40`(NIST sc14 126프레임 과대표집 다운샘플) 재측정 → **깨끗 recall 0.813/scene 0.846**(synth 0.357). 깨끗 실화재 13/16장면 0.72~1.0, 유일 약점=NIST 창백극소형불(sc14 0.175·sc15 0.438).
3. **conf 스윕**(`colab_realtest_confsweep.py`): 문턱조정만으론 불충분(recall>0.80 & fpr<0.10 지점 없음).
4. **FP 진단**(`colab_diag_cook_fp.py`): 급식실 헛불 47/181 전부 **비화염**=불색 조리음식(볶음)·스팀·스테인리스 반사(실 배포주방=한국로보틱스 조리로봇, 개방화염 없음). ck03/07/08 100%. → ⑤ 정당(비화염이라 recall 위험↓·§F와 달리 28영상 다양성).
5. **⑤ 재학습**(`colab_build_cook_negs.py`→`HN_TAG=ck` split_audit→eval): 조리CCTV 영상단위 train/held-out split(누수0), 조리네거 231장 주입. **같은 held-out(안 본 8영상·63프레임): fpr_급식실 0.302→0.032(10×), recall scene 0.846→0.815.** §F식 일반화 증명(ck03/07/08 학습→안 본 ck02/11/18 전이). §F가 실패한 일반화가 여기선 성공(장면 다양성 덕).
6. **conf 스윕(2ck)**: 2ck가 2g/real_only를 전 구간 지배. 운영점 conf 0.25=스위트스폿(0.15로 낮춰도 recall +0.8점뿐·fpr 2배).

**정직한 잔여 한계(향후과제)**: ⓐ sc14 창백극소형불 recall→0(문턱으로 회복 안 됨·안전상 흐린 초기불 놓침 → **시간축 성장감지**가 보완 여지) ⓑ held-out N 작음(fpr 2/63·CI 큼, 효과 10×라 방향은 확실) ⓒ fpr_발화전 0.336 불변(발화 직전=정상조리 헛불과 성격 다름·조기경보로 볼 수도) ⓓ 미검증=28영상 밖 다른 급식실 일반화·시간축 필터·실제 급식실 화재·frame-level 프록시. **다음 방향 후보=시간축 필터(밀집프레임 test 필요·sc14+스팀 동시 보완)·다른 급식실 수집·실 급식실 화재 확보.**

**⚠️ 재현 함정(변함없음)**: (1)Drive `oilfire_raw`/작업본 FUSE 유실→로컬 `/content` (2)`/content` 세션리셋 소멸→**한 세션에 build~train~measure 연속** (3)`%run -i` 필수(RANGES 파이썬변수). ⑤ 재학습은 Indoor 재분할 포함 ~40분(L4).

---
### (참고) A안 ④→⑤ 재현 레시피 — 초기 진행용 원문 (이제 결과 확정)

**⚠️ 중대 함정 3개(꼭 지킬 것)**:
1. **Drive `oilfire_raw` 반복 유실**(Colab+Drive FUSE 비동기 쓰기 손실 추정) → **작업본은 로컬 `/content`에서.** 소스 영상은 Drive 루트에서 읽어 로컬로 복사(루트 85 mp4는 안정). 모델·평가결과 등 기존 산출물은 Drive에 멀쩡.
2. **`/content`는 세션 리셋 시 소멸** → **rebuild→build→measure를 한 세션에 연속** 실행.
3. **`%run -i` 필수**(그냥 `%run` 아님) — `%run`은 새 네임스페이스라 셀에서 정의한 `RANGES`(파이썬 변수)를 못 봄. `-i`라야 보임. (env는 프로세스전역이라 `%run`도 됨.)

**전체 재현 레시피 (한 세션, 순서대로)**:
```python
# (1) 재clone + 로컬 oilfire_raw 재구성(루트 화재영상 복사 + NIST 다운로드)
!rm -rf /content/kitchen-fire-noise-poc && git clone -q https://github.com/K-H-MOON/kitchen-fire-noise-poc.git /content/kitchen-fire-noise-poc
from google.colab import drive; drive.mount('/content/drive')
import os, glob, re, shutil
ROOT='/content/drive/MyDrive'; L='/content/oilfire_raw'; os.makedirs(L, exist_ok=True)
stock=re.compile(r'(istockphoto|watermarked|-hd_|-uhd_|_\d{3,4}_\d{3,4}_\d+fps|^267-)')
for p in glob.glob(f'{ROOT}/*.mp4'):
    b=os.path.basename(p)
    if not stock.search(b) and '360___Video' not in b and '(1)' not in b:
        shutil.copy(p, f'{L}/{b}')
os.system('pip -q install -U yt-dlp')
for u in ['https://www.nist.gov/video/cooktop-reignition-oil','https://www.nist.gov/video/cooktop-ignition-prevention-technology-evaluation-ignition-not-prevented']:
    os.system(f'yt-dlp -o "{L}/NIST_%(title)s.%(ext)s" "{u}"')
print('local oilfire_raw:', len(os.listdir(L)))   # ~68

# (2) test 빌드 — 교정 RANGES(오염 제거본) + FIRE_CAP=40(과대표집 완화) + %run -i
os.environ['RAW_DIR']='/content/oilfire_raw'; os.environ['OUT_DIR']='/content/oilfire_realtest'
os.environ['FIRE_CAP']='40'
RANGES = {   # ★교정본(셀2b 오염검증 반영). 원본 대비 좁힌 것: How to Prevent·Cooking·Grease·IH·恐怖·NIST ignition
    'How to Prevent':[(160,171)], 'Chip pan':[(13,15),(21,25),(29,32)], 'Cooking Fire Safety':[(12,17)],
    'Kitchen Grease Fire Safety':[(30,32),(46,85)], '2 東京防災':[(91,138),(147,151)], 'IHこんろ「4':[(7,8)],
    '発生':[(185,256)], 'シミュレーション':[(33,36),(55,85)], '恐怖':[(12,13),(42,43),(54,57)], '1637681405':[(11,21)],
    '401469436':[(3,27)], '774563476':[(3,22)], '32125355803':[(9,42)], '34938882503':[(7,24)],
    'NIST_Cooktop Reignition':[(24,136),(150,162)], 'NIST_Cooktop ignition':[(24,39)],
}
%run -i /content/kitchen-fire-noise-poc/scripts/colab_build_firetest.py
# → fire(양성)~359 · nofire_kitchen(급식실조리 CCTV)~181 · nofire_presrc(발화전)~301

# (2b) 양성 프레임 육안 검증(셀2b) — 이미 반영됨. 재교정 시만 재실행.
# os.environ['INSP_DIR']='/content/inspect'; %run /content/kitchen-fire-noise-poc/scripts/colab_inspect_firetest_pos.py

# (3) 측정 (2g/2ck 등 전 모델)
os.environ['OUT_DIR']='/content/oilfire_realtest'; os.environ['EVAL_OUT']='/content'
%run /content/kitchen-fire-noise-poc/scripts/colab_realtest_eval.py

# ===== ⑤ 급식실 조리 하드네거 재학습 (fpr 공략) =====
# (5a) 조리 네거티브 빌드 — 영상단위 split(누수0). HELDOUT 미지정 시 목록만 출력.
os.environ['HELDOUT']='ck02,ck09,ck11,ck13,ck16,ck18,ck20,ck25'   # held-out(fpr 테스트)·나머지=train
%run /content/kitchen-fire-noise-poc/scripts/colab_build_cook_negs.py
# → /content/cook_neg_train/nofire (학습주입) · /content/cook_neg_test/nofire (held-out fpr)

# (5b) 재학습(~40분, Indoor 재분할+60ep) → runs_if/real_only_grouped_ck
os.environ['HARDNEG']='1'; os.environ['HARDNEG_TRAIN_DIR']='/content/cook_neg_train/nofire'
os.environ['HN_TAG']='ck'; os.environ['EVAL_MIXED']='0'
%run /content/kitchen-fire-noise-poc/scripts/colab_indoorfire_split_audit.py

# (5c) 재측정 — recall(oilfire_realtest) + held-out 조리 fpr(누수 차단)
os.environ['OUT_DIR']='/content/oilfire_realtest'; os.environ['EVAL_OUT']='/content'
os.environ['COOK_TEST_DIR']='/content/cook_neg_test/nofire'
%run /content/kitchen-fire-noise-poc/scripts/colab_realtest_eval.py         # 표에 2ck 행
%run /content/kitchen-fire-noise-poc/scripts/colab_realtest_confsweep.py    # 2g/2ck 곡선(운영 conf)
```
**결과(확정)**: 위 레시피대로 2g recall 0.813/scene 0.846·fpr_급식실 0.260(오염교정 후) → ⑤ 후 2ck fpr_급식실 held-out 0.032·recall scene 0.815. 운영 conf 0.25. json: Drive `indoorfire_eval/{oilfire_realtest_eval_clean,indoorfire_regroup_ck}.json`·모델 `runs_if/real_only_grouped_ck/best.pt`.

**스크립트(A안 ④→⑤ 신규/수정, 전부 커밋·푸시됨)**:
- `colab_build_firetest.py` — frame-level test 빌더(2모드·env RAW_DIR/OUT_DIR/INSP_DIR·**FIRE_CAP**=장면당 양성 상한 균등서브샘플)
- `colab_inspect_firetest_pos.py` — 빌드된 양성 장면별 몽타주(셀2b·오염검증·env OUT_DIR/INSP_DIR)
- `colab_realtest_eval.py` — 재측정(env OUT_DIR/EVAL_OUT/**COOK_TEST_DIR**·2g/2ck 등 전 모델)
- `colab_realtest_confsweep.py` — conf 스윕(recall vs fpr 곡선·2g/2ck·COOK_TEST_DIR)
- `colab_diag_cook_fp.py` — 급식실 FP 진단(예측박스 몽타주·env CONF/MODEL)
- `colab_build_cook_negs.py` — ⑤ 조리네거 영상단위 split(2모드·env HELDOUT/COOK_TRAIN_OUT/COOK_TEST_OUT)
- `colab_indoorfire_split_audit.py` — 재학습에 **HN_TAG**(접미사·_hn/_ck 분리) 추가
채택 16 선정기준·최종풀 = `DATA_collection_spec.md §11`.


> **v1·v2·v3 모두 실제 전이 약함(불꽃·표현 레버 아님) → 병목 = 데이터/도메인.** 회의 후 = **실제 데이터로 공략** → **실험 A: 실데이터 학습이 놓침 해소**(Indoor recall 0.235→0.985). **누수 통제 0.899로 견고**(누수 ~8.6점). **도메인 이동 파일럿: 유류화재 test 큰불(§C) 실 0.985·초기작은불(§D) 실 0.97~1.0 vs 합성 0.52·0.27 → 실데이터가 유류화재 큰불·초기불 모두에 전이**(누수 원천 불가 독립 test). §AFTER_meeting §5·§6.
> **★2026-08-24 §F 헛불 고치기(하드네거 재학습) 완료 = 사실상 무효. 주황조명(13476222)이 모든 실모델에서 fpr 0.875~1.0로 요지부동 = "주황조명 FP는 데이터 문제" 확정(1장면뿐 유형은 일반화 못 함, 실증). 스팀은 이미 그룹모델이 해결. 다음 진짜 레버 = 주황 하드네거 장면 다수 수집. 상세 §F.**
> **★2026-08-25 A안(전이학습) 데이터 수집 ✅종료: 4배치 총 69영상(영·일 소방데모+중국 실주방 CCTV+NIST lab)→전수 검수→깨끗 21장면(test·train겸용 16+세로 train전용 5)+경계 7 = 28장면. fine-tune 문턱(≥20~25) 여유 초과·도메인 다양성 확보(실 in-situ 주방·단체급식류·NIST고해상). ⚠️누수 2개(기존 oilfire_pilot/early 원본=일본 事故再現 2편) 학습 절대금지. 최종 풀·파일명 `DATA_collection_spec.md §11 "최종 풀"`.**
> **★2026-08-25 A안 ④→⑤ ✅완료(상세=문서 최상단 블록). ④ 측정: 오염교정 후 2g recall 0.813/scene 0.846·fpr_급식실 0.260 → recall은 실데이터로 해결, 병목은 조리 헛불. FP 진단=불색음식·스팀·반사(비화염). ⑤ 급식실 조리 하드네거 재학습: held-out fpr 0.302→0.032(10×)·recall −3점(약한 흐린불 집중). 배포후보=`real_only_grouped_ck`(2ck) @ conf 0.25(recall_sc 0.815·fpr 0.032). 잔여=sc14 창백극소형불·시간축 미검증·다른 급식실 일반화.**
> 완료: realfire 오염·Indoor 첫측정(0.20)·**실험 A(랜덤 합성0.235/실0.985)**·**누수 통제(그룹 합성0.237/실0.899)**·**도메인 이동(§C 큰불 합성0.523/실0.985 · §D 초기작은불 합성0.267/실1.000)**·**§E 헛불(하드네거130장: 실 fpr0.223/혼합0.238/그룹0.085/합성0.092)**·**§F 헛불 고치기(hardneg 재학습: grouped_hn held-out fpr 0.209→0.186·주황 0.875불변·recall유지)**·**A안 데이터 1차 수집·검수(49→9)**·미팅 문서.

## ▶ 새 세션 첫 작업 (2026-08-24 §F 헛불 고치기까지 완료 후)
1. ~~실험 A~~✅ · ~~누수통제~~✅(그룹0.899) · ~~§C큰불~~✅ · ~~§D초기작은불~~✅ · ~~§E 헛불 검증~~✅ · ~~§F 헛불 고치기(hardneg 재학습)~~✅.
2. **§F 결론 = 헛불 고치기(hardneg 재학습)는 사실상 무효.** 배포후보 grouped_hn: held-out fpr 0.209→0.186(개선폭 1프레임·노이즈)·목표recall 유지(pilot0.985·early1.000)·Indoor recall만 0.899→0.821 하락(유류엔 전이 안 됨). **주황(13476222) 0.875 요지부동** = 모든 실모델 공통 → **"주황조명 FP=데이터 문제" 확정.** 상세 §F.
3. **주황조명 헛불 진단 완료(2026-08-24, `colab_diag_hardneg.py`) → 핵심 다음 = 제품 범위 결정 후 방향 확정.**
   - **진단 결과**: 헛불 정체 = 배경의 작은 **"따뜻한 장식 전구"(점광원)** 를 소형 화염으로 착각(넓은 조명 아님·실제불/로고 아님). 3장면: `13476222`(안 본 레스토랑) 7/8 지속헛불 · `13578888`(학습에 본 주황) **0/5 완전억제** · `94587527` 1/11 경미. → **학습에 본 주황장면은 억제·안 본 장면은 실패 = "장면 다양성 부족" 문제(고칠 수 있음, 직접 증거).** 몽타주 `inspect/diag_<scene>_real_only_grouped_hn.jpg`. 상세 `AFTER_meeting.md §F 진단`.
   - **✅ 결정됨 (2026-08-24, 제품 = 급식실 전용)**: 급식실 주방은 밝은 산업용 형광등이라 주황 조명 안 씀 → **주황-전구 헛불은 out-of-domain, 저순위(수집 안 함).** (상업주방까지 확장 시에만 재개할 값싼 카드.)
   - **★ 부수 함의(기록)**: §E/§F fpr(그룹 0.085/held-out 0.209)의 최대 기여자가 out-of-domain 레스토랑 조명(13476222) → **배포 도메인(밝은 주방·스팀)에선 헛불 실제로 더 작음**(순수 스팀·대량조리 fpr 0). **급식실 배포 관점 헛불은 사실상 관리됨.**
   - **→ 확정된 핵심 다음 = A안(전이학습)**: 헛불은 접고, 남은 진짜 과제는 **배포-대표 양성 데이터**(급식실/유류 실화재)로 목표 도메인 성능 확보. 상세 아래 4번.

   **오버피팅/도메인갭 정리(사용자 질문 답)**: ⓐ 에폭 과적합=없음(val 평탄·best.pt·grouped_hn은 ep59 조기종료 ep44 선택). ⓑ 랜덤분할 val mAP0.53 vs 그룹0.40 격차=**누수 인플레이션**(test 73% train근접중복)이지 과적합 아님 → 그룹수치가 정직. ⓒ **진짜 병목=도메인 갭**(정직한 그룹모델도 못 본 유류-도메인 주황엔 헛불 = 데이터로만 고침). 상세는 이 세션 대화.
4. **~~그 후 A안(전이학습)~~ (위 3-③로 이동)**: 일반 화재(Indoor) 사전학습 → 소량 급식실/유류 fine-tune. test=목표 도메인·씬분리·불가침(소량이면 LOSO CV).
4. **후순위**: New_sample(야외 JSON) 변환 · (합성 추가는 "무기여"라 낮음).
5. 모델: `runs_if/{real_only,mixed,real_only_grouped}/weights/best.pt` · 결과 `indoorfire_eval/{indoorfire_train,indoorfire_regroup,indoorfire_split_audit,oilfire_pilot_eval,oilfire_early_eval,oilfire_hardneg_eval}.json` · 스크립트 `colab_hardneg_split.py`(하드네거 장면 split·2026-08-24 신규)·`colab_indoorfire_train.py`/`colab_indoorfire_split_audit.py`(둘 다 `HARDNEG=1` env로 하드네거 주입→`_hn` 모델)·`colab_oilfire_eval.py`(EVAL_SET·소스별 fpr·`_hn` 모델 포함)·`colab_build_hardneg.py`(루트 조리영상→하드네거)·`colab_inspect_newdata.py`(INSPECT_ALL). **test: `oilfire_pilot.zip`(큰불65/14)·`oilfire_early.zip`(초기불30/10)·하드네거 = Drive 루트 조리영상16개→build로 `fire_frames/oilfire_hardneg`(nofire130/sanity10). 로컬 scratchpad에 원본·큐레이션.**
   - 데이터 수집 함정: **YouTube는 Colab(데이터센터 IP) 봇 차단** → 로컬 PC(주거IP)에서 yt-dlp 받고 ffmpeg로 프레임 추출·검증 후 zip 업로드. Drive 커넥터는 **blessmoonkh** 계정(seochorobotics 루트 새 파일 검색 안 됨, fire_frames는 공유돼 열림).
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
  - json `indoorfire_eval/indoorfire_train.json` · 모델 `runs_if/(real_only|mixed)/weights/best.pt`.

- **누수 통제 재측정 (그룹 분할, `colab_indoorfire_split_audit.py`)**:
  - 누수 감사: **test fire 73%(286/392)가 train에 근접중복**(dHash≤6) → 랜덤분할 누수 심각.
  - 그룹 분할(클러스터 split 경계 안 넘게, test 근접중복 0) 재측정:

    | 조건 | recall | precision | fpr | 검출(fire) |
    |---|---:|---:|---:|---:|
    | ① 합성-only | 0.237 | 0.842 | 0.041 | 85/358 |
    | ② 실-only | **0.899** | 0.979 | 0.018 | 322/358 |

  - **판정**: 실 0.985→**0.899**(누수 ~8.6점) 하지만 견고 · 합성 0.235→0.237 **불변**(Indoor 미학습 → 분할 무영향 = sanity 통과). → **결론(실데이터가 놓침 해소)은 누수 제거해도 유지. 누수는 끝난 이슈, 남은 경계=도메인.**
  - json `indoorfire_eval/indoorfire_regroup.json`·`indoorfire_split_audit.json` · 모델 `runs_if/real_only_grouped/weights/best.pt`.

### §F 헛불 고치기 — 하드네거 재학습 (완료, 2026-08-24)
- **질문**: §E 헛불(주황조명·스팀)을 하드네거를 학습에 넣어 고칠 수 있나. **급식실 화재 데이터 불요·값쌈**이라 첫 시도.
- **설계**: 하드네거 130장(16장면)을 **장면 단위** train12/held-out4 분리(프레임 누수 0). held-out = `13476222`(주황)·`8094275`(스팀)·`94587527`·`267`. **주황 일반화 증명 = 13578888(train·5프레임)→13476222(test).** train-hardneg 87장을 빈 라벨(음성)로 주입해 `real_only_hn`(랜덤)·`real_only_grouped_hn`(그룹·배포후보) 재학습(`HARDNEG=1`). 스크립트 `colab_hardneg_split.py`+`colab_indoorfire_train.py`/`_split_audit.py`+`colab_oilfire_eval.py`.
- **결과 (held-out fpr, 43장·4장면·frame-level)**:

  | 장면(프레임) | ①synth | ②real | ②h real_hn | ②g grouped | ②gh grouped_hn |
  |---|---:|---:|---:|---:|---:|
  | 13476222 주황(8) | 0.000 | 0.875 | **1.000** | 0.875 | **0.875** |
  | 8094275 스팀(12) | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 |
  | 94587527 (11) | 0.000 | 0.273 | 0.545 | 0.182 | 0.091 |
  | 267 (12) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
  | **전체(43)** | 0.000 | 0.419 | 0.326 | 0.209 | **0.186** |
  | recall pilot | 0.523 | 0.985 | 0.985 | 0.985 | 0.985 |
  | recall early | 0.267 | 1.000 | 0.967 | 0.967 | 1.000 |
  | Indoor(그룹)recall | — | — | — | 0.899 | 0.821 |

  - **판정 = 사실상 무효.** 전체 fpr 미세개선(그룹 0.209→0.186)은 전부 94587527 1프레임(2/11→1/11)=노이즈. **주황(13476222)은 모든 실모델에서 0.875~1.0 요지부동** — 13578888(주황 1장면) 학습이 다른 주황 장면에 전이 안 됨. 스팀(8094275)은 그룹모델이 이미 0(hardneg 불요). recall은 목표도메인(pilot/early) 유지·early는 오히려↑. grouped_hn은 Indoor recall만 0.899→0.821 하락(유류엔 전이 안 됨, 무시가능).
  - **결론: "주황조명 FP=데이터 문제" 확정**(1장면뿐 유형은 일반화 못 함, 우리가 사전 경고한 한계의 실증). hardneg 재학습은 다양성 있는 유형(스팀)엔 통하나 주황엔 무력. **다음 레버=주황 하드네거 장면 다수 수집**(§ 새 세션 첫 작업 3번).
  - **오버피팅 아님(사용자 질문)**: real_only_hn val ep58 best·평탄, grouped_hn **ep59 조기종료→ep44 best**(patience 작동). 랜덤val 0.53 vs 그룹val 0.40 격차=누수 인플레이션(test 73% train근접중복)이지 과적합 아님. 진짜 병목=도메인 갭.
  - json `indoorfire_eval/{indoorfire_train_hn,indoorfire_regroup_hn,oilfire_hardneg_test_eval,oilfire_pilot_eval,oilfire_early_eval}.json` · 모델 `runs_if/{real_only_hn,real_only_grouped_hn}/weights/best.pt` · manifest `hardneg_split.json`. (mixed_hn·synth_hn 미학습.)
  - **진단(`colab_diag_hardneg.py`, 예측박스 몽타주)**: 헛불 정체 = 배경 작은 **따뜻한 장식 전구(점광원)** 착각. `13476222`(안 본) 7/8 지속 · `13578888`(학습에 본) **0/5 억제** · `94587527` 1/11 경미. → **"장면 다양성 부족"(1장면 학습→그 장면만 억제·전이 실패)이지 "못 죽이는 특징" 아님 = 고칠 수 있음.** ⚠️ 단 13476222=레스토랑 무드조명→급식실엔 out-of-domain(수집 저순위, 상업주방 범위면 in-domain). 몽타주 `inspect/diag_<scene>_real_only_grouped_hn.jpg`. env: `SCENE`(장면토큰)·`MODEL`·`SRC`(train장면은 `oilfire_hardneg_train/nofire` 지정 필요).

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
