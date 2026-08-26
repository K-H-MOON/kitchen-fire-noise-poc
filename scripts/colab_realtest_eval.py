# ===== 새 실화재 test 로 현재 모델 재측정 (A안 ④) · Colab =====
#
# 목적: colab_build_firetest.py 로 만든 oilfire_realtest 에서 현재 모델의 frame-level 성능 측정.
#   - recall: 양성(불) 프레임 검출률 — 장면(sc##) 단위 평균±std 도 함께(유효 N=장면수).
#   - fpr: ① nofire_kitchen(급식실 조리 CCTV=배포-대표 헛불률) · ② nofire_presrc(동일소스 발화 전).
#   양성/음성을 분리 보고 → 소스 스타일 혼입 상쇄. 박스 아님(frame-level 프록시).
#
# 모델(Drive): 합성 v8_C0_s1 · real_only · real_only_grouped(배포후보) · (_hn 있으면 포함).
# 경계: frame-level 은 위치정확도 무시(관대) · 장면 N 작음(CI 큼) · 저해상 다수 · 급식실 실화재 아님(근접).

import os, glob, json, subprocess, sys
import numpy as np
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE  = '/content/drive/MyDrive/fire_frames'
RUNS  = f'{FIRE}/runs_phaseB'
IFRUN = f'{FIRE}/runs_if'
TEST  = os.environ.get('OUT_DIR', f'{FIRE}/oilfire_realtest')   # build 와 동일 경로(로컬 가능)
OUT   = os.environ.get('EVAL_OUT', f'{FIRE}/indoorfire_eval')   # 결과 json(작은 1파일)
CONF  = float(os.environ.get('CONF', '0.25'))
# 급식실(조리) 음성 경로 override: ⑤ 재학습 후 held-out 조리영상으로만 fpr 측정(누수 차단).
COOK_TEST_DIR = os.environ.get('COOK_TEST_DIR', f'{TEST}/nofire_kitchen')
# 배찬우 팀원 검수 반영(2026-08-26): recall clean/all 이중집계 · presrc 오라벨 장면 제외.
#   기본 빈값 → 기존 동작 불변. 제외 기준=불 가리는 비-배포 아티팩트만(체리피킹 금지).
COMPROMISED = {s for s in os.environ.get('COMPROMISED_SCENES', '').split(',') if s}
PRESRC_DROP = {s for s in os.environ.get('PRESRC_DROP', '').split(',') if s}

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

fire_imgs = sorted(glob.glob(f'{TEST}/fire/*.jpg'))
cook_imgs = sorted(glob.glob(f'{COOK_TEST_DIR}/*.jpg'))
pre_imgs  = sorted(glob.glob(f'{TEST}/nofire_presrc/*.jpg'))
assert fire_imgs, f'양성 없음: {TEST}/fire — colab_build_firetest.py(빌드 모드) 먼저'
print(f'test: fire {len(fire_imgs)} · nofire_kitchen {len(cook_imgs)} · nofire_presrc {len(pre_imgs)}')


def source(p):
    return os.path.basename(p).rsplit('_', 1)[0]


# presrc 오라벨(불) 장면 제외 — 파일 삭제 아님, 이 실행에서만 fpr_발화전 계산서 빼는 필터.
if PRESRC_DROP:
    _n0 = len(pre_imgs)
    pre_imgs = [p for p in pre_imgs if source(p) not in PRESRC_DROP]
    print(f'[PRESRC_DROP] {sorted(PRESRC_DROP)} 제외(오라벨) → nofire_presrc {_n0} → {len(pre_imgs)}')


models = {
    '1_synth (v8_C0_s1)':      f'{RUNS}/v8_C0_s1/best.pt',
    '2_real_only':             f'{IFRUN}/real_only/weights/best.pt',
    '2g_real_only_grouped':    f'{IFRUN}/real_only_grouped/weights/best.pt',
    '2gh_real_only_grouped_hn': f'{IFRUN}/real_only_grouped_hn/weights/best.pt',
    '2ck_real_only_grouped_ck': f'{IFRUN}/real_only_grouped_ck/weights/best.pt',  # ⑤ 조리 네거티브(영상단위 held-out)
    '2cks_real_only_grouped_cksite': f'{IFRUN}/real_only_grouped_cksite/weights/best.pt',  # ⑤ 사이트단위 held-out(교차-주방)
    '2df_real_only_grouped_df': f'{IFRUN}/real_only_grouped_df/weights/best.pt',  # D-Fire 실데이터 보강(배찬우 제안)
}


def detected(model, paths):
    flags = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=CONF, verbose=False):
            flags.append(len(r.boxes) > 0)
    return flags


def by_source_rate(paths, flags):
    d = {}
    for p, f in zip(paths, flags):
        d.setdefault(source(p), []).append(f)
    return {s: float(np.mean(v)) for s, v in d.items()}


rows = {}
for key, best in models.items():
    if not os.path.exists(best):
        print(f'  [없음] {key}: {best}'); continue
    m = YOLO(best)
    fdet = detected(m, fire_imgs)
    cdet = detected(m, cook_imgs) if cook_imgs else []
    pdet = detected(m, pre_imgs) if pre_imgs else []
    recall = float(np.mean(fdet)) if fdet else 0.0
    fpr_cook = float(np.mean(cdet)) if cdet else None
    fpr_pre  = float(np.mean(pdet)) if pdet else None
    scene_rec = by_source_rate(fire_imgs, fdet)
    sc_mean = float(np.mean(list(scene_rec.values()))) if scene_rec else 0.0
    sc_std  = float(np.std(list(scene_rec.values()))) if scene_rec else 0.0
    clean_rec = {s: v for s, v in scene_rec.items() if s not in COMPROMISED}   # 이중집계: compromised 제외
    sc_mean_clean = float(np.mean(list(clean_rec.values()))) if clean_rec else 0.0
    rows[key] = dict(recall=recall, scene_mean=sc_mean, scene_std=sc_std, n_scene=len(scene_rec),
                     scene_mean_clean=sc_mean_clean, n_scene_clean=len(clean_rec),
                     fpr_kitchen=fpr_cook, fpr_presrc=fpr_pre,
                     scene_recall=scene_rec,
                     fpr_kitchen_by_src=by_source_rate(cook_imgs, cdet) if cdet else {})

print('\n' + '=' * 78)
print(f'A안 ④ — 새 실화재 test · frame-level (conf {CONF})')
print('=' * 78)
print(f'{"모델":<28}{"recall":>8}{"scene_mean±std":>17}{"fpr_급식실":>11}{"fpr_발화전":>11}')
for k, r in rows.items():
    fk = f'{r["fpr_kitchen"]:.3f}' if r["fpr_kitchen"] is not None else '  -  '
    fp = f'{r["fpr_presrc"]:.3f}' if r["fpr_presrc"] is not None else '  -  '
    print(f'{k:<28}{r["recall"]:>8.3f}{r["scene_mean"]:>11.3f}±{r["scene_std"]:.3f}{fk:>11}{fp:>11}')

if COMPROMISED:
    print(f'\n[이중집계] compromised 장면 제외({sorted(COMPROMISED)}) → clean-only 대비:')
    print(f'{"모델":<28}{"scene_all":>11}{"scene_clean":>13}{"Δ":>8}')
    for k, r in rows.items():
        d = r["scene_mean_clean"] - r["scene_mean"]
        print(f'{k:<28}{r["scene_mean"]:>11.3f}{r["scene_mean_clean"]:>13.3f}{d:>+8.3f}')
    print('  기준: 제외는 "불을 가리는 비-배포 아티팩트"만(체리피킹 금지). Δ 작으면 "recall 낮음=테스트탓" 반증.')

print(f'\n장면별 recall (유효 N=장면수):')
for k, r in rows.items():
    print(f'  {k}  (N={r["n_scene"]})')
    for s, v in sorted(r['scene_recall'].items()):
        print(f'      {s:<8} {v:.3f}')

print('\n해석: recall↑=실화재 잘 잡음 · fpr_급식실↓=실제 조리에 헛불 적음(배포 핵심) · fpr_발화전=동일소스 헛불.')
print('경계: frame-level(위치 무시·관대) · 장면 N 작음(CI 큼) · 저해상 다수 · 급식실 근접(실 급식실 화재 아님).')
print('판정: 현재 모델(2g)이 이미 recall 높고 fpr_급식실 낮으면 → fine-tune 불요 가능. 낮으면 → ⑤ fine-tune 근거.')

json.dump({'conf': CONF, 'n_fire': len(fire_imgs), 'n_cook': len(cook_imgs), 'n_pre': len(pre_imgs),
           'compromised_scenes': sorted(COMPROMISED), 'presrc_drop': sorted(PRESRC_DROP),
           'rows': rows}, open(f'{OUT}/oilfire_realtest_eval.json', 'w'),
          ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/oilfire_realtest_eval.json')
