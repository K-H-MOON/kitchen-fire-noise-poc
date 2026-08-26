# ===== A안 ④-후속: conf 문턱 스윕 (fpr 병목 진단) · Colab =====
#
# 목적: 배포후보 2g(및 real_only)를 대상으로, oilfire_realtest 에서
#   conf 문턱을 쓸며 recall(frame·scene) vs fpr(급식실·발화전) 트레이드오프를 본다.
#   → fpr_급식실이 배포선(<0.05~0.1)까지 떨어지는 conf 에서 recall_scene 이 유지되나?
#     유지되면 fine-tune 불요(문턱만 올리면 됨) · 안 되면 ⑤ fine-tune(급식실 네거티브) 근거.
#
# 방식: 이미지당 '최대 박스 conf' 를 conf=0.01 로 1회 추론해 확보 → numpy 로 문턱 스윕(빠름).
#   recall/fpr(문턱 c) = mean(max_conf >= c). recall_scene = 장면(sc##)별 recall 평균.
#   (conf=0.25 값은 colab_realtest_eval.py 결과와 일치해야 함 = sanity.)

import os, glob, json, sys
import numpy as np
from google.colab import drive

# #3 엣지/색 전처리 — 학습·평가와 동일한 scripts/edge_preproc.py 공유(변환 일치 보장).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/content/kitchen-fire-noise-poc/scripts')   # %run 폴백
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE  = '/content/drive/MyDrive/fire_frames'
IFRUN = f'{FIRE}/runs_if'
TEST  = os.environ.get('OUT_DIR', f'{FIRE}/oilfire_realtest')
OUT   = os.environ.get('EVAL_OUT', '/content')
COOK_TEST_DIR = os.environ.get('COOK_TEST_DIR', f'{TEST}/nofire_kitchen')   # ⑤ held-out 조리로 fpr
# #3 엣지/색: 설정 시 엣지·색 모델을 동일 변환한 test 로 스윕(2g 곡선은 EDGE 미설정 실행에서 얻어 비교).
EDGE_MODE  = os.environ.get('EDGE_MODE', '').strip()
EDGE_GAIN  = float(os.environ.get('EDGE_GAIN', '0.5'))
EDGE_ALPHA = float(os.environ.get('EDGE_ALPHA', '0.4'))
drive.mount('/content/drive')

fire_imgs = sorted(glob.glob(f'{TEST}/fire/*.jpg'))
cook_imgs = sorted(glob.glob(f'{COOK_TEST_DIR}/*.jpg'))
pre_imgs  = sorted(glob.glob(f'{TEST}/nofire_presrc/*.jpg'))
assert fire_imgs, f'양성 없음: {TEST}/fire'
print(f'test: fire {len(fire_imgs)} · nofire_kitchen {len(cook_imgs)} · nofire_presrc {len(pre_imgs)}')

# #3: test 3종을 학습과 동일 변환 → 엣지·색 모델만 스윕. realtest_eval 과 같은 경로({TEST}_edge_<mode>)라
#     이미 변환돼 있으면 재사용(동일 이미지 보장 = confsweep conf0.25 가 eval 과 일치하는 sanity).
if EDGE_MODE:
    from edge_preproc import edge_transform_file, MODES
    assert EDGE_MODE in MODES, f'EDGE_MODE={EDGE_MODE!r} 무효 (택1: {MODES})'
    _base = f'{TEST}_edge_{EDGE_MODE}'
    def _prep(paths, sub):
        d = f'{_base}/{sub}'; os.makedirs(d, exist_ok=True)
        out = []
        for p in paths:
            q = f'{d}/{os.path.basename(p)}'
            if not os.path.exists(q):
                edge_transform_file(p, q, EDGE_MODE, EDGE_GAIN, EDGE_ALPHA)
            out.append(q)
        return out
    fire_imgs = _prep(fire_imgs, 'fire')
    cook_imgs = _prep(cook_imgs, 'nofire_kitchen') if cook_imgs else []
    pre_imgs  = _prep(pre_imgs, 'nofire_presrc') if pre_imgs else []
    print(f'[EDGE] {EDGE_MODE} 전처리본 → {_base} (fire {len(fire_imgs)}·조리 {len(cook_imgs)}·발화전 {len(pre_imgs)})')


def source(p):
    return os.path.basename(p).rsplit('_', 1)[0]


def maxconf(model, paths, base=0.01):
    out = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=base, verbose=False):
            out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)


models = {
    '2g_real_only_grouped':    f'{IFRUN}/real_only_grouped/weights/best.pt',
    '2ck_real_only_grouped_ck': f'{IFRUN}/real_only_grouped_ck/weights/best.pt',
    '2cks_real_only_grouped_cksite': f'{IFRUN}/real_only_grouped_cksite/weights/best.pt',
    '2_real_only':             f'{IFRUN}/real_only/weights/best.pt',
}
# #3 엣지/색: 엣지·색 모델은 변환된 test 에서만 유효(RGB 모델을 얹으면 무의미) → EDGE_MODE 시 해당 모델만.
#   2g 등 RGB 곡선은 EDGE 미설정 실행에서 얻어, 같은 recall_scene 지점의 fpr 을 나란히 비교(recall-matched).
if EDGE_MODE:
    models = {f'2edge_{EDGE_MODE}': f'{IFRUN}/real_only_grouped_edge_{EDGE_MODE}/weights/best.pt'}
    print(f'[EDGE] 스윕 대상 = real_only_grouped_edge_{EDGE_MODE} 만. '
          f'2g 곡선은 EDGE 미설정 confsweep 결과와 recall_scene 매칭해 비교.')
CONFS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

all_rows = {}
for key, best in models.items():
    if not os.path.exists(best):
        print(f'  [없음] {key}'); continue
    m = YOLO(best)
    fc = maxconf(m, fire_imgs)
    cc = maxconf(m, cook_imgs) if cook_imgs else np.array([])
    pc = maxconf(m, pre_imgs) if pre_imgs else np.array([])
    srcs = {}
    for p, c in zip(fire_imgs, fc):
        srcs.setdefault(source(p), []).append(c)

    print('\n' + '=' * 66)
    print(f'{key} — conf 스윕')
    print('=' * 66)
    print(f'{"conf":>6}{"recall_f":>10}{"recall_sc":>11}{"fpr_급식실":>11}{"fpr_발화전":>11}')
    rows = []
    for c in CONFS:
        rf  = float(np.mean(fc >= c))
        rsc = float(np.mean([np.mean(np.array(v) >= c) for v in srcs.values()]))
        fk  = float(np.mean(cc >= c)) if cc.size else None
        fp  = float(np.mean(pc >= c)) if pc.size else None
        rows.append(dict(conf=c, recall_frame=rf, recall_scene=rsc, fpr_kitchen=fk, fpr_pre=fp))
        fks = f'{fk:.3f}' if fk is not None else '  -  '
        fps = f'{fp:.3f}' if fp is not None else '  -  '
        print(f'{c:>6.2f}{rf:>10.3f}{rsc:>11.3f}{fks:>11}{fps:>11}')
    all_rows[key] = rows

_jsuf = f'_edge_{EDGE_MODE}' if EDGE_MODE else ''
json.dump(all_rows, open(f'{OUT}/confsweep{_jsuf}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/confsweep{_jsuf}.json')
print('해석: fpr_급식실이 배포선(예: <0.10)으로 떨어지는 conf 에서 recall_scene 이 얼마나 유지되나.')
print('  유지되면(예 recall_scene>0.80 & fpr_급식실<0.10) → 문턱 조정만으로 배포가능·fine-tune 불요.')
print('  트레이드오프가 나쁘면 → ⑤ fine-tune(급식실 조리 네거티브) 로 곡선 자체를 밀어야 함.')
