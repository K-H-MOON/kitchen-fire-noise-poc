# ===== 6-Phase B: 노이즈 곡선 평가 + seed 평균±CI (Colab) =====
#
# runs_phaseB 에 저장된 모델들에 노이즈 곡선을 돌려 config별(baseline/modelA/modelB)
# **seed 평균±95%CI** 를 내고, 저하·극복·일반화를 한눈에 보이는 그래프를 만든다.
#
# 증분(incremental) — 모델별 결과를 캐시(noise_eval.json)에 저장해, 학습이 진행되며
# 모델이 늘면 이 셀만 다시 돌려 새 모델만 평가한다.
#
# 공정성 — 모든 모델을 **동일한 노이즈 실현**으로 테스트(테스트 rng 고정 777,
# 증강 rng 12345 와 독립). Phase A 손상 = Phase B 증강 동일 구현(noise_lib).
#
# held-out(steam·grayscale·random_erasing) 은 그래프에 표시 — modelB 가 학습 안 한 것.

import os, glob, json, sys
import numpy as np, cv2
from google.colab import drive

REPO = '/content/kitchen-fire-noise-poc/scripts'
sys.path.insert(0, REPO)
import noise_lib as NL

FIRE = '/content/drive/MyDrive/fire_frames'
SYN  = f'{FIRE}/synth/test'
RUNS = f'{FIRE}/runs_phaseB'
OUT  = f'{FIRE}/phaseB_eval'
CONF = 0.25
TEST_RNG_SEED = 777
CONFIG_ORDER = ['baseline', 'modelA', 'modelB']
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}

drive.mount('/content/drive')
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

# ---------------------------------------------------------------------------
# test 이미지 적재 (RAM, 1회)
# ---------------------------------------------------------------------------
def load_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

pos, neg = [], []
for p in sorted(glob.glob(f'{SYN}/images/*.jpg')):
    lab = f'{SYN}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
    (pos if os.path.getsize(lab) > 0 else neg).append(p)
POS = [load_rgb(p) for p in pos]
NEG = [load_rgb(p) for p in neg]
print(f'test 양성 {len(POS)} · 음성 {len(NEG)} 적재')

# ---------------------------------------------------------------------------
# 모델 발견 — runs_phaseB/{tag}_{config}_s{seed}/best.pt
# ---------------------------------------------------------------------------
models = []
for bp in sorted(glob.glob(f'{RUNS}/*/best.pt')):
    name = os.path.basename(os.path.dirname(bp))
    parts = name.split('_')
    if len(parts) >= 3 and parts[-1].startswith('s'):
        tag, seed = parts[0], parts[-1][1:]
        cfg = '_'.join(parts[1:-1])
        models.append({'name': name, 'tag': tag, 'cfg': cfg, 'seed': seed, 'best': bp})
if not models:
    raise SystemExit(f'{RUNS} 에 모델이 없음 — colab_phaseB_train.py 를 먼저')
print(f'모델 {len(models)}개 발견')

# ---------------------------------------------------------------------------
# 모델별 노이즈 곡선 (캐시)
# ---------------------------------------------------------------------------
def curve_for(best):
    model = YOLO(best)

    def det(rgb):
        r = model.predict(rgb[..., ::-1], conf=CONF, verbose=False)[0]
        return len(r.boxes) > 0

    res = {}
    for nm in NL.ALL9:
        fn = NL.NOISE[nm]
        rng = np.random.RandomState(TEST_RNG_SEED)      # 모든 모델 동일 실현
        fr, fp = [], []
        for s in NL.SEVS:
            fr.append(float(np.mean([det(fn(im, s, rng)) for im in POS])))
            fp.append(float(np.mean([det(fn(im, s, rng)) for im in NEG])))
        res[nm] = {'flame': fr, 'fp': fp}
    return res

per_model = {}
for m in models:
    cache = f'{RUNS}/{m["name"]}/noise_eval.json'
    if os.path.exists(cache):
        per_model[m['name']] = json.load(open(cache))
    else:
        print(f'  평가 {m["name"]} ...')
        r = curve_for(m['best'])
        json.dump(r, open(cache, 'w'))
        per_model[m['name']] = r

# ---------------------------------------------------------------------------
# 집계 — (tag, config, noise) 별 seed 평균±CI
# ---------------------------------------------------------------------------
tags = sorted({m['tag'] for m in models})
agg = {}   # tag -> cfg -> noise -> {'flame_mean','flame_ci','fp_mean','fp_ci'} (길이 6)
for tag in tags:
    agg[tag] = {}
    for cfg in CONFIG_ORDER:
        seeds = [m for m in models if m['tag'] == tag and m['cfg'] == cfg]
        if not seeds:
            continue
        agg[tag][cfg] = {}
        for nm in NL.ALL9:
            fl = np.array([per_model[m['name']][nm]['flame'] for m in seeds])   # (n_seed, 6)
            fpv = np.array([per_model[m['name']][nm]['fp'] for m in seeds])
            n = len(seeds); t = T95.get(n - 1, 2.0) if n > 1 else 0.0
            agg[tag][cfg][nm] = {
                'n': n,
                'flame_mean': fl.mean(0).tolist(),
                'flame_ci': (t * fl.std(0, ddof=1) / np.sqrt(n)).tolist() if n > 1 else [0] * 6,
                'fp_mean': fpv.mean(0).tolist(),
                'fp_ci': (t * fpv.std(0, ddof=1) / np.sqrt(n)).tolist() if n > 1 else [0] * 6,
            }

os.makedirs(OUT, exist_ok=True)
json.dump(agg, open(f'{OUT}/phaseB_agg.json', 'w'), ensure_ascii=False, indent=1)

# 진행 상황 표
print('\n' + '=' * 66)
for tag in tags:
    have = {cfg: len([m for m in models if m['tag'] == tag and m['cfg'] == cfg])
            for cfg in CONFIG_ORDER}
    print(f'  {tag}: ' + ' · '.join(f'{c} {have[c]}/5' for c in CONFIG_ORDER))

# ---------------------------------------------------------------------------
# 그래프 — 노이즈 9개 그리드, config별 평균±CI 밴드. held-out 표시.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR = {'baseline': 'tab:gray', 'modelA': 'tab:green', 'modelB': 'tab:blue'}

def grid(tag, metric, title, fname):
    a = agg.get(tag, {})
    fig, ax = plt.subplots(3, 3, figsize=(15, 12))
    for i, nm in enumerate(NL.ALL9):
        p = ax[i // 3][i % 3]
        for cfg in CONFIG_ORDER:
            if cfg not in a or nm not in a[cfg]:
                continue
            mean = np.array(a[cfg][nm][f'{metric}_mean'])
            ci = np.array(a[cfg][nm][f'{metric}_ci'])
            p.plot(NL.SEVS, mean, marker='o', color=COLOR[cfg], label=cfg)
            p.fill_between(NL.SEVS, mean - ci, mean + ci, color=COLOR[cfg], alpha=0.15)
        held = ' [held-out]' if nm in NL.HELDOUT else ''
        p.set_title(nm + held, fontweight='bold' if held else 'normal')
        p.set_xlabel('severity'); p.set_ylim(0, 1); p.grid(alpha=0.3)
        if i == 0:
            p.legend(fontsize=8)
    fig.suptitle(f'{tag} — {title}', fontsize=14)
    plt.tight_layout(); plt.savefig(f'{OUT}/{fname}', dpi=100); plt.close()

for tag in tags:
    grid(tag, 'flame', 'flame_rate vs severity (baseline vs A vs B, ±95%CI)',
         f'{tag}_flame.png')
    grid(tag, 'fp', 'fp_rate vs severity (up = false alarms)', f'{tag}_fp.png')

print(f'\n-> {OUT}/phaseB_agg.json · <tag>_flame.png · <tag>_fp.png')
print('읽기: baseline(회색) 대비 modelA(초록)·modelB(파랑)가 위로 오르면 극복.')
print('held-out 노이즈에서 modelB 가 오르면 = 안 배운 노이즈에도 일반화.')
