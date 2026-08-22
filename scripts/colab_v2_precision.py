# ===== v2 realfire precision + recall (프레임 단위 · v1과 동일 정의) — C0/C3 =====
#
# v1 은 recall(=flame_rate) 0.31 · precision 0.80 을 프레임 단위로 봤다. v2 도 같은 정의로 재
# 비교 가능하게 한다.
#   recall    = fire_det / fire_tot                  (실제 불 프레임 중 검출 = flame_rate)
#   precision = fire_det / (fire_det + nof_det)       (검출한 프레임 중 진짜 불 비율)
#     · fire_det = 불 프레임 중 검출된 수 · nof_det = 무화재 프레임 중 (오)검출된 수
# 프레임 단위 · 5영상 풀링 · seed 평균±표준편차. colab_v2_eval 과 같은 캐시/CONF 재사용.
#
# 선행: colab_v2_train.py 로 {TAG}_C0_s1..5 · {TAG}_C3_s1..5 학습 · real_fire.json shots 채움.

import os, glob, json, subprocess, sys, unicodedata
import numpy as np
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO
import torch

REPO = '/content/kitchen-fire-noise-poc/scripts'
FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'
OUT  = f'{FIRE}/v2_eval'
CONF = 0.25
TAG   = os.environ.get('V2_TAG', 'v8')
CONDS = ['C0', 'C3']
SEEDS = [1, 2, 3, 4, 5]
CACHE = '/content/_rf_v2'                     # colab_v2_eval 과 동일 캐시(재추출 방지)

drive.mount('/content/drive')
inv = json.load(open(f'{REPO}/real_fire.json', encoding='utf-8'))
SRC, FPS = inv['src_dir'], inv['fps']
allf = glob.glob(f'{SRC}/*')

def norm(s):
    return unicodedata.normalize('NFC', s)

def extract():
    frames = {}
    for s in inv['sources']:
        if not s.get('fire_shots'):
            continue
        hit = [p for p in allf if norm(s['file']) in norm(os.path.basename(p))]
        if len(hit) != 1:
            continue
        src = hit[0]; frames[s['key']] = {'fire': [], 'nofire': []}
        for kind, shots in (('fire', s['fire_shots']), ('nofire', s.get('nofire_shots', []))):
            for a, b in shots:
                d = f'{CACHE}/{s["key"]}/{kind}/{a}_{b}'
                os.makedirs(d, exist_ok=True)
                if not glob.glob(f'{d}/*.jpg'):
                    vf = f'fps={FPS}'; cr = s.get('crop')
                    if cr:
                        vf = f'crop=iw:ih*{1 - cr[0] - cr[1]}:0:ih*{cr[0]},' + vf
                    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                                    '-t', str(round(b - a + 1, 2)), '-vf', vf, '-q:v', '2',
                                    f'{d}/%05d.jpg'], check=False)
                frames[s['key']][kind] += sorted(glob.glob(f'{d}/*.jpg'))
    return frames

print('realfire 프레임(캐시)...')
FR = extract()
VIDEOS = list(FR.keys())
if not VIDEOS:
    raise SystemExit('realfire 영상 없음 — real_fire.json 확인')

def ndet(model, paths):
    n = 0
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=CONF, verbose=False):
            n += int(len(r.boxes) > 0)
    return n

rows = {c: {'recall': [], 'precision': []} for c in CONDS}
for cond in CONDS:
    for seed in SEEDS:
        best = f'{RUNS}/{TAG}_{cond}_s{seed}/best.pt'
        if not os.path.exists(best):
            print(f'  [없음] {TAG}_{cond}_s{seed}'); continue
        m = YOLO(best)
        fire_det = fire_tot = nof_det = nof_tot = 0
        for v in VIDEOS:
            fire_det += ndet(m, FR[v]['fire']); fire_tot += len(FR[v]['fire'])
            nof_det  += ndet(m, FR[v]['nofire']); nof_tot += len(FR[v]['nofire'])
        rec = fire_det / fire_tot if fire_tot else 0.0
        prec = fire_det / (fire_det + nof_det) if (fire_det + nof_det) else 0.0
        rows[cond]['recall'].append(rec); rows[cond]['precision'].append(prec)
        print(f'  {TAG}_{cond}_s{seed}  recall {rec:.3f} · precision {prec:.3f}  '
              f'(불 {fire_det}/{fire_tot} · 헛불 {nof_det}/{nof_tot})')
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print('\n' + '=' * 64)
print(f'{TAG} realfire — 프레임단위·5영상 풀링·seed 평균  (참고 v1: recall 0.31 · precision 0.80)')
print('=' * 64)
summary = {}
def ms(x):
    x = np.array(x)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0)
for cond in CONDS:
    if rows[cond]['recall']:
        rm, rs = ms(rows[cond]['recall']); pm, ps = ms(rows[cond]['precision'])
        summary[cond] = {'recall': [rm, rs], 'precision': [pm, ps], 'n_seed': len(rows[cond]['recall'])}
        print(f'  {cond}  recall {rm:.3f} ± {rs:.3f}  ·  precision {pm:.3f} ± {ps:.3f}')

json.dump({'tag': TAG, 'conf': CONF,
           'def': 'recall=fire_det/fire_tot · precision=fire_det/(fire_det+nof_det) · frame-level · 5video pooled · seed mean',
           'summary': summary}, open(f'{OUT}/v2_precision_{TAG}.json', 'w'),
          ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/v2_precision_{TAG}.json')
print('주의: 프레임단위 precision 은 fire:nofire 프레임 수 비율에 의존 · v1 0.80 과 정의는 같으나 표본이 다름.')
