# ===== v2 Phase 1: realfire 영상단위 C0 vs C3 비교 (Colab, GPU 권장) =====
#
# pre-reg v2 §6·§7 — 주지표 realfire real_flame_rate 를 **영상 단위**로 본다
# (영상 N개 = 독립 N개, 프레임은 상관 → 의사반복 금지). 각 조건 5 seed.
#
# 절차: realfire 프레임 1회 추출(캐시) → 10모델 예측 → 영상별 rate →
#       조건별 [영상들 사이] 평균 ± 군집 CI(t, df=영상수-1). C0 vs C3 델타·CI겹침 판정.
#
# 선행 — colab_v2_train.py 로 {tag}_C0_s1..5 · {tag}_C3_s1..5 학습됨.
#        real_fire.json 의 fire_shots/nofire_shots 채워짐.
# 환경 — V2_TAG 로 아키텍처 선택(기본 v8; v11 미러는 'v11').

import os, glob, json, subprocess, unicodedata, sys
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
CACHE = '/content/_rf_v2'

drive.mount('/content/drive')
inv = json.load(open(f'{REPO}/real_fire.json', encoding='utf-8'))
SRC, FPS = inv['src_dir'], inv['fps']
allf = glob.glob(f'{SRC}/*')

def norm(s):
    return unicodedata.normalize('NFC', s)

# ---------------------------------------------------------------------------
# 1) realfire 프레임 1회 추출 (영상별 fire/nofire, 캐시)
# ---------------------------------------------------------------------------
def extract():
    frames = {}
    for s in inv['sources']:
        if not s.get('fire_shots'):
            continue
        hit = [p for p in allf if norm(s['file']) in norm(os.path.basename(p))]
        if len(hit) != 1:
            print(f'  [건너뜀] {s["key"]}: 영상 매칭 {len(hit)}개'); continue
        src = hit[0]; frames[s['key']] = {'fire': [], 'nofire': []}
        for kind, shots in (('fire', s['fire_shots']), ('nofire', s.get('nofire_shots', []))):
            for a, b in shots:
                d = f'{CACHE}/{s["key"]}/{kind}/{a}_{b}'
                os.makedirs(d, exist_ok=True)
                if not glob.glob(f'{d}/*.jpg'):
                    vf = f'fps={FPS}'
                    cr = s.get('crop')
                    if cr:
                        vf = f'crop=iw:ih*{1 - cr[0] - cr[1]}:0:ih*{cr[0]},' + vf
                    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                                    '-t', str(round(b - a + 1, 2)), '-vf', vf, '-q:v', '2',
                                    f'{d}/%05d.jpg'], check=False)
                frames[s['key']][kind] += sorted(glob.glob(f'{d}/*.jpg'))
        print(f'  {s["key"]:<14} fire {len(frames[s["key"]]["fire"]):>3} · '
              f'nofire {len(frames[s["key"]]["nofire"]):>3}')
    return frames

print('realfire 프레임 추출(캐시)...')
FR = extract()
VIDEOS = list(FR.keys())
if not VIDEOS:
    raise SystemExit('realfire 영상이 없음 — real_fire.json src_dir/파일 확인')

# ---------------------------------------------------------------------------
# 2) 각 모델 예측 → 영상별 검출률
# ---------------------------------------------------------------------------
def rate(model, paths):
    if not paths:
        return None
    n = 0
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=CONF, verbose=False):
            n += int(len(r.boxes) > 0)
    return n / len(paths)

res = {c: {} for c in CONDS}
for cond in CONDS:
    for seed in SEEDS:
        name = f'{TAG}_{cond}_s{seed}'; best = f'{RUNS}/{name}/best.pt'
        if not os.path.exists(best):
            print(f'  [없음] {name} — 학습 먼저'); continue
        m = YOLO(best); res[cond][seed] = {}
        for v in VIDEOS:
            res[cond][seed][v] = {'flame': rate(m, FR[v]['fire']),
                                  'fp': rate(m, FR[v]['nofire'])}
        print(f'  {name} 예측 완료')
        del m                                        # GPU 메모리 해제 (10모델 순차)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 3) 집계 — 영상단위 평균 ± 군집 CI (영상별 seed평균 → 영상들 사이)
# ---------------------------------------------------------------------------
def tcrit(df):
    return {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
            6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(df, 1.96)

def agg(cond, key):
    pervid = []
    for v in VIDEOS:
        vals = [res[cond][s][v][key] for s in res[cond]
                if res[cond].get(s, {}).get(v, {}).get(key) is not None]
        if vals:
            pervid.append(float(np.mean(vals)))
    if not pervid:
        return None
    pv = np.array(pervid); mean = pv.mean()
    sd = pv.std(ddof=1) if len(pv) > 1 else 0.0
    ci = tcrit(len(pv) - 1) * sd / np.sqrt(len(pv)) if len(pv) > 1 else 0.0
    return float(mean), float(ci), pervid

def seed_pooled(cond, key):                       # seed별 영상평균(안정성 참고)
    out = []
    for s in res[cond]:
        vals = [res[cond][s][v][key] for v in VIDEOS if res[cond][s][v][key] is not None]
        if vals:
            out.append(round(float(np.mean(vals)), 3))
    return out

print('\n' + '=' * 68)
print(f'{TAG} — realfire 영상단위 (영상 {len(VIDEOS)}개 · 각 조건 {len(SEEDS)} seed 평균)')
print('=' * 68)
summary = {}
for cond in CONDS:
    fa = agg(cond, 'flame'); pa = agg(cond, 'fp')
    summary[cond] = {'flame': fa[:2] if fa else None, 'fp': pa[:2] if pa else None,
                     'flame_pervid': fa[2] if fa else None,
                     'flame_perseed': seed_pooled(cond, 'flame')}
    if fa:
        print(f'  {cond}  real_flame_rate {fa[0]:.3f} ± {fa[1]:.3f}  [높아야] · '
              f'seed별 {summary[cond]["flame_perseed"]}')
    if pa:
        print(f'       real_fp_rate    {pa[0]:.3f} ± {pa[1]:.3f}  [낮아야]')

# ---------------------------------------------------------------------------
# 4) 판정 (pre-reg §7)
# ---------------------------------------------------------------------------
if summary['C0']['flame'] and summary['C3']['flame']:
    c0m, c0ci = summary['C0']['flame']; c3m, c3ci = summary['C3']['flame']
    delta = c3m - c0m
    overlap = (c3m - c3ci) <= (c0m + c0ci)
    print('\n' + '-' * 68)
    print(f'C3 - C0 = {delta:+.3f}  (영상단위 real_flame_rate)')
    if delta > 0 and not overlap:
        print('판정: C3 ≫ C0 (CI 비겹침) — 현실성 개선이 전이를 올림(가설 확인 → Phase 2).')
    elif delta < 0 and not overlap:
        print('판정: C3 ≪ C0 (CI 비겹침) — 현실성이 전이를 오히려 낮춤(예상 밖 · 원인 재점검).')
    elif abs(delta) <= max(c0ci, c3ci):
        print('판정: C3 ≈ C0 — 병목은 불꽃이 아니었음(값진 음성 → 방향 전환).')
    else:
        print('판정: 델타는 있으나 CI 겹침 — 유의성 약함(realfire·seed 확대 검토).')
    if summary['C0']['fp'] and summary['C3']['fp']:
        if summary['C3']['fp'][0] > summary['C0']['fp'][0] + max(summary['C0']['fp'][1], summary['C3']['fp'][1]):
            print('  ⚠ real_fp 악화 — 검출↑라도 헛불이 늘었는지 함께 볼 것(§6).')

os.makedirs(OUT, exist_ok=True)
json.dump({'tag': TAG, 'conf': CONF, 'videos': VIDEOS, 'conds': CONDS, 'seeds': SEEDS,
           'per_model': res, 'summary': summary},
          open(f'{OUT}/v2_eval_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/v2_eval_{TAG}.json  (시각화·문서용)')
