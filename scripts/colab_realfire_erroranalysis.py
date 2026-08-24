# ===== realfire 오류 분석 — 놓침(miss)·헛불(FP)을 원인 단서와 함께 (Colab) =====
#
# 목적: "무엇을 더 모아야 하나"를 데이터로 조준. 현 모델이 실제 화재에서 무엇을 놓치는지
#       (밝은 배경? 특정 영상? 낮은 conf?) 패턴을 보고 수집 사양(DATA_collection_spec.md)을 갱신.
# 재는 것: per-video miss/FP율 · 프레임 밝기(luma) 버킷별 miss율 · 검출 conf 분포 ·
#         놓침/헛불 컨택트 시트(육안).
# 환경: ERR_MODEL(기본 v8_C0_s1) — 다른 모델로 보려면 지정.
# 선행: runs_phaseB/<model>/best.pt · real_fire.json shots. GPU 권장.

import os, glob, json, subprocess, sys, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

REPO = '/content/kitchen-fire-noise-poc/scripts'
FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'
OUT  = f'{FIRE}/err_analysis'
CONF = 0.25
MODEL = os.environ.get('ERR_MODEL', 'v8_C0_s1')
BEST = f'{RUNS}/{MODEL}/best.pt'
CACHE = '/content/_rf_v2'                     # colab_v2_eval 과 동일 캐시 재사용

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 없음 — ERR_MODEL 확인')
model = YOLO(BEST)
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

print(f'realfire 프레임(캐시)... · 모델 {MODEL}')
FR = extract()
VIDEOS = list(FR.keys())
if not VIDEOS:
    raise SystemExit('realfire 영상 없음')
os.makedirs(OUT, exist_ok=True)

def luma(p):
    a = np.asarray(Image.open(p).convert('RGB'), np.float32)
    return float(a[..., 0].mean() * 0.299 + a[..., 1].mean() * 0.587 + a[..., 2].mean() * 0.114)

def infer(p):
    r = model.predict(p, conf=CONF, verbose=False)[0]
    return (len(r.boxes) > 0), (float(r.boxes.conf.max()) if len(r.boxes) else 0.0)

recs = []                                    # (video, kind, path, detected, conf, luma)
for v in VIDEOS:
    for kind in ('fire', 'nofire'):
        for p in FR[v][kind]:
            det, cf = infer(p)
            recs.append((v, kind, p, det, cf, luma(p)))

# ---------------------------------------------------------------------------
# per-video miss/FP
# ---------------------------------------------------------------------------
print('\n' + '=' * 66)
print(f'realfire 오류 분석 — {MODEL}')
print('=' * 66)
print('per-video (miss = 놓친 불 · FP = 헛불):')
for v in VIDEOS:
    fire = [r for r in recs if r[0] == v and r[1] == 'fire']
    nof  = [r for r in recs if r[0] == v and r[1] == 'nofire']
    miss = [r for r in fire if not r[3]]; fp = [r for r in nof if r[3]]
    mr = len(miss) / len(fire) if fire else 0
    fr = len(fp) / len(nof) if nof else 0
    print(f'  {v:<14} miss {len(miss):>3}/{len(fire):<3} ({mr:.2f})  ·  FP {len(fp):>3}/{len(nof):<3} ({fr:.2f})')

# ---------------------------------------------------------------------------
# 밝기 버킷별 miss율 (fire) — "밝은 배경에서 더 놓치나?"
# ---------------------------------------------------------------------------
fire_all = [r for r in recs if r[1] == 'fire']
lums = np.array([r[5] for r in fire_all])
if len(lums):
    q1, q2 = np.percentile(lums, [33, 66])
    def bucket(l):
        return '어두움' if l < q1 else ('중간' if l < q2 else '밝음')
    print(f'\n밝기별 miss율 (fire · 컷 {q1:.0f}/{q2:.0f}):')
    for b in ('어두움', '중간', '밝음'):
        grp = [r for r in fire_all if bucket(r[5]) == b]
        miss = [r for r in grp if not r[3]]
        print(f'  {b:<5} miss {len(miss):>3}/{len(grp):<3} ({len(miss) / len(grp) if grp else 0:.2f})')

hits = [r[4] for r in fire_all if r[3]]
if hits:
    print(f'\n검출된 불 conf: 중앙 {np.median(hits):.2f} · 최소 {min(hits):.2f} '
          f'(0.25 근처면 아슬아슬하게 잡는 중)')

# ---------------------------------------------------------------------------
# 컨택트 시트 — 놓침·헛불 (육안으로 공통점 찾기)
# ---------------------------------------------------------------------------
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

def sheet(items, name, label, cap=16):
    items = items[:cap]
    if not items:
        return
    CW = 320; cols = 4; rows = (len(items) + cols - 1) // cols
    im0 = Image.open(items[0][2]).convert('RGB'); ch = round(CW * im0.height / im0.width)
    sh = Image.new('RGB', (cols * CW, rows * (ch + 22)), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for j, r in enumerate(items):
        rr = model.predict(r[2], conf=CONF, verbose=False)[0]
        im = Image.fromarray(rr.plot()[..., ::-1])
        c, rw = j % cols, j // cols; y = rw * (ch + 22)
        dr.text((c * CW + 4, y + 2), f'{r[0]} {label} L{r[5]:.0f} c{r[4]:.2f}',
                fill=(0, 255, 0), font=F)
        sh.paste(im.resize((CW, ch)), (c * CW, y + 22))
    sh.save(f'{OUT}/{name}.jpg', quality=85)
    print(f'-> {OUT}/{name}.jpg  ({len(items)}장, L=밝기·c=conf)')

print()
# 영상별 시트 — 각 영상의 실패 유형을 따로 판독 (jikken이 전체를 채우는 문제 방지)
for v in VIDEOS:
    sheet([r for r in recs if r[0] == v and r[1] == 'fire' and not r[3]], f'_miss_{v}', '놓침')
    sheet([r for r in recs if r[0] == v and r[1] == 'nofire' and r[3]], f'_fp_{v}', '헛불')
# 전체 통합 시트(참고)
sheet([r for r in fire_all if not r[3]], '_miss_all', '놓침', cap=24)
sheet([r for r in recs if r[1] == 'nofire' and r[3]], '_fp_all', '헛불', cap=24)

json.dump({'model': MODEL, 'conf': CONF, 'videos': VIDEOS,
           'per_frame': [(r[0], r[1], r[3], round(r[4], 3), round(r[5], 1)) for r in recs]},
          open(f'{OUT}/err_{MODEL}.json', 'w'), ensure_ascii=False, default=float)
print(f'\n-> {OUT}/err_{MODEL}.json')
print('육안 판독: 놓친 불(_miss.jpg)의 공통점 — 작다? 밝은 배경? 연기 가림? 색? →')
print('  그 부족한 유형을 DATA_collection_spec.md §9 기준으로 집중 수집.')
