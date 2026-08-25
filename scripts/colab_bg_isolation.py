# ===== 배경-격리 스트레스 테스트 (A) · Colab =====
#
# 질문: "급식실 배경이 실불꽃 검출을 막나?" — 실 급식실 화재 데이터가 없어(구할 수도 실연할 수도 없음)
#   직접 증명은 불가. 대신 배경 '한 변수'만 격리해 갭을 좁힌다:
#     같은 실불꽃(flamelib 아틀라스·C3 합성)을 ① 급식실 배경 ② 대조(비-급식실 주방 배경) 에 얹어
#     배포후보 2ck 의 검출률을 비교. 급식실에서 대조와 비슷하게 잡히면
#     → "급식실 배경이 검출을 막지는 않는다" 배제(rule-out).
#
# ⚠️ 정직한 한계(반드시 병기): 합성 불꽃 ≠ 실 유류불(연기·반사·유류불 외형 없음).
#   이건 "실 급식실 화재가 된다"의 증명이 아니라 "배경이 방해하진 않는다"의 반증. 갭을 좁히지 닫지 못함.
#
# 선행: kitchen-fire-poc clone(ATLAS=flamelib). env:
#   CAFE_BG_DIR(급식실 프레임·기본 oilfire_realtest/nofire_kitchen) · CTRL_BG_DIR(대조·기본 nofire_presrc)
#   N(팔당 합성 수·기본 120) · SEED(기본 0 대신 프레임순서로 결정론)

import os, glob, json, subprocess, sys
import numpy as np
try:
    import cv2
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python-headless'], check=True)
    import cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE  = '/content/drive/MyDrive/fire_frames'
IFRUN = f'{FIRE}/runs_if'; RUNS = f'{FIRE}/runs_phaseB'
ATLAS = '/content/kitchen-fire-poc/assets/flamelib'
REAL  = os.environ.get('OUT_DIR', '/content/oilfire_realtest')
CAFE_BG = os.environ.get('CAFE_BG_DIR', f'{REAL}/nofire_kitchen')
CTRL_BG = os.environ.get('CTRL_BG_DIR', f'{REAL}/nofire_presrc')
OUTDIR  = os.environ.get('BGISO_OUT', '/content/bg_isolation')
N       = int(os.environ.get('N', '120'))
CONF    = float(os.environ.get('CONF', '0.25'))
drive.mount('/content/drive')
os.makedirs(OUTDIR, exist_ok=True)

atlas = sorted(glob.glob(f'{ATLAS}/*.webp'))
assert atlas, f'{ATLAS} 비었음 — kitchen-fire-poc clone 필요(HANDOFF)'
# test 풀(v05·v09) 만 사용 = 어느 실모델도 학습에 안 쓴 held-out 불꽃(합성모델 누수도 차단)
atlas = [f for f in atlas if os.path.basename(f).split('_')[0] in ('v05', 'v09')] or atlas
print(f'flamelib 불꽃 {len(atlas)}개(held-out v05·v09) · 급식실bg {len(glob.glob(f"{CAFE_BG}/*.jpg"))} · 대조bg {len(glob.glob(f"{CTRL_BG}/*.jpg"))}')

# ---- C3 합성 연산자 (colab_synth.py 이식, 스크린+페더+색보정+코어블룸+스필) ----
FLAME_H_RANGE = (0.15, 0.45); PLACE_CX = (0.30, 0.70); PLACE_CY = (0.45, 0.78)
ATHR = 0.06; VIS_FLOOR = 10.0; RETRY = 8
_cache = {}
def spr(p):
    if p not in _cache: _cache[p] = np.asarray(Image.open(p).convert('RGBA'))
    return _cache[p]
def scale_h(s, th):
    h, w = s.shape[:2]; sc = th / max(h, 1)
    return cv2.resize(s, (max(1, int(w*sc)), max(1, int(h*sc))), interpolation=cv2.INTER_AREA)
def _screen(bg, fg): return 255.0 - (255.0-bg)*(255.0-fg)/255.0
def _luma(x): return x[...,0]*0.299 + x[...,1]*0.587 + x[...,2]*0.114
def _feather(a, s): return cv2.GaussianBlur(a, (0,0), sigmaX=max(s, 0.1))
def _cc(fg, bgm, k=0.15): return np.clip(fg + (bgm-bgm.mean())[None,None,:]*k, 0, 255)
def _bloom(out, fg, a3, bloom=0.9, glow=0.35):
    ci = np.clip((_luma(fg)/255.0 - 0.6)/0.4, 0, 1) * a3[...,0]
    out = out + ci[...,None]*(255.0-out)*bloom
    g = _feather(ci, max(fg.shape[:2])*0.05); warm = np.array([255,180,90], np.float32)
    return out + g[...,None]*warm[None,None,:]/255.0*glow*60.0
def _abbox(a):
    ys, xs = np.where(a > ATHR)
    return None if len(xs)==0 else (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
def composite(bg, s, cx, cy):
    H, W = bg.shape[:2]; h, w = s.shape[:2]
    x0 = min(max(0, int(cx-w/2)), max(0, W-w)); y0 = min(max(0, int(cy-h/2)), max(0, H-h))
    out = bg.astype(np.float32); rgb = s[...,:3].astype(np.float32)
    a_orig = s[...,3].astype(np.float32)/255.0
    a = _feather(a_orig, max(h,w)*0.02); rgb = _cc(rgb, out.reshape(-1,3).mean(0)); a3 = a[...,None]
    reg = out[y0:y0+h, x0:x0+w]
    comp = reg*(1-a3) + _screen(reg, rgb)*a3
    comp = _bloom(comp, rgb, a3)
    out[y0:y0+h, x0:x0+w] = comp
    af = np.zeros((H, W), np.float32); af[y0:y0+h, x0:x0+w] = a
    sp = _feather(af, max(H,W)*0.04); warm = np.array([255,140,45], np.float32)
    out = out + sp[...,None]*warm[None,None,:]/255.0*0.5*90.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    bb = _abbox(a_orig)
    return (out, (x0, y0, x0+w, y0+h)) if bb is None else (out, (x0+bb[0], y0+bb[1], x0+bb[2], y0+bb[3]))
def visibility(bg, s, cx, cy):
    H, W = bg.shape[:2]; h, w = s.shape[:2]
    x0 = min(max(0, int(cx-w/2)), max(0, W-w)); y0 = min(max(0, int(cy-h/2)), max(0, H-h))
    a = s[...,3].astype(np.float32)/255.0; bb = _abbox(a)
    if bb is None: return 0.0
    rgb = _cc(s[...,:3].astype(np.float32), bg.reshape(-1,3).mean(0))
    bx0, by0, bx1, by1 = bb
    bgb = bg[y0+by0:y0+by1, x0+bx0:x0+bx1].astype(np.float32); fgb = rgb[by0:by1, bx0:bx1]
    ab = a[by0:by1, bx0:bx1][...,None]
    return float((np.abs(_screen(bgb, fgb)-bgb)*ab).mean())

def build_arm(bg_dir, tag, rngseed):
    bgs = sorted(glob.glob(f'{bg_dir}/*.jpg'))
    assert bgs, f'{tag} 배경 없음: {bg_dir}'
    d = f'{OUTDIR}/{tag}'; os.makedirs(d, exist_ok=True)
    import random; rng = random.Random(rngseed)
    recs = []
    for i in range(N):
        bp = bgs[i % len(bgs)]
        bg = np.asarray(Image.open(bp).convert('RGB')); H, W = bg.shape[:2]
        best = None
        for _ in range(RETRY):                          # 가시성 게이트(무신호 배치 차단)
            base = spr(atlas[rng.randrange(len(atlas))])
            th = int(H * rng.uniform(*FLAME_H_RANGE)); s = scale_h(base, th)
            if s.shape[1] > 0.9*W: s = scale_h(s, int(s.shape[0]*0.9*W/s.shape[1]))
            cx = rng.uniform(*PLACE_CX)*W; cy = rng.uniform(*PLACE_CY)*H
            vis = visibility(bg, s, cx, cy)
            if best is None or vis > best[-1]: best = (s, cx, cy, vis)
            if vis >= VIS_FLOOR: break
        s, cx, cy, vis = best
        comp, box = composite(bg, s, cx, cy)
        op = f'{d}/{i:04d}.jpg'; Image.fromarray(comp).save(op, quality=90)
        recs.append(dict(path=op, box=box))
    return recs

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

models = {
    '2ck_grouped_ck': f'{IFRUN}/real_only_grouped_ck/weights/best.pt',
    '2g_grouped':     f'{IFRUN}/real_only_grouped/weights/best.pt',
    '2_real_only':    f'{IFRUN}/real_only/weights/best.pt',
    '1_synth':        f'{RUNS}/v8_C0_s1/best.pt',
}

print('\n합성 중...')
arms = {'cafe(급식실)': build_arm(CAFE_BG, 'cafe', 0),
        'ctrl(대조주방)': build_arm(CTRL_BG, 'ctrl', 0)}   # 같은 seed·같은 불꽃 순서 → 배경만 다름

rows = {}
for key, best in models.items():
    if not os.path.exists(best):
        print(f'  [없음] {key}'); continue
    m = YOLO(best); rows[key] = {}
    for arm, recs in arms.items():
        paths = [r['path'] for r in recs]; det = hit = 0
        preds = []
        for i in range(0, len(paths), 64):
            for r in m.predict(paths[i:i+64], conf=CONF, verbose=False):
                preds.append(r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0,4)))
        for rec, pb in zip(recs, preds):
            if len(pb): det += 1
            if len(pb) and max(iou(rec['box'], b) for b in pb) >= 0.1: hit += 1
        rows[key][arm] = dict(detect=det/len(recs), localize=hit/len(recs), n=len(recs))

print('\n' + '='*72)
print(f'배경-격리 스트레스 테스트 · 같은 실불꽃(flamelib), 배경만 교체 (conf {CONF})')
print('='*72)
print(f'{"모델":<18}{"검출률 cafe":>13}{"검출률 ctrl":>13}{"위치 cafe":>11}{"위치 ctrl":>11}')
for k, r in rows.items():
    c, t = r.get('cafe(급식실)', {}), r.get('ctrl(대조주방)', {})
    print(f'{k:<18}{c.get("detect",0):>13.3f}{t.get("detect",0):>13.3f}{c.get("localize",0):>11.3f}{t.get("localize",0):>11.3f}')

print('\n판독(2ck 기준):')
print('  · 검출률 cafe ≈ ctrl 이고 높으면 → "급식실 배경은 실불꽃 검출을 막지 않는다" 배제(rule-out).')
print('  · cafe ≪ ctrl 이면 → 급식실 배경(스테인리스 반사·밝은 조명)이 검출을 저해 → 문제.')
print('  ⚠️ 한계: 합성불꽃 ≠ 실 유류불(연기·반사·유류외형 없음). 배경변수만 격리 — 갭 좁힘, 못 닫음.')

json.dump(rows, open(f'{OUTDIR}/bg_isolation.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUTDIR}/bg_isolation.json · 합성본 {OUTDIR}/cafe·ctrl/*.jpg')

# 몽타주(2ck 예측박스) — cafe 팔 눈확인
try: F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except Exception: F = ImageFont.load_default()
mk = '2ck_grouped_ck'
if mk in [k for k in models if os.path.exists(models[k])]:
    m = YOLO(models[mk]); recs = arms['cafe(급식실)'][:24]
    cols = 6; cw = 220; im0 = Image.open(recs[0]['path']); ch = round(cw*im0.height/im0.width)
    rows_n = (len(recs)+cols-1)//cols
    sh = Image.new('RGB', (cols*cw, rows_n*(ch+4)), (16,16,16))
    for j, rec in enumerate(recs):
        im = Image.open(rec['path']).convert('RGB'); sx = cw/im.width; sy = ch/im.height
        im = im.resize((cw, ch)); dr = ImageDraw.Draw(im)
        gt = rec['box']; dr.rectangle([gt[0]*sx, gt[1]*sy, gt[2]*sx, gt[3]*sy], outline=(0,255,0), width=1)
        for r in m.predict(rec['path'], conf=CONF, verbose=False):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                dr.rectangle([b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy], outline=(255,60,60), width=2)
                dr.text((b[0]*sx+1, b[1]*sy+1), f'{c:.2f}', fill=(255,220,0), font=F)
        sh.paste(im, ((j%cols)*cw, (j//cols)*(ch+4)))
    out = f'{OUTDIR}/cafe_2ck_pred.jpg'; sh.save(out, quality=86)
    print(f'-> {out} (초록=합성 불꽃 GT · 빨강=2ck 예측 · 노랑=conf)')
