# ===== Indoor Fire Smoke(Roboflow)로 깨끗한 실화재 eval — 기존 모델 recall/precision (Colab) =====
#
# 목적: 편집 PSA 영상(오염)이 아니라 **실촬영 실내 화재/연기 데이터셋**으로 우리 합성-학습
#       모델의 실제 전이(recall/precision)를 오염 없이 처음 측정.
# 데이터: Drive '/content/drive/MyDrive/Indoor Fire Smoke.zip' (YOLO 포맷, 2클래스 fire/smoke).
# 방법(프레임 단위, realfire와 동일 정의):
#   - fire 프레임 = GT에 fire 클래스 박스가 있는 이미지 · nofire = fire 박스 없는 이미지(연기만/배경).
#   - recall = fire_det/fire_tot · precision = fire_det/(fire_det+nof_det).
#   - fire 클래스 id(0/1)는 자동 추정(검출률 높은 쪽=fire) + 클래스 샘플 시트로 육안 확인. FIRE_CLASS 로 강제 가능.
# 경계: Roboflow 커뮤니티셋이라 소수 편집/그래픽 프레임 섞일 수 있음(~10%) · 주방/유류 아닌 일반 실내 화재.
# 환경: EVAL_MODEL(기본 v8_C0_s1) · FIRE_CLASS(0/1, 미지정 시 자동) · MAX_IMG(기본 0=전체)

import os, glob, json, zipfile, subprocess, sys, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO
import torch

FIRE   = '/content/drive/MyDrive/fire_frames'
RUNS   = f'{FIRE}/runs_phaseB'
ZIP    = '/content/drive/MyDrive/Indoor Fire Smoke.zip'
DSET   = '/content/Indoor_Fire_Smoke'
OUT    = f'{FIRE}/indoorfire_eval'
CONF   = 0.25
MODEL  = os.environ.get('EVAL_MODEL', 'v8_C0_s1')
BEST   = f'{RUNS}/{MODEL}/best.pt'
FIRE_CLASS = os.environ.get('FIRE_CLASS', '')          # '', '0', '1'
MAX_IMG = int(os.environ.get('MAX_IMG', '0'))
random.seed(0)

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 없음 — EVAL_MODEL 확인')
os.makedirs(DSET, exist_ok=True)
if not os.listdir(DSET):
    print('압축 해제...'); zipfile.ZipFile(ZIP).extractall(DSET)
os.makedirs(OUT, exist_ok=True)
model = YOLO(BEST)

# 이미지 + 라벨(YOLO txt) 수집
imgs = [p for p in glob.glob(f'{DSET}/**/*.jpg', recursive=True) if os.sep + 'images' + os.sep in p]
if not imgs:
    imgs = glob.glob(f'{DSET}/**/*.jpg', recursive=True)      # 폴백
if MAX_IMG:
    random.shuffle(imgs); imgs = sorted(imgs[:MAX_IMG])
print(f'이미지 {len(imgs)}장 · 모델 {MODEL}')

def label_of(imgp):
    lp = imgp.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')
    lp = os.path.splitext(lp)[0] + '.txt'
    return lp

def gt_classes(imgp):
    lp = label_of(imgp)
    if not os.path.exists(lp):
        return set()
    cs = set()
    for line in open(lp):
        line = line.strip()
        if line:
            cs.add(int(float(line.split()[0])))
    return cs

# 모델 검출(프레임 단위) — 배치
det = {}
for i in range(0, len(imgs), 64):
    batch = imgs[i:i + 64]
    for p, r in zip(batch, model.predict(batch, conf=CONF, verbose=False)):
        det[p] = (len(r.boxes) > 0, float(r.boxes.conf.max()) if len(r.boxes) else 0.0)

gts = {p: gt_classes(p) for p in imgs}
present = sorted({c for cs in gts.values() for c in cs})
print(f'GT 클래스: {present}')

# ---------------------------------------------------------------------------
# 클래스별 검출률 → fire 클래스 자동 추정 (fire일수록 우리 flame 모델이 더 잘 잡음)
# ---------------------------------------------------------------------------
def stats_for(fire_cls):
    fire_imgs = [p for p in imgs if fire_cls in gts[p]]
    nof_imgs  = [p for p in imgs if fire_cls not in gts[p]]
    fd = sum(det[p][0] for p in fire_imgs)
    nd = sum(det[p][0] for p in nof_imgs)
    rec = fd / len(fire_imgs) if fire_imgs else 0.0
    prec = fd / (fd + nd) if (fd + nd) else 0.0
    fpr = nd / len(nof_imgs) if nof_imgs else 0.0
    return dict(fire_cls=fire_cls, n_fire=len(fire_imgs), n_nof=len(nof_imgs),
                recall=rec, precision=prec, fpr=fpr, fire_det=fd, nof_det=nd)

print('\n클래스별(가정) 검출률 — 높은 쪽이 fire일 가능성:')
cand = {}
for c in present:
    s = stats_for(c); cand[c] = s
    print(f'  class {c} 를 fire로 보면: recall {s["recall"]:.3f} · precision {s["precision"]:.3f} '
          f'(fire 프레임 {s["n_fire"]} · det {s["fire_det"]})')

if FIRE_CLASS != '':
    fc = int(FIRE_CLASS)
else:
    fc = max(present, key=lambda c: cand[c]['recall']) if present else 0
print(f'\n=> fire 클래스 = {fc} ({"env 지정" if FIRE_CLASS!="" else "자동: 검출률 최대"})  '
      f'· 반대면 FIRE_CLASS 로 강제')

# 클래스 샘플 시트(육안 확인용) — 각 클래스만 든 이미지 6장
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

def class_sheet(c):
    only = [p for p in imgs if gts[p] == {c}][:6] or [p for p in imgs if c in gts[p]][:6]
    if not only:
        return
    CW = 300; im0 = Image.open(only[0]).convert('RGB'); ch = round(CW * im0.height / im0.width)
    sh = Image.new('RGB', (3 * CW, 2 * (ch + 20)), (16, 16, 16)); dr = ImageDraw.Draw(sh)
    for j, p in enumerate(only):
        im = Image.open(p).convert('RGB'); c2, r2 = j % 3, j // 3; y = r2 * (ch + 20)
        dr.text((c2 * CW + 4, y + 2), f'class {c}', fill=(0, 255, 0), font=F)
        sh.paste(im.resize((CW, ch)), (c2 * CW, y + 20))
    sh.save(f'{OUT}/_class{c}.jpg', quality=85); print(f'-> {OUT}/_class{c}.jpg')

for c in present:
    class_sheet(c)

# ---------------------------------------------------------------------------
# 최종 수치 (선택된 fire 클래스)
# ---------------------------------------------------------------------------
s = stats_for(fc)
print('\n' + '=' * 64)
print(f'Indoor Fire Smoke — 깨끗한 실화재 eval · 모델 {MODEL} (fire=class {fc})')
print('=' * 64)
print(f'  recall    {s["recall"]:.3f}  (fire 프레임 {s["fire_det"]}/{s["n_fire"]})')
print(f'  precision {s["precision"]:.3f}')
print(f'  fpr       {s["fpr"]:.3f}  (fire 아닌 {s["nof_det"]}/{s["n_nof"]}에서 오검 — 연기만/배경)')
print(f'  참고: 오염된 realfire(PSA)에서는 recall 0.27~0.31 이었음 → 깨끗한 셋과 비교.')

# miss/fp 시트
fire_imgs = [p for p in imgs if fc in gts[p]]
nof_imgs  = [p for p in imgs if fc not in gts[p]]
def sheet(items, name, label):
    items = items[:16]
    if not items:
        return
    CW = 320; cols = 4; rows = (len(items) + cols - 1) // cols
    im0 = Image.open(items[0]).convert('RGB'); ch = round(CW * im0.height / im0.width)
    sh = Image.new('RGB', (cols * CW, rows * (ch + 20)), (16, 16, 16)); dr = ImageDraw.Draw(sh)
    for j, p in enumerate(items):
        r = model.predict(p, conf=CONF, verbose=False)[0]
        im = Image.fromarray(r.plot()[..., ::-1]); c2, r2 = j % cols, j // cols; y = r2 * (ch + 20)
        dr.text((c2 * CW + 4, y + 2), f'{label} c{det[p][1]:.2f}', fill=(0, 255, 0), font=F)
        sh.paste(im.resize((CW, ch)), (c2 * CW, y + 20))
    sh.save(f'{OUT}/{name}.jpg', quality=85); print(f'-> {OUT}/{name}.jpg')

sheet([p for p in fire_imgs if not det[p][0]], '_miss', '놓친 실화재')
sheet([p for p in nof_imgs if det[p][0]], '_fp', '헛불(연기/배경)')

json.dump({'model': MODEL, 'fire_class': fc, 'conf': CONF, 'n_images': len(imgs),
           'candidates': cand, 'final': s}, open(f'{OUT}/indoorfire_eval.json', 'w'),
          ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/indoorfire_eval.json')
print('※ 경계: 일반 실내 화재(주방/유류 아님) · Roboflow 소수 편집 프레임 섞일 수 있음 · fire 클래스 육안 확인(_class*.jpg).')
