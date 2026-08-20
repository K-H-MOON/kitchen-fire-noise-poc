# ===== 불꽃 현실성 3단 비교 (문서용, Colab) =====
#
# A 우리 합성 불꽃(플랫 컷아웃) · B 실제 유류 불꽃(발광·입체감) · C 실사 화염 아틀라스(v2 후보).
# "병목 진단 관찰" 삽화용. 라벨 한글(NanumGothic 자동 설치, 없으면 영어 대체).
#
# 설정(env):
#   FLAME_ATLAS = 아틀라스 스프라이트 파일 또는 폴더(png/jpg). 없으면 A·B 2단만.
#   RF_VIDEO    = realfire 영상 key (기본 jikken_douga)
#   SYN_SAMPLE  = A 패널 synth 이미지 경로 (기본: 불꽃 박스 가장 큰 양성)

import os, glob, sys, json, shutil, subprocess, unicodedata
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

REPO = '/content/kitchen-fire-noise-poc/scripts'
FIRE = '/content/drive/MyDrive/fire_frames'
SYN  = f'{FIRE}/synth/test'
OUT  = f'{FIRE}/flame_compare'
PANEL = 460                                   # 각 패널 정사각 px
drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

# --- 폰트 ---
def _find(c):
    for p in c:
        if os.path.exists(p):
            return p
    return None
_DF = _find(['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'])
try:
    import matplotlib
    _b = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    _DF = _DF or _find([f'{_b}/DejaVuSans-Bold.ttf'])
except Exception:
    pass
_KF = _find(['/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
             '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'])
if not _KF:
    subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-nanum'],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _KF = _find(['/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
                 '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'])
print('한글폰트:', _KF or '없음 → 라벨 영어')
def lfont(sz):
    return ImageFont.truetype(_KF or _DF, sz)

def square_crop(img, box, pad=0.3):           # box=(x0,y0,x1,y1)px → 정사각 확대 크롭
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), h, w)
    X0 = int(max(0, min(cx - s / 2, w - s)))
    Y0 = int(max(0, min(cy - s / 2, h - s)))
    return cv2.resize(img[Y0:Y0 + int(s), X0:X0 + int(s)], (PANEL, PANEL))

# --- A: 우리 합성 불꽃 (synth 박스 크롭) ---
def labpath(p):
    return f'{SYN}/labels/' + os.path.splitext(os.path.basename(p))[0] + '.txt'
def flame_boxes(p):
    b, lab = [], labpath(p)
    if os.path.exists(lab):
        for ln in open(lab):
            v = ln.split()
            if len(v) >= 5:
                b.append(tuple(map(float, v[1:5])))
    return b
sample = os.environ.get('SYN_SAMPLE')
if not sample:
    pos = [p for p in sorted(glob.glob(f'{SYN}/images/*.jpg')) if flame_boxes(p)]
    sample = max(pos, key=lambda p: max(w * h for _, _, w, h in flame_boxes(p)))
imgA = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB); h, w = imgA.shape[:2]
cx, cy, bw, bh = max(flame_boxes(sample), key=lambda t: t[2] * t[3])
panelA = square_crop(imgA, (int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                            int((cx + bw / 2) * w), int((cy + bh / 2) * h)), pad=0.35)
print('A synth:', os.path.basename(sample))

# --- B: 실제 유류 불꽃 (fire_shots 프레임들 중 불꽃 가장 큰 순간 자동 선택 + 크롭) ---
def norm(s):
    return unicodedata.normalize('NFC', s)
def warm(im):                                  # 주황·고온 마스크 (얇은 자막 글자는 open 으로 제거)
    R, G, Bl = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    m = (((R > 160) & (R > Bl + 55) & ((R + G + Bl) > 340)).astype(np.uint8)) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
inv = json.load(open(f'{REPO}/real_fire.json', encoding='utf-8'))
key = os.environ.get('RF_VIDEO', 'grease_spread')      # 불꽃 크고 주황 자막 없는 기본
rec = next(s for s in inv['sources'] if s['key'] == key)
vid = [p for p in glob.glob(f"{inv['src_dir']}/*") if norm(rec['file']) in norm(os.path.basename(p))][0]
tmp = '/content/_fc'
best = None                                    # (area, frame_rgb, (x,y,w,h), sec) — 불꽃 가장 큰 프레임
for a, b in rec['fire_shots']:
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', vid, '-t', str(b - a + 1),
                    '-vf', 'fps=1', '-q:v', '2', f'{tmp}/%03d.jpg'], check=False)
    for i, fp in enumerate(sorted(glob.glob(f'{tmp}/*.jpg'))):
        im = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        cnts, _ = cv2.findContours(warm(im), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea); ar = cv2.contourArea(c)
        if best is None or ar > best[0]:
            best = (ar, im, cv2.boundingRect(c), int(a) + i)
if best:
    _, imgB, (x, y, ww, hh), secB = best
    boxB = (x, y, x + ww, y + hh)
else:                                          # 못 찾으면 첫 shot 중앙
    a, b = rec['fire_shots'][0]; secB = (a + b) // 2
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(secB), '-i', vid, '-frames:v', '1', f'{tmp}/f.jpg'], check=False)
    imgB = cv2.cvtColor(cv2.imread(f'{tmp}/f.jpg'), cv2.COLOR_BGR2RGB)
    H2, W2 = imgB.shape[:2]; boxB = (W2 // 4, H2 // 4, W2 * 3 // 4, H2 * 3 // 4)
panelB = square_crop(imgB, boxB, pad=0.5)
print(f'B realfire: {key} @ {secB}s (flame area {int(best[0]) if best else 0})')

# --- C: 아틀라스 스프라이트 (FLAME_ATLAS 지정 시) ---
panels = [('합성 — 플랫 컷아웃', panelA, (185, 70, 0)),
          ('실제 유류 불꽃 — 발광·입체감', panelB, (0, 110, 0))]
atlas = os.environ.get('FLAME_ATLAS')
if atlas:
    try:
        if os.path.isfile(atlas):
            af = atlas
        else:                                  # 폴더면 불꽃(warm) 큰 이미지 자동 선택
            cand = sorted(glob.glob(f'{atlas}/**/*.png', recursive=True) +
                          glob.glob(f'{atlas}/**/*.jpg', recursive=True))[:400]
            af, bestw = cand[0], -1
            for c in cand[::max(1, len(cand) // 24)]:      # 최대 ~24장만 샘플
                im = cv2.imread(c)
                if im is None:
                    continue
                a = int((warm(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) > 0).sum())
                if a > bestw:
                    bestw, af = a, c
        sp = Image.open(af)
        if sp.mode == 'RGBA':                  # 알파 스프라이트 → 회색 위 합성 + 알파 bbox
            bg = Image.new('RGBA', sp.size, (210, 210, 210, 255)); bg.alpha_composite(sp)
            arr = np.array(bg.convert('RGB'))
            al = np.array(sp.split()[-1]); ys, xs = np.where(al > 20)
            boxC = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, sp.width, sp.height)
        else:                                  # 합성 이미지(RGB) → 불꽃 영역 자동 크롭
            arr = np.array(sp.convert('RGB'))
            cc, _ = cv2.findContours(warm(arr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cc:
                x, y, ww, hh = cv2.boundingRect(max(cc, key=cv2.contourArea)); boxC = (x, y, x + ww, y + hh)
            else:
                boxC = (0, 0, sp.width, sp.height)
        panelC = square_crop(arr, boxC, pad=0.3)
        panels.append(('실사 아틀라스 합성 — v2 후보', panelC, (0, 70, 150)))
        print('C atlas:', af)
    except Exception as e:
        print('C atlas 실패 → A·B 2단만:', e)
else:
    print('C atlas: FLAME_ATLAS 미지정 → A·B 2단만')

# --- 조립 ---
HDR, GAP = 56, 10
n = len(panels); W = n * PANEL + (n - 1) * GAP; H = HDR + PANEL
canvas = Image.new('RGB', (W, H), (245, 245, 245)); d = ImageDraw.Draw(canvas)
LF = lfont(24)
for i, (lab, arr, col) in enumerate(panels):
    x = i * (PANEL + GAP)
    canvas.paste(Image.fromarray(arr), (x, HDR))
    d.text((x + PANEL / 2 - d.textlength(lab, font=LF) / 2, 16), lab, fill=col, font=LF)
canvas.save(f'{OUT}/flame_compare.jpg', quality=92)
print(f'-> {OUT}/flame_compare.jpg  ({W}x{H})')
