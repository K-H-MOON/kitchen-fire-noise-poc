# ===== S0 — 학습 불꽃 ↔ 검증 불꽃 0-overlap 점검 (pre-reg v2 §4) =====
#
# v2 의 진짜 sim2real 검증이 성립하려면, 학습에 쓴 불꽃(flamelib 아틀라스)과
# realfire 검증 불꽃이 **한 영상도 겹치지 않아야** 한다(§2 "같은 텍스처, 다른 배경"
# 착시 차단). flamelib = 범용 스톡 화염(모닥불·토치류), realfire = 실제 튀김유 화재 →
# 소스 풀이 원래 다르다. 이 시트로 **"이전 아틀라스가 혹시 튀김유 영상을 포함했는지"**
# 를 눈으로 검증한다.
#
# 산출: docs/img/overlap_check.jpg (문서 근거) + Drive 사본(다운로드용) + 인라인 표시.
# 선행: kitchen-fire-poc clone(아틀라스) · Drive 에 realfire 영상(smoke_frames).

import os, glob, json, subprocess, unicodedata
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

REPO  = '/content/kitchen-fire-noise-poc'
ATLAS = '/content/kitchen-fire-poc/assets/flamelib'
RF    = json.load(open(f'{REPO}/scripts/real_fire.json', encoding='utf-8'))
TILE  = 240
DRIVE_OUT = '/content/drive/MyDrive/fire_frames/overlap_check.jpg'
drive.mount('/content/drive')


def norm(s):
    return unicodedata.normalize('NFC', s)


def warm(im):                                    # 주황·고온 마스크 (colab_flame_compare 와 동일)
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    m = (((R > 160) & (R > B + 55) & ((R + G + B) > 340)).astype(np.uint8)) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))


def crop_sq(img, box, pad=0.3):
    h, w = img.shape[:2]; x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), h, w)
    X0 = int(max(0, min(cx - s / 2, w - s))); Y0 = int(max(0, min(cy - s / 2, h - s)))
    return cv2.resize(img[Y0:Y0 + int(s), X0:X0 + int(s)], (TILE, TILE))


def font(sz):
    p = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


if not glob.glob(f'{ATLAS}/*.webp'):
    raise SystemExit(f'{ATLAS} 비었음 — kitchen-fire-poc clone 필요')

# --- flamelib: 소스별 대표 스프라이트 1장 (warm 영역 큰 것) ---
imgs = sorted(glob.glob(f'{ATLAS}/*.webp'))
srcs = sorted(set(os.path.basename(f).split('_')[0] for f in imgs))
atlas_tiles = []
for s in srcs:
    fs = [f for f in imgs if os.path.basename(f).startswith(s + '_')]
    best, bestw = fs[0], -1
    for f in fs[::max(1, len(fs) // 8)]:
        sp = np.array(Image.open(f).convert('RGBA'))
        wa = int((warm(sp[..., :3]) > 0).sum())
        if wa > bestw:
            bestw, best = wa, f
    sp = np.array(Image.open(best).convert('RGBA'))
    a = sp[..., 3:4].astype(np.float32) / 255
    comp = (sp[..., :3] * a + np.full((*sp.shape[:2], 3), 210, np.float32) * (1 - a)).astype(np.uint8)
    ys, xs = np.where(sp[..., 3] > 20)
    box = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, sp.shape[1], sp.shape[0])
    atlas_tiles.append((s, crop_sq(comp, box, 0.3)))
    print(f'flamelib {s}: {os.path.basename(best)} ({len(fs)}장 중)')

# --- realfire: 소스별 불꽃 프레임 1장 ---
tmp = '/content/_ov'
rf_tiles = []
for rec in RF['sources']:
    if not rec['fire_shots']:
        continue
    vids = [p for p in glob.glob(f"{RF['src_dir']}/*") if norm(rec['file']) in norm(os.path.basename(p))]
    if not vids:
        rf_tiles.append((rec['key'] + '(no video)', np.full((TILE, TILE, 3), 60, np.uint8)))
        print(f'realfire {rec["key"]}: 영상 없음 — {RF["src_dir"]} 확인')
        continue
    a, b = rec['fire_shots'][0]; sec = (a + b) // 2
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(sec), '-i', vids[0],
                    '-frames:v', '1', f'{tmp}/f.jpg'], check=False)
    im = cv2.cvtColor(cv2.imread(f'{tmp}/f.jpg'), cv2.COLOR_BGR2RGB)
    cc, _ = cv2.findContours(warm(im), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cc:
        x, y, w, h = cv2.boundingRect(max(cc, key=cv2.contourArea)); box = (x, y, x + w, y + h)
    else:
        H, W = im.shape[:2]; box = (W // 4, H // 4, W * 3 // 4, H * 3 // 4)
    rf_tiles.append((rec['key'], crop_sq(im, box, 0.5)))
    print(f'realfire {rec["key"]}: @ {sec}s')

# --- 조립: 위 = 학습 불꽃(flamelib), 아래 = 검증 불꽃(realfire) ---
def strip(tiles, title, tint):
    n = len(tiles); HDR, LBL = 44, 24
    canvas = Image.new('RGB', (n * TILE, HDR + TILE + LBL), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((10, 12), title, fill=tint, font=font(22))
    for i, (name, t) in enumerate(tiles):
        canvas.paste(Image.fromarray(t), (i * TILE, HDR))
        d.text((i * TILE + 6, HDR + TILE + 2), name, fill=(60, 60, 60), font=font(16))
    return np.array(canvas)

top = strip(atlas_tiles, 'TRAIN flames - flamelib atlas (stock: campfire/torch)', (0, 70, 150))
bot = strip(rf_tiles,    'VAL flames - realfire (real tempura-oil fire)', (185, 70, 0))
Wm = max(top.shape[1], bot.shape[1])
pad = lambda a: np.pad(a, ((0, 0), (0, Wm - a.shape[1]), (0, 0)), constant_values=245)
grid = np.vstack([pad(top), np.full((10, Wm, 3), 245, np.uint8), pad(bot)])

os.makedirs(f'{REPO}/docs/img', exist_ok=True)
repo_out = f'{REPO}/docs/img/overlap_check.jpg'
for p in (repo_out, DRIVE_OUT):
    cv2.imwrite(p, grid[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])

print('\nflamelib 소스:', srcs)
print('realfire 소스:', [r['key'] for r in RF['sources'] if r['fire_shots']])
print(f'-> {repo_out}\n-> {DRIVE_OUT} (다운로드해 로컬 docs/img 에 넣고 커밋)')
print('\n육안 판정: 위(스톡 화염) 와 아래(튀김유 화재)가 서로 다른 영상이면 0-overlap 성립.')
from IPython.display import Image as IPImage, display
display(IPImage(repo_out))
