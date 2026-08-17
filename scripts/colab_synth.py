# ===== 4단계: 배경 위 불꽃 합성 + YOLO 자동 박스 (Colab) =====
#
# bg 프레임(2단계) 위에 불꽃 스프라이트(3c단계)를 얹어 검출 학습용 데이터셋을 만든다.
# 스프라이트는 이미 불꽃 bbox 로 잘려 있으므로 **얹은 사각형이 곧 정답 박스**다 —
# 수작업 주석 0.
#
# 이 스크립트가 지키는 설계(docs/PREREGISTER.md):
#
#  1) 풀 매칭 (누수 방지) — split 별로 쓸 불꽃 풀을 고정한다:
#       train 합성 = train 배경 + train 불꽃
#       val   합성 = val   배경 + train 불꽃   (test 불꽃은 절대 안 씀)
#       test  합성 = test  배경 + test  불꽃
#     test 는 배경(사이트)도 불꽃도 학습에서 한 번도 안 본 것 → 진짜 일반화를 잰다.
#
#  2) 제약된 무작위 배치 (B) — 불꽃을 화면 하단~중앙 영역에만 무작위로 놓는다.
#     '공중의 불' 은 피하되 조리면 ROI 지정(A)까지는 안 간다. 검출은 위치 불변이라
#     이 정도로 충분하고, 오히려 위치 증강이 된다.
#
#  3) glow 블렌딩 — 그냥 붙이면 경계선(seam)이 남아 모델이 '붙인 흔적'을 배운다
#     (합성 아티팩트 지름길). 불꽃 둘레에 난색 광원 스필을 더해 경계를 녹인다.
#
#  4) 하드네거티브 — 같은 방식으로 '회색으로 죽인 블롭'을 붙이되 정답은 비움.
#     "붙인 것 ≠ 무조건 불" 을 학습시켜, 불꽃색·밝기로 판단하게 만든다.
#
# 나오는 것 — YOLO 데이터셋(images/labels) + data.yaml + manifest_synth.json + QC 시트.
# **학습 전 QC 시트로 박스가 불꽃에 맞는지, 경계가 자연스러운지 반드시 확인.**
#
# 선행 — 3c(colab_extract_flames.py)로 flame_matte 가 만들어져 있어야 한다.

import os, glob, json, random, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

FIRE       = '/content/drive/MyDrive/fire_frames'
BG_ROOT    = f'{FIRE}/bg'            # bg/<split>/<site>/*.jpg
FLAME_ROOT = f'{FIRE}/flame_matte'   # <pool>/<key>/*.png (RGBA)
OUT        = f'{FIRE}/synth'         # 결과 YOLO 데이터셋

SEED          = 1
POS_FRAC      = 0.60      # 양성(불꽃 합성) 비율
HARDNEG_FRAC  = 0.15      # 하드네거티브(회색 블롭) 비율 — 나머지(0.25)는 평범한 음성
FLAME_H_RANGE = (0.15, 0.45)   # 불꽃 높이 = 배경 높이의 이 비율 (무작위)
PLACE_CX      = (0.30, 0.70)   # 불꽃 중심 x 범위 (배경 폭 비율)
PLACE_CY      = (0.45, 0.78)   # 불꽃 중심 y 범위 (하단~중앙)
GLOW          = 0.5            # 난색 광원 스필 세기 (0=끔)
JPG_Q         = 92

# split → 쓸 불꽃 풀. val 이 train 풀을 쓰는 것이 핵심 — test 풀은 test 에만.
SPLIT_POOL = {'train': 'train', 'val': 'train', 'test': 'test'}

drive.mount('/content/drive')
rng = random.Random(SEED)

# ---------------------------------------------------------------------------
# 스프라이트 적재 (풀별, 캐시)
# ---------------------------------------------------------------------------
def load_pool(pool):
    return sorted(glob.glob(f'{FLAME_ROOT}/{pool}/*/*.png'))

_cache = {}
def sprite(path):
    if path not in _cache:
        _cache[path] = np.asarray(Image.open(path).convert('RGBA'))
    return _cache[path]

def scale_h(spr, target_h):
    h, w = spr.shape[:2]
    s = target_h / max(h, 1)
    return cv2.resize(spr, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)

def paste(bg, spr, cx, cy, glow=GLOW):
    """bg(RGB uint8) 위에 spr(RGBA) 를 (cx,cy) 중심으로 얹는다. 얹은 사각형을 반환."""
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    out = bg.astype(np.float32)
    rgb = spr[..., :3].astype(np.float32)
    a = spr[..., 3].astype(np.float32) / 255.0
    if glow > 0:
        m = int(max(h, w) * 0.35)
        gy0, gy1 = max(0, y0 - m), min(H, y0 + h + m)
        gx0, gx1 = max(0, x0 - m), min(W, x0 + w + m)
        pad = np.zeros((gy1 - gy0, gx1 - gx0), np.float32)
        pad[y0 - gy0:y0 - gy0 + h, x0 - gx0:x0 - gx0 + w] = a
        gb = cv2.GaussianBlur(pad, (0, 0), sigmaX=max(h, w) * 0.12)
        warm = np.array([255, 150, 45], np.float32)          # RGB 난색
        out[gy0:gy1, gx0:gx1] += gb[..., None] * warm * glow
    reg = out[y0:y0 + h, x0:x0 + w]
    out[y0:y0 + h, x0:x0 + w] = rgb * a[..., None] + reg * (1 - a[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), (x0, y0, x0 + w, y0 + h)

def grayblob(spr):
    """불꽃 스프라이트를 회색·어둡게 죽여 '불 아닌 붙임' 으로. 모양만 남고 색·밝기는 뺌."""
    g = spr.copy()
    lum = g[..., :3].astype(np.float32).mean(2, keepdims=True)
    g[..., :3] = np.clip(lum * 0.55, 0, 255).astype(np.uint8)
    return g

def place_one(bg, base_spr):
    H, W = bg.shape[:2]
    th = int(H * rng.uniform(*FLAME_H_RANGE))
    spr = scale_h(base_spr, th)
    if spr.shape[1] > 0.9 * W:                               # 너무 넓으면 줄임
        spr = scale_h(spr, int(spr.shape[0] * 0.9 * W / spr.shape[1]))
    cx = rng.uniform(*PLACE_CX) * W
    cy = rng.uniform(*PLACE_CY) * H
    return spr, cx, cy

# ---------------------------------------------------------------------------
# 합성
# ---------------------------------------------------------------------------
if not os.path.isdir(FLAME_ROOT):
    raise SystemExit(f'{FLAME_ROOT} 가 없음 — 3c(colab_extract_flames.py)를 먼저 돌릴 것')

shutil.rmtree(OUT, ignore_errors=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
except Exception:
    F = ImageFont.load_default()

manifest = {'seed': SEED, 'pos_frac': POS_FRAC, 'hardneg_frac': HARDNEG_FRAC,
            'glow': GLOW, 'split_pool': SPLIT_POOL, 'splits': {}}
qc = []

print('=' * 70)
print('합성 — 배경 위 불꽃 + 자동 박스')
print('=' * 70)

for split in ('train', 'val', 'test'):
    pool = SPLIT_POOL[split]
    sprites = load_pool(pool)
    bgs = sorted(glob.glob(f'{BG_ROOT}/{split}/*/*.jpg'))
    img_dir = f'{OUT}/{split}/images'; lab_dir = f'{OUT}/{split}/labels'
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)

    print(f'\n[{split}] 배경 {len(bgs)}장 · 불꽃 풀 "{pool}" {len(sprites)}개')
    if not sprites:
        print(f'  **불꽃 풀이 빔 — 이 split 은 전부 음성으로 만든다** '
              f'(flame_split.json 배정을 확인할 것)')

    n_pos = n_hn = n_neg = 0
    for i, bp in enumerate(bgs):
        bg = np.asarray(Image.open(bp).convert('RGB'))
        H, W = bg.shape[:2]
        r = rng.random()
        role = ('pos' if (r < POS_FRAC and sprites) else
                'hardneg' if r < POS_FRAC + HARDNEG_FRAC else 'neg')

        label = ''
        if role == 'pos':
            spr, cx, cy = place_one(bg, sprite(rng.choice(sprites)))
            bg, (x0, y0, x1, y1) = paste(bg, spr, cx, cy, glow=GLOW)
            cxn, cyn = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            label = f'0 {cxn:.6f} {cyn:.6f} {bw:.6f} {bh:.6f}\n'
            n_pos += 1
            if len(qc) < 12 and split != 'val':
                qc.append((split, bg.copy(), (x0, y0, x1, y1)))
        elif role == 'hardneg' and sprites:
            spr, cx, cy = place_one(bg, grayblob(sprite(rng.choice(sprites))))
            bg, _ = paste(bg, spr, cx, cy, glow=0)           # 광원 없이 — 불 아님
            n_hn += 1
        else:
            n_neg += 1

        stem = os.path.splitext(os.path.basename(bp))[0]
        name = f'{i:05d}_{stem}'
        Image.fromarray(bg).save(f'{img_dir}/{name}.jpg', quality=JPG_Q)
        open(f'{lab_dir}/{name}.txt', 'w').write(label)      # 음성은 빈 파일

    # 검산 — 이미지 수 == 라벨 수
    ni = len(glob.glob(f'{img_dir}/*.jpg')); nl = len(glob.glob(f'{lab_dir}/*.txt'))
    ok = ni == nl == len(bgs)
    print(f'  양성 {n_pos} · 하드네거 {n_hn} · 음성 {n_neg}  '
          f'(이미지 {ni} · 라벨 {nl})  [검산 {"통과" if ok else "**실패**"}]')
    assert ok, '이미지/라벨 수 불일치'
    manifest['splits'][split] = {'pool': pool, 'n_sprites': len(sprites),
                                 'pos': n_pos, 'hardneg': n_hn, 'neg': n_neg,
                                 'total': len(bgs)}

# ---------------------------------------------------------------------------
# data.yaml + manifest
# ---------------------------------------------------------------------------
open(f'{OUT}/data.yaml', 'w').write(
    f'path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n'
    f'nc: 1\nnames: [\'fire\']\n')
json.dump(manifest, open(f'{OUT}/manifest_synth.json', 'w'),
          ensure_ascii=False, indent=1)

print('\n' + '=' * 70)
print('요약')
print('=' * 70)
for s, d in manifest['splits'].items():
    print(f'  {s:<6} 양성 {d["pos"]:>5} · 하드네거 {d["hardneg"]:>4} · 음성 {d["neg"]:>5} '
          f'· 합 {d["total"]:>5}  (불꽃풀 {d["pool"]} {d["n_sprites"]}개)')
print(f'\n-> 데이터셋: {OUT}/<split>/images · labels')
print(f'-> {OUT}/data.yaml · manifest_synth.json')

# ---------------------------------------------------------------------------
# QC 시트 — 박스가 불꽃에 맞는지, 경계가 자연스러운지
# ---------------------------------------------------------------------------
if qc:
    CW = 360; cols = 3; rows = (len(qc) + cols - 1) // cols
    h0, w0 = qc[0][1].shape[:2]; ch = round(CW * h0 / w0)
    sheet = Image.new('RGB', (cols * CW, rows * (ch + 26)), (16, 16, 16))
    dr = ImageDraw.Draw(sheet)
    for j, (split, img, box) in enumerate(qc):
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle(box, outline=(0, 255, 0), width=3)       # 자동 박스
        r, c = divmod(j, cols); y = r * (ch + 26)
        dr.text((c * CW + 6, y + 3), f'{split} 양성+박스', fill=(0, 255, 0), font=F)
        sheet.paste(im.resize((CW, ch)), (c * CW, y + 26))
    sheet.save(f'{OUT}/_check.jpg', quality=88)
    print(f'\n확인용 시트 -> {OUT}/_check.jpg')
    print('  녹색 박스가 불꽃에 딱 맞는지, 경계가 티 나게 붙었는지(seam) 확인.')
    print('  경계가 심하면 GLOW 를 올리고, 불꽃이 어색하게 크면 FLAME_H_RANGE 를 줄인다.')

print('\n다음 — 학습(YOLO) 전에 Grad-CAM 으로 모델이 불꽃을 보는지 검증(사전 등록).')
