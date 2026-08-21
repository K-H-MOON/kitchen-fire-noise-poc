# ===== 4단계: 배경 위 불꽃 합성 + YOLO 자동 박스 (Colab) — v2 통합 =====
#
# v1: 불꽃 스프라이트를 알파 오버로 얹어 검출 학습 데이터를 만듦(수작업 주석 0).
#     결론 — 실제 불 전이 약함(realfire 0.31). 병목 = 합성 불꽃의 현실성·다양성
#     (소재 4종 · 납작한 컷아웃).
#
# v2: **불 합성만 바꾸고 나머지(배경·분할·장수·seed·배치)는 전부 고정**하는 단일변수
#     A/B. C0~C3 를 한 스크립트의 SYNTH_MODE 로 돌려 같은 하네스로 생성한다.
#     상세·성공기준·해석트리 = docs/PREREGISTER_v2.md.
#
#     | 모드 | 불꽃 소스        | 합성 방식                         | 직전 대비 |
#     | C0   | v1 매트 4종      | 알파 오버                         | (기준선)  |
#     | C1   | 아틀라스 390     | 알파 오버                         | 소스만    |
#     | C2   | 아틀라스 390     | +스크린·엣지페더·색보정·코어블룸   | 합성 통합 |
#     | C3   | 아틀라스 390     | +가짜 조명 스필                   | 조명 스필 |
#
# 확정한 두 세부 결정 (docs/PREREGISTER_v2.md §5.5):
#   (가) 라벨 박스 = **알파 bbox 고정** — 블룸·스필이 배경을 밝혀도 박스는 안 넓힘.
#        모든 모드 동일 → 단일변수 보존 · realfire 채점 기준(불꽃 자체)과 정합.
#   (나) 블렌딩 = **스크린 + 코어 블룸** — 밝은 급식실 배경(~150+)에서 가산은 순백
#        클리핑(흰 컷아웃). 스크린은 색·계조 살림. 코어 블룸으로 백열 코어만 보완.
#
# 아틀라스(flamelib)는 RGBA WebP 390장(8소스). 알파가 실재 → 휘도키잉 불필요.
# 선행 — kitchen-fire-poc 를 clone 해 ATLAS 경로가 존재해야 함(C1~C3).
#        C0 는 v1 flame_matte(3c: colab_extract_flames.py) 가 있어야 함.
#
# 나오는 것 — synth_<MODE>/ YOLO 데이터셋 + data.yaml + manifest_synth.json + QC 시트.
# **학습 전 _check.jpg 로 박스·경계·발광이 자연스러운지 반드시 확인(Phase 0).**

import os, glob, json, random, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

FIRE       = '/content/drive/MyDrive/fire_frames'
BG_ROOT    = f'{FIRE}/bg'                                   # bg/<split>/<site>/*.jpg
FLAME_V1   = f'{FIRE}/flame_matte'                          # C0: <pool>/<key>/*.png (RGBA)
ATLAS      = '/content/kitchen-fire-poc/assets/flamelib'    # C1~C3: vNN_###.webp (RGBA)

MODE       = os.environ.get('SYNTH_MODE', 'C3').upper()     # C0 / C1 / C2 / C3
OUT        = f'{FIRE}/synth_{MODE}'                         # 모드별 분리 저장

SEED          = 1
POS_FRAC      = 0.60
HARDNEG_FRAC  = 0.15
FLAME_H_RANGE = (0.15, 0.45)
PLACE_CX      = (0.30, 0.70)
PLACE_CY      = (0.45, 0.78)
JPG_Q         = 92
ATHR          = 0.06          # 알파 문턱(박스·가시성)
VIS_FLOOR     = 10.0          # 박스 영역 평균 |Δ| 바닥 — 미만이면 '무신호'로 보고 재배치
RETRY         = 8             # 가시성 재시도 횟수(스케일·위치·소재 바꿈). 형제 smoke repo 교훈

# split → 쓸 불꽃 풀. val 이 train 풀을 쓰는 것이 핵심 — test 풀은 test 에만.
SPLIT_POOL = {'train': 'train', 'val': 'train', 'test': 'test'}

# 아틀라스 소스(vNN) → train/test 풀 배정. **확정 2026-08-21 · 근거 pre-reg §5.6.**
#   ① 비율 348/42(≈89/11)는 이미지 분할 2278/309 과 매칭 → 재사용 균형(~4× 양쪽).
#   ② test 소스 불꽃은 학습에서 한 번도 안 쓰이는 held-out(합성 test 가드·텍스처 착시 차단).
#   ③ realfire 와의 0-overlap 은 소스 풀이 원래 다름(스톡 화염 vs 유류화재)으로 충족,
#      S0 에서 실제 확인.
# 소스별 장수: v01 39·v02 4·v03 117·v05 19·v06 20·v07 136·v09 23·v10 32 (합 390).
ATLAS_SPLIT = {'v01': 'train', 'v02': 'train', 'v03': 'train', 'v06': 'train',
               'v07': 'train', 'v10': 'train',                       # 348장
               'v05': 'test',  'v09': 'test'}                        #  42장 held-out

CFG = {
    'C0': dict(source='v1',    blend='alpha',  feather=False, colorcorr=False, bloom=False, spill=False),
    'C1': dict(source='atlas', blend='alpha',  feather=False, colorcorr=False, bloom=False, spill=False),
    'C2': dict(source='atlas', blend='screen', feather=True,  colorcorr=True,  bloom=True,  spill=False),
    'C3': dict(source='atlas', blend='screen', feather=True,  colorcorr=True,  bloom=True,  spill=True),
}[MODE]

drive.mount('/content/drive')
rng = random.Random(SEED)          # 배치(스케일·위치·소재)
rng_role = random.Random(SEED)     # 역할(양성/음성) — 별도 스트림이라 재시도와 무관, 전 모드 동일 배정

# ---------------------------------------------------------------------------
# 스프라이트 적재 (풀별) — C0 은 v1 매트, C1~C3 은 아틀라스. 둘 다 RGBA.
# ---------------------------------------------------------------------------
def load_pool(pool):
    if CFG['source'] == 'v1':
        return sorted(glob.glob(f'{FLAME_V1}/{pool}/*/*.png'))
    out = []
    for f in sorted(glob.glob(f'{ATLAS}/*.webp')):
        if ATLAS_SPLIT.get(os.path.basename(f).split('_')[0]) == pool:
            out.append(f)
    return out

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

# ---------------------------------------------------------------------------
# 합성 연산자 (로컬 검증 완료 — scratchpad/demo_synth.py)
# ---------------------------------------------------------------------------
def _screen(bg, fg):
    return 255.0 - (255.0 - bg) * (255.0 - fg) / 255.0

def _luma(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114

def _feather(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigmaX=max(sigma, 0.1))

def _color_correct(fg, bg_mean, strength=0.15):
    cast = bg_mean - bg_mean.mean()                 # 배경 색편향으로 살짝 당김
    return np.clip(fg + cast[None, None, :] * strength, 0, 255)

def _core_bloom(out, fg, a3, bloom=0.9, glow=0.35):
    ci = np.clip((_luma(fg) / 255.0 - 0.6) / 0.4, 0, 1) * a3[..., 0]
    out = out + ci[..., None] * (255.0 - out) * bloom          # headroom 비례 → 클리핑 없음
    g = _feather(ci, sigma=max(fg.shape[:2]) * 0.05)
    warm = np.array([255, 180, 90], np.float32)
    return out + g[..., None] * warm[None, None, :] / 255.0 * glow * 60.0

def _alpha_bbox(a):
    ys, xs = np.where(a > ATHR)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def composite(bg, spr, cx, cy):
    """bg(RGB uint8) 위에 spr(RGBA) 를 (cx,cy) 중심으로 CFG 방식대로 얹는다.
       반환: (합성 uint8, 알파 bbox 박스). 박스는 스프라이트 원본 알파 기준(고정)."""
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    out = bg.astype(np.float32)
    rgb = spr[..., :3].astype(np.float32)
    a_orig = spr[..., 3].astype(np.float32) / 255.0            # 박스는 이 원본 알파로
    a = a_orig
    if CFG['feather']:
        a = _feather(a_orig, sigma=max(h, w) * 0.02)
    if CFG['colorcorr']:
        rgb = _color_correct(rgb, out.reshape(-1, 3).mean(0))
    a3 = a[..., None]

    reg = out[y0:y0 + h, x0:x0 + w]
    if CFG['blend'] == 'screen':
        comp = reg * (1 - a3) + _screen(reg, rgb) * a3
    else:                                                       # alpha over
        comp = reg * (1 - a3) + rgb * a3
    if CFG['bloom']:
        comp = _core_bloom(comp, rgb, a3)
    out[y0:y0 + h, x0:x0 + w] = comp

    if CFG['spill']:                                           # 조명 스필(캔버스 전체·박스 밖)
        af = np.zeros((H, W), np.float32)
        af[y0:y0 + h, x0:x0 + w] = a
        sp = _feather(af, sigma=max(H, W) * 0.04)
        warm = np.array([255, 140, 45], np.float32)
        out = out + sp[..., None] * warm[None, None, :] / 255.0 * 0.5 * 90.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    bb = _alpha_bbox(a_orig)
    if bb is None:
        return out, (x0, y0, x0 + w, y0 + h)
    return out, (x0 + bb[0], y0 + bb[1], x0 + bb[2], y0 + bb[3])

def grayblob(spr):
    """불꽃 스프라이트를 회색·어둡게 죽여 '불 아닌 붙임' 으로. 모양만 남고 색·밝기는 뺌."""
    g = spr.copy()
    lum = g[..., :3].astype(np.float32).mean(2, keepdims=True)
    g[..., :3] = np.clip(lum * 0.55, 0, 255).astype(np.uint8)
    return g

def paste_gray(bg, spr, cx, cy):
    """하드네거티브 — 회색 블롭을 알파 오버(발광·스필 없음). 반환: 합성 uint8."""
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    out = bg.astype(np.float32)
    rgb = spr[..., :3].astype(np.float32)
    a3 = (spr[..., 3].astype(np.float32) / 255.0)[..., None]
    out[y0:y0 + h, x0:x0 + w] = out[y0:y0 + h, x0:x0 + w] * (1 - a3) + rgb * a3
    return np.clip(out, 0, 255).astype(np.uint8)

def place_one(bg, base_spr):
    H, W = bg.shape[:2]
    th = int(H * rng.uniform(*FLAME_H_RANGE))
    spr = scale_h(base_spr, th)
    if spr.shape[1] > 0.9 * W:                                 # 너무 넓으면 줄임
        spr = scale_h(spr, int(spr.shape[0] * 0.9 * W / spr.shape[1]))
    cx = rng.uniform(*PLACE_CX) * W
    cy = rng.uniform(*PLACE_CY) * H
    return spr, cx, cy

# ---------------------------------------------------------------------------
# 가시성 게이트 — 무신호 라벨 차단 (형제 smoke repo 교훈). 전 모드 공통 → 단일변수 유지.
#   스크린은 밝은 배경에서 씻겨 불꽃이 안 보이는 배치가 생김 → 박스에 신호 없는 라벨.
#   그런 배치를 버리고 스케일·위치·소재를 바꿔 재시도(장수 유지). C0(불투명)은 거의 안 걸림.
# ---------------------------------------------------------------------------
def _place_geom(bg, spr, cx, cy):
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    a_orig = spr[..., 3].astype(np.float32) / 255.0
    return x0, y0, a_orig, _alpha_bbox(a_orig)

def visibility(bg, spr, x0, y0, a_orig, bb):
    """박스(tight) 영역에서 블렌딩이 만드는 평균 |Δ|. feather/bloom/spill 전이라 저렴.
       Δ = (블렌드 - 배경)·알파. 스크린이 밝은 배경에서 무신호가 된 배치를 걸러내는 값."""
    rgb = spr[..., :3].astype(np.float32)
    if CFG['colorcorr']:
        rgb = _color_correct(rgb, bg.reshape(-1, 3).mean(0))
    bx0, by0, bx1, by1 = bb
    bgbox = bg[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1].astype(np.float32)
    fgbox = rgb[by0:by1, bx0:bx1]
    abox = a_orig[by0:by1, bx0:bx1][..., None]
    blended = _screen(bgbox, fgbox) if CFG['blend'] == 'screen' else fgbox
    return float((np.abs(blended - bgbox) * abox).mean())

def place_gated(bg, sprites):
    """RETRY 안에 바닥(VIS_FLOOR)을 넘으면 그 배치를, 못 넘으면 그중 가장 잘 보이는 것(최선)을
       쓴다. 반환: (spr, cx, cy, 가시성, 바닥미달여부)."""
    best = None
    for _ in range(RETRY):
        spr, cx, cy = place_one(bg, sprite(rng.choice(sprites)))
        x0, y0, a_orig, bb = _place_geom(bg, spr, cx, cy)
        if bb is None:
            continue
        vis = visibility(bg, spr, x0, y0, a_orig, bb)
        if best is None or vis > best[3]:
            best = (spr, cx, cy, vis)
        if vis >= VIS_FLOOR:
            return spr, cx, cy, vis, False
    if best is None:
        spr, cx, cy = place_one(bg, sprite(rng.choice(sprites)))
        return spr, cx, cy, 0.0, True
    return best[0], best[1], best[2], best[3], True

# ---------------------------------------------------------------------------
# 합성
# ---------------------------------------------------------------------------
if CFG['source'] == 'atlas' and not glob.glob(f'{ATLAS}/*.webp'):
    raise SystemExit(f'{ATLAS} 가 비었음 — kitchen-fire-poc 를 clone 할 것(HANDOFF)')
if CFG['source'] == 'v1' and not os.path.isdir(FLAME_V1):
    raise SystemExit(f'{FLAME_V1} 가 없음 — 3c(colab_extract_flames.py)를 먼저 돌릴 것')

shutil.rmtree(OUT, ignore_errors=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
except Exception:
    F = ImageFont.load_default()

manifest = {'mode': MODE, 'cfg': CFG, 'seed': SEED, 'pos_frac': POS_FRAC,
            'hardneg_frac': HARDNEG_FRAC, 'split_pool': SPLIT_POOL,
            'atlas_split': ATLAS_SPLIT if CFG['source'] == 'atlas' else None, 'splits': {}}
qc = []

print('=' * 70)
print(f'합성 v2 — MODE {MODE}  {CFG}')
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
              f'(ATLAS_SPLIT / flame_matte 배정 확인)')

    n_pos = n_hn = n_neg = 0
    vis_list = []; n_fallback = 0
    for i, bp in enumerate(bgs):
        bg = np.asarray(Image.open(bp).convert('RGB'))
        H, W = bg.shape[:2]
        r = rng_role.random()
        role = ('pos' if (r < POS_FRAC and sprites) else
                'hardneg' if r < POS_FRAC + HARDNEG_FRAC else 'neg')

        label = ''
        if role == 'pos':
            spr, cx, cy, vis, fb = place_gated(bg, sprites)
            bg, (x0, y0, x1, y1) = composite(bg, spr, cx, cy)
            cxn, cyn = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            label = f'0 {cxn:.6f} {cyn:.6f} {bw:.6f} {bh:.6f}\n'
            n_pos += 1; vis_list.append(vis); n_fallback += int(fb)
            if len(qc) < 12 and split != 'val':
                qc.append((split, bg.copy(), (x0, y0, x1, y1)))
        elif role == 'hardneg' and sprites:
            spr, cx, cy = place_one(bg, grayblob(sprite(rng.choice(sprites))))
            bg = paste_gray(bg, spr, cx, cy)
            n_hn += 1
        else:
            n_neg += 1

        stem = os.path.splitext(os.path.basename(bp))[0]
        name = f'{i:05d}_{stem}'
        Image.fromarray(bg).save(f'{img_dir}/{name}.jpg', quality=JPG_Q)
        open(f'{lab_dir}/{name}.txt', 'w').write(label)        # 음성은 빈 파일

    ni = len(glob.glob(f'{img_dir}/*.jpg')); nl = len(glob.glob(f'{lab_dir}/*.txt'))
    ok = ni == nl == len(bgs)
    print(f'  양성 {n_pos} · 하드네거 {n_hn} · 음성 {n_neg}  '
          f'(이미지 {ni} · 라벨 {nl})  [검산 {"통과" if ok else "**실패**"}]')
    if n_pos:
        print(f'  가시성 중앙 {np.median(vis_list):.1f} · 바닥({VIS_FLOOR:.0f}) 미달로 최선 사용 '
              f'{n_fallback}장 ({n_fallback / n_pos:.0%})')
    assert ok, '이미지/라벨 수 불일치'
    manifest['splits'][split] = {'pool': pool, 'n_sprites': len(sprites),
                                 'pos': n_pos, 'hardneg': n_hn, 'neg': n_neg,
                                 'total': len(bgs),
                                 'vis_median': float(np.median(vis_list)) if n_pos else None,
                                 'fallback': n_fallback}

# ---------------------------------------------------------------------------
# data.yaml + manifest
# ---------------------------------------------------------------------------
open(f'{OUT}/data.yaml', 'w').write(
    f'path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n'
    f'nc: 1\nnames: [\'fire\']\n')
json.dump(manifest, open(f'{OUT}/manifest_synth.json', 'w'), ensure_ascii=False, indent=1)

print('\n' + '=' * 70)
print(f'요약 — MODE {MODE}')
print('=' * 70)
for s, d in manifest['splits'].items():
    print(f'  {s:<6} 양성 {d["pos"]:>5} · 하드네거 {d["hardneg"]:>4} · 음성 {d["neg"]:>5} '
          f'· 합 {d["total"]:>5}  (불꽃풀 {d["pool"]} {d["n_sprites"]}개)')
print(f'\n-> {OUT}/<split>/images · labels · data.yaml · manifest_synth.json')

# ---------------------------------------------------------------------------
# QC 시트 — 박스가 불꽃에 맞는지, 발광·경계가 자연스러운지 (Phase 0 육안)
# ---------------------------------------------------------------------------
if qc:
    CW = 360; cols = 3; rows = (len(qc) + cols - 1) // cols
    h0, w0 = qc[0][1].shape[:2]; ch = round(CW * h0 / w0)
    sheet = Image.new('RGB', (cols * CW, rows * (ch + 26)), (16, 16, 16))
    dr = ImageDraw.Draw(sheet)
    for j, (split, img, box) in enumerate(qc):
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        d.rectangle(box, outline=(0, 255, 0), width=3)
        r, c = divmod(j, cols); y = r * (ch + 26)
        dr.text((c * CW + 6, y + 3), f'{MODE} {split} 양성+박스', fill=(0, 255, 0), font=F)
        sheet.paste(im.resize((CW, ch)), (c * CW, y + 26))
    sheet.save(f'{OUT}/_check.jpg', quality=88)
    print(f'\n확인용 시트 -> {OUT}/_check.jpg')
    print('  녹색 박스가 불꽃에 맞는지(스필·블룸은 박스 밖), 발광이 자연스러운지 확인.')
    print('  C0 와 C3 시트를 나란히 비교 — 컷아웃 vs 발광(Phase 0 게이트).')

print('\n다음 — C0 와 C3 를 각각 생성(SYNTH_MODE 로 재실행) 후 Phase 0 육안 비교.')
