# ===== 방안 #4: 합성 도메인 랜덤화(Domain Randomization) — Colab, GPU =====
#
# 질문(미확정): 합성 불꽃 현실성(v2 C0≈C3)·혼합(무기여)·커리큘럼(#2 무효)은 다 실패했다.
#   그럼 sim2real 의 마지막 안 해본 레버 — **장면·스케일·조명·센서를 극단적으로 randomize**
#   해서 "실제가 그냥 또 하나의 randomization"처럼 보이게 하면 전이가 되나?
#
# 설계(오류 최소화):
#   - colab_synth.py 의 **검증된 합성 코어를 그대로 재사용**(composite/place_gated/visibility).
#     colab_synth.py 는 건드리지 않음(회귀 0). 이 스크립트는 self-contained.
#   - 그 위에 **DR 층**을 얹는다 = ① 넓은 스케일·위치 ② 불꽃 색/밝기 지터
#     ③ 실 CCTV 도메인 쪽 센서 열화(저해상 down-up · grain · blur · jpeg) ④ 광학 지터.
#   - **박스 안전**: DR 열화는 전부 박스-보존(광학/센서)만. 기하변형은 배치범위 확대뿐
#     (composite 가 프레임 클램프). 다중 불꽃·flip·perspective 는 박스 버그 위험이라 제외
#     (단일 불꽃 유지). → 박스는 기존과 동일한 알파 bbox 로직, 새 버그 표면 없음.
#   - 결정적(시드 고정). QC 시트로 **열화 후에도 박스 정합** 육안 확인(Phase 0-DR).
#
# 파이프라인(이 스크립트가 둘 다 함):
#   (1) synth_DR 데이터셋 생성(YOLO·nc=1·fire) — synth_C0 와 동일 포맷.
#   (2) TRAIN=1 이면 synth-only 모델 학습 → runs_if/synth_dr/weights/best.pt.
#       = DR-synth-only 앵커(실 proxy 서 C0 synth-only 0.357 과 대조) + 커리큘럼 BASE.
#
# 다음(별도 셀): split_audit BASE_YOLO=runs_if/synth_dr/best.pt · BASE_TAG=dr
#   → real_only_grouped_dr (DR 커리큘럼). eval 에 1dr·2dr 행 추가돼 자동 대조.
#
# 환경변수:
#   BASE_MODE   : 합성 코어 모드(기본 C3 = screen+bloom+spill, 가장 사실적; C0≈C3라 무관)
#   TRAIN       : '1'(기본) 이면 synth-only 학습까지. '0' 이면 데이터셋만.
#   SYNTH_EPOCHS: synth-only 학습 epoch(기본 60, v8_C0_s1 과 동일)
#   SEED        : 시드(기본 2)
#   DR_STRENGTH : 열화 강도 스케일 0~1(기본 1.0). 너무 세면 불 신호 죽음 → QC 로 조절.

import os, glob, json, random, shutil, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

FIRE     = '/content/drive/MyDrive/fire_frames'
BG_ROOT  = f'{FIRE}/bg'
FLAME_V1 = f'{FIRE}/flame_matte'
ATLAS    = '/content/kitchen-fire-poc/assets/flamelib'

BASE_MODE   = os.environ.get('BASE_MODE', 'C3').upper()
TRAIN       = os.environ.get('TRAIN', '1') == '1'
SYNTH_EPOCHS= int(os.environ.get('SYNTH_EPOCHS', '60'))
SEED        = int(os.environ.get('SEED', '2'))
DR          = float(os.environ.get('DR_STRENGTH', '1.0'))
OUT         = os.environ.get('SYNTH_DR_OUT', '/content/synth_DR')   # 기본=로컬(빠름·세션스코프). Drive 원하면 env 로 지정.

# --- 기존 합성 상수(colab_synth.py 와 동일 개념) + DR 확대 범위 ---
POS_FRAC      = 0.60
HARDNEG_FRAC  = 0.15
FLAME_H_RANGE = (0.10, 0.75)      # DR: 기존 (0.15,0.45) → tiny~large 로 확대
PLACE_CX      = (0.15, 0.85)      # DR: 기존 (0.30,0.70) → 넓게
PLACE_CY      = (0.35, 0.85)      # DR: 기존 (0.45,0.78) → 넓게
JPG_Q         = 92
ATHR          = 0.06
VIS_FLOOR     = 10.0
RETRY         = 8
SPLIT_POOL = {'train': 'train', 'val': 'train', 'test': 'test'}
ATLAS_SPLIT = {'v01': 'train', 'v02': 'train', 'v03': 'train', 'v06': 'train',
               'v07': 'train', 'v10': 'train',
               'v05': 'test',  'v09': 'test'}
CFG = {
    'C0': dict(source='v1',    blend='alpha',  feather=False, colorcorr=False, bloom=False, spill=False),
    'C3': dict(source='atlas', blend='screen', feather=True,  colorcorr=True,  bloom=True,  spill=True),
}[BASE_MODE]

drive.mount('/content/drive')
rng      = random.Random(SEED)          # 배치(스케일·위치·소재)
rng_role = random.Random(SEED)          # 역할(양성/음성)
rng_dr   = random.Random(SEED + 999)    # DR 지터/광학(배치 스트림과 분리)
np_rng   = np.random.default_rng(SEED)  # DR grain(배열 노이즈)

# ---------------------------------------------------------------------------
# 합성 코어 — colab_synth.py 에서 검증된 함수 그대로(단일변수·박스 로직 동일)
# ---------------------------------------------------------------------------
def load_pool(pool):
    if CFG['source'] == 'v1':
        return sorted(glob.glob(f'{FLAME_V1}/{pool}/*/*.png'))
    return [f for f in sorted(glob.glob(f'{ATLAS}/*.webp'))
            if ATLAS_SPLIT.get(os.path.basename(f).split('_')[0]) == pool]

_cache = {}
def sprite(path):
    if path not in _cache:
        _cache[path] = np.asarray(Image.open(path).convert('RGBA'))
    return _cache[path]

def scale_h(spr, target_h):
    h, w = spr.shape[:2]
    s = target_h / max(h, 1)
    return cv2.resize(spr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

def _screen(bg, fg):
    return 255.0 - (255.0 - bg) * (255.0 - fg) / 255.0
def _luma(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
def _feather(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), sigmaX=max(sigma, 0.1))
def _color_correct(fg, bg_mean, strength=0.15):
    cast = bg_mean - bg_mean.mean()
    return np.clip(fg + cast[None, None, :] * strength, 0, 255)
def _core_bloom(out, fg, a3, bloom=0.9, glow=0.35):
    ci = np.clip((_luma(fg) / 255.0 - 0.6) / 0.4, 0, 1) * a3[..., 0]
    out = out + ci[..., None] * (255.0 - out) * bloom
    g = _feather(ci, sigma=max(fg.shape[:2]) * 0.05)
    warm = np.array([255, 180, 90], np.float32)
    return out + g[..., None] * warm[None, None, :] / 255.0 * glow * 60.0
def _alpha_bbox(a):
    ys, xs = np.where(a > ATHR)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def composite(bg, spr, cx, cy):
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    out = bg.astype(np.float32)
    rgb = spr[..., :3].astype(np.float32)
    a_orig = spr[..., 3].astype(np.float32) / 255.0
    a = a_orig
    if CFG['feather']:
        a = _feather(a_orig, sigma=max(h, w) * 0.02)
    if CFG['colorcorr']:
        rgb = _color_correct(rgb, out.reshape(-1, 3).mean(0))
    a3 = a[..., None]
    reg = out[y0:y0 + h, x0:x0 + w]
    if CFG['blend'] == 'screen':
        comp = reg * (1 - a3) + _screen(reg, rgb) * a3
    else:
        comp = reg * (1 - a3) + rgb * a3
    if CFG['bloom']:
        comp = _core_bloom(comp, rgb, a3)
    out[y0:y0 + h, x0:x0 + w] = comp
    if CFG['spill']:
        af = np.zeros((H, W), np.float32); af[y0:y0 + h, x0:x0 + w] = a
        sp = _feather(af, sigma=max(H, W) * 0.04)
        warm = np.array([255, 140, 45], np.float32)
        out = out + sp[..., None] * warm[None, None, :] / 255.0 * 0.5 * 90.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    bb = _alpha_bbox(a_orig)
    if bb is None:
        return out, (x0, y0, x0 + w, y0 + h)
    return out, (x0 + bb[0], y0 + bb[1], x0 + bb[2], y0 + bb[3])

def grayblob(spr):
    g = spr.copy()
    lum = g[..., :3].astype(np.float32).mean(2, keepdims=True)
    g[..., :3] = np.clip(lum * 0.55, 0, 255).astype(np.uint8)
    return g

def paste_gray(bg, spr, cx, cy):
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
    spr = scale_h(base_spr, max(th, 8))
    if spr.shape[1] > 0.9 * W:
        spr = scale_h(spr, int(spr.shape[0] * 0.9 * W / spr.shape[1]))
    cx = rng.uniform(*PLACE_CX) * W
    cy = rng.uniform(*PLACE_CY) * H
    return spr, cx, cy

def _place_geom(bg, spr, cx, cy):
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    a_orig = spr[..., 3].astype(np.float32) / 255.0
    return x0, y0, a_orig, _alpha_bbox(a_orig)

def visibility(bg, spr, x0, y0, a_orig, bb):
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
# DR 층 — 박스 보존 연산만(광학/센서). 기하변형 없음(배치범위만 확대).
# ---------------------------------------------------------------------------
def jitter_sprite(spr, r):
    """불꽃 스프라이트 RGB 를 hue/밝기 지터(알파 보존). 색·밝기 다양화."""
    out = spr.copy()
    rgb = out[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + r.uniform(-8, 8) * DR) % 180                 # hue(불꽃색)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + r.uniform(-0.15, 0.15) * DR), 0, 255)  # 밝기
    out[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out

def photometric(img, r):
    """전역 광학 지터(밝기·대비·채도·hue·감마). 박스 불변."""
    x = img.astype(np.float32)
    x = x + r.uniform(-25, 25) * DR                                          # 밝기
    x = np.clip((x - 128) * (1 + r.uniform(-0.2, 0.2) * DR) + 128, 0, 255)   # 대비
    hsv = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + r.uniform(-0.3, 0.2) * DR), 0, 255)  # 채도
    hsv[..., 0] = (hsv[..., 0] + r.uniform(-6, 6) * DR) % 180                 # hue(색온도)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    g = 1 + r.uniform(-0.2, 0.25) * DR                                       # 감마
    x = 255.0 * np.power(np.clip(x, 0, 255) / 255.0, g)
    return np.clip(x, 0, 255).astype(np.uint8)

def degrade(img, r):
    """실 CCTV 도메인 쪽 센서 열화 — 저해상 down-up · blur · grain · jpeg. 박스 불변."""
    H, W = img.shape[:2]; x = img
    if r.random() < 0.85 * DR + 0.05:                                        # 저해상(down-up)
        f = r.uniform(0.40, 0.85)
        small = cv2.resize(x, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    if r.random() < 0.5 * DR:                                                # blur
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=r.uniform(0.4, 1.6) * DR + 0.1)
    if r.random() < 0.85:                                                     # grain(가우시안 노이즈)
        noise = np_rng.normal(0, r.uniform(4, 16) * DR, x.shape)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(r.uniform(70 - 45 * DR, 75))                                      # jpeg 압축 아티팩트
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), max(20, q)])
    if ok:
        x = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x

# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------
if CFG['source'] == 'atlas' and not glob.glob(f'{ATLAS}/*.webp'):
    raise SystemExit(f'{ATLAS} 비었음 — kitchen-fire-poc clone(HANDOFF)')
if CFG['source'] == 'v1' and not os.path.isdir(FLAME_V1):
    raise SystemExit(f'{FLAME_V1} 없음 — colab_extract_flames.py 먼저')

# 파괴삭제 가드(script-safety): OUT 바구니 이름에 반드시 'synth_DR' 포함일 때만 rmtree.
assert 'synth_DR' in os.path.basename(OUT.rstrip('/\\')), f'안전가드: 예상치 못한 OUT={OUT}'
shutil.rmtree(OUT, ignore_errors=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
except Exception:
    F = ImageFont.load_default()

manifest = {'mode': f'DR(base={BASE_MODE})', 'seed': SEED, 'dr_strength': DR,
            'flame_h': FLAME_H_RANGE, 'place_cx': PLACE_CX, 'place_cy': PLACE_CY,
            'pos_frac': POS_FRAC, 'hardneg_frac': HARDNEG_FRAC, 'splits': {}}
qc = []
print('=' * 70); print(f'합성 도메인 랜덤화(#4) — base {BASE_MODE} · DR {DR} · seed {SEED}'); print('=' * 70)

for split in ('train', 'val', 'test'):
    pool = SPLIT_POOL[split]
    sprites = load_pool(pool)
    bgs = sorted(glob.glob(f'{BG_ROOT}/{split}/*/*.jpg'))
    img_dir = f'{OUT}/{split}/images'; lab_dir = f'{OUT}/{split}/labels'
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    print(f'\n[{split}] 배경 {len(bgs)} · 불꽃풀 "{pool}" {len(sprites)}')
    if not sprites:
        print('  **불꽃풀 빔 — 이 split 전부 음성**')

    n_pos = n_hn = n_neg = 0; vis_list = []; n_fb = 0
    for i, bp in enumerate(bgs):
        bg = np.asarray(Image.open(bp).convert('RGB')); H, W = bg.shape[:2]
        r = rng_role.random()
        role = ('pos' if (r < POS_FRAC and sprites) else
                'hardneg' if r < POS_FRAC + HARDNEG_FRAC else 'neg')
        label = ''; box = None
        if role == 'pos':
            spr, cx, cy, vis, fb = place_gated(bg, sprites)
            spr = jitter_sprite(spr, rng_dr)                       # DR: 불꽃 색/밝기 지터
            bg, (x0, y0, x1, y1) = composite(bg, spr, cx, cy)
            cxn, cyn = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            label = f'0 {cxn:.6f} {cyn:.6f} {bw:.6f} {bh:.6f}\n'
            box = (x0, y0, x1, y1); n_pos += 1; vis_list.append(vis); n_fb += int(fb)
        elif role == 'hardneg' and sprites:
            spr, cx, cy = place_one(bg, grayblob(sprite(rng.choice(sprites))))
            bg = paste_gray(bg, spr, cx, cy); n_hn += 1
        else:
            n_neg += 1

        bg = photometric(bg, rng_dr)                               # DR: 전역 광학
        bg = degrade(bg, rng_dr)                                   # DR: 센서 열화(→ 박스 그대로)

        stem = os.path.splitext(os.path.basename(bp))[0]; name = f'{i:05d}_{stem}'
        Image.fromarray(bg).save(f'{img_dir}/{name}.jpg', quality=JPG_Q)
        open(f'{lab_dir}/{name}.txt', 'w').write(label)
        if box and len(qc) < 12 and split != 'val':
            qc.append((split, bg.copy(), box))                     # 열화 후 이미지+박스(정합 확인)

    ni = len(glob.glob(f'{img_dir}/*.jpg')); nl = len(glob.glob(f'{lab_dir}/*.txt'))
    ok = ni == nl == len(bgs)
    print(f'  양성 {n_pos} · 하드네거 {n_hn} · 음성 {n_neg}  (img {ni}·lab {nl}) [검산 {"통과" if ok else "**실패**"}]')
    if n_pos:
        print(f'  가시성 중앙 {np.median(vis_list):.1f} · 바닥미달 {n_fb}장 ({n_fb/n_pos:.0%})')
    assert ok, '이미지/라벨 수 불일치'
    manifest['splits'][split] = dict(pool=pool, n_sprites=len(sprites), pos=n_pos,
                                     hardneg=n_hn, neg=n_neg, total=len(bgs),
                                     vis_median=float(np.median(vis_list)) if n_pos else None, fallback=n_fb)

open(f'{OUT}/data.yaml', 'w').write(
    f'path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: [\'fire\']\n')
json.dump(manifest, open(f'{OUT}/manifest_synth_dr.json', 'w'), ensure_ascii=False, indent=1)

print('\n' + '=' * 70); print(f'요약 — 도메인 랜덤화(base {BASE_MODE})'); print('=' * 70)
for s, d in manifest['splits'].items():
    print(f'  {s:<6} 양성 {d["pos"]:>5}·하드네거 {d["hardneg"]:>4}·음성 {d["neg"]:>5}·합 {d["total"]:>5} (불꽃풀 {d["pool"]} {d["n_sprites"]})')

# QC 시트 — 열화 후에도 녹색 박스가 불꽃에 맞는지(Phase 0-DR 게이트)
if qc:
    CW = 360; cols = 3; rows = (len(qc) + cols - 1) // cols
    h0, w0 = qc[0][1].shape[:2]; ch = round(CW * h0 / w0)
    sheet = Image.new('RGB', (cols * CW, rows * (ch + 26)), (16, 16, 16)); dr_ = ImageDraw.Draw(sheet)
    for j, (split, img, bx) in enumerate(qc):
        im = Image.fromarray(img); ImageDraw.Draw(im).rectangle(bx, outline=(0, 255, 0), width=3)
        rr, cc = divmod(j, cols); y = rr * (ch + 26)
        dr_.text((cc * CW + 6, y + 3), f'DR {split} 양성+박스', fill=(0, 255, 0), font=F)
        sheet.paste(im.resize((CW, ch)), (cc * CW, y + 26))
    sheet.save(f'{OUT}/_check_dr.jpg', quality=88)
    print(f'\n확인용 시트 -> {OUT}/_check_dr.jpg  (열화 후 박스 정합·불 신호 살았는지 확인)')
    print('  너무 세면(불 안 보임) DR_STRENGTH 낮춰 재생성. 적당하면 아래 학습 진행.')

# ---------------------------------------------------------------------------
# (2) synth-only 학습 — DR 앵커 + 커리큘럼 BASE. runs_if/synth_dr/best.pt (Drive)
# ---------------------------------------------------------------------------
if TRAIN:
    try:
        from ultralytics import YOLO
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
        from ultralytics import YOLO
    print('\n' + '=' * 70); print(f'synth-only 학습(DR) — yolov8s · {SYNTH_EPOCHS}ep → runs_if/synth_dr'); print('=' * 70)
    YOLO('yolov8s.pt').train(data=f'{OUT}/data.yaml', epochs=SYNTH_EPOCHS, imgsz=640, patience=15,
                             project=f'{FIRE}/runs_if', name='synth_dr', exist_ok=True, verbose=False, plots=False)
    best = f'{FIRE}/runs_if/synth_dr/weights/best.pt'
    print(f'\n-> DR-synth-only 모델: {best}')
    print('다음(별도 셀) — 커리큘럼:')
    print("  os.environ['BASE_YOLO']='"+best+"'; os.environ['BASE_TAG']='dr'; os.environ['RUN']='both'")
    print("  %run -i /content/kitchen-fire-noise-poc/scripts/colab_indoorfire_split_audit.py")
    print("  → real_only_grouped_dr · 이후 colab_realtest_eval.py 로 1dr·2dr 대조")
else:
    print('\nTRAIN=0 → 데이터셋만 생성. 학습은 TRAIN=1 로 재실행하거나 split_audit BASE 로 직접.')
