# ===== v2 예시 그림 — 같은 배경·같은 불꽃을 [알파(플랫)] vs [발광] 두 방식으로 렌더 (Colab) =====
#
# 목적: 팀 설명용. **같은 배경 + 같은 불꽃 스프라이트 + 같은 위치**에 렌더링 방식만 바꿔
#       "발광 합성이 뭘 바꿨나"를 단일변수로 한눈에 보여준다.
#       (synth_C0/synth_C3 는 소스·배치가 달라 불꽃이 안 맞음 → 여기서 직접 렌더)
# 방법: colab_synth.py 의 블렌딩 로직(스크린·코어블룸·조명스필)을 그대로 재사용.
# 산출: Drive fire_frames/v2_examples/v2_c0_c3.jpg
#       → 다운로드해 repo docs/img/v2_c0_c3.jpg 로 커밋하면 SUMMARY 에 표시됨.
#
# 선행 clone(둘 다): kitchen-fire-noise-poc(이 repo) + kitchen-fire-poc(아틀라스 소스).
# 조절 env: EX_BG(배경 파일 일부) · EX_SPRITE(스프라이트 파일 일부) ·
#          EX_H(불꽃 높이 비율 0.42) · EX_CX·EX_CY(중심 위치 0~1)

import os, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

FIRE  = '/content/drive/MyDrive/fire_frames'
BG    = f'{FIRE}/bg'
ATLAS = '/content/kitchen-fire-poc/assets/flamelib'
OUT   = f'{FIRE}/v2_examples'

EX_BG     = os.environ.get('EX_BG', '')
EX_SPRITE = os.environ.get('EX_SPRITE', '')
EX_H      = float(os.environ.get('EX_H', '0.42'))
EX_CX     = float(os.environ.get('EX_CX', '0.50'))
EX_CY     = float(os.environ.get('EX_CY', '0.62'))
SPLIT     = os.environ.get('EX_SPLIT', 'test')

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

bgs = sorted(glob.glob(f'{BG}/{SPLIT}/*/*.jpg'))
if not bgs:
    raise SystemExit(f'{BG}/{SPLIT} 배경 없음')
bg_path = next((p for p in bgs if EX_BG and EX_BG in p), bgs[0])

sprites = sorted(glob.glob(f'{ATLAS}/*.webp'))
if not sprites:
    raise SystemExit(f'{ATLAS} 비었음 — kitchen-fire-poc 를 clone 할 것')
sp_path = next((p for p in sprites if EX_SPRITE and EX_SPRITE in os.path.basename(p)),
               sprites[len(sprites) // 2])
print(f'배경: {os.path.basename(bg_path)}')
print(f'불꽃: {os.path.basename(sp_path)}  (총 {len(sprites)}개 · EX_SPRITE 로 변경)')
print(f'배치: 높이 {EX_H:.2f} · 중심 ({EX_CX:.2f},{EX_CY:.2f})  (EX_H·EX_CX·EX_CY 로 변경)')

# --- 블렌딩 (colab_synth.py 발췌) ---
def _screen(bg, fg):
    return 255.0 - (255.0 - bg) * (255.0 - fg) / 255.0

def _luma(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114

def _feather(a, s):
    return cv2.GaussianBlur(a, (0, 0), sigmaX=max(s, 0.1))

def _color_correct(fg, bgm, strength=0.15):
    cast = bgm - bgm.mean()
    return np.clip(fg + cast[None, None, :] * strength, 0, 255)

def _core_bloom(out, fg, a3, bloom=0.9, glow=0.35):
    ci = np.clip((_luma(fg) / 255.0 - 0.6) / 0.4, 0, 1) * a3[..., 0]
    out = out + ci[..., None] * (255.0 - out) * bloom
    g = _feather(ci, sigma=max(fg.shape[:2]) * 0.05)
    warm = np.array([255, 180, 90], np.float32)
    return out + g[..., None] * warm[None, None, :] / 255.0 * glow * 60.0

def scale_h(spr, th):
    h, w = spr.shape[:2]; s = th / max(h, 1)
    return cv2.resize(spr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

def render(bg, spr, cx, cy, glow):
    """glow=False → 알파 오버(v1 플랫) · glow=True → 스크린+코어블룸+조명스필(v2)."""
    H, W = bg.shape[:2]; h, w = spr.shape[:2]
    x0 = min(max(0, int(cx - w / 2)), max(0, W - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, H - h))
    out = bg.astype(np.float32); rgb = spr[..., :3].astype(np.float32)
    a = spr[..., 3].astype(np.float32) / 255.0
    if glow:
        a = _feather(a, sigma=max(h, w) * 0.02)
        rgb = _color_correct(rgb, out.reshape(-1, 3).mean(0))
    a3 = a[..., None]; reg = out[y0:y0 + h, x0:x0 + w]
    comp = reg * (1 - a3) + (_screen(reg, rgb) if glow else rgb) * a3
    if glow:
        comp = _core_bloom(comp, rgb, a3)
    out[y0:y0 + h, x0:x0 + w] = comp
    if glow:
        af = np.zeros((H, W), np.float32); af[y0:y0 + h, x0:x0 + w] = a
        sp = _feather(af, sigma=max(H, W) * 0.04)
        warm = np.array([255, 140, 45], np.float32)
        out = out + sp[..., None] * warm[None, None, :] / 255.0 * 0.5 * 90.0
    return np.clip(out, 0, 255).astype(np.uint8)

bg = np.asarray(Image.open(bg_path).convert('RGB'))
H, W = bg.shape[:2]
spr = scale_h(np.asarray(Image.open(sp_path).convert('RGBA')), int(H * EX_H))
cx, cy = EX_CX * W, EX_CY * H
A = render(bg, spr, cx, cy, glow=False)
B = render(bg, spr, cx, cy, glow=True)

def panel(arr, label):
    im = Image.fromarray(arr); Wp = 560; hp = round(Wp * im.height / im.width)
    im = im.resize((Wp, hp))
    c = Image.new('RGB', (Wp, hp + 36), (20, 20, 20)); c.paste(im, (0, 36))
    try:
        Fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except Exception:
        Fnt = ImageFont.load_default()
    ImageDraw.Draw(c).text((8, 7), label, fill=(255, 255, 255), font=Fnt)
    return c

pa = panel(A, 'C0-style  alpha-over (v1 flat)')
pb = panel(B, 'C3-style  glow: screen+bloom+spill (v2)')
G = 12
sheet = Image.new('RGB', (pa.width + pb.width + G, max(pa.height, pb.height)), (20, 20, 20))
sheet.paste(pa, (0, 0)); sheet.paste(pb, (pa.width + G, 0))
dst = f'{OUT}/v2_c0_c3.jpg'
sheet.save(dst, quality=90)
print(f'-> {dst}')
print('  같은 배경·같은 불꽃·같은 위치 → 렌더링 방식만 다름(진짜 단일변수).')
print('  안 보이면 EX_CY(더 어두운 곳)·EX_H(크기)·EX_SPRITE(다른 불꽃) 조절 후 재실행.')
print('  이 파일을 다운로드해 repo docs/img/v2_c0_c3.jpg 로 커밋하세요.')
