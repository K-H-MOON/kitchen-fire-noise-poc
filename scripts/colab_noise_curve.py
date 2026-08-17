# ===== 6단계: 노이즈 저하 곡선 (Phase A, Colab) =====
#
# 믿을 수 있는 기준점(강도 0, ablation 통과한 모델)에서 노이즈를 단계별로 올리며
# 인식률이 어디서 무너지는지 잰다. 학습은 그대로(깨끗한 합성으로만), 노이즈는
# **test 에만** 입힌다.
#
# 재는 값 (이미지 단위 — 화재경보 관점):
#   flame_rate = 노이즈 입힌 test 양성에서 불꽃 검출되는 비율  (떨어지는 걸 봄)
#   fp_rate    = 노이즈 입힌 test 음성에서 오탐 비율            (오르는 걸 봄)
#
# 노이즈 종류 (전부 numpy/cv2 자작 — 외부 의존성 없음, 강도 0=원본 ~ 5):
#   화질   gaussian · jpeg · motion_blur · defocus · low_light
#   의미적 steam (흰 수증기 오버레이)
#
# GPU 권장. best.pt(5단계)가 있어야 한다.

import os, glob, json
import numpy as np, cv2
from google.colab import drive
from ultralytics import YOLO

FIRE = '/content/drive/MyDrive/fire_frames'
BEST = f'{FIRE}/runs/fire_s/best.pt'
SYN  = f'{FIRE}/synth/test'
OUT  = f'{FIRE}/noise'
CONF = 0.25
SEVS = [0, 1, 2, 3, 4, 5]
SEED = 1

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 가 없음 — 5단계를 먼저 돌릴 것')
model = YOLO(BEST)
rng = np.random.RandomState(SEED)

# ---------------------------------------------------------------------------
# 노이즈 함수들 — 입력/출력 모두 RGB uint8. 강도 0 은 항상 원본.
# ---------------------------------------------------------------------------
def n_gaussian(img, s):
    std = [0, 8, 16, 28, 44, 64][s]
    if std == 0:
        return img
    return np.clip(img + rng.normal(0, std, img.shape), 0, 255).astype(np.uint8)

def n_jpeg(img, s):
    q = [100, 40, 25, 15, 9, 5][s]
    if q >= 100:
        return img
    bgr = img[..., ::-1]
    ok, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]

def n_motion(img, s):
    k = [0, 5, 9, 13, 19, 27][s]
    if k <= 1:
        return img
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker)

def n_defocus(img, s):
    sig = [0, 1, 2, 3.5, 5, 7][s]
    if sig == 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sig)

def n_lowlight(img, s):
    f = [1.0, 0.7, 0.5, 0.35, 0.22, 0.13][s]
    if f >= 1.0:
        return img
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

def n_steam(img, s):
    if s == 0:
        return img
    nb, alpha = [(0, 0), (2, 0.20), (3, 0.30), (4, 0.42), (5, 0.55), (6, 0.68)][s]
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.float32)
    for _ in range(nb):
        cx, cy = rng.randint(0, W), rng.randint(int(H * 0.3), H)
        r = rng.randint(int(min(H, W) * 0.15), int(min(H, W) * 0.4))
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(H, W) * 0.06) * alpha
    white = np.full_like(img, 235)
    return np.clip(img * (1 - mask[..., None]) + white * mask[..., None], 0, 255).astype(np.uint8)

def n_contrast(img, s):                                    # 대비 낮추기 (CCTV 저대비 모사)
    c = [1.0, 0.75, 0.58, 0.42, 0.30, 0.20][s]
    if c >= 1.0:
        return img
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)

def n_grayscale(img, s):                                   # 회색으로 섞기 (학습 안 됨→저하 볼 것)
    f = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0][s]
    if f == 0:
        return img
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
    return np.clip(img * (1 - f) + g * f, 0, 255).astype(np.uint8)

def n_erasing(img, s):                                     # 무작위 사각형 가림 (occlusion)
    if s == 0:
        return img
    nb, frac = [(0, 0), (1, 0.10), (2, 0.15), (3, 0.20), (4, 0.28), (5, 0.35)][s]
    H, W = img.shape[:2]
    out = img.copy()
    for _ in range(nb):
        bw, bh = int(W * frac), int(H * frac)
        x, y = rng.randint(0, max(1, W - bw)), rng.randint(0, max(1, H - bh))
        out[y:y + bh, x:x + bw] = 128                      # 회색으로 지움
    return out

NOISE = {'gaussian': n_gaussian, 'jpeg': n_jpeg, 'motion_blur': n_motion,
         'defocus': n_defocus, 'low_light': n_lowlight, 'steam': n_steam,
         'contrast': n_contrast, 'grayscale': n_grayscale, 'random_erasing': n_erasing}

# ---------------------------------------------------------------------------
# test 이미지 미리 적재 (RAM) — 반복 Drive 읽기 방지
# ---------------------------------------------------------------------------
def load_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

pos, neg = [], []
for p in sorted(glob.glob(f'{SYN}/images/*.jpg')):
    lab = f'{SYN}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
    (pos if os.path.getsize(lab) > 0 else neg).append(p)
print(f'test 양성 {len(pos)}장 · 음성 {len(neg)}장 적재 중...')
POS = [load_rgb(p) for p in pos]
NEG = [load_rgb(p) for p in neg]


def detected(rgb):
    r = model.predict(rgb[..., ::-1], conf=CONF, verbose=False)[0]   # RGB→BGR
    return len(r.boxes) > 0

# ---------------------------------------------------------------------------
# 곡선 측정
# ---------------------------------------------------------------------------
res = {}
print('\n' + '=' * 74)
print('노이즈 저하 곡선 — flame_rate(양성 검출) · fp_rate(음성 오탐)')
print('=' * 74)
for name, fn in NOISE.items():
    fr_row, fp_row = [], []
    for s in SEVS:
        fr = np.mean([detected(fn(im, s)) for im in POS])
        fp = np.mean([detected(fn(im, s)) for im in NEG])
        fr_row.append(round(float(fr), 3)); fp_row.append(round(float(fp), 3))
    res[name] = {'flame_rate': fr_row, 'fp_rate': fp_row}
    print(f'\n[{name}]')
    print('  강도      ' + ''.join(f'{s:>7}' for s in SEVS))
    print('  flame_rate' + ''.join(f'{v:>7.2f}' for v in fr_row))
    print('  fp_rate   ' + ''.join(f'{v:>7.2f}' for v in fp_row))

# ---------------------------------------------------------------------------
# 저장 + 그래프
# ---------------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
json.dump({'sevs': SEVS, 'conf': CONF, 'results': res,
           'n_pos': len(pos), 'n_neg': len(neg)},
          open(f'{OUT}/noise_curve.json', 'w'), ensure_ascii=False, indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name in NOISE:
    ax[0].plot(SEVS, res[name]['flame_rate'], marker='o', label=name)
    ax[1].plot(SEVS, res[name]['fp_rate'], marker='o', label=name)
ax[0].set_title('flame_rate vs noise severity (down = worse)')
ax[0].set_xlabel('severity'); ax[0].set_ylabel('flame detect rate'); ax[0].set_ylim(0, 1)
ax[1].set_title('fp_rate vs noise severity (up = worse)')
ax[1].set_xlabel('severity'); ax[1].set_ylabel('false-positive rate'); ax[1].set_ylim(0, 1)
for a in ax:
    a.grid(alpha=0.3); a.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/noise_curve.png', dpi=110)
print(f'\n-> {OUT}/noise_curve.json · noise_curve.png')
print('\n임계점 — flame_rate 가 기준(강도 0) 대비 크게 떨어지기 시작하는 강도를 본다.')
print('다음(Phase B) — 그 노이즈들을 학습 증강으로 넣어 재학습 → 곡선이 올라오면 극복.')
