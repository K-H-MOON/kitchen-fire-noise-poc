# ===== 5.6단계: 불꽃 제거(ablation) 검증 — 모델이 불꽃을 '필요'로 하는가 (Colab) =====
#
# EigenCAM 은 클래스를 구별 못 해 판정에 부적합했다. 여기서는 직접 잰다.
# 우리는 불꽃을 합성했으므로 각 test 양성의 '원본 배경(불꽃 없는 그 프레임)' 을
# 그대로 갖고 있다. 같은 장면을 불꽃 있음/없음으로 짝지어 비교한다:
#
#   flame_rate  = 불꽃 있는 합성 이미지에서 검출되는 비율   (높아야 함 = recall)
#   bg_fp_rate  = **같은 배경(불꽃 제거)** 에서 검출되는 비율 (낮아야 함)
#   neg_fp_rate = test 음성(원래 불꽃 없이 만든) 전체에서 오탐 비율 (낮아야 함)
#
# 판정 — 모델이 불꽃을 필요로 하면 flame_rate ≫ bg_fp_rate 여야 한다.
#   bg_fp_rate 가 높으면 = 배경만으로도 불이라 함 = 배경/합성 아티팩트 지름길.
#
# GPU 권장. best.pt(5단계)가 있어야 한다.

import os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE = '/content/drive/MyDrive/fire_frames'
# 깨끗한 Phase B baseline 으로 검증 (구 runs/fire_s 는 konro LED 오염 시절 산출물).
# 다른 모델로 보려면 BEST_MODEL 만 바꾼다 (예: 'v8_modelA_s1').
BEST_MODEL = 'v8_baseline_s1'
BEST = f'{FIRE}/runs_phaseB/{BEST_MODEL}/best.pt'
SYN  = f'{FIRE}/synth/test'
BG   = f'{FIRE}/bg/test'
OUT  = f'{FIRE}/ablation'
CONF = 0.25
SEED = 1

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 가 없음 — 5단계를 먼저 돌릴 것')
model = YOLO(BEST)

# 원본 배경을 stem 으로 찾는 표. 합성 파일명은 'NNNNN_<stem>.jpg'.
bg_by_stem = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob(f'{BG}/*/*.jpg')}


def max_conf(path):
    r = model.predict(path, conf=CONF, verbose=False)[0]
    return float(r.boxes.conf.max()) if len(r.boxes) else 0.0


# test 양성(라벨 있는 것) / 음성(라벨 빈 것) 나누기
imgs = sorted(glob.glob(f'{SYN}/images/*.jpg'))
pos, neg = [], []
for p in imgs:
    lab = f'{SYN}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
    (pos if os.path.getsize(lab) > 0 else neg).append(p)

print('=' * 66)
print('불꽃 제거(ablation) 검증')
print('=' * 66)
print(f'  test 양성 {len(pos)}장 · 음성 {len(neg)}장')

# ---------------------------------------------------------------------------
# 짝 비교 — 양성(불꽃) vs 같은 배경(불꽃 제거)
# ---------------------------------------------------------------------------
paired, cf_flame, cf_bg = 0, [], []
sheets = []
for p in pos:
    name = os.path.splitext(os.path.basename(p))[0]
    stem = name[6:] if name[:5].isdigit() and name[5] == '_' else name
    bgp = bg_by_stem.get(stem)
    if bgp is None:
        continue
    paired += 1
    cf = max_conf(p); cb = max_conf(bgp)
    cf_flame.append(cf); cf_bg.append(cb)
    if len(sheets) < 8:
        sheets.append((p, bgp, cf, cb))

cf_flame, cf_bg = np.array(cf_flame), np.array(cf_bg)
flame_rate = float((cf_flame >= CONF).mean()) if len(cf_flame) else 0
bg_fp_rate = float((cf_bg >= CONF).mean()) if len(cf_bg) else 0

# ---------------------------------------------------------------------------
# test 음성 전체 오탐
# ---------------------------------------------------------------------------
neg_hits = sum(max_conf(p) >= CONF for p in neg)
neg_fp_rate = neg_hits / len(neg) if neg else 0

print(f'\n짝 비교 (양성 {paired}쌍, conf≥{CONF})')
print(f'  flame_rate (불꽃 있음→검출)   {flame_rate:.3f}   [높아야 함]')
print(f'  bg_fp_rate (불꽃 제거→검출)   {bg_fp_rate:.3f}   [낮아야 함]')
print(f'  평균 conf  불꽃 {cf_flame.mean():.3f}  vs  배경 {cf_bg.mean():.3f}')
print(f'\ntest 음성 전체 오탐  {neg_fp_rate:.3f}  ({neg_hits}/{len(neg)})   [낮아야 함]')

print('\n' + '=' * 66)
gap = flame_rate - bg_fp_rate
if flame_rate >= 0.6 and bg_fp_rate <= 0.2 and neg_fp_rate <= 0.2:
    print(f'판정: 통과 — 모델이 불꽃을 필요로 함 (flame {flame_rate:.2f} ≫ bg {bg_fp_rate:.2f}).')
    print('      노이즈 실험으로 진행 가능.')
elif bg_fp_rate >= 0.4 or neg_fp_rate >= 0.4:
    print(f'판정: **지름길 의심** — 배경만으로도 자주 검출(bg {bg_fp_rate:.2f} · 음성 {neg_fp_rate:.2f}).')
    print('      하드네거티브 강화·블렌딩 개선 후 재학습이 필요.')
else:
    print(f'판정: 애매 — flame {flame_rate:.2f} · bg {bg_fp_rate:.2f} · 음성 {neg_fp_rate:.2f}.')
    print('      시트를 보고 오탐이 어디서 나는지 확인.')

# ---------------------------------------------------------------------------
# 짝 시트 — [불꽃 이미지+검출 | 같은 배경+검출]
# ---------------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

def draw(path, lab):
    r = model.predict(path, conf=CONF, verbose=False)[0]
    im = Image.fromarray(r.plot()[..., ::-1])          # BGR→RGB, 박스 그려짐
    d = ImageDraw.Draw(im); d.text((6, 6), lab, fill=(0, 255, 0), font=F)
    return im

if sheets:
    CW = 380
    rows = []
    for p, bgp, cf, cb in sheets:
        rows.append([draw(p, f'불꽃 conf={cf:.2f}'), draw(bgp, f'배경 conf={cb:.2f}')])
    h0, w0 = np.asarray(rows[0][0]).shape[:2]; ch = round(CW * h0 / w0)
    sh = Image.new('RGB', (2 * CW, len(rows) * (ch + 26)), (16, 16, 16))
    for r_i, row in enumerate(rows):
        for c_i, im in enumerate(row):
            sh.paste(im.resize((CW, ch)), (c_i * CW, r_i * (ch + 26) + 26))
    sh.save(f'{OUT}/_ablation.jpg', quality=88)
    print(f'\n짝 시트 -> {OUT}/_ablation.jpg  (왼=불꽃 · 오=같은 배경)')
    print('  왼쪽만 박스가 뜨고 오른쪽은 비면 정상. 오른쪽에도 박스가 뜨면 지름길.')
