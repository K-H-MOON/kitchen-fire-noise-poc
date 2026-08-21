# ===== v2 Phase 1 가드: 불꽃 제거(ablation) — 모델이 불꽃을 '필요'로 하는가 (Colab) =====
#
# pre-reg v2 §6 가드 — C3(발광 합성)이 배경/합성 아티팩트 지름길을 만들지 않았나 확인.
# v1 방식 그대로: 각 test 양성(불꽃 합성)을 **그 원본 배경(불꽃 없는 프레임)** 과 짝지어,
# 불꽃 있음/없음의 검출을 비교한다.
#
#   flame_rate  = 불꽃 있는 합성 이미지에서 검출 비율   (높아야 = recall)
#   bg_fp_rate  = **같은 배경(불꽃 제거)** 에서 검출 비율 (낮아야)
#   neg_fp_rate = test 음성 전체 오탐 (낮아야)
#   판정: flame_rate ≫ bg_fp_rate 여야 "불꽃을 봄"(배경 지름길 아님).
#
# 모델·데이터셋 자동 연결 — ABLATION_MODEL 로 지정(기본 v8_C3_s1).
#   MODEL 이름의 조건(C0/C3)에서 그 조건의 test 셋(synth_<cond>/test)을 고른다.
#   그래야 모델과 test 이미지가 같은 합성 방식으로 맞물림.
# 비교(선택): ABLATION_MODEL=v8_C0_s1 로 다시 돌려 C0 와 나란히 볼 수 있음.
# 선행: colab_v2_train.py 로 해당 best.pt 학습됨. GPU 권장.

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

FIRE  = '/content/drive/MyDrive/fire_frames'
MODEL = os.environ.get('ABLATION_MODEL', 'v8_C3_s1')
COND  = MODEL.split('_')[1] if '_' in MODEL else 'C3'      # v8_C3_s1 → C3
BEST  = f'{FIRE}/runs_phaseB/{MODEL}/best.pt'
SYN   = f'{FIRE}/synth_{COND}/test'                        # 모델 조건에 맞는 test 셋
BG    = f'{FIRE}/bg/test'                                  # 원본 배경(불꽃 제거) — v2 공통
OUT   = f'{FIRE}/ablation_{MODEL}'
CONF  = 0.25

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 가 없음 — colab_v2_train.py 로 {MODEL} 먼저 학습')
if not os.path.isdir(SYN):
    raise SystemExit(f'{SYN} 가 없음 — synth_{COND} 생성 확인')
model = YOLO(BEST)
print(f'ablation — 모델 {MODEL} · 조건 {COND} · test {SYN}')

bg_by_stem = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob(f'{BG}/*/*.jpg')}


def max_conf(path):
    r = model.predict(path, conf=CONF, verbose=False)[0]
    return float(r.boxes.conf.max()) if len(r.boxes) else 0.0


imgs = sorted(glob.glob(f'{SYN}/images/*.jpg'))
pos, neg = [], []
for p in imgs:
    lab = f'{SYN}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
    (pos if os.path.exists(lab) and os.path.getsize(lab) > 0 else neg).append(p)

print('=' * 66)
print(f'불꽃 제거(ablation) 검증 — {MODEL}')
print('=' * 66)
print(f'  test 양성 {len(pos)}장 · 음성 {len(neg)}장')

paired, cf_flame, cf_bg = 0, [], []
sheets = []
for p in pos:
    name = os.path.splitext(os.path.basename(p))[0]
    stem = name[6:] if name[:5].isdigit() and name[5:6] == '_' else name
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
neg_hits = sum(max_conf(p) >= CONF for p in neg)
neg_fp_rate = neg_hits / len(neg) if neg else 0

print(f'\n짝 비교 (양성 {paired}쌍, conf≥{CONF})')
print(f'  flame_rate (불꽃 있음→검출)   {flame_rate:.3f}   [높아야]')
print(f'  bg_fp_rate (불꽃 제거→검출)   {bg_fp_rate:.3f}   [낮아야]')
if len(cf_flame):
    print(f'  평균 conf  불꽃 {cf_flame.mean():.3f}  vs  배경 {cf_bg.mean():.3f}')
print(f'\ntest 음성 전체 오탐  {neg_fp_rate:.3f}  ({neg_hits}/{len(neg)})   [낮아야]')

print('\n' + '=' * 66)
if flame_rate >= 0.6 and bg_fp_rate <= 0.2 and neg_fp_rate <= 0.2:
    print(f'판정: 통과 — {COND} 모델이 불꽃을 필요로 함 (flame {flame_rate:.2f} ≫ bg {bg_fp_rate:.2f}).')
    print('      배경/합성 아티팩트 지름길 아님 → 가드 유지.')
elif bg_fp_rate >= 0.4 or neg_fp_rate >= 0.4:
    print(f'판정: **지름길 의심** — 배경만으로도 자주 검출(bg {bg_fp_rate:.2f} · 음성 {neg_fp_rate:.2f}).')
    print(f'      {COND} 합성이 아티팩트를 남겼는지 시트로 확인 필요.')
else:
    print(f'판정: 애매 — flame {flame_rate:.2f} · bg {bg_fp_rate:.2f} · 음성 {neg_fp_rate:.2f}. 시트 확인.')

os.makedirs(OUT, exist_ok=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

def draw(path, lab):
    r = model.predict(path, conf=CONF, verbose=False)[0]
    im = Image.fromarray(r.plot()[..., ::-1])
    ImageDraw.Draw(im).text((6, 6), lab, fill=(0, 255, 0), font=F)
    return im

if sheets:
    CW = 380
    rows = [[draw(p, f'불꽃 conf={cf:.2f}'), draw(bgp, f'배경 conf={cb:.2f}')]
            for p, bgp, cf, cb in sheets]
    h0, w0 = np.asarray(rows[0][0]).shape[:2]; ch = round(CW * h0 / w0)
    sh = Image.new('RGB', (2 * CW, len(rows) * (ch + 26)), (16, 16, 16))
    for r_i, row in enumerate(rows):
        for c_i, im in enumerate(row):
            sh.paste(im.resize((CW, ch)), (c_i * CW, r_i * (ch + 26) + 26))
    sh.save(f'{OUT}/_ablation.jpg', quality=88)
    print(f'\n짝 시트 -> {OUT}/_ablation.jpg  (왼=불꽃 · 오=같은 배경)')
    print('  왼쪽만 박스가 뜨고 오른쪽은 비면 정상. 오른쪽에도 박스면 지름길.')
