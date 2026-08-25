# ===== 급식실 조리 CCTV false-positive 진단 · Colab =====
#
# 목적: 배포후보 2g 가 '정상 조리' 프레임의 무엇에 헛불(FP)을 내는지 예측박스 몽타주로 육안 확인.
#   → FP 정체에 따라 다음 수가 갈림:
#     · 실제 조리 불꽃(가스버너·wok flambé)에 뜬다 = 외형상 맞음 → ⑤(네거티브)면 recall 깎일 위험,
#        시간축/문맥 로직으로 풀 문제(§F 교훈: 실재 시각특징은 학습으로 못 죽임).
#     · 스팀·반사·그림자 등 비화염에 뜬다 = ⑤(급식실 조리 네거티브 추가)가 깨끗하게 통함.
#
# env: OUT_DIR(test 경로) · INSP_DIR(몽타주 출력) · CONF(문턱, 기본 0.25) · MODEL(기본 real_only_grouped)

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
IFRUN = f'{FIRE}/runs_if'
TEST  = os.environ.get('OUT_DIR', f'{FIRE}/oilfire_realtest')
INSP  = os.environ.get('INSP_DIR', '/content/inspect_fp')
CONF  = float(os.environ.get('CONF', '0.25'))
MODEL = os.environ.get('MODEL', 'real_only_grouped')
drive.mount('/content/drive')
os.makedirs(INSP, exist_ok=True)

cook = sorted(glob.glob(f'{TEST}/nofire_kitchen/*.jpg'))
assert cook, f'급식실 음성 없음: {TEST}/nofire_kitchen'
m = YOLO(f'{IFRUN}/{MODEL}/weights/best.pt')


def source(p):
    return os.path.basename(p).rsplit('_', 1)[0]


fps = []                 # (path, xyxy, conf)
persrc = {}
for i in range(0, len(cook), 64):
    batch = cook[i:i + 64]
    for p, r in zip(batch, m.predict(batch, conf=CONF, verbose=False)):
        hit = len(r.boxes) > 0
        persrc.setdefault(source(p), []).append(hit)
        if hit:
            fps.append((p, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))

print(f'급식실 음성 {len(cook)}장 · FP {len(fps)}장 (conf {CONF} · {MODEL}) · fpr {len(fps)/len(cook):.3f}')
print('소스(조리영상 ck##)별 fpr — 특정 영상에 몰리나:')
for s, v in sorted(persrc.items()):
    if sum(v):
        print(f'  {s}  {np.mean(v):.3f}  ({sum(v)}/{len(v)})')

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
except Exception:
    F = ImageFont.load_default()

if fps:
    cols = 6; cw = 300
    im0 = Image.open(fps[0][0]); ch = round(cw * im0.height / im0.width)
    rows = (len(fps) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * (ch + 16) + 4), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for j, (p, xy, cf) in enumerate(fps):
        im = Image.open(p).convert('RGB'); sx = cw / im.width; sy = ch / im.height
        im = im.resize((cw, ch)); d = ImageDraw.Draw(im)
        for b, c in zip(xy, cf):
            d.rectangle([b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy], outline=(255, 60, 60), width=2)
            d.text((b[0]*sx + 2, b[1]*sy + 2), f'{c:.2f}', fill=(255, 220, 0), font=F)
        col, row = j % cols, j // cols
        sh.paste(im, (col * cw, row * (ch + 16)))
        dr.text((col * cw + 2, row * (ch + 16) + ch + 1), source(p), fill=(120, 220, 255), font=F)
    out = f'{INSP}/cook_fp_{MODEL}_c{int(CONF*100)}.jpg'
    sh.save(out, quality=84)
    print(f'\n-> {out}   (빨간박스=헛불 위치 · 노랑=conf · 파랑=조리영상 id)')
    print('   판독: 박스가 가스버너/wok 실화염 위=외형상 맞음(시간축로직) · 스팀/반사/그림자/음식 위=비화염 오탐(⑤ 유효)')
else:
    print('FP 없음 — 이 conf 에선 헛불 0.')
