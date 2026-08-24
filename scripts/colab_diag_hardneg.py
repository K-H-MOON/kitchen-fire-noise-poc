# ===== 헛불 진단: 하드네거 장면에 모델 예측박스 얹은 몽타주 (Colab) =====
#
# 목적: §F 에서 주황조명(13476222)이 모든 실모델에서 헛불(fpr~0.875)인데 '왜' 뜨는지 육안 확인.
#   → 모델이 무엇을 불로 보는지(예측박스+conf)를 프레임에 그려 몽타주로 저장.
#   판단: (a) 단지 주황 조명(넓은 배경/조명 영역에 박스) → 주황 하드네거 다수 수집해 재학습으로 고침
#         (b) 진짜 불꽃 유사(작은 화염형 박스·화면 속 불) → 데이터로 안 고쳐짐, 애매 케이스로 별도
#
# ⚠ Drive 커넥터는 이미지 픽셀을 못 봄 → 저장된 몽타주(inspect/diag_*.jpg)를 사용자가 채팅에 첨부해야 함.
#
# 환경:
#   SCENE   대상 장면 토큰(파일명 substring, 기본 '13476222')
#   MODEL   모델 이름(runs_if/<MODEL>/weights/best.pt, 기본 'real_only_grouped_hn')
#   SRC     프레임 소스 폴더(기본: oilfire_hardneg_test/nofire 있으면 그것, 없으면 oilfire_hardneg/nofire)
#   CONF    검출 임계(기본 0.25 — 평가와 동일)

import os, glob, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE  = '/content/drive/MyDrive/fire_frames'
IFRUN = f'{FIRE}/runs_if'
INSP  = f'{FIRE}/inspect'
SCENE = os.environ.get('SCENE', '13476222')
MODEL = os.environ.get('MODEL', 'real_only_grouped_hn')
CONF  = float(os.environ.get('CONF', '0.25'))

# 소스 폴더 선택
if 'SRC' in os.environ:
    SRC = os.environ['SRC']
elif glob.glob(f'{FIRE}/oilfire_hardneg_test/nofire/*.jpg'):
    SRC = f'{FIRE}/oilfire_hardneg_test/nofire'
else:
    SRC = f'{FIRE}/oilfire_hardneg/nofire'

os.makedirs(INSP, exist_ok=True)
best = f'{IFRUN}/{MODEL}/weights/best.pt'
assert os.path.exists(best), f'모델 없음: {best}'
imgs = sorted(p for p in glob.glob(f'{SRC}/*.jpg') if SCENE in os.path.basename(p))
assert imgs, f'장면 프레임 없음: SCENE="{SCENE}" in {SRC}'
print(f'모델 {MODEL} · 장면 "{SCENE}" · {len(imgs)}프레임 · conf {CONF}\n소스 {SRC}')

m = YOLO(best)

# --- 추론 + 프레임별 텍스트 리포트(박스 수·conf·정규화 면적·중심) ---
# 픽셀을 못 보는 상황 대비: 박스가 큰 배경/조명(넓은 면적)인지 작은 화염형(작은 면적)인지 수치로도 판단.
rendered = []
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
except Exception:
    F = ImageFont.load_default()

print('\n프레임별 검출:')
for p in imgs:
    r = m.predict(p, conf=CONF, verbose=False)[0]
    im = Image.open(p).convert('RGB')
    W, H = im.size
    dr = ImageDraw.Draw(im)
    boxes = r.boxes
    n = len(boxes)
    info = []
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        c = float(b.conf[0])
        area = ((x2 - x1) * (y2 - y1)) / (W * H)            # 화면 대비 면적
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H       # 정규화 중심
        info.append((c, area, cx, cy))
        dr.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
        dr.text((x1 + 2, max(0, y1 - 18)), f'{c:.2f} a{area:.2f}', fill=(255, 200, 0), font=F)
    tag = 'HEO' if n else 'ok '   # HEO=헛불(검출됨), ok=검출 없음
    if info:
        s = ' '.join(f'[c{c:.2f} 면적{a:.2f} @({cx:.2f},{cy:.2f})]' for c, a, cx, cy in info)
    else:
        s = '(검출 없음)'
    print(f'  {tag} {os.path.basename(p):<34} 박스{n} {s}')
    rendered.append(im)

# --- 몽타주 ---
cols = 4
cw = 320
ch = round(cw * rendered[0].height / rendered[0].width)
rows = (len(rendered) + cols - 1) // cols
sheet = Image.new('RGB', (cols * cw, rows * (ch + 20)), (16, 16, 16))
sd = ImageDraw.Draw(sheet)
for j, im in enumerate(rendered):
    x = im.resize((cw, ch))
    c, rr = j % cols, j // cols
    y = rr * (ch + 20)
    sd.text((c * cw + 3, y + 2), os.path.basename(imgs[j])[:34], fill=(120, 200, 255), font=F)
    sheet.paste(x, (c * cw, y + 20))
out = f'{INSP}/diag_{SCENE}_{MODEL}.jpg'
sheet.save(out, quality=85)
print(f'\n-> 몽타주 {out}')
print('  빨간 박스=모델이 불로 본 곳 · a=화면대비 면적(크면 배경/조명, 작으면 화염형).')
print('  ⚠ 이 몽타주를 채팅에 첨부해야 육안 판단 가능(Drive 커넥터는 픽셀 못 봄).')
print('  판단: 박스가 넓은 주황 배경/조명이면 → 주황 하드네거 수집으로 고침 / 작은 화염형이면 → 애매 케이스.')
