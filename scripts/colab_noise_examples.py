# ===== 노이즈 예시 그리드 (문서용, Colab) =====
#
# noise_lib 9종을 test 이미지 한 장에 강도별로 적용해 라벨 그리드를 만든다 (문서 삽화용).
# 기본 샘플 = **불꽃 박스가 가장 큰 양성**(가장 뚜렷). 라벨은 영어(Colab 폰트 한계).
#
# 샘플 고르기:
#   os.environ['NOISE_PICK']='1'        → 불꽃 큰 top-16 후보 시트 + 콘솔 경로목록만 내고 종료
#   os.environ['NOISE_SAMPLE']='<경로>'  → 그 이미지로 그리드
#   (둘 다 없으면 불꽃 가장 큰 양성 자동 선택)

import os, glob, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

REPO = '/content/kitchen-fire-noise-poc/scripts'
sys.path.insert(0, REPO)
import noise_lib as NL

FIRE = '/content/drive/MyDrive/fire_frames'
SYN  = f'{FIRE}/synth/test'
OUT  = f'{FIRE}/noise_examples'
SEVS = [0, 2, 4, 5]           # 보여줄 강도 (clean · 약 · 강 · 최대)
SEED = 777
TILE = 300
PAD_L = 210
HDR = 34

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

def font(sz):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', sz)
    except Exception:
        return ImageFont.load_default()

def labpath(p):
    return f'{SYN}/labels/' + os.path.splitext(os.path.basename(p))[0] + '.txt'

def flame_area(p):                              # 불꽃 박스 넓이 합(정규화) — 클수록 뚜렷
    lab = labpath(p)
    if not (os.path.exists(lab) and os.path.getsize(lab) > 0):
        return 0.0
    a = 0.0
    for line in open(lab):
        v = line.split()
        if len(v) >= 5:
            a += float(v[3]) * float(v[4])
    return a

allimg = sorted(glob.glob(f'{SYN}/images/*.jpg'))
pos = sorted([p for p in allimg if flame_area(p) > 0], key=flame_area, reverse=True)

# --- 후보 고르기 모드: 불꽃 큰 top-16 시트 + 콘솔 경로목록 ---
if os.environ.get('NOISE_PICK'):
    top = pos[:16]
    print('불꽃 큰 순 후보 — 마음에 드는 것의 경로를 NOISE_SAMPLE 에 넣어 재실행:')
    for i, p in enumerate(top):
        print(f'  [{i}] area={flame_area(p):.3f}  {p}')
    im0 = cv2.imread(top[0]); CW = 300; ch = round(CW * im0.shape[0] / im0.shape[1])
    Fp = font(13)
    sh = Image.new('RGB', (4 * CW, 4 * (ch + 22)), (245, 245, 245)); dd = ImageDraw.Draw(sh)
    for i, p in enumerate(top):
        r, c = divmod(i, 4)
        t = Image.fromarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)).resize((CW, ch))
        sh.paste(t, (c * CW, r * (ch + 22) + 22))
        dd.text((c * CW + 4, r * (ch + 22) + 4), f'[{i}] {os.path.basename(p)[:32]}', fill=(0, 0, 0), font=Fp)
    sh.save(f'{OUT}/_candidates.jpg', quality=90)
    print(f'-> {OUT}/_candidates.jpg  (시트에서 번호 보고 위 경로로 NOISE_SAMPLE 지정)')
    raise SystemExit(0)

SAMPLE = os.environ.get('NOISE_SAMPLE') or (pos[0] if pos else allimg[0])
print('샘플:', os.path.basename(SAMPLE), f'(flame area {flame_area(SAMPLE):.3f})')

rgb0 = cv2.cvtColor(cv2.imread(SAMPLE), cv2.COLOR_BGR2RGB)
h0, w0 = rgb0.shape[:2]; th = round(TILE * h0 / w0)
F, Fs = font(18), font(13)

cols, rows = len(SEVS), len(NL.ALL9)
W = PAD_L + cols * TILE
H = HDR + rows * (th + 4)
canvas = Image.new('RGB', (W, H), (245, 245, 245)); d = ImageDraw.Draw(canvas)

for c, s in enumerate(SEVS):                       # 강도 헤더
    d.text((PAD_L + c * TILE + TILE // 2 - 26, 8), f'severity {s}', fill=(0, 0, 0), font=F)

for r, nm in enumerate(NL.ALL9):
    fn = NL.NOISE[nm]; y = HDR + r * (th + 4)
    d.text((8, y + th // 2 - 16), nm, fill=(0, 0, 0), font=F)
    if nm in NL.HELDOUT:
        d.text((8, y + th // 2 + 4), '[held-out]', fill=(180, 60, 0), font=Fs)
    rng = np.random.RandomState(SEED); tiles = {}
    for s in range(6):                             # 0..5 순서대로 rng 소비(재현성)
        out = fn(rgb0, s, rng)
        if s in SEVS:
            tiles[s] = out
    for c, s in enumerate(SEVS):
        canvas.paste(Image.fromarray(tiles[s]).resize((TILE, th)), (PAD_L + c * TILE, y))

canvas.save(f'{OUT}/noise_grid.jpg', quality=90)
print(f'-> {OUT}/noise_grid.jpg  ({W}x{H})')
