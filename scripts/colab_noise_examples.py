# ===== 노이즈 예시 그리드 (문서용, Colab) =====
#
# noise_lib 의 9종을 test 이미지 한 장에 강도별로 적용해, 라벨 붙인 그리드 이미지를 만든다.
# 독자가 "어떤 노이즈가 어떻게 보이는지"를 한눈에 보게 하는 용도(README/TIMELINE 삽화).
#
# 라벨은 영어(Colab 기본 폰트가 한글/일본어를 못 그림). 노이즈명은 그대로 코드명.
# 샘플 바꾸려면 os.environ['NOISE_SAMPLE'] = '<이미지 절대경로>' 후 실행.

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
TILE = 300                    # 타일 가로 px
PAD_L = 210                   # 왼쪽 노이즈 이름 칸
HDR = 34                      # 위 강도 헤더

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

# 양성(불꽃 있는) test 이미지 하나 고름 (기본: 중간 순번). NOISE_SAMPLE 로 지정 가능.
SAMPLE = os.environ.get('NOISE_SAMPLE')
if not SAMPLE:
    def has_flame(p):
        lab = f'{SYN}/labels/' + os.path.splitext(os.path.basename(p))[0] + '.txt'
        return os.path.exists(lab) and os.path.getsize(lab) > 0
    allimg = sorted(glob.glob(f'{SYN}/images/*.jpg'))
    pos = [p for p in allimg if has_flame(p)]
    SAMPLE = pos[len(pos) // 2] if pos else allimg[0]
print('샘플:', os.path.basename(SAMPLE))

rgb0 = cv2.cvtColor(cv2.imread(SAMPLE), cv2.COLOR_BGR2RGB)
h0, w0 = rgb0.shape[:2]; th = round(TILE * h0 / w0)

def font(sz):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', sz)
    except Exception:
        return ImageFont.load_default()
F, Fs = font(18), font(13)

cols, rows = len(SEVS), len(NL.ALL9)
W = PAD_L + cols * TILE
H = HDR + rows * (th + 4)
canvas = Image.new('RGB', (W, H), (245, 245, 245))
d = ImageDraw.Draw(canvas)

for c, s in enumerate(SEVS):                       # 강도 헤더
    d.text((PAD_L + c * TILE + TILE // 2 - 26, 8), f'severity {s}', fill=(0, 0, 0), font=F)

for r, nm in enumerate(NL.ALL9):
    fn = NL.NOISE[nm]
    y = HDR + r * (th + 4)
    d.text((8, y + th // 2 - 16), nm, fill=(0, 0, 0), font=F)
    if nm in NL.HELDOUT:
        d.text((8, y + th // 2 + 4), '[held-out]', fill=(180, 60, 0), font=Fs)
    rng = np.random.RandomState(SEED)              # 행마다 동일 실현
    tiles = {}
    for s in range(6):                             # 0..5 순서대로 rng 소비(재현성)
        out = fn(rgb0, s, rng)
        if s in SEVS:
            tiles[s] = out
    for c, s in enumerate(SEVS):
        canvas.paste(Image.fromarray(tiles[s]).resize((TILE, th)), (PAD_L + c * TILE, y))

canvas.save(f'{OUT}/noise_grid.jpg', quality=90)
print(f'-> {OUT}/noise_grid.jpg  ({W}x{H})')
print('강도 6종 화질계는 위 6줄, held-out 3종(steam·grayscale·random_erasing)은 아래.')
