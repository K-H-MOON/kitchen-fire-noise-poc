# ===== 5.5단계: Grad-CAM(EigenCAM) 검증 — 모델이 '불꽃'을 보는가 (Colab, GPU) =====
#
# 사전 등록 관문: 노이즈 실험 전에 **모델이 불꽃을 보고 판단하는지** 확인한다.
# kitchen-fire-poc 는 불꽃이 아니라 밝은 블롭(유니폼)을 봤다. 우리 설계는 합성
# 아티팩트(경계선)를 볼 위험이 있다 — 그러면 노이즈 저하 곡선이 '틀린 이유'로 나온다.
#
# 방법 — EigenCAM: 백본 깊은 층(SPPF)의 활성을 PCA 로 요약해 히트맵을 만든다.
# 그라디언트가 필요 없어 검출 모델에 안정적으로 붙는다.
#
# 두 가지로 판정:
#   (1) 히트맵 시트 — 열이 불꽃(정답 박스) 위에 있는가, 아니면 유니폼·솥·경계에 있는가
#   (2) 박스 내부/외부 에너지비 — 박스 안 평균 CAM / 박스 밖 평균 CAM.
#       1보다 충분히 크면 불꽃에 주목하는 것. test 양성 여러 장의 평균으로 본다.
#
# GPU 권장. best.pt(5단계)가 있어야 한다.

import os, glob, random
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE = '/content/drive/MyDrive/fire_frames'
BEST = f'{FIRE}/runs/fire_s/best.pt'
TEST_IMG = f'{FIRE}/synth/test/images'
TEST_LAB = f'{FIRE}/synth/test/labels'
OUT = f'{FIRE}/gradcam'
N = 20                     # 볼 표본 수 (test 양성 중)
SZ = 640
SEED = 1

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 가 없음 — 5단계(colab_train.py)를 먼저 돌릴 것')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

ym = YOLO(BEST)
net = ym.model.to(dev).eval()

# 대상 층 — SPPF(백본 마지막). 없으면 Detect 직전 층.
target = None
for mod in net.model:
    if mod.__class__.__name__ == 'SPPF':
        target = mod
if target is None:
    target = net.model[-2]
print(f'CAM 대상 층 — {target.__class__.__name__}')

acts = {}
target.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o.detach()))


def eigencam(a):                       # a: (C,H,W) tensor
    c, h, w = a.shape
    feat = a.reshape(c, h * w).T.float()
    feat = feat - feat.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(feat, full_matrices=False)
    cam = (feat @ vt[0]).reshape(h, w).abs()          # PC 부호 모호성 → abs
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam.cpu().numpy(), (SZ, SZ))


def gt_boxes(lab_path):                # YOLO 라벨 → 640 픽셀 박스들
    out = []
    if os.path.exists(lab_path):
        for line in open(lab_path):
            p = line.split()
            if len(p) == 5:
                _, cx, cy, bw, bh = map(float, p)
                out.append([(cx - bw / 2) * SZ, (cy - bh / 2) * SZ,
                            (cx + bw / 2) * SZ, (cy + bh / 2) * SZ])
    return out


# test 양성(라벨 있는 것)만
pos = [p for p in sorted(glob.glob(f'{TEST_IMG}/*.jpg'))
       if os.path.getsize(f'{TEST_LAB}/{os.path.splitext(os.path.basename(p))[0]}.txt') > 0]
random.Random(SEED).shuffle(pos)
pos = pos[:N]
print(f'test 양성 {len(pos)}장으로 검증')

os.makedirs(OUT, exist_ok=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

tiles, ratios = [], []
for p in pos:
    bgr = cv2.resize(cv2.imread(p), (SZ, SZ))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad():
        net(x)
    cam = eigencam(acts['a'][0])

    boxes = gt_boxes(f'{TEST_LAB}/{os.path.splitext(os.path.basename(p))[0]}.txt')
    mask = np.zeros((SZ, SZ), bool)
    for x0, y0, x1, y1 in boxes:
        mask[int(y0):int(y1), int(x0):int(x1)] = True
    if mask.any() and (~mask).any():
        ratios.append(float(cam[mask].mean() / (cam[~mask].mean() + 1e-8)))

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0)
    over = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(over); d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(b, outline=(0, 255, 0), width=3)
    tiles.append(im)

# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------
if ratios:
    arr = np.array(ratios)
    print('\n' + '=' * 60)
    print('박스 내부/외부 CAM 에너지비 (1보다 크면 불꽃에 주목)')
    print('=' * 60)
    print(f'  평균 {arr.mean():.2f} · 중앙값 {np.median(arr):.2f} · '
          f'>1 비율 {100 * (arr > 1).mean():.0f}%  (n={len(arr)})')
    if arr.mean() >= 1.3:
        print('  -> 불꽃에 주목하는 것으로 보임. 노이즈 실험으로 진행 가능.')
    elif arr.mean() >= 1.0:
        print('  -> 약함. 시트를 눈으로 확인 — 경계선·솥에 새는지 볼 것.')
    else:
        print('  -> **불꽃 밖을 봄.** 합성 아티팩트/배경 지름길 의심 — 시트 확인 필수.')

# 시트
if tiles:
    cols = 4; rows = (len(tiles) + cols - 1) // cols
    tw = 320; th = round(tw * SZ / SZ)
    sheet = Image.new('RGB', (cols * tw, rows * (th + 24)), (16, 16, 16))
    dr = ImageDraw.Draw(sheet)
    for j, im in enumerate(tiles):
        r, c = divmod(j, cols); y = r * (th + 24)
        dr.text((c * tw + 6, y + 3), f'r={ratios[j]:.2f}' if j < len(ratios) else '',
                fill=(0, 255, 0), font=F)
        sheet.paste(im.resize((tw, th)), (c * tw, y + 24))
    sheet.save(f'{OUT}/_cam.jpg', quality=88)
    print(f'\n히트맵 시트 -> {OUT}/_cam.jpg')
    print('  녹색 박스(불꽃) 위에 붉은 열이 몰리면 좋음. 유니폼·솥·경계선에 몰리면 지름길.')
