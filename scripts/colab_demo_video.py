# ===== 발표 데모 영상 — 검출 박스 + 시간축(지속성) 경보 오버레이 → MP4 · Colab =====
#
# 목적: 영상 구간을 재생하며 프레임마다 2ck 검출(박스)을 얹고, 그 위에
#   시간축 지속성 경보(최근 PERSIST_SEC 중 PERSIST_FRAC 이상 검출 시 ALARM)를 표시해
#   "이 모델이 이렇게 작동한다"를 한눈에 보여주는 MP4 를 만든다.
#   → 단일 프레임이 아니라 '구간 재생 + 지속성' = 우리가 검증한 배포 로직(헛불 억제).
#
# env(필수): DEMO_VIDEO(영상 전체경로) · START·END(초)
# env(선택): MODEL(기본 real_only_grouped_ck) · CONF(0.25) · FPS(12) ·
#   PERSIST_SEC(1.0)·PERSIST_FRAC(0.5) — 지속성 경보 창/문턱 · DEMO_LABEL(우상단 캡션, 예 "실화재")
#   OUT_MP4(기본 /content/demo_out.mp4)

import os, glob, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

VIDEO = os.environ['DEMO_VIDEO']
START = float(os.environ.get('START', '0'))
END   = float(os.environ.get('END', '10'))
FPS   = float(os.environ.get('FPS', '12'))
CONF  = float(os.environ.get('CONF', '0.25'))
MODEL = os.environ.get('MODEL', 'real_only_grouped_ck')
PERSIST_SEC  = float(os.environ.get('PERSIST_SEC', '1.0'))
PERSIST_FRAC = float(os.environ.get('PERSIST_FRAC', '0.5'))
LABEL = os.environ.get('DEMO_LABEL', '')
OUT   = os.environ.get('OUT_MP4', '/content/demo_out.mp4')
WORK  = '/content/demo_work'
drive.mount('/content/drive')
IFRUN = '/content/drive/MyDrive/fire_frames/runs_if'
assert os.path.exists(VIDEO), f'영상 없음: {VIDEO}'

# --- 구간 프레임 추출(FPS 로 균등) ---
os.makedirs(WORK, exist_ok=True)
for f in glob.glob(f'{WORK}/*.jpg'):
    os.remove(f)
subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{START:.2f}', '-i', VIDEO, '-t', f'{max(0.0,END-START):.2f}',
                '-vf', f'fps={FPS:.4f}', '-q:v', '2', f'{WORK}/%04d.jpg'], check=False)
frames = sorted(glob.glob(f'{WORK}/*.jpg'))
assert frames, '프레임 추출 실패 — START/END·경로 확인'
print(f'구간 {START:.1f}~{END:.1f}s · {len(frames)}프레임 @ {FPS}fps · 모델 {MODEL}')

# --- 프레임별 추론 ---
m = YOLO(f'{IFRUN}/{MODEL}/weights/best.pt')
dets = []; flags = []
for i in range(0, len(frames), 64):
    for r in m.predict(frames[i:i + 64], conf=CONF, verbose=False):
        b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        c = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
        dets.append((b, c)); flags.append(int(len(b) > 0))
flags = np.array(flags)

# --- 시간축 지속성 경보 (최근 W프레임 중 PERSIST_FRAC 이상 검출 시 ON) ---
W = max(1, int(round(PERSIST_SEC * FPS)))
alarm = np.zeros(len(frames), dtype=int)
for i in range(len(frames)):
    if flags[max(0, i - W + 1):i + 1].mean() >= PERSIST_FRAC:
        alarm[i] = 1

# --- 오버레이 렌더 ---
def font(sz):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', sz)
    except Exception:
        return ImageFont.load_default()
Fbig, Fsm = font(34), font(22)
od = f'{WORK}/out'; os.makedirs(od, exist_ok=True)
for f in glob.glob(f'{od}/*.jpg'):
    os.remove(f)
for i, fp in enumerate(frames):
    im = Image.open(fp).convert('RGB'); d = ImageDraw.Draw(im); Wd, Hd = im.size
    b, c = dets[i]
    for box, cf in zip(b, c):
        d.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])], outline=(255, 40, 40), width=4)
        d.text((float(box[0]) + 3, float(box[1]) + 3), f'fire {cf:.2f}', fill=(255, 220, 0), font=Fsm)
    on = alarm[i] == 1
    d.rectangle([0, 0, Wd, 56], fill=(200, 0, 0) if on else (35, 35, 35))
    d.ellipse([14, 16, 40, 42], fill=(255, 255, 255) if on else (120, 120, 120))
    d.text((50, 12), 'FIRE ALARM' if on else 'MONITORING', fill=(255, 255, 255), font=Fbig)
    d.text((Wd - 150, 20), f'{START + i / FPS:5.1f}s', fill=(230, 230, 230), font=Fsm)
    if LABEL:
        d.text((Wd - 150, 64), LABEL, fill=(120, 220, 255), font=Fsm)
    im.save(f'{od}/{i:04d}.jpg', quality=92)

# --- MP4 인코딩 ---
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', f'{FPS:.4f}', '-i', f'{od}/%04d.jpg',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', OUT], check=False)
first = int(np.argmax(alarm)) if alarm.any() else -1
print(f'-> {OUT}')
print(f'   경보 ON {int(alarm.sum())}/{len(frames)}프레임' +
      (f' · 첫 경보 {START + first / FPS:.1f}s(구간 시작 +{first / FPS:.1f}s)' if first >= 0 else ' · 경보 없음'))

# --- 몽타주 스트립(채팅 첨부·육안 확인용 이미지) : 오버레이 프레임 균등 10장 ---
ovf = sorted(glob.glob(f'{od}/*.jpg'))
NP = min(10, len(ovf))
pick = [ovf[int(round(k * (len(ovf) - 1) / max(1, NP - 1)))] for k in range(NP)]
cols = 5; cw = 300
im0 = Image.open(pick[0]); ch = round(cw * im0.height / im0.width)
rows = (len(pick) + cols - 1) // cols
sh = Image.new('RGB', (cols * cw, rows * ch), (16, 16, 16))
for j, p in enumerate(pick):
    sh.paste(Image.open(p).convert('RGB').resize((cw, ch)), ((j % cols) * cw, (j // cols) * ch))
strip = OUT.rsplit('.', 1)[0] + '_strip.jpg'
sh.save(strip, quality=88)
print(f'   몽타주(첨부용) -> {strip}')
print('   빨간박스=검출·상단바 빨강=ALARM/회색=MONITORING.')
