# ===== 5단계: YOLO 불꽃 검출 학습 (Colab, GPU) =====
#
# synth 데이터셋(4단계)으로 YOLOv8s 를 학습한다.
#   train  = train 배경(9사이트) + train 불꽃 풀
#   val    = val 배경(인화여중·로봇고) + train 불꽃 풀 → **모델 선택(best.pt)**
#   test   = test 배경(개원중 CCTV·내곡중) + test 불꽃 풀 → **최종 평가**
#
# val 이 train 과 분리돼 있어(연기 프로젝트와 달리) val mAP 로 조기 종료가 유효하다.
# 사전학습은 일반 COCO(yolov8s.pt)에서 시작 — kitchen-fire-poc 의 급식실-학습
# 가중치는 쓰지 않는다(독립 프로젝트).
#
# **GPU 필요** — 런타임 유형을 L4/A100/T4 로 바꿀 것.
#
# 학습 전 사전 등록 사항: 학습 후 Grad-CAM 으로 '모델이 불꽃을 보는지' 검증한다
# (colab_gradcam.py, 다음 단계). 경계선을 보고 있으면 노이즈 곡선이 틀린 이유로 나온다.

import os, time, shutil, subprocess, sys
from google.colab import drive

FIRE  = '/content/drive/MyDrive/fire_frames'
DATA  = f'{FIRE}/synth/data.yaml'
ROUT  = f'{FIRE}/runs'                 # 결과 저장 (Drive)
WORK  = '/content/runs'                # 작업 (런타임)
MODEL = 'yolov8s.pt'
EPOCHS, IMGSZ, BATCH, SEED = 80, 640, 16, 1
PATIENCE = 20                          # val mAP 가 이만큼 안 오르면 조기 종료
NAME  = 'fire_s'

drive.mount('/content/drive')

try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import torch
from ultralytics import YOLO

print('=' * 72)
print('불꽃 검출 학습 — YOLOv8s · COCO 사전학습에서 시작')
print('=' * 72)
print(f'  ultralytics {ultralytics.__version__} · torch {torch.__version__}')
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(f'  GPU {gpu or "**없음 — 런타임을 GPU 로 바꿀 것**"}')
if not torch.cuda.is_available():
    raise SystemExit('GPU 가 없음')
if not os.path.exists(DATA):
    raise SystemExit(f'{DATA} 가 없음 — 4단계(colab_synth.py)를 먼저 돌릴 것')
print(f'\n{open(DATA).read().strip()}')

# ---------------------------------------------------------------------------
# 학습 — val 로 best.pt 선택
# ---------------------------------------------------------------------------
shutil.rmtree(f'{WORK}/{NAME}', ignore_errors=True)
t0 = time.time()
m = YOLO(MODEL)
m.train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
        project=WORK, name=NAME, exist_ok=True, patience=PATIENCE, verbose=True)
dt = time.time() - t0
print(f'\n학습 시간 {dt / 60:.1f}분')

rd = f'{WORK}/{NAME}'
best = f'{rd}/weights/best.pt'
print(f'  best.pt {"있음" if os.path.exists(best) else "**없음**"}')

# 실제 쓰인 증강 값 남기기
args = f'{rd}/args.yaml'
if os.path.exists(args):
    print('\n실제 쓰인 증강·학습 인자')
    keep = ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'fliplr',
            'mosaic', 'mixup', 'erasing', 'close_mosaic', 'lr0', 'optimizer',
            'patience', 'epochs', 'batch', 'imgsz', 'seed')
    for line in open(args):
        if line.split(':')[0].strip() in keep:
            print('    ' + line.rstrip())

# ---------------------------------------------------------------------------
# 최종 평가 — test (미학습 사이트 + test 불꽃)
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('test 평가 — 개원중(CCTV)·내곡중 + test 불꽃 (학습에서 한 번도 안 본 조합)')
print('=' * 72)
mt = YOLO(best)
mv = mt.val(data=DATA, split='val', imgsz=IMGSZ, verbose=False)
me = mt.val(data=DATA, split='test', imgsz=IMGSZ, verbose=False)
print(f'  val  mAP@0.5 {mv.box.map50:.3f} · mAP@0.5:0.95 {mv.box.map:.3f} · '
      f'P {mv.box.mp:.3f} · R {mv.box.mr:.3f}')
print(f'  test mAP@0.5 {me.box.map50:.3f} · mAP@0.5:0.95 {me.box.map:.3f} · '
      f'P {me.box.mp:.3f} · R {me.box.mr:.3f}')
print('  **안전상 recall(R) 이 핵심 — 불을 놓치는지.**')

# ---------------------------------------------------------------------------
# Drive 로 저장
# ---------------------------------------------------------------------------
out = f'{ROUT}/{NAME}'
os.makedirs(out, exist_ok=True)
for f in ('weights/best.pt', 'weights/last.pt', 'results.csv', 'args.yaml'):
    p = f'{rd}/{f}'
    if os.path.exists(p):
        shutil.copy(p, f'{out}/{os.path.basename(f)}')
import json
json.dump({'model': MODEL, 'epochs': EPOCHS, 'imgsz': IMGSZ, 'batch': BATCH,
           'seed': SEED, 'patience': PATIENCE, 'minutes': dt / 60, 'gpu': gpu,
           'val': {'map50': float(mv.box.map50), 'map': float(mv.box.map),
                   'P': float(mv.box.mp), 'R': float(mv.box.mr)},
           'test': {'map50': float(me.box.map50), 'map': float(me.box.map),
                    'P': float(me.box.mp), 'R': float(me.box.mr)}},
          open(f'{out}/train_meta.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n-> {out} 에 best.pt · results.csv · train_meta.json 저장')
print('\n다음 — Grad-CAM 검증(모델이 불꽃을 보는가) → 이상 없으면 6단계 노이즈 저하 곡선.')
