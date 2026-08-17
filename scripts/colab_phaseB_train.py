# ===== 6-Phase B: 학습 오케스트레이션 (Colab, GPU) =====
#
# 3개 config × 5 seed = 15회 학습을 한 아키텍처에서 돌린다. resumable — 각 모델을
# Drive 에 저장하고 이미 있으면 건너뛴다. 런타임이 끊겨도 이 셀만 다시 돌리면 이어진다.
#
#   config:
#     baseline — 노이즈 증강 없음 (clean)
#     modelA   — 9개 노이즈 전부 증강 (in-distribution 상한)
#     modelB   — 화질 6개만 증강, steam·grayscale·random_erasing 는 held-out (일반화)
#
# **공정성** — 모든 config 의 학습 이미지 수를 동일(2×)하게 맞춘다:
#     baseline = 원본 train 을 두 번(clean+clean)
#     modelA/B = 원본 train(clean) + 노이즈 구운 사본
#   그래서 config 간 차이는 '둘째 사본이 clean 이냐 noised 이냐' 뿐 → 증강 효과만 격리.
#   모든 config 가 clean 원본을 전부 유지하므로 clean 성능도 보존된다.
#
# **누수 위생** — 증강 rng(고정 12345)는 테스트 노이즈 rng 와 독립 스트림. 노이즈는
#   train 이미지에만 굽고 val/test 는 깨끗(테스트 노이즈는 평가 때 별도로 입힘).
#
# 아키텍처 미러링 — ARCH 한 줄만 바꿔 v8 → v11 로 같은 프로토콜을 반복한다.

import os, glob, shutil, subprocess, sys
import numpy as np, cv2
from google.colab import drive

REPO = '/content/kitchen-fire-noise-poc/scripts'
sys.path.insert(0, REPO)
import noise_lib as NL

# ---------------------------------------------------------------------------
# 설정 — 미러링 시 ARCH 만 바꾼다
# ---------------------------------------------------------------------------
ARCH = 'yolov8s.pt'          # ← v11 차례엔 'yolo11s.pt' 로만 변경
SEEDS = [1, 2, 3, 4, 5]
EPOCHS, IMGSZ, BATCH, PATIENCE = 80, 640, 16, 20

FIRE = '/content/drive/MyDrive/fire_frames'
SYN  = f'{FIRE}/synth'
RUNS = f'{FIRE}/runs_phaseB'          # Drive 저장 (resumable 근거)
WORK = '/content/runs_phaseB'
tag = 'v11' if '11' in ARCH else 'v8'
CONFIGS = {'baseline': None, 'modelA': NL.ALL9, 'modelB': NL.QUALITY}

drive.mount('/content/drive')
try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    import ultralytics
import torch
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise SystemExit('GPU 가 없음 — 런타임을 GPU 로')
if not os.path.exists(f'{SYN}/train/images'):
    raise SystemExit(f'{SYN}/train 이 없음 — 4단계(합성)를 먼저')
print(f'아키텍처 {ARCH} (tag={tag}) · GPU {torch.cuda.get_device_name(0)}')

# ---------------------------------------------------------------------------
# 1) 오프라인 증강셋 생성 (config별 1회, 아키텍처 무관 → v8/v11 공유)
# ---------------------------------------------------------------------------
def gen_aug(cfg, noises):
    d = f'{SYN}/train_aug_{cfg}'
    src = sorted(glob.glob(f'{SYN}/train/images/*.jpg'))
    have = glob.glob(f'{d}/images/*.jpg')
    if len(have) == len(src) and len(src) > 0:
        print(f'  [{cfg}] 증강셋 이미 있음 ({len(have)}장) — 건너뜀'); return
    os.makedirs(f'{d}/images', exist_ok=True); os.makedirs(f'{d}/labels', exist_ok=True)
    rng = np.random.RandomState(12345)          # 증강 rng — 테스트 노이즈와 독립
    for p in src:
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        out = NL.apply_random(rgb, noises, rng)
        stem = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(f'{d}/images/{stem}.jpg', out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy(f'{SYN}/train/labels/{stem}.txt', f'{d}/labels/{stem}.txt')
    print(f'  [{cfg}] 증강셋 생성 {len(src)}장 -> {d}')

print('\n증강셋 준비')
for cfg, noises in [('modelA', NL.ALL9), ('modelB', NL.QUALITY)]:
    gen_aug(cfg, noises)

# ---------------------------------------------------------------------------
# 2) config별 data.yaml — train 을 두 경로 리스트로(2× 로 수 맞춤)
# ---------------------------------------------------------------------------
os.makedirs(WORK, exist_ok=True)
def data_yaml(cfg):
    second = 'train/images' if cfg == 'baseline' else f'train_aug_{cfg}/images'
    path = f'{WORK}/data_{cfg}.yaml'
    open(path, 'w').write(
        f"path: {SYN}\ntrain: [train/images, {second}]\nval: val/images\n"
        f"test: test/images\nnc: 1\nnames: ['fire']\n")
    return path

# ---------------------------------------------------------------------------
# 3) 학습 루프 — resumable (Drive 에 best.pt 있으면 건너뜀)
# ---------------------------------------------------------------------------
done, trained = [], []
for cfg in CONFIGS:
    dy = data_yaml(cfg)
    for seed in SEEDS:
        name = f'{tag}_{cfg}_s{seed}'
        drive_best = f'{RUNS}/{name}/best.pt'
        if os.path.exists(drive_best):
            done.append(name); continue
        print(f'\n[train] {name}  ({ARCH} · epochs {EPOCHS} · seed {seed})')
        shutil.rmtree(f'{WORK}/{name}', ignore_errors=True)
        m = YOLO(ARCH)
        m.train(data=dy, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=seed,
                project=WORK, name=name, exist_ok=True, patience=PATIENCE, verbose=False)
        rd = f'{WORK}/{name}'
        os.makedirs(f'{RUNS}/{name}', exist_ok=True)
        for f in ('weights/best.pt', 'results.csv', 'args.yaml'):
            pth = f'{rd}/{f}'
            if os.path.exists(pth):
                shutil.copy(pth, f'{RUNS}/{name}/{os.path.basename(f)}')
        trained.append(name)
        print(f'  -> {RUNS}/{name}/best.pt 저장')

print('\n' + '=' * 60)
print(f'{tag} — 이미 있던 {len(done)}개 · 이번에 학습 {len(trained)}개 (총 {len(done)+len(trained)}/15)')
if len(done) + len(trained) < len(CONFIGS) * len(SEEDS):
    print('아직 남음 — 런타임 유지되면 계속, 끊기면 이 셀만 다시 돌리면 이어짐.')
else:
    print(f'{tag} 15회 완료. 다음 — ARCH 를 다른 아키텍처로 바꿔 반복하거나,')
    print('colab_phaseB_eval.py 로 노이즈 곡선+CI 평가.')
