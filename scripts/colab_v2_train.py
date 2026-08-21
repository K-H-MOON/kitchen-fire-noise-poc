# ===== v2 Phase 1: C0 vs C3 학습 (Colab, GPU) =====
#
# 단일변수 A/B — 합성 방식(C0 컷아웃 vs C3 발광)만 다르고 나머지는 v1 baseline 과
# 동일. 각 조건 5 seed = 10 모델. resumable(Drive 에 best.pt 있으면 건너뜀).
#
# v1 하네스 재사용:
#   - baseline 레시피 그대로 — 노이즈 증강 없음(노이즈는 v2 변수 아님).
#   - train 을 두 경로 리스트로(2×) — v1 baseline 과 같은 스텝 수. 두 조건 모두 동일 →
#     차이는 오직 데이터셋(synth_C0 vs synth_C3).
#   - EPOCHS·IMGSZ·BATCH·PATIENCE·SEEDS v1 과 동일.
#
# 선행 — colab_synth.py 로 synth_C0 · synth_C3 가 생성돼 있어야 함.
# 미러링 — ARCH 한 줄만 바꿔 v8 → v11 로 반복(Phase 2).

import os, glob, shutil, subprocess, sys
from google.colab import drive

ARCH = 'yolov8s.pt'                  # ← v11 차례엔 'yolo11s.pt'
CONDS = ['C0', 'C3']                 # 단일변수: 합성 방식
SEEDS = [1, 2, 3, 4, 5]
EPOCHS, IMGSZ, BATCH, PATIENCE = 80, 640, 16, 20

FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'         # Drive 저장 (resumable)
WORK = '/content/runs_v2'
tag = 'v11' if '11' in ARCH else 'v8'

drive.mount('/content/drive')
try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
import torch
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise SystemExit('GPU 가 없음 — 런타임을 GPU 로')
for c in CONDS:
    if not os.path.exists(f'{FIRE}/synth_{c}/train/images'):
        raise SystemExit(f'{FIRE}/synth_{c}/train 이 없음 — colab_synth.py (SYNTH_MODE={c}) 먼저')
print(f'아키텍처 {ARCH} (tag={tag}) · GPU {torch.cuda.get_device_name(0)}')
print(f'조건 {CONDS} × seed {SEEDS} = {len(CONDS) * len(SEEDS)} 모델')

os.makedirs(WORK, exist_ok=True)

def data_yaml(cond):
    syn = f'{FIRE}/synth_{cond}'
    path = f'{WORK}/data_{cond}.yaml'
    open(path, 'w').write(                           # baseline: train 을 두 번(2× clean)
        f"path: {syn}\ntrain: [train/images, train/images]\nval: val/images\n"
        f"test: test/images\nnc: 1\nnames: ['fire']\n")
    return path

done, trained = [], []
for cond in CONDS:
    dy = data_yaml(cond)
    for seed in SEEDS:
        name = f'{tag}_{cond}_s{seed}'
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
tot = len(CONDS) * len(SEEDS)
print(f'{tag} — 이미 있던 {len(done)}개 · 이번 학습 {len(trained)}개 (총 {len(done) + len(trained)}/{tot})')
if len(done) + len(trained) < tot:
    print('아직 남음 — 런타임 끊기면 이 셀만 다시 돌리면 이어짐(resumable).')
else:
    print(f'{tag} {tot}개 완료. 다음 — colab_v2_eval.py 로 realfire 영상단위 C0 vs C3 비교.')
