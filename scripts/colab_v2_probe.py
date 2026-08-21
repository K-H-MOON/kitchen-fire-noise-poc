# ===== v2 Phase 1 프로브 — 수렴 epoch 측정 (Colab, GPU) =====
#
# 목적: EPOCHS 상한을 추측하지 않고 측정으로 정한다. 한 모델(v8_C0_s1)을 80ep·
# patience 20 로 돌려 ① early-stop 이 언제 걸리는지 ② best epoch·수렴점이 어디인지
# 를 results.csv 에서 읽는다. 이 값으로 전체 10모델의 상한을 확정한다.
#
# 주의(정직): 여기 수렴점은 **합성 val 기준**이다. 이는 "합성 과제를 다 배운 지점"이지
# realfire 전이 정점이 아니다(그건 held-out 이라 못 봄). 상한을 이보다 크게 잡을 근거가
# 없다는 뜻으로 쓴다 — "배울 만큼만, 임의의 과잉 없이".
#
# 이 모델은 측정용(별도 dir /content/probe) — runs_phaseB 를 건드리지 않는다.
# 상한 확정 후 colab_v2_train.py 로 10모델을 fresh 로 돌린다.

import os, subprocess, sys
from google.colab import drive

FIRE = '/content/drive/MyDrive/fire_frames'
WORK = '/content/probe'
PROBE_MAX, PATIENCE, IMGSZ, BATCH, SEED = 80, 20, 640, 16, 1

drive.mount('/content/drive')
try:
    import ultralytics
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
import torch, pandas as pd
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise SystemExit('GPU 가 없음 — 런타임을 GPU 로')
if not os.path.exists(f'{FIRE}/synth_C0/train/images'):
    raise SystemExit(f'{FIRE}/synth_C0 이 없음 — colab_synth.py (SYNTH_MODE=C0) 먼저')

os.makedirs(WORK, exist_ok=True)
dy = f'{WORK}/data_C0.yaml'
open(dy, 'w').write(                                 # v1 baseline 레시피(2× clean)
    f"path: {FIRE}/synth_C0\ntrain: [train/images, train/images]\nval: val/images\n"
    f"test: test/images\nnc: 1\nnames: ['fire']\n")

print(f'프로브 학습 — v8_C0_s1 · 최대 {PROBE_MAX}ep · patience {PATIENCE} (측정용)')
m = YOLO('yolov8s.pt')
m.train(data=dy, epochs=PROBE_MAX, imgsz=IMGSZ, batch=BATCH, seed=SEED,
        project=WORK, name='probe_C0_s1', exist_ok=True, patience=PATIENCE, verbose=False)

# ---------------------------------------------------------------------------
# results.csv 분석 — 실제 epoch·best·수렴점
# ---------------------------------------------------------------------------
df = pd.read_csv(f'{WORK}/probe_C0_s1/results.csv')
df.columns = [c.strip() for c in df.columns]
ran = len(df)
mcol = (next((c for c in df.columns if 'mAP50-95' in c), None)
        or next((c for c in df.columns if 'mAP50' in c), None))
if mcol is None or 'epoch' not in df.columns:      # ultralytics 버전 바뀌어 열 이름 예상 밖이면
    raise SystemExit(f'results.csv 형식 예상 밖(ultralytics 버전?) — 열 목록: {list(df.columns)}')
best_i = int(df[mcol].idxmax())
best_ep = int(df['epoch'].iloc[best_i]); best_v = float(df[mcol].max())
conv = int(df[df[mcol] >= best_v * 0.99]['epoch'].min())     # best 의 99% 첫 도달 = 사실상 수렴

print('\n' + '=' * 60)
print(f'실제 학습 epoch : {ran} / {PROBE_MAX}  → early-stop '
      f'{"걸림(합성 val 수렴)" if ran < PROBE_MAX else "안 걸림(상한까지 감)"}')
print(f'best epoch      : {best_ep}   ({mcol} = {best_v:.4f})')
print(f'수렴점(99% 첫)  : epoch {conv}')
step = max(1, ran // 10)
print('\nepoch별 val 지표(발췌):')
for i in list(range(0, ran, step)) + [ran - 1]:
    print(f'  ep {int(df["epoch"].iloc[i]):3d}   {mcol} {df[mcol].iloc[i]:.4f}')
print('\n' + '-' * 60)
if ran < PROBE_MAX:
    print(f'해석: patience 로 {ran}ep 에서 자동 정지 → 상한 {PROBE_MAX} 는 안 쓰였음.')
    print(f'      전체 10모델도 patience {PATIENCE} 면 각자 수렴에서 자동으로 잘림.')
    print(f'      권장 상한 = 수렴 여유값(예 {conv + PATIENCE + 5}) · patience 유지.')
else:
    print(f'해석: {PROBE_MAX}ep 까지 안 멈춤 — 합성 val 이 계속 개선(과적합 구간 진입 가능).')
    print(f'      권장 상한 = best epoch 근처({best_ep}) 로 낮춰 과학습 방지.')
print('이 값으로 colab_v2_train.py 의 EPOCHS 를 확정한다. (프로브 모델은 버림)')
