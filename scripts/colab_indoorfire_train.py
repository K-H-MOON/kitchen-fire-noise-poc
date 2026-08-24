# ===== 실험 A: 합성 vs 실데이터 vs 혼합 학습 → 같은 실제 test 비교 (Colab, GPU) =====
#
# 질문: "실제 화재 데이터를 학습에 넣으면 recall(놓침)이 오르나?"
# 데이터: Indoor Fire Smoke(실촬영, YOLO, fire=0/smoke=1). fire-only(1클래스)로 통일해 합성과 공정 비교.
# 3조건 · 같은 held-out test(Indoor test 750, fire 392):
#   1) 합성-only  = 기존 v8_C0_s1 (새 학습 없음) 을 test 에서 재측정
#   2) 실-only    = Indoor train 3500 로 학습(val=Indoor valid)
#   3) 혼합       = 합성(synth_C0 train) + Indoor train 로 학습
# 지표: frame-level recall = fire_det/fire_tot · precision = fire_det/(fire_det+nof_det) · fpr.
#
# 경계: Indoor 는 일반 실내 화재(주방/유류 아님) · Roboflow 랜덤분할 약한 누수 가능 · 라벨은 육안 감사 통과.
# 환경: EPOCHS(기본 60) · SYNTH_COND(기본 C0) · RUN(기본 'both'; 'realonly'/'mixed'/'evalonly') · BASE_MODEL(합성-only, 기본 v8_C0_s1)

import os, glob, json, zipfile, shutil, subprocess, sys
import numpy as np
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'
ZIP  = '/content/drive/MyDrive/Indoor Fire Smoke.zip'
RAW  = '/content/Indoor_Fire_Smoke'
IFD  = '/content/if_fire'                 # fire-only 재구성 Indoor
MIX  = '/content/mixed'                   # 합성 + Indoor train
PROJ = f'{FIRE}/runs_if'                  # 학습 산출(Drive 보존)
OUT  = f'{FIRE}/indoorfire_eval'
CONF = 0.25
EPOCHS = int(os.environ.get('EPOCHS', '60'))
SYN_COND = os.environ.get('SYNTH_COND', 'C0')
SYN_TRAIN = f'{FIRE}/synth_{SYN_COND}/train'
BASE_MODEL = os.environ.get('BASE_MODEL', 'v8_C0_s1')
RUN = os.environ.get('RUN', 'both')       # both / realonly / mixed / evalonly
HARDNEG = os.environ.get('HARDNEG', '0') == '1'   # 하드네거 주입(헛불 고치기) → 모델명 _hn
HN_TRAIN = os.environ.get('HARDNEG_TRAIN_DIR', f'{FIRE}/oilfire_hardneg_train/nofire')
SUF = '_hn' if HARDNEG else ''             # _hn 접미사(기존 baseline 안 덮음 · before/after 비교)

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(RAW) or not os.listdir(RAW):
    os.makedirs(RAW, exist_ok=True); print('Indoor 압축 해제...'); zipfile.ZipFile(ZIP).extractall(RAW)

# ---------------------------------------------------------------------------
# fire-only 재구성 — smoke(1) 라벨 제거, fire(0) 만. 이미지 심링크 + 필터 라벨 작성.
# ---------------------------------------------------------------------------
def build_fireonly():
    for sp_src, sp_dst in [('train', 'train'), ('valid', 'valid'), ('test', 'test')]:
        imgs = glob.glob(f'{RAW}/**/{sp_src}/images/*.jpg', recursive=True)
        di, dl = f'{IFD}/{sp_dst}/images', f'{IFD}/{sp_dst}/labels'
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for p in imgs:
            name = os.path.basename(p); stem = os.path.splitext(name)[0]
            dst = f'{di}/{name}'
            if not os.path.exists(dst):
                os.symlink(p, dst)
            lp = p.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')[:-4] + '.txt'
            lines = []
            if os.path.exists(lp):
                lines = [l for l in open(lp) if l.split() and l.split()[0] == '0']
            open(f'{dl}/{stem}.txt', 'w').writelines(lines)          # 빈 파일 = 불 없음(음성)
    yaml = f"path: {IFD}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fire']\n"
    open(f'{IFD}/data.yaml', 'w').write(yaml)
    print(f'fire-only 재구성 완료 -> {IFD}')

def build_mixed():
    # train = if_fire/train + synth train(이미 fire 1클래스). valid/test = if_fire 것 심링크.
    di, dl = f'{MIX}/train/images', f'{MIX}/train/labels'
    os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
    for p in glob.glob(f'{IFD}/train/images/*.jpg'):
        n = 'if_' + os.path.basename(p)
        if not os.path.exists(f'{di}/{n}'):
            os.symlink(os.path.realpath(p), f'{di}/{n}')
        shutil.copy(f'{IFD}/train/labels/{os.path.splitext(os.path.basename(p))[0]}.txt',
                    f'{dl}/{os.path.splitext(n)[0]}.txt')
    for p in glob.glob(f'{SYN_TRAIN}/images/*.jpg'):
        n = 'sy_' + os.path.basename(p)
        if not os.path.exists(f'{di}/{n}'):
            os.symlink(p, f'{di}/{n}')
        lp = f'{SYN_TRAIN}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
        open(f'{dl}/{os.path.splitext(n)[0]}.txt', 'w').write(open(lp).read() if os.path.exists(lp) else '')
    for sp in ('valid', 'test'):                                     # val/test 는 실제(if_fire) 그대로
        d = f'{MIX}/{sp}'
        if os.path.islink(d) or os.path.exists(d):
            pass
        else:
            os.symlink(f'{IFD}/{sp}', d)
    yaml = f"path: {MIX}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fire']\n"
    open(f'{MIX}/data.yaml', 'w').write(yaml)
    ni = len(glob.glob(f'{di}/*.jpg'))
    print(f'혼합 train 구성 완료 {ni}장 -> {MIX}')

def inject_hardneg(imgs_dir, labels_dir):
    """train-hardneg 프레임을 빈 라벨(0바이트=불 없음) 음성으로 train 에 주입."""
    if not HARDNEG:
        return 0
    n = 0
    for p in sorted(glob.glob(f'{HN_TRAIN}/*.jpg')):
        name = 'hn_' + os.path.basename(p)
        dst = f'{imgs_dir}/{name}'
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(p), dst)
        open(f'{labels_dir}/{os.path.splitext(name)[0]}.txt', 'w').close()   # 0바이트 = 음성
        n += 1
    print(f'[HARDNEG] 하드네거 {n}장 train 주입 -> {imgs_dir}')
    return n

build_fireonly()
inject_hardneg(f'{IFD}/train/images', f'{IFD}/train/labels')   # real_only(및 mixed 가 IFD 상속)
if RUN in ('both', 'mixed'):
    build_mixed()

# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def train(name, data_yaml):
    print(f'\n=== 학습: {name} (epochs {EPOCHS}) ===')
    YOLO('yolov8s.pt').train(data=data_yaml, epochs=EPOCHS, imgsz=640, patience=15,
                             project=PROJ, name=name, exist_ok=True, verbose=False, plots=False)
    return f'{PROJ}/{name}/weights/best.pt'

models = {'1_synth_only': f'{RUNS}/{BASE_MODEL}/best.pt'}
if RUN in ('both', 'realonly'):
    models['2_real_only'] = train('real_only' + SUF, f'{IFD}/data.yaml')
if RUN in ('both', 'mixed'):
    models['3_mixed'] = train('mixed' + SUF, f'{MIX}/data.yaml')
if RUN == 'evalonly':                                               # 이미 학습된 것 평가만
    if os.path.exists(f'{PROJ}/real_only{SUF}/weights/best.pt'):
        models['2_real_only'] = f'{PROJ}/real_only{SUF}/weights/best.pt'
    if os.path.exists(f'{PROJ}/mixed{SUF}/weights/best.pt'):
        models['3_mixed'] = f'{PROJ}/mixed{SUF}/weights/best.pt'

# ---------------------------------------------------------------------------
# 평가 — 같은 Indoor test 에서 frame-level recall/precision/fpr
# ---------------------------------------------------------------------------
test_imgs = sorted(glob.glob(f'{IFD}/test/images/*.jpg'))
def is_fire(imgp):
    lp = f'{IFD}/test/labels/{os.path.splitext(os.path.basename(imgp))[0]}.txt'
    return os.path.exists(lp) and os.path.getsize(lp) > 0
fire_imgs = [p for p in test_imgs if is_fire(p)]
nof_imgs  = [p for p in test_imgs if not is_fire(p)]
print(f'\ntest: fire {len(fire_imgs)} · nofire {len(nof_imgs)}')

def ndet(model, paths):
    n = 0
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=CONF, verbose=False):
            n += int(len(r.boxes) > 0)
    return n

rows = {}
for key, best in models.items():
    if not best or not os.path.exists(best):
        print(f'  [없음] {key}: {best}'); continue
    m = YOLO(best)
    fd = ndet(m, fire_imgs); nd = ndet(m, nof_imgs)
    rec = fd / len(fire_imgs) if fire_imgs else 0
    prec = fd / (fd + nd) if (fd + nd) else 0
    fpr = nd / len(nof_imgs) if nof_imgs else 0
    rows[key] = dict(recall=rec, precision=prec, fpr=fpr, fire_det=fd, nof_det=nd)

# ---------------------------------------------------------------------------
# 학습 곡선 — epoch별 train/val loss (results.csv/png). "에폭마다 loss 측정" 충족.
# ---------------------------------------------------------------------------
def show_curves(name):
    import csv as _csv
    d = f'{PROJ}/{name}'; c = f'{d}/results.csv'; p = f'{d}/results.png'
    if not os.path.exists(c):
        return
    rows = [{k.strip(): v for k, v in r.items()} for r in _csv.DictReader(open(c))]
    def g(r, k):
        try:
            return float(r[k])
        except Exception:
            return None
    last = rows[-1]
    tl = sum(x for x in [g(last, 'train/box_loss'), g(last, 'train/cls_loss'), g(last, 'train/dfl_loss')] if x)
    vl = sum(x for x in [g(last, 'val/box_loss'), g(last, 'val/cls_loss'), g(last, 'val/dfl_loss')] if x)
    K = 'metrics/mAP50-95(B)'
    bi = max(range(len(rows)), key=lambda i: (g(rows[i], K) or -1))
    print(f'[{name}] 실행 epoch {len(rows)} · 최종 train_loss {tl:.3f} · val_loss {vl:.3f} '
          f'· best epoch {int(g(rows[bi], "epoch"))} (mAP50-95 {g(rows[bi], K):.3f})')
    if os.path.exists(p):                    # plots=False 면 results.png 없음(수치는 위 줄에 있음)
        try:
            from IPython.display import Image as _I, display as _d
            _d(_I(p))
        except Exception:
            print(f'  곡선 그림: {p}')

print('\n' + '=' * 66)
print('학습 곡선 (train/val loss · best epoch)')
print('=' * 66)
for n in ('real_only' + SUF, 'mixed' + SUF):
    show_curves(n)

print('\n' + '=' * 66)
print('실험 A — 같은 Indoor test(fire 392) · frame-level')
print('=' * 66)
print(f'{"조건":<16}{"recall":>9}{"precision":>11}{"fpr":>8}')
for k in ['1_synth_only', '2_real_only', '3_mixed']:
    if k in rows:
        r = rows[k]
        print(f'{k:<16}{r["recall"]:>9.3f}{r["precision"]:>11.3f}{r["fpr"]:>8.3f}')
print('\n해석: 2·3 이 1보다 recall↑면 "실데이터가 놓침을 줄임" · 3 vs 2 로 합성 기여 확인.')
print('경계: 일반 실내 화재(주방 아님) · Roboflow 분할 약한 누수 가능.')

json.dump({'epochs': EPOCHS, 'synth_cond': SYN_COND, 'base': BASE_MODEL, 'hardneg': HARDNEG,
           'n_fire_test': len(fire_imgs), 'n_nof_test': len(nof_imgs), 'rows': rows},
          open(f'{OUT}/indoorfire_train{SUF}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/indoorfire_train{SUF}.json · 모델 {PROJ}/(real_only{SUF}|mixed{SUF})/weights/best.pt')
if HARDNEG:
    print('  [HARDNEG] Indoor test 는 헛불 원본이 아님 → 여기 fpr 은 회귀 확인용(낮게 유지돼야).')
    print('  실제 헛불 개선은 EVAL_SET=oilfire_hardneg_test 로 colab_oilfire_eval.py 에서 측정.')
