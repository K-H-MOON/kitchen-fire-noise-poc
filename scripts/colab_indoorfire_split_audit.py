# ===== 실험 A 후속: 누수 통제 재측정 (Colab, GPU) =====
#
# 질문: 실험 A의 recall 0.985가 '진짜'인가, 아니면 프레임 랜덤분할 '누수'로 부풀었나?
#   (같은 영상의 거의 똑같은 인접 프레임이 train·test에 동시에 들어가면 test가 부풀 수 있음)
#
# 데이터셋에 영상 ID가 없으므로, 근접중복(perceptual hash)로 '거의 동일한 프레임'을
# 클러스터링하고, 그 클러스터가 train/test에 걸치지 않게 그룹 단위로 재분할한다
# (= 가장 강한 형태의 누수 제거). 그 후 재학습·재측정해 원본(0.985)과 비교.
#
# Phase A (감사, GPU 불필요): 원본 split의 누수율 측정
#     = test 프레임 중 train에 근접중복 클러스터 동료가 있는 비율(특히 fire).
# Phase B (재측정, GPU): 그룹 단위 재분할 → real_only 재학습 → 같은 frame-level 지표 재측정.
#
# 환경변수:
#   RUN='audit'  → Phase A 만 (빠름, GPU 불필요; 먼저 이걸로 누수 크기 확인 권장)
#   RUN='both'   → Phase A + B (기본)
#   HAM_THRESH   → pHash(64bit) Hamming 임계(기본 6; 작을수록 '거의 동일'만 묶음)
#   EPOCHS       → 재학습 epoch (기본 60)
#   EVAL_MIXED   → '1'이면 mixed 도 재학습·재측정(기본 0 = real_only 만; 누수 질문엔 real_only 로 충분)
#   SEED         → 재분할 시드(기본 0)
#
# 경계: 근접중복 클러스터는 '인접 프레임 누수'를 잡지만, 육안상 다른 같은-영상 프레임까지
#       완벽히 묶지는 못함(진짜 영상 단위 분할의 근사). 그래도 가장 부풀리기 쉬운 누수를 제거함.

import os, glob, json, zipfile, shutil, subprocess, sys
import numpy as np

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
try:
    import imagehash
    from PIL import Image
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imagehash', 'Pillow'], check=True)
    import imagehash
    from PIL import Image

FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'
ZIP  = '/content/drive/MyDrive/Indoor Fire Smoke.zip'
RAW  = '/content/Indoor_Fire_Smoke'
NEW  = '/content/if_fire_grouped'          # 그룹 단위 재분할본(fire-only)
PROJ = f'{FIRE}/runs_if'
OUT  = f'{FIRE}/indoorfire_eval'
CONF = 0.25
RUN        = os.environ.get('RUN', 'both')
HAM_THRESH = int(os.environ.get('HAM_THRESH', '6'))
EPOCHS     = int(os.environ.get('EPOCHS', '60'))
EVAL_MIXED = os.environ.get('EVAL_MIXED', '0') == '1'
SEED       = int(os.environ.get('SEED', '0'))
BASE_MODEL = os.environ.get('BASE_MODEL', 'v8_C0_s1')   # 합성-only 앵커
SYN_COND   = os.environ.get('SYNTH_COND', 'C0')
SYN_TRAIN  = f'{FIRE}/synth_{SYN_COND}/train'
HARDNEG    = os.environ.get('HARDNEG', '0') == '1'      # 하드네거 주입(헛불 고치기) → 모델명 _hn
HN_TRAIN   = os.environ.get('HARDNEG_TRAIN_DIR', f'{FIRE}/oilfire_hardneg_train/nofire')
SUF        = '_hn' if HARDNEG else ''

os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(RAW) or not os.listdir(RAW):
    os.makedirs(RAW, exist_ok=True); print('Indoor 압축 해제...'); zipfile.ZipFile(ZIP).extractall(RAW)

# ---------------------------------------------------------------------------
# 원본 split의 모든 이미지 수집 (경로·원split·라벨경로·fire여부)
# ---------------------------------------------------------------------------
def is_fire(labelpath):
    if not (os.path.exists(labelpath) and os.path.getsize(labelpath) > 0):
        return False
    return any(l.split() and l.split()[0] == '0' for l in open(labelpath))   # fire=class0 존재

recs = []      # dict: img, split(orig), label, fire
for sp in ('train', 'valid', 'test'):
    for p in glob.glob(f'{RAW}/**/{sp}/images/*.jpg', recursive=True):
        lp = p.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')[:-4] + '.txt'
        recs.append(dict(img=p, split=sp, label=lp, fire=is_fire(lp)))
N = len(recs)
print(f'총 {N}장 (원본 split): '
      f"train {sum(r['split']=='train' for r in recs)} · "
      f"valid {sum(r['split']=='valid' for r in recs)} · "
      f"test {sum(r['split']=='test' for r in recs)}")

# ---------------------------------------------------------------------------
# perceptual hash (dHash 64bit) → uint64  (dHash: scipy 불필요·근접중복 탐지에 적합)
# ---------------------------------------------------------------------------
print('dHash 계산...')
def phash_u64(path):
    h = imagehash.dhash(Image.open(path).convert('RGB'))   # 8x8 bool
    bits = h.hash.flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)

hashes = np.empty(N, dtype=np.uint64)
for i, r in enumerate(recs):
    try:
        hashes[i] = phash_u64(r['img'])
    except Exception:
        hashes[i] = np.uint64(0)
    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{N}')

# ---------------------------------------------------------------------------
# Hamming 임계 클러스터링 (union-find) — 근접중복 그룹
# ---------------------------------------------------------------------------
POP = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.uint8)
def hamm_to_all(h):
    x = hashes ^ h
    return (POP[np.asarray(x & np.uint64(0xFFFF), dtype=np.uint32)]
            + POP[np.asarray((x >> np.uint64(16)) & np.uint64(0xFFFF), dtype=np.uint32)]
            + POP[np.asarray((x >> np.uint64(32)) & np.uint64(0xFFFF), dtype=np.uint32)]
            + POP[np.asarray((x >> np.uint64(48)) & np.uint64(0xFFFF), dtype=np.uint32)])

parent = list(range(N))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

print(f'근접중복 클러스터링 (Hamming ≤ {HAM_THRESH})...')
for i in range(N):
    d = hamm_to_all(hashes[i])
    for j in np.where(d <= HAM_THRESH)[0]:
        if j > i:
            union(i, int(j))
    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{N}')

gid = np.array([find(i) for i in range(N)])
groups = {}
for i, g in enumerate(gid):
    groups.setdefault(int(g), []).append(i)
sizes = np.array([len(v) for v in groups.values()])
print(f'클러스터 {len(groups)}개 · 다중원소 클러스터 {int((sizes>1).sum())}개 · '
      f'최대 {int(sizes.max())} · 근접중복에 속한 프레임 {int(sizes[sizes>1].sum())}/{N}')

# ---------------------------------------------------------------------------
# Phase A — 원본 split 누수율: test 프레임의 클러스터 동료가 train 에 있나
# ---------------------------------------------------------------------------
split_of = np.array([r['split'] for r in recs])
fire_of  = np.array([r['fire'] for r in recs])
gmembers = {int(g): set(v) for g, v in groups.items()}

def leak_rate(sel_mask):
    idxs = np.where(sel_mask)[0]
    leaked = 0
    for i in idxs:
        fam = gmembers[int(gid[i])]
        if any(split_of[j] == 'train' for j in fam if j != i):
            leaked += 1
    return leaked, len(idxs)

t_all = leak_rate(split_of == 'test')
t_fire = leak_rate((split_of == 'test') & fire_of)
print('\n' + '=' * 66)
print('Phase A — 원본 split 누수 감사')
print('=' * 66)
print(f'test 전체: train 근접중복 있는 프레임 {t_all[0]}/{t_all[1]} ({t_all[0]/max(t_all[1],1):.1%})')
print(f'test fire: train 근접중복 있는 프레임 {t_fire[0]}/{t_fire[1]} ({t_fire[0]/max(t_fire[1],1):.1%})')
print('→ 이 비율이 높을수록 recall 0.985가 누수로 부풀었을 가능성이 큼.')

audit = dict(n=N, ham_thresh=HAM_THRESH, n_clusters=len(groups),
             n_multi=int((sizes > 1).sum()), max_cluster=int(sizes.max()),
             test_leak=t_all, test_fire_leak=t_fire)
json.dump(audit, open(f'{OUT}/indoorfire_split_audit.json', 'w'), ensure_ascii=False, indent=1, default=int)
print(f'-> {OUT}/indoorfire_split_audit.json')

if RUN == 'audit':
    print('\nRUN=audit → 여기서 종료. 누수율 보고 Phase B(재학습) 여부 판단.')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Phase B — 그룹 단위 재분할 (클러스터가 split 경계 안 넘게) → fire-only 재구성
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
group_ids = list(groups.keys())
rng.shuffle(group_ids)
# 그룹을 이미지 수 기준 70/15/15 로 배분(그룹 통째로)
tot = N; tr_cap, va_cap = int(tot * 0.70), int(tot * 0.15)
assign = {}; c_tr = c_va = 0
for g in group_ids:
    n = len(groups[g])
    if c_tr + n <= tr_cap or c_tr == 0:
        assign[g] = 'train'; c_tr += n
    elif c_va + n <= va_cap or c_va == 0:
        assign[g] = 'valid'; c_va += n
    else:
        assign[g] = 'test'
new_split = np.array([assign[int(gid[i])] for i in range(N)])
for sp in ('train', 'valid', 'test'):
    m = new_split == sp
    print(f'재분할 {sp}: {int(m.sum())}장 (fire {int((m & fire_of).sum())})')

# 누수 0 확인
def leak_after(sp):
    idxs = np.where(new_split == sp)[0]
    bad = 0
    for i in idxs:
        fam = gmembers[int(gid[i])]
        if any(new_split[j] == 'train' for j in fam if j != i):
            bad += 1
    return bad
print(f'재분할 후 test의 train-근접중복: {leak_after("test")} (0이어야 정상)')

# fire-only 디렉터리 작성 (smoke 제거, fire만; 빈 라벨 = 음성)
if os.path.isdir(NEW):
    shutil.rmtree(NEW)
for i, r in enumerate(recs):
    sp = new_split[i]
    di, dl = f'{NEW}/{sp}/images', f'{NEW}/{sp}/labels'
    os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
    name = os.path.basename(r['img']); stem = os.path.splitext(name)[0]
    dst = f'{di}/{name}'
    if not os.path.exists(dst):
        os.symlink(r['img'], dst)
    lines = []
    if os.path.exists(r['label']):
        lines = [l for l in open(r['label']) if l.split() and l.split()[0] == '0']
    open(f'{dl}/{stem}.txt', 'w').writelines(lines)
yaml = f"path: {NEW}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fire']\n"
open(f'{NEW}/data.yaml', 'w').write(yaml)
print(f'fire-only 재분할본 -> {NEW}')

# 하드네거 주입(opt-in): 빈 라벨 음성으로 그룹-train 에 추가. valid/test 는 불변(=fpr 회귀는 별도 셋).
if HARDNEG:
    hn = 0
    for p in sorted(glob.glob(f'{HN_TRAIN}/*.jpg')):
        name = 'hn_' + os.path.basename(p)
        dst = f'{NEW}/train/images/{name}'
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(p), dst)
        open(f'{NEW}/train/labels/{os.path.splitext(name)[0]}.txt', 'w').close()   # 0바이트 = 음성
        hn += 1
    print(f'[HARDNEG] 하드네거 {hn}장 그룹-train 주입 (mixed 도 상속)')

# ---------------------------------------------------------------------------
# 재학습
# ---------------------------------------------------------------------------
def train(name, data_yaml):
    print(f'\n=== 재학습: {name} (epochs {EPOCHS}) ===')
    YOLO('yolov8s.pt').train(data=data_yaml, epochs=EPOCHS, imgsz=640, patience=15,
                             project=PROJ, name=name, exist_ok=True, verbose=False, plots=False)
    return f'{PROJ}/{name}/weights/best.pt'

# mixed 재학습(선택): synth 를 재분할 train 에 더함
def build_mixed_grouped():
    MIX = '/content/mixed_grouped'
    if os.path.isdir(MIX):
        shutil.rmtree(MIX)
    di, dl = f'{MIX}/train/images', f'{MIX}/train/labels'
    os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
    for p in glob.glob(f'{NEW}/train/images/*.jpg'):
        n = 'if_' + os.path.basename(p)
        if not os.path.exists(f'{di}/{n}'):
            os.symlink(os.path.realpath(p), f'{di}/{n}')
        shutil.copy(f'{NEW}/train/labels/{os.path.splitext(os.path.basename(p))[0]}.txt',
                    f'{dl}/{os.path.splitext(n)[0]}.txt')
    for p in glob.glob(f'{SYN_TRAIN}/images/*.jpg'):
        n = 'sy_' + os.path.basename(p)
        if not os.path.exists(f'{di}/{n}'):
            os.symlink(p, f'{di}/{n}')
        lp = f'{SYN_TRAIN}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
        open(f'{dl}/{os.path.splitext(n)[0]}.txt', 'w').write(open(lp).read() if os.path.exists(lp) else '')
    for sp in ('valid', 'test'):
        d = f'{MIX}/{sp}'
        if not (os.path.islink(d) or os.path.exists(d)):
            os.symlink(f'{NEW}/{sp}', d)
    open(f'{MIX}/data.yaml', 'w').write(
        f"path: {MIX}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fire']\n")
    return f'{MIX}/data.yaml'

models = {'1_synth_only': f'{RUNS}/{BASE_MODEL}/best.pt',
          '2_real_only': train('real_only_grouped' + SUF, f'{NEW}/data.yaml')}
if EVAL_MIXED:
    models['3_mixed'] = train('mixed_grouped' + SUF, build_mixed_grouped())

# ---------------------------------------------------------------------------
# 재측정 — 그룹 단위 test 에서 frame-level recall/precision/fpr
# ---------------------------------------------------------------------------
test_imgs = sorted(glob.glob(f'{NEW}/test/images/*.jpg'))
def is_fire_new(imgp):
    lp = f'{NEW}/test/labels/{os.path.splitext(os.path.basename(imgp))[0]}.txt'
    return os.path.exists(lp) and os.path.getsize(lp) > 0
fire_imgs = [p for p in test_imgs if is_fire_new(p)]
nof_imgs  = [p for p in test_imgs if not is_fire_new(p)]
print(f'\n재분할 test: fire {len(fire_imgs)} · nofire {len(nof_imgs)}')

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

print('\n' + '=' * 66)
print('Phase B — 누수 통제(그룹 분할) 재측정 · frame-level')
print('=' * 66)
print(f'{"조건":<16}{"recall":>9}{"precision":>11}{"fpr":>8}')
for k in ['1_synth_only', '2_real_only', '3_mixed']:
    if k in rows:
        r = rows[k]
        print(f'{k:<16}{r["recall"]:>9.3f}{r["precision"]:>11.3f}{r["fpr"]:>8.3f}')
print('\n비교: 실험 A(랜덤분할) real_only recall 0.985.')
print('  → 재측정 recall 이 여전히 높으면(예 0.85+) 누수 걱정 대부분 해소·in-domain 견고.')
print('  → 크게 떨어지면 0.985 는 누수로 부풀었던 것.')

json.dump({'ham_thresh': HAM_THRESH, 'epochs': EPOCHS, 'seed': SEED,
           'audit': audit,
           'resplit_counts': {sp: int((new_split == sp).sum()) for sp in ('train', 'valid', 'test')},
           'n_fire_test': len(fire_imgs), 'n_nof_test': len(nof_imgs), 'hardneg': HARDNEG,
           'rows': rows, 'original_random_split_real_only_recall': 0.9847},
          open(f'{OUT}/indoorfire_regroup{SUF}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/indoorfire_regroup{SUF}.json · 모델 {PROJ}/real_only_grouped{SUF}/weights/best.pt')
if HARDNEG:
    print('  [HARDNEG] 여기 fpr 은 Indoor(헛불 원본 아님) 회귀용. 헛불 개선은 oilfire_hardneg_test 로 측정.')
