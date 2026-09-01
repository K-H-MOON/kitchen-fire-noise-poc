# ===== Roboflow YOLO export 품질·누수 검수 (Colab, %run -i) =====
#
# 하는 일:
#   1) zip 해제 + split 구조 자동 탐지(train/valid|val/test, images/·labels/)
#   2) 라벨 통계 — 클래스 분포 · 이미지당 박스 수(0박스=음성) · 박스 크기 분포(초소형/거대) · 좌표 이탈
#   3) dHash 근접중복 → 장면 수(≈고유 장면) · **교차-split 누수**(val/test 가 train 과 근접중복인가)
#   4) 몽타주 — 박스 얹은 랜덤/최소박스/최대박스/근접중복대표 를 인라인 표시
#
# 사용(별도 셀):
#   import os
#   os.environ['QC_ZIP'] = '/content/drive/MyDrive/fire-indoor.zip'   # Drive 에 올린 zip
#   %run -i /content/kitchen-fire-noise-poc/scripts/colab_qc_dataset.py
#
# 환경변수:
#   QC_ZIP    : Drive/로컬 zip 경로. 주면 /content/_qc_data 로 해제 후 data.yaml 루트 자동탐색.
#   QC_ROOT   : (zip 대신) 이미 풀린 데이터셋 루트(data.yaml 있는 폴더).
#   QC_OUT    : 몽타주 출력 폴더(기본 /content/qc_out).
#   QC_SAMPLE : 랜덤 몽타주 표본 수(기본 100).
#   QC_HAM    : 근접중복 Hamming 임계(기본 6).
#   QC_NOHASH : '1' 이면 pHash(장면수·누수) 건너뜀(빠름).
#
# 주의: 이 데이터셋 train 은 ×3 증강이 구워져 3장이 한 클러스터로 묶임(정상).
#       누수는 val/test ↔ train 사이만 문제(증강은 train-only).
import os, sys, glob, random, json, zipfile, subprocess
import numpy as np

for pkg in ('PIL', 'imagehash'):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'Pillow' if pkg == 'PIL' else pkg], check=True)
from PIL import Image, ImageDraw, ImageFont
import imagehash
try:
    from IPython.display import display, Image as IPImage
    _INLINE = True
except Exception:
    _INLINE = False
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print('drive.mount 건너뜀:', e)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
QC_ZIP    = os.environ.get('QC_ZIP', '').strip()
QC_ROOT   = os.environ.get('QC_ROOT', '').strip()
QC_OUT    = os.environ.get('QC_OUT', '/content/qc_out')
QC_SAMPLE = int(os.environ.get('QC_SAMPLE', '100'))
QC_HAM    = int(os.environ.get('QC_HAM', '6'))
QC_NOHASH = os.environ.get('QC_NOHASH', '0') == '1'
os.makedirs(QC_OUT, exist_ok=True)


def resolve_root():
    if QC_ZIP:
        ex = '/content/_qc_data'
        import shutil; shutil.rmtree(ex, ignore_errors=True); os.makedirs(ex, exist_ok=True)
        print(f'[zip 해제] {QC_ZIP} → {ex}')
        with zipfile.ZipFile(QC_ZIP) as z:
            z.extractall(ex)
        # data.yaml 있는 폴더를 루트로
        for r, _, fs in os.walk(ex):
            if 'data.yaml' in fs:
                return r
        return ex
    assert QC_ROOT, 'QC_ZIP 또는 QC_ROOT 를 지정하세요'
    return QC_ROOT


def find_splits(root):
    splits = {}
    for name in ('train', 'valid', 'val', 'test'):
        if os.path.isdir(os.path.join(root, name, 'images')):
            splits[name] = os.path.join(root, name)
    if not splits and os.path.isdir(os.path.join(root, 'images')):
        splits['(root)'] = root
    return splits


def list_pairs(split_dir):
    imgs = []
    for e in IMG_EXTS:
        imgs += glob.glob(os.path.join(split_dir, 'images', '*' + e))
        imgs += glob.glob(os.path.join(split_dir, 'images', '*' + e.upper()))
    imgs = sorted(set(imgs))
    out = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(split_dir, 'labels', stem + '.txt')
        out.append((p, lp if os.path.exists(lp) else None))
    return out


def read_boxes(lp):
    if not lp:
        return []
    out = []
    for ln in open(lp, encoding='utf-8', errors='ignore'):
        t = ln.split()
        if len(t) < 5:
            continue
        try:
            c = int(float(t[0])); cx, cy, w, h = (float(x) for x in t[1:5])
        except ValueError:
            continue
        out.append((c, cx, cy, w, h))
    return out


def read_names(root):
    y = os.path.join(root, 'data.yaml'); names = {}
    if os.path.exists(y):
        import re
        m = re.search(r'names:\s*\[([^\]]*)\]', open(y, encoding='utf-8', errors='ignore').read())
        if m:
            items = [s.strip().strip("'\"") for s in m.group(1).split(',') if s.strip()]
            names = {i: v for i, v in enumerate(items)}
    return names


def dhash_u64(path):
    h = imagehash.dhash(Image.open(path).convert('RGB')); v = 0
    for b in h.hash.flatten():
        v = (v << 1) | int(b)
    return np.uint64(v)


def montage(items, out_path, cols=10, cell=200, title=''):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell, rows * cell + 24), (18, 18, 18))
    drw = ImageDraw.Draw(sheet)
    try: F = ImageFont.load_default()
    except Exception: F = None
    drw.text((6, 6), title, fill=(255, 255, 255), font=F)
    for j, (p, boxes) in enumerate(items):
        try: im = Image.open(p).convert('RGB')
        except Exception: continue
        W, H = im.size; d = ImageDraw.Draw(im)
        for (c, cx, cy, w, h) in boxes:
            x0 = (cx - w / 2) * W; y0 = (cy - h / 2) * H
            x1 = (cx + w / 2) * W; y1 = (cy + h / 2) * H
            d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0) if c == 0 else (255, 120, 0),
                        width=max(2, W // 200))
        im = im.resize((cell, cell)); r, cc = divmod(j, cols)
        sheet.paste(im, (cc * cell, 24 + r * cell))
    sheet.save(out_path, quality=85)
    print('  몽타주:', out_path)
    if _INLINE:
        display(IPImage(filename=out_path))


# ---- 실행 ----------------------------------------------------------------
root = resolve_root()
names = read_names(root)
splits = find_splits(root)
assert splits, f'split 못 찾음: {root} 아래 train/valid/test 의 images/ 필요'

print('=' * 70)
print('데이터셋:', root)
print('클래스(data.yaml):', names or '(못 읽음)')
print('split:', ', '.join(splits))
print('=' * 70)

allrec = []   # (img, boxes, split)
for sp, d in splits.items():
    pairs = list_pairs(d)
    n_img = len(pairs); n_lbl = sum(1 for _, lp in pairs if lp)
    cls_cnt = {}; boxes_per = []; areas = []; oob = 0; empty = 0
    for p, lp in pairs:
        bxs = read_boxes(lp); allrec.append((p, bxs, sp)); boxes_per.append(len(bxs))
        if not bxs:
            empty += 1
        for (c, cx, cy, w, h) in bxs:
            cls_cnt[c] = cls_cnt.get(c, 0) + 1; areas.append(w * h)
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1.01 and 0 < h <= 1.01):
                oob += 1
    areas = np.array(areas) if areas else np.array([0.0])
    bp = np.array(boxes_per) if boxes_per else np.array([0])
    print(f'\n[{sp}] 이미지 {n_img} · 라벨파일 {n_lbl} · 라벨없음/빈(=음성) {empty}')
    print('   클래스 박스수: ' + ', '.join(f'{names.get(c, c)}={n}' for c, n in sorted(cls_cnt.items())))
    print(f'   이미지당 박스: 중앙 {int(np.median(bp))} · 최대 {int(bp.max())} · 평균 {bp.mean():.2f}')
    print(f'   박스면적(정규화): 중앙 {np.median(areas):.4f} · '
          f'초소형(<0.2%) {int((areas < 0.002).sum())} · 거대(>50%) {int((areas > 0.5).sum())}')
    if oob:
        print(f'   ⚠️ 좌표 이탈/이상 박스 {oob}개')

print('\n몽타주 생성...')
rng = random.Random(0)
withbox = [(p, b) for (p, b, s) in allrec if b]
if withbox:
    montage(rng.sample(withbox, min(QC_SAMPLE, len(withbox))),
            f'{QC_OUT}/montage_random.jpg', title='RANDOM (green=fire)')
    sm = sorted(withbox, key=lambda r: min(w*h for (_,_,_,w,h) in r[1]))[:50]
    lg = sorted(withbox, key=lambda r: max(w*h for (_,_,_,w,h) in r[1]), reverse=True)[:50]
    montage(sm, f'{QC_OUT}/montage_tiny_boxes.jpg', title='SMALLEST BOXES (noise-label suspect)')
    montage(lg, f'{QC_OUT}/montage_huge_boxes.jpg', title='LARGEST BOXES (lazy-label suspect)')

if QC_NOHASH:
    print('\nQC_NOHASH=1 → pHash(장면수·누수) 건너뜀'); sys.exit(0) if not _INLINE else None
else:
    print('\ndHash 계산(이미지 로드라 수 분 가능)...')
    N = len(allrec); hashes = np.empty(N, dtype=np.uint64); sp_of = [s for (_, _, s) in allrec]
    for i, (p, _, _) in enumerate(allrec):
        try: hashes[i] = dhash_u64(p)
        except Exception: hashes[i] = np.uint64(0)
        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{N}')
    POP = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.uint8)
    def hamm_all(h):
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
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    print(f'클러스터링 Hamming ≤ {QC_HAM} ...')
    for i in range(N):
        d = hamm_all(hashes[i])
        for j in np.where(d <= QC_HAM)[0]:
            if j > i: union(i, int(j))
    gid = [find(i) for i in range(N)]
    clusters = {}
    for i, g in enumerate(gid):
        clusters.setdefault(g, []).append(i)
    sizes = np.array([len(v) for v in clusters.values()])
    print(f'\n장면(클러스터) {len(clusters)}개 / 전체 {N}장 · '
          f'다중원소 {int((sizes>1).sum())} · 최대 {int(sizes.max())}')
    print(f'→ 고유 장면이 {len(clusters)}개뿐이면 그만큼만 다양하다는 뜻(6.8k 뻥튀기 여부).')

    leak = {s: [0, 0] for s in ('valid', 'val', 'test')}
    for g, mem in clusters.items():
        if any(sp_of[i] == 'train' for i in mem):
            for i in mem:
                if sp_of[i] in leak:
                    leak[sp_of[i]][0] += 1
    for s in ('valid', 'val', 'test'):
        tot = sum(1 for x in sp_of if x == s)
        if tot:
            leak[s][1] = tot; n = leak[s][0]
            print(f'   누수 {s}: train 근접중복 {n}/{tot} ({n/tot:.1%})'
                  + ('  ⚠️ 랜덤split 누수 → 재분할 필요' if n/tot > 0.02 else '  (양호)'))

    reps = [(allrec[mem[0]][0], allrec[mem[0]][1])
            for g, mem in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:50] if len(mem) > 1]
    montage(reps, f'{QC_OUT}/montage_dup_clusters.jpg', title='TOP DUP CLUSTERS (video frames?)')

    json.dump({'root': root, 'names': {str(k): v for k, v in names.items()},
               'N': N, 'n_scenes': len(clusters), 'leak': leak},
              open(f'{QC_OUT}/qc_summary.json', 'w'), ensure_ascii=False, indent=1)
    print('\n-> 요약:', f'{QC_OUT}/qc_summary.json')
    print('   몽타주 4장(random/tiny/huge/dup) 육안 확인 → 라벨 품질·누수 감.')
