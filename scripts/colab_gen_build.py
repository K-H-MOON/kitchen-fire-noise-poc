# ===== 방안 #6: 생성형 합성(gen) 데이터 빌더 + 사전학습 앵커 — Colab, GPU =====
#
# 목적(미확정): NB Pro·Codex(GPT image)로 생성한 급식실/상업주방 CCTV 유류불 이미지를
#   YOLO 셋으로 만들고, 그 위에 yolov8s 를 사전학습(gen-synth 앵커)한다.
#   이 앵커가 커리큘럼 BASE 가 되어(→ split_audit BASE_YOLO) 실 파인튜닝으로 이어짐.
#   ★"이미지 좋음"은 필요조건일 뿐 — 전이 개선(2gencurr ≷ 2g)은 eval 로만 앎(단정 금지).
#
# 파이프라인(이 스크립트가 (1)(2) 둘 다):
#   (1) gen YOLO 데이터셋 빌드 — Roboflow 로 박스 친 export(이미지+라벨)를 재귀 수집.
#       · 하위폴더 접두어로 파일명 충돌 방지(p01_fire_001.jpg …)
#       · 클래스 전부 0(fire)로 강제(단일 클래스 감지기·nc=1)
#       · 라벨 없는 이미지 = 미주석 → 기본 제외(경고). fire 이미지를 배경으로 가르치면 유해.
#       · train/val 자체 분할(시드) — 이건 gen 앵커의 과적합 점검용. 실 test(oilfire_realtest)와 무관 = 누수 아님.
#   (2) TRAIN=1 이면 gen-only 학습 → runs_if/gen_synth/weights/best.pt (Drive·생존).
#       = 커리큘럼 BASE + (원하면) 실 proxy 서 gen-only 앵커로도 대조 가능.
#
# 다음(별도 셀) — 커리큘럼(배관은 #2·#4 로 검증됨, split_audit 재사용):
#   os.environ['BASE_YOLO']='/content/drive/MyDrive/fire_frames/runs_if/gen_synth/weights/best.pt'
#   os.environ['BASE_TAG']='gencurr'
#   %run -i /content/kitchen-fire-noise-poc/scripts/colab_indoorfire_split_audit.py
#   → real_only_grouped_gencurr · 이후 colab_realtest_eval.py 가 2gencurr 행 자동 대조.
#
# ★선행(수동): 생성 이미지 → (1x2 그리드면 slice_grid.py) → Roboflow 박스 → YOLO export.
#   그 export 폴더/zip 을 Drive 의 GEN_SRC 로 올린 뒤 이 스크립트 실행. 라벨 없으면 학습 불가.
#
# 환경변수:
#   GEN_SRC   : Roboflow YOLO export 루트(폴더 또는 .zip). 기본 = {FIRE}/gen_export
#   GEN_OUT   : 빌드 출력 YOLO 셋(기본 /content/gen_synth · 로컬·세션스코프·빠름). 파괴가드로 'gen_synth' 이름 강제.
#   GEN_VALFRAC : val 비율(기본 0.15)
#   GEN_SEED  : 분할 시드(기본 2)
#   GEN_EPOCHS: gen-only 학습 epoch(기본 60 · v8_C0_s1/synth_dr 와 동일)
#   TRAIN     : '1'(기본) gen-only 학습까지. '0' 이면 데이터셋만(QC 후 판단).
#   GEN_KEEP_UNLABELED : '1' 이면 라벨 없는 이미지도 배경(음성)으로 포함. 기본 '0'(제외).

import os, glob, json, random, shutil, zipfile, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

FIRE = '/content/drive/MyDrive/fire_frames'

GEN_SRC   = os.environ.get('GEN_SRC', f'{FIRE}/gen_export')
GEN_OUT   = os.environ.get('GEN_OUT', '/content/gen_synth')
VALFRAC   = float(os.environ.get('GEN_VALFRAC', '0.15'))
SEED      = int(os.environ.get('GEN_SEED', '2'))
GEN_EPOCHS= int(os.environ.get('GEN_EPOCHS', '60'))
TRAIN     = os.environ.get('TRAIN', '1') == '1'
KEEP_UNLABELED = os.environ.get('GEN_KEEP_UNLABELED', '0') == '1'

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

drive.mount('/content/drive')
rng = random.Random(SEED)

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except Exception:
    F = ImageFont.load_default()


# ---------------------------------------------------------------------------
# 소스 준비 — zip 이면 로컬로 풀고, 폴더면 그대로. __MACOSX/숨김 무시.
# ---------------------------------------------------------------------------
def resolve_src(src):
    if src.lower().endswith('.zip'):
        ex = '/content/_gen_src_unzipped'
        shutil.rmtree(ex, ignore_errors=True)
        os.makedirs(ex, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(ex)
        print(f'[zip 해제] {src} → {ex}')
        return ex
    return src


def label_for(img_path, src_root):
    """이미지의 YOLO 라벨(.txt) 경로를 찾는다.
    (a) Roboflow 표준: .../images/<stem>.png ↔ .../labels/<stem>.txt
    (b) 같은 폴더 <stem>.txt (flat export)."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    d = os.path.dirname(img_path)
    # (a) 형제 labels/ 디렉터리
    if os.path.basename(d).lower() == 'images':
        cand = os.path.join(os.path.dirname(d), 'labels', stem + '.txt')
        if os.path.exists(cand):
            return cand
    # (b) 같은 폴더
    cand = os.path.join(d, stem + '.txt')
    if os.path.exists(cand):
        return cand
    return None


def prefix_for(img_path, src_root):
    """하위폴더 경로를 파일명 접두어로 → 충돌 방지. images/ 계층은 접두어에서 생략."""
    rel = os.path.relpath(os.path.dirname(img_path), src_root)
    parts = [p for p in rel.replace('\\', '/').split('/')
             if p not in ('.', '', 'images')]
    pref = '_'.join(parts)
    # 파일명 안전화
    pref = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in pref)
    return (pref + '_') if pref else ''


def read_yolo_label(path):
    """YOLO txt 읽어 (cls, cx, cy, w, h) 리스트. cls 는 뒤에서 0으로 강제."""
    rows = []
    for ln in open(path, encoding='utf-8', errors='ignore'):
        t = ln.split()
        if len(t) < 5:
            continue
        try:
            cx, cy, w, h = (float(x) for x in t[1:5])
        except ValueError:
            continue
        rows.append((cx, cy, w, h))
    return rows


# ---------------------------------------------------------------------------
# (1) 수집 — 재귀. (out_name, img_path, label_path|None)
# ---------------------------------------------------------------------------
SRC = resolve_src(GEN_SRC)
assert os.path.isdir(SRC), f'GEN_SRC 없음: {SRC}'

all_imgs = []
for root, _, files in os.walk(SRC):
    if '__MACOSX' in root:
        continue
    for fn in files:
        if fn.startswith('._') or fn.startswith('.'):
            continue
        if os.path.splitext(fn)[1].lower() in IMG_EXTS:
            all_imgs.append(os.path.join(root, fn))
all_imgs = sorted(set(all_imgs))
assert all_imgs, f'이미지 0개: {SRC} (Roboflow export 를 GEN_SRC 로 올렸는지 확인)'

items, names_seen, n_unlabeled, per_dir = [], set(), 0, {}
for p in all_imgs:
    lp = label_for(p, SRC)
    pref = prefix_for(p, SRC)
    out_name = pref + os.path.basename(p)
    # 접두어에도 불구하고 충돌하면 카운터
    base, ext = os.path.splitext(out_name)
    k = 1
    while out_name in names_seen:
        out_name = f'{base}__{k}{ext}'; k += 1
    names_seen.add(out_name)
    d = os.path.relpath(os.path.dirname(p), SRC)
    per_dir[d] = per_dir.get(d, 0) + 1
    if lp is None:
        n_unlabeled += 1
        if not KEEP_UNLABELED:
            continue      # fire 이미지를 배경으로 가르치는 것 방지 — 기본 제외
    items.append((out_name, p, lp))

print(f'\n수집: 이미지 {len(all_imgs)}장 · 라벨없음 {n_unlabeled}장'
      + ('(음성으로 포함)' if KEEP_UNLABELED else '(제외)')
      + f' · 사용 {len(items)}장')
for d in sorted(per_dir):
    print(f'   {d:<40} {per_dir[d]:>4}장')
if n_unlabeled and not KEEP_UNLABELED:
    print(f'⚠️ 라벨 없는 {n_unlabeled}장 제외됨 — Roboflow 박스 미완이면 그 이미지들 마저 라벨 후 재실행.')
assert items, '사용 가능한(라벨 있는) 이미지 0장 — Roboflow 박스/​export 확인.'


# ---------------------------------------------------------------------------
# 파괴삭제 가드(script-safety): 바구니 이름에 'gen_synth' 포함일 때만 rmtree.
# ---------------------------------------------------------------------------
assert 'gen_synth' in os.path.basename(GEN_OUT.rstrip('/\\')), f'안전가드: 예상치 못한 GEN_OUT={GEN_OUT}'
shutil.rmtree(GEN_OUT, ignore_errors=True)

# 분할(시드)
rng.shuffle(items)
n_val = max(1, round(len(items) * VALFRAC)) if len(items) > 1 else 0
split_of = {}
for i, it in enumerate(items):
    split_of[it[0]] = 'val' if i < n_val else 'train'

for s in ('train', 'val'):
    os.makedirs(f'{GEN_OUT}/{s}/images', exist_ok=True)
    os.makedirs(f'{GEN_OUT}/{s}/labels', exist_ok=True)

qc, n_pos, n_neg, n_box, box_per = [], 0, 0, 0, []
for out_name, img_path, lp in items:
    s = split_of[out_name]
    stem = os.path.splitext(out_name)[0]
    # 이미지 복사(형식 유지·재인코딩 없음)
    dst_img = f'{GEN_OUT}/{s}/images/{out_name}'
    shutil.copy(img_path, dst_img)
    # 라벨: 클래스 전부 0 으로 강제해 재작성
    boxes = read_yolo_label(lp) if lp else []
    with open(f'{GEN_OUT}/{s}/labels/{stem}.txt', 'w') as f:
        for (cx, cy, w, h) in boxes:
            f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
    if boxes:
        n_pos += 1; n_box += len(boxes); box_per.append(len(boxes))
    else:
        n_neg += 1
    # QC 후보(양성만·최대 9)
    if boxes and len(qc) < 9:
        qc.append((s, dst_img, boxes))

open(f'{GEN_OUT}/data.yaml', 'w').write(
    f"path: {GEN_OUT}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['fire']\n")

json.dump({'src': GEN_SRC, 'seed': SEED, 'valfrac': VALFRAC,
           'n_images_seen': len(all_imgs), 'n_unlabeled': n_unlabeled,
           'keep_unlabeled': KEEP_UNLABELED, 'n_used': len(items),
           'n_pos': n_pos, 'n_neg': n_neg, 'n_box': n_box,
           'n_train': sum(1 for v in split_of.values() if v == 'train'),
           'n_val': sum(1 for v in split_of.values() if v == 'val'),
           'per_dir': per_dir},
          open(f'{GEN_OUT}/manifest_gen.json', 'w'), ensure_ascii=False, indent=1)

print('\n' + '=' * 70); print('요약 — gen 데이터셋'); print('=' * 70)
print(f'  train {sum(1 for v in split_of.values() if v=="train")} · val {sum(1 for v in split_of.values() if v=="val")}')
print(f'  양성 {n_pos}장 · 배경 {n_neg}장 · 박스 {n_box}개'
      + (f' · 장당 중앙 {int(np.median(box_per))}' if box_per else ''))
print(f'  → {GEN_OUT}/data.yaml · manifest_gen.json')


# ---------------------------------------------------------------------------
# QC 시트 — Roboflow 박스가 좌표/포맷 오류 없이 얹혔는지 육안 게이트(빌드 오염 방지)
# ---------------------------------------------------------------------------
if qc:
    CW = 360; cols = 3; rows = (len(qc) + cols - 1) // cols
    im0 = Image.open(qc[0][1]); w0, h0 = im0.size; ch = round(CW * h0 / w0)
    sheet = Image.new('RGB', (cols * CW, rows * (ch + 26)), (16, 16, 16))
    drw = ImageDraw.Draw(sheet)
    for j, (s, path, boxes) in enumerate(qc):
        im = Image.open(path).convert('RGB'); W, H = im.size
        d = ImageDraw.Draw(im)
        for (cx, cy, w, h) in boxes:                      # YOLO norm → 픽셀
            x0 = (cx - w / 2) * W; y0 = (cy - h / 2) * H
            x1 = (cx + w / 2) * W; y1 = (cy + h / 2) * H
            d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        rr, cc = divmod(j, cols); y = rr * (ch + 26)
        drw.text((cc * CW + 6, y + 3), f'gen {s} +box', fill=(0, 255, 0), font=F)
        sheet.paste(im.resize((CW, ch)), (cc * CW, y + 26))
    sheet.save(f'{GEN_OUT}/_check_gen.jpg', quality=88)
    print(f'\n확인용 시트 -> {GEN_OUT}/_check_gen.jpg  (녹색 박스가 불꽃에 정합하는지 육안 확인)')
    print('  박스가 어긋나면 Roboflow export 좌표계/클래스 확인 후 재빌드.')


# ---------------------------------------------------------------------------
# (2) gen-only 학습 — 커리큘럼 BASE. runs_if/gen_synth/best.pt (Drive)
# ---------------------------------------------------------------------------
if TRAIN:
    try:
        from ultralytics import YOLO
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
        from ultralytics import YOLO
    print('\n' + '=' * 70)
    print(f'gen-only 학습 — yolov8s · {GEN_EPOCHS}ep → runs_if/gen_synth')
    print('=' * 70)
    YOLO('yolov8s.pt').train(data=f'{GEN_OUT}/data.yaml', epochs=GEN_EPOCHS, imgsz=640,
                             patience=15, project=f'{FIRE}/runs_if', name='gen_synth',
                             exist_ok=True, verbose=False, plots=False)
    best = f'{FIRE}/runs_if/gen_synth/weights/best.pt'
    print(f'\n-> gen-only 앵커: {best}')
    print('다음(별도 셀) — 커리큘럼:')
    print("  os.environ['BASE_YOLO']='" + best + "'; os.environ['BASE_TAG']='gencurr'")
    print("  for k in ('HARDNEG','DFIRE_DIR','EDGE_MODE'): os.environ.pop(k, None)")
    print("  %run -i /content/kitchen-fire-noise-poc/scripts/colab_indoorfire_split_audit.py")
    print("  → real_only_grouped_gencurr · 이후 colab_realtest_eval.py 로 2gencurr vs 2g 대조")
else:
    print('\nTRAIN=0 → 데이터셋만 생성. QC(_check_gen.jpg) 확인 후 TRAIN=1 재실행하거나 '
          'split_audit BASE_YOLO 로 직접.')
