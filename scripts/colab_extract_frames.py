# ===== 조리 영상 → 배경 프레임 추출 (Colab) =====
#
# 이 단계는 **불꽃이 없는 배경 프레임**만 만든다. 불꽃 합성은 다음 단계다.
# split.json 의 사이트 배정을 그대로 따라 train/val/test 폴더로 나눈다 —
# 같은 주방이 두 세트에 섞이지 않는다.
#
# 확정한 샘플링 규칙 (docs/PREREGISTER.md 와 같음):
#   - 추출 레이트  1 fps   (조리는 느려서 초당 1장이면 충분)
#   - 중복 제거    직전 채택 프레임과 유사도 > DEDUP_SIM 이면 버림
#   - 사이트당 상한 SITE_CAP 장  (한 주방이 데이터를 독식하지 못하게)
#   - 영상당 하위 상한 = ceil(SITE_CAP / 그 사이트의 영상 수)
#     ('full' 한 영상이 사이트 몫을 독식하지 못하게)
#
# 상한은 '할당량'이 아니라 '상한'이다 — 짧은 영상은 나오는 만큼만 낸다.
# 실제 사이트별 수율은 끝에 표로 찍는다. 이 값을 보고 상한을 조정하면 된다.

import os, sys, json, math, glob, unicodedata
from google.colab import drive

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
# 원본 영상이 있는 폴더 (Drive 마운트 경로).
#   주의 — 원본 영상은 hankookro@gmail.com 소유라 blessmoonkh 계정에서는
#   '공유 문서함(Shared with me)'에 있다. Colab 의 drive.mount 는 '내 드라이브'만
#   마운트하고 공유 문서함은 마운트하지 않는다. 그래서 둘 중 하나를 먼저 해야 한다:
#     (A) 공유 문서함의 그 폴더에서 우클릭 → '바로가기 추가' → 내 드라이브에 넣기
#     (B) 폴더를 내 드라이브로 복사
#   그런 다음 아래 SRC_DIR 을 그 위치로 맞춘다.
SRC_DIR   = '/content/drive/MyDrive/kitchen_src'      # 원본 영상 폴더
OUT_ROOT  = '/content/drive/MyDrive/fire_frames'      # 결과 저장 루트
REPO      = '/content/kitchen-fire-noise-poc/scripts' # videos.json · split.json 이 있는 곳

FPS_EXTRACT = 1.0     # 초당 뽑을 장수
DEDUP_SIM   = 0.90    # 직전 채택본과 유사도 이 값 초과면 버림 (0~1)
SITE_CAP    = 400     # 사이트당 최대 프레임
JPG_QUALITY = 95

drive.mount('/content/drive')

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 인벤토리·분할 적재
# ---------------------------------------------------------------------------
inv   = json.load(open(f'{REPO}/videos.json', encoding='utf-8'))
split = json.load(open(f'{REPO}/split.json',  encoding='utf-8'))

# 파일명 -> (site, dish, set) 매핑. 파일명 정규화(NFC)로 자모 분리 문제를 피한다.
def norm(s):
    return unicodedata.normalize('NFC', s)

meta = {}
for v in inv['videos']:
    meta[norm(v['file'])] = {'site': v['site'], 'dish': v['dish'], 'cctv': v['cctv']}
setof = {}
for st in ('train', 'val', 'test'):
    for f in split['videos'][st]:
        setof[norm(f)] = st

# 사이트별 영상 수 (영상당 하위 상한 계산용)
site_nvideos = {}
for v in inv['videos']:
    site_nvideos[v['site']] = site_nvideos.get(v['site'], 0) + 1

# ---------------------------------------------------------------------------
# SRC_DIR 안의 실제 파일을 인벤토리와 대조
# ---------------------------------------------------------------------------
exts = ('.mp4', '.MOV', '.mov', '.avi', '.mkv', '.MP4', '.AVI', '.MKV')
found = []
for p in sorted(glob.glob(f'{SRC_DIR}/*')):
    if p.endswith(exts):
        found.append(p)

found_names = {norm(os.path.basename(p)) for p in found}
inv_names   = set(meta)
missing = inv_names - found_names            # 인벤토리엔 있는데 폴더엔 없음
extra   = found_names - inv_names            # 폴더엔 있는데 인벤토리엔 없음

print('=' * 74)
print('원본 대조')
print('=' * 74)
print(f'  인벤토리 {len(inv_names)}개 · 폴더에서 찾음 {len(found_names)}개')
if missing:
    print(f'  **폴더에 없음 {len(missing)}개** — SRC_DIR 확인:')
    for m in sorted(missing): print(f'      {m}')
if extra:
    print(f'  인벤토리에 없는 파일 {len(extra)}개 (건너뜀):')
    for e in sorted(extra): print(f'      {e}')
if missing:
    raise SystemExit('원본이 다 있지 않음 — 위 목록을 채우고 다시 돌릴 것')


# ---------------------------------------------------------------------------
# 유사도 (중복 제거) — 32x32 회색조로 줄여 정규화 L1 로 잼
# ---------------------------------------------------------------------------
def signature(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g

def similarity(a, b):
    return 1.0 - float(np.mean(np.abs(a - b))) / 255.0


# ---------------------------------------------------------------------------
# 한 영상에서 프레임 뽑기 — 1fps 로 훑고, 중복 버리고, subcap 까지만
# ---------------------------------------------------------------------------
def extract_one(path, out_dir, subcap):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'    **열기 실패** {path}'); return 0, 0
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / FPS_EXTRACT)))   # 몇 프레임마다 한 장 볼지
    stem = os.path.splitext(os.path.basename(path))[0]

    idx, kept, seen = 0, 0, 0
    last_sig = None
    while kept < subcap:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            seen += 1
            sig = signature(frame)
            if last_sig is not None and similarity(sig, last_sig) > DEDUP_SIM:
                idx += 1
                continue                      # 직전과 너무 닮음 — 버림
            last_sig = sig
            fn = f'{stem}__{idx:07d}.jpg'
            cv2.imwrite(os.path.join(out_dir, fn),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
            kept += 1
        idx += 1
    cap.release()
    return kept, seen


# ---------------------------------------------------------------------------
# 사이트별로 돌리기 — subcap 은 사이트 영상 수로 나눔, 사이트 총량은 SITE_CAP 로 캡
# ---------------------------------------------------------------------------
os.makedirs(OUT_ROOT, exist_ok=True)
manifest = {'fps': FPS_EXTRACT, 'dedup_sim': DEDUP_SIM, 'site_cap': SITE_CAP, 'frames': []}
yields = {}   # site -> kept

# 파일을 사이트별로 묶기 (인벤토리 순서 유지)
by_site = {}
for v in inv['videos']:
    by_site.setdefault(v['site'], []).append(v['file'])

print('\n' + '=' * 74)
print('프레임 추출')
print('=' * 74)
for site, files in by_site.items():
    st = setof[norm(files[0])]
    subcap = math.ceil(SITE_CAP / site_nvideos[site])
    out_dir = f'{OUT_ROOT}/bg/{st}/{site}'
    os.makedirs(out_dir, exist_ok=True)
    site_kept = 0
    print(f'\n[{st}] {site}  (영상 {len(files)}개 · 영상당 상한 {subcap} · 사이트 상한 {SITE_CAP})')
    for f in files:
        if site_kept >= SITE_CAP:
            print(f'    사이트 상한 도달 — 나머지 영상 건너뜀'); break
        path = f'{SRC_DIR}/{f}'
        room = min(subcap, SITE_CAP - site_kept)
        kept, seen = extract_one(path, out_dir, room)
        site_kept += kept
        print(f'    {f:<34} 본 {seen:>4} · 채택 {kept:>4}')
        manifest['frames'].append({'set': st, 'site': site, 'dish': meta[norm(f)]['dish'],
                                   'cctv': meta[norm(f)]['cctv'], 'video': f, 'kept': kept})
    yields[site] = site_kept

# ---------------------------------------------------------------------------
# 수율 표 + 저장
# ---------------------------------------------------------------------------
print('\n' + '=' * 74)
print('사이트별 수율 (이 값을 보고 SITE_CAP 을 조정하면 됨)')
print('=' * 74)
tot = {'train': 0, 'val': 0, 'test': 0}
for site, files in by_site.items():
    st = setof[norm(files[0])]
    k = yields[site]
    tot[st] += k
    cap_hit = ' (상한)' if k >= SITE_CAP else ''
    print(f'  [{st:<5}] {site:<10} {k:>4}{cap_hit}')
print('-' * 40)
for st in ('train', 'val', 'test'):
    print(f'  {st:<7} 합계 {tot[st]:>5}')
print(f'  전체    합계 {sum(tot.values()):>5}')

json.dump(manifest, open(f'{OUT_ROOT}/manifest_bg.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\n-> 프레임: {OUT_ROOT}/bg/<set>/<site>/*.jpg')
print(f'-> 매니페스트: {OUT_ROOT}/manifest_bg.json')
print('\n다음 — 불꽃 소재 수집 후 합성 스크립트 (배경 위에 불꽃 + 자동 박스).')
