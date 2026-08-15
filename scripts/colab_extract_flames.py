# ===== 불꽃 소재 만들기 — 실제 화재 영상에서 불꽃 매트 추출 (Colab) =====
#
# 실제 유류화재 영상의 '불꽃이 보이는 구간'에서 프레임을 뽑고, 휘도·난색 기반으로
# 알파 매트를 만들어 RGBA 스프라이트(PNG)를 낸다. 이 스프라이트를 나중에 주방 배경
# 위에 합성한다. 합성할 때 알파 마스크가 곧 정답 박스가 된다.
#
# 왜 플레이트 차감이 아니라 색·휘도 매트인가 — 연기(colab_make_matte.py)는 배경에서
# 옅게 번지므로 '판(첫 프레임) 차감'이 맞았다. 불꽃은 스스로 빛나고 난색으로
# 포화되므로, 발화 전 깨끗한 판이 없어도 밝기+난색만으로 딸 수 있다. 판이 필요 없어
# 발화 이후 구간을 그대로 쓸 수 있다.
#
# 나오는 것 — RGBA 조각(PNG) + flame_matte_manifest.json + 확인용 시트 _check.jpg.
# **매트 파라미터(v0/core0/thr)는 출처마다 다르다.** 먼저 돌려서 _check.jpg 를 보고,
# 불꽃이 덜 따지거나 배경이 새면 flames.json 의 항목별 matte 로 덮어써서 다시 돌린다.
#
# Colab 새 노트북에 이 셀 하나. GPU 불필요.

import os, glob, json, shutil, subprocess, unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

REPO = '/content/kitchen-fire-noise-poc/scripts'   # flames.json · flame_split.json 위치
FPS  = 2                                            # 불꽃 구간에서 초당 뽑을 장수
OUT  = '/content/drive/MyDrive/fire_frames/flame_matte'

drive.mount('/content/drive')

inv = json.load(open(f'{REPO}/flames.json', encoding='utf-8'))
SRC_DIR = inv['src_dir']
MD = inv['matte_default']

# 풀 배정(있으면) — 스프라이트를 train/test 풀 폴더로 나눠 저장
pool_of = {}
fsplit = f'{REPO}/flame_split.json'
if os.path.exists(fsplit):
    fs = json.load(open(fsplit, encoding='utf-8'))
    for p in ('train', 'test'):
        for k in fs['pools'][p]:
            pool_of[k] = p
    print(f'풀 배정 읽음 — train {len(fs["pools"]["train"])} · test {len(fs["pools"]["test"])}')
else:
    print('[주의] flame_split.json 이 없음 — 전부 unassigned 로 저장. '
          'assign_flame_split.py 를 먼저 돌리는 것이 맞음')


def norm(s):
    return unicodedata.normalize('NFC', s)


# ---------------------------------------------------------------------------
# 불꽃 알파 — 밝고 난색이면 불꽃. 백열(전 채널 밝음) 코어는 따로 살림.
# ---------------------------------------------------------------------------
def flame_alpha(img, v0, core0, thr):
    r, b = img[..., 0], img[..., 2]
    v = img.max(2)                                   # 휘도
    warmth = np.clip((r - b) / 255.0, 0, 1)          # 오렌지·붉은 불꽃 (R 이 B 보다 큼)
    bright = np.clip((v - v0) / max(255.0 - v0, 1.0), 0, 1)
    core   = np.clip((img.min(2) - core0) / max(255.0 - core0, 1.0), 0, 1)  # 백열 코어
    a = np.clip(np.maximum(warmth * bright, core), 0, 1)
    a[a < thr] = 0
    return a


def frames_of(src, a, b, crop):
    tmp = '/content/_f'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    vf = f'fps={FPS}'
    if crop:                                          # (위, 아래) 잘라내기 비율
        vf = f'crop=iw:ih*{1 - crop[0] - crop[1]}:0:ih*{crop[0]},' + vf
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                    '-t', str(round(b - a + 1, 2)), '-vf', vf,
                    '-q:v', '2', f'{tmp}/%05d.jpg'], check=False)
    out = []
    for j, p in enumerate(sorted(glob.glob(f'{tmp}/*.jpg'))):
        sec = round(a + j / FPS, 1)
        if sec <= b + 1e-6:
            out.append((sec, np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)))
    return out


def dhash(a, size=8):
    x = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size + 1, size),
                   Image.LANCZOS), dtype=np.int16)
    return np.packbits((x[:, 1:] > x[:, :-1]).flatten())


def ham(p, q):
    return int(np.unpackbits(p ^ q).sum())


# ---------------------------------------------------------------------------
# 사전 검사 — shots 채워진 출처만. 예상 장수 = Σ (끝−시작) × FPS
# ---------------------------------------------------------------------------
ready = [s for s in inv['sources'] if s.get('shots')]
if not ready:
    raise SystemExit('flames.json 에 shots 가 채워진 출처가 없음 — 불꽃 구간을 먼저 확인할 것')
EXPECT = {s['key']: sum(int(round((b - a) * FPS)) for a, b in s['shots']) for s in ready}
print(f'\n대상 출처 {len(ready)}개 · 예상 장수 합 {sum(EXPECT.values())}장')
print(f'  {" · ".join(f"{k} {v}" for k, v in EXPECT.items())}')

drive.mount('/content/drive')
allf = glob.glob(f'{SRC_DIR}/*')
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except Exception:
    F = ImageFont.load_default()

print(f'\n{"출처":<14}{"풀":<11}{"장":>5}{"알파평균":>9}{"면적비":>8}{"서로다름":>9}')
print('-' * 60)

manifest, sheets, total, mismatch = {}, [], 0, []
for s in ready:
    key, frag, shots, crop = s['key'], s['file'], s['shots'], s.get('crop')
    mp = {**MD, **(s.get('matte') or {})}             # 항목별 matte 로 덮어씀
    v0, core0, thr = mp['v0'], mp['core0'], mp['thr']
    pool = pool_of.get(key, 'unassigned')

    hit = [p for p in allf if norm(frag) in norm(os.path.basename(p))]
    if len(hit) != 1:
        print(f'{key:<14}[건너뜀] "{frag}" 에 맞는 파일 {len(hit)}개'); continue
    src = hit[0]
    d = f'{OUT}/{pool}/{key}'; os.makedirs(d, exist_ok=True)

    recs, alphas = [], []
    for a, b in shots:
        for sec, img in frames_of(src, a, b, crop):
            al = flame_alpha(img, v0, core0, thr)
            ys, xs = np.nonzero(al)
            if len(ys) == 0:
                recs.append({'sec': sec, 'shot': [a, b], 'file': None, 'alpha_mean': 0.0,
                             'skip': '알파 0'}); continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            rgba = np.dstack([img[y0:y1, x0:x1], (al[y0:y1, x0:x1] * 255)]).astype(np.uint8)
            name = f'{key}_{sec:07.1f}.png'.replace(' ', '0')
            Image.fromarray(rgba, 'RGBA').save(f'{d}/{name}')
            alphas.append(dhash(al))
            recs.append({'sec': sec, 'shot': [a, b], 'file': name,
                         'alpha_mean': round(float(al.mean()), 4),
                         'bbox': [int(x0), int(y0), int(x1), int(y1)],
                         'area_ratio': round(float((x1 - x0) * (y1 - y0)) / al.size, 4)})
            if len(sheets) < 12 and abs(sec - (shots[0][0] + shots[0][1]) / 2) < 1.0:
                sheets.append((key, sec, img, al))

    n = sum(1 for r in recs if r.get('file'))
    total += n
    if n != EXPECT[key]:
        mismatch.append((key, EXPECT[key], n))
    uniq = 0
    if alphas:
        keep = [alphas[0]]
        for h in alphas[1:]:
            if min(ham(h, g) for g in keep) > 0:
                keep.append(h)
        uniq = len(keep)
    am = np.mean([r['alpha_mean'] for r in recs if r.get('file')]) if n else 0
    ar = np.mean([r['area_ratio'] for r in recs if r.get('file')]) if n else 0
    manifest[key] = {'file': os.path.basename(src), 'pool': pool, 'crop': crop,
                     'matte': {'v0': v0, 'core0': core0, 'thr': thr},
                     'shots': shots, 'frames': recs}
    print(f'{key:<14}{pool:<11}{n:>5}{am:>9.3f}{ar:>8.2f}{uniq:>9}')

json.dump(manifest, open(f'{OUT}/flame_matte_manifest.json', 'w'),
          ensure_ascii=False, indent=1)

print('-' * 60)
print(f'{"합":<14}{"":<11}{total:>5}장   -> {OUT}')
if mismatch:
    print('\n[확인 필요] 장수가 예상과 다름')
    for k, e, g in mismatch:
        print(f'  {k:<14}예상 {e} · 실제 {g} ({g - e:+d})')

# 확인용 시트 — 원본 / 알파 / 검은 배경 위 합성 미리보기
if sheets:
    CW = 360
    rows = []
    for key, sec, img, al in sheets[:12]:
        rgb = Image.fromarray(img.astype(np.uint8))
        am = Image.fromarray((al * 255).astype(np.uint8)).convert('RGB')
        comp = (img * al[..., None]).astype(np.uint8)          # 검은 배경 위 불꽃만
        rows.append([(f'{key} {sec:g}s', rgb), ('알파', am), ('합성', Image.fromarray(comp))])
    h0, w0 = np.asarray(rows[0][0][1]).shape[:2]
    ch = round(CW * h0 / w0)
    sh = Image.new('RGB', (3 * CW, len(rows) * (ch + 28)), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for r, row in enumerate(rows):
        for c, (lab, im) in enumerate(row):
            y = r * (ch + 28)
            dr.text((c * CW + 6, y + 4), lab, fill=(255, 190, 0), font=F)
            sh.paste(im.resize((CW, ch), Image.LANCZOS), (c * CW, y + 28))
    sh.save(f'{OUT}/_check.jpg', quality=88)
    print(f'\n확인용 시트 -> {OUT}/_check.jpg  ({len(rows)}줄)')
    print('  [원본 · 알파 · 합성] 셋을 보고 불꽃이 잘 따졌는지 판단.')
    print('  배경이 새면 v0/thr 를 올리고, 불꽃이 잘리면 내린다 (flames.json 의 항목별 matte).')

print('\n다음 — 배경 프레임(bg) 위에 이 불꽃 스프라이트를 합성 + 자동 박스 생성.')
