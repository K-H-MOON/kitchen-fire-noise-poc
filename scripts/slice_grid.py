# ===== 1×2(가로) CCTV 그리드 이미지 슬라이서 (로컬 · 라벨링 전단계) =====
#
# 생성형(#6) 워크플로우: Nano Banana Pro 가 한 파일에 2장면(좌/우 카메라)을 담은
#   1×2 그리드를 저장 → 이 스크립트가 각 패널을 개별 이미지로 잘라 라벨링 대상으로 만든다.
#   그리드 원본 그대로 학습 금지(가운데 seam 아티팩트·스케일 불일치) → 반드시 슬라이스 후 학습.
#
# 동작(비파괴): 검은 여백(상하 레터박스)·가운데 세로 gutter 를 밝기로 자동 감지 →
#   좌/우 패널을 content 바운딩박스로 크롭 → OUT_DIR 에 <stem>_L.jpg · <stem>_R.jpg 저장.
#   입력 파일은 절대 삭제/수정하지 않음. OUT_DIR 만 새로 씀.
#
# 환경변수:
#   GRID_DIR : 그리드 원본 폴더(기본 = 아래 DEFAULT_GRID)
#   OUT_DIR  : 슬라이스 출력 폴더(기본 = GRID_DIR/../sliced)
#   DARK     : '검다' 판정 밝기 임계(0~255, 기본 45)
#   ROWFRAC  : 행이 레터박스로 판정되는 최소 dark 비율(기본 0.85)
#   COLFRAC  : 열이 여백/gutter 로 판정되는 최소 dark 비율(기본 0.85)
#   NCOLS    : 가로 분할 수(기본 2). 3 이상도 gutter 다중감지로 지원(6분할 등).
#   MINPANEL : 패널 최소 폭(px, 기본 120) — 오검출 gutter 무시용.
#
# 결과 요약과 감지 박스를 출력. 잘림이 이상하면 DARK/ROWFRAC/COLFRAC 조정.

import os, glob
import numpy as np
from PIL import Image

DEFAULT_GRID = r"C:\Users\jhmoo\OneDrive\바탕 화면\생성 이미지\학교급식실 · 웍 · 중간불 · 천장 내려보기\1x2 그리드"
GRID_DIR = os.environ.get('GRID_DIR', DEFAULT_GRID)
OUT_DIR  = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(GRID_DIR.rstrip('/\\')), 'sliced'))
DARK     = int(os.environ.get('DARK', '45'))
ROWFRAC  = float(os.environ.get('ROWFRAC', '0.85'))
COLFRAC  = float(os.environ.get('COLFRAC', '0.85'))
NCOLS    = int(os.environ.get('NCOLS', '2'))
MINPANEL = int(os.environ.get('MINPANEL', '120'))

EXTS = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')


def content_bounds(dark_1d, frac):
    """dark 비율이 frac 미만인 인덱스의 [min, max+1] (content 범위·바깥 여백 트림용)."""
    idx = np.where(dark_1d < frac)[0]
    if idx.size == 0:
        return None
    return int(idx.min()), int(idx.max()) + 1


def largest_run(dark_1d, frac):
    """dark 비율 < frac 인 '가장 긴 연속 구간' [start, end).
    상하 레터박스 + 분리된 캡션 띠(CAMERA 1 ...)를 함께 배제하고 주방 본문만 잡음."""
    content = dark_1d < frac
    best_s = best_e = 0; best_len = 0
    s = None
    for i, c in enumerate(content):
        if c and s is None:
            s = i
        elif not c and s is not None:
            if i - s > best_len:
                best_len, best_s, best_e = i - s, s, i
            s = None
    if s is not None and len(content) - s > best_len:
        best_len, best_s, best_e = len(content) - s, s, len(content)
    return (best_s, best_e) if best_len else None


def find_gutters(col_dark, x0, x1, ncols):
    """content 열 범위 [x0,x1) 안에서 ncols-1 개의 세로 gutter(어두운 띠) 중심을 찾는다."""
    if ncols <= 1:
        return []
    seg = x1 - x0
    gutters = []
    # 각 예상 분할 경계 부근(등간격)에서 가장 어두운 열 run 을 찾음
    for k in range(1, ncols):
        center = x0 + int(seg * k / ncols)
        lo, hi = max(x0, center - seg // (2 * ncols)), min(x1, center + seg // (2 * ncols))
        band = col_dark[lo:hi]
        cand = np.where(band >= COLFRAC)[0]
        if cand.size:
            g0 = lo + int(cand.min()); g1 = lo + int(cand.max()) + 1
            gutters.append((g0, g1))
        else:  # gutter 없으면 등간격 지점에서 0폭 분할
            gutters.append((center, center))
    return gutters


def slice_one(path):
    im = Image.open(path).convert('RGB')
    g = np.asarray(im.convert('L'), dtype=np.uint8)
    H, W = g.shape
    dark = g < DARK

    rb = largest_run(dark.mean(axis=1), ROWFRAC)         # 주방 본문 = 가장 긴 연속 content 행 구간
    if rb is None:
        return []
    y0, y1 = rb
    col_dark = dark[y0:y1].mean(axis=0)                  # content 행에서 열 프로파일
    cb = content_bounds(col_dark, COLFRAC)               # 좌우 바깥 여백 트림
    if cb is None:
        return []
    x0, x1 = cb

    gutters = find_gutters(col_dark, x0, x1, NCOLS)
    # 패널 경계 = [x0, gutter0.start], [gutter0.end, gutter1.start], ...
    xs = [x0]
    for (ga, gb) in gutters:
        xs.append(ga); xs.append(gb)
    xs.append(x1)
    panels = [(xs[i], xs[i + 1]) for i in range(0, len(xs), 2)]

    boxes = []
    for (px0, px1) in panels:
        if px1 - px0 >= MINPANEL:
            boxes.append((px0, y0, px1, y1))
    return [(im.crop(b), b) for b in boxes]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = []
    for e in EXTS:
        files += glob.glob(os.path.join(GRID_DIR, e))
    files = sorted(set(files))
    if not files:
        print(f'[없음] 그리드 파일 0개: {GRID_DIR}')
        return
    print(f'그리드 {len(files)}개 · NCOLS={NCOLS} · DARK={DARK} → 출력 {OUT_DIR}\n')
    suffix = ['_L', '_R'] if NCOLS == 2 else [f'_c{i}' for i in range(NCOLS)]
    total = 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        crops = slice_one(path)
        for i, (crop, box) in enumerate(crops):
            sfx = suffix[i] if i < len(suffix) else f'_c{i}'
            out = os.path.join(OUT_DIR, f'{stem}{sfx}.jpg')
            crop.save(out, quality=95)
            total += 1
        dims = ' · '.join(f'{c.width}x{c.height}@{b}' for c, b in crops)
        print(f'{stem[:16]:<18} → {len(crops)}패널  {dims}')
    print(f'\n완료: 패널 {total}장 → {OUT_DIR}')
    print('육안 확인 후 이상 없으면 이 폴더를 Roboflow 라벨링에 사용.')


if __name__ == '__main__':
    main()
