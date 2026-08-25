# ===== 셀 2b: 빌드된 양성(fire) 프레임 육안 검증 · Colab =====
#
# 목적: colab_build_firetest.py(빌드 모드)가 만든 oilfire_realtest/fire/*.jpg 를
#       장면(sc##) 단위 몽타주로 펼쳐, "불 아닌 프레임"이 섞였는지 눈으로 확인한다.
#       RANGES 가 ±4s 근사라 불 시작 전/꺼진 뒤 프레임이 혼입될 수 있음 → recall 과소평가 위험.
#       비화염이 많이 섞인 장면은 그 RANGES 만 좁혀 재빌드(build 재실행) 후 측정.
#
# 사용:
#   os.environ['OUT_DIR']='/content/oilfire_realtest'   # build 와 동일 경로
#   %run scripts/colab_inspect_firetest_pos.py
#   # → INSP_DIR(기본 /content/inspect)/firetest_pos_<sc##>.jpg 몽타주. 채팅에 첨부해 확인.
#
# 프레임 파일명 규칙(build): {sid}_{t:06.1f}s.jpg  (예: sc03_0012.0s.jpg) — 라벨에 초 표시.

import os, glob, re
from PIL import Image, ImageDraw, ImageFont

OUT  = os.environ.get('OUT_DIR', '/content/oilfire_realtest')
INSP = os.environ.get('INSP_DIR', '/content/inspect')
FIREDIR = f'{OUT}/fire'
os.makedirs(INSP, exist_ok=True)

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except Exception:
    F = ImageFont.load_default()

imgs = sorted(glob.glob(f'{FIREDIR}/*.jpg'))
assert imgs, f'양성 프레임 없음: {FIREDIR} — colab_build_firetest.py(빌드 모드) 먼저 실행.'

# 장면(sc##)별 그룹
scenes = {}
for p in imgs:
    b = os.path.basename(p)
    sid = b.split('_', 1)[0]          # sc03
    scenes.setdefault(sid, []).append(p)

# manifest 로 원본 파일명(어떤 영상인지) 매핑 — 있으면 라벨에 붙임
import json
mani = {}
mp = f'{OUT}/realtest_manifest.json'
if os.path.exists(mp):
    try:
        mani = json.load(open(mp)).get('accepted', {})
    except Exception:
        mani = {}

print(f'양성 검증 몽타주 — {len(imgs)}프레임 / {len(scenes)}장면\n')
for sid in sorted(scenes):
    fs = sorted(scenes[sid])
    cols = 8; cw = 200
    im0 = Image.open(fs[0]); ch = round(cw * im0.height / im0.width)
    rows = (len(fs) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * (ch + 18) + 20), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    src = mani.get(sid, '')[:60]
    dr.text((3, 2), f'{sid}  ({len(fs)} pos)   {src}', fill=(120, 220, 255), font=F)
    for j, p in enumerate(fs):
        x = Image.open(p).convert('RGB').resize((cw, ch)); c, r = j % cols, j // cols
        sh.paste(x, (c * cw, 20 + r * (ch + 18)))
        m = re.search(r'_(\d+\.\d)s', os.path.basename(p))
        lab = f'{float(m.group(1)):.0f}s' if m else os.path.basename(p)[:8]
        dr.text((c * cw + 2, 20 + r * (ch + 18) + 1), lab, fill=(255, 210, 0), font=F)
    out = f'{INSP}/firetest_pos_{sid}.jpg'
    sh.save(out, quality=84)
    print(f'  {sid:<6} {len(fs):>3} pos -> {out}   {src}')

print('\n→ 각 몽타주에서 "불 없음/꺼짐" 프레임이 많이 섞인 장면 확인.')
print('  섞였으면: RANGES 에서 그 장면 구간만 좁힌 뒤 colab_build_firetest.py 재실행 → 다시 이 셀 → 측정.')
print('  거의 다 불이면 통과 → colab_realtest_eval.py 로 측정 진행.')
