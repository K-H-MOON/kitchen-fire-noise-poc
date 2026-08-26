# ===== presrc(발화전 음성) 프레임 육안 검증 — 오라벨(불) 찾기 · Colab =====
#
# 목적: colab_build_firetest.py 가 만든 oilfire_realtest/nofire_presrc/*.jpg 를
#       장면(sc##) 단위 몽타주로 펼쳐, "발화 전(불 없음)"으로 라벨됐지만 실제로
#       불이 타는 프레임이 섞였는지 눈으로 확인한다(배찬우 팀원 검수 §3 sc10 오라벨).
#       편집 뉴스 클립은 앞부분부터 이미 화재라 "원본 앞=발화전" 가정이 깨질 수 있음.
#       → 불 섞인 장면은 eval 에서 PRESRC_DROP 으로 제외(파일 삭제 아님, eval 시점 필터).
#
# 사용:
#   os.environ['OUT_DIR']='/content/oilfire_realtest'   # build 와 동일 경로(읽기 전용)
#   %run scripts/colab_inspect_presrc.py
#   # → INSP_DIR(기본 /content/inspect_presrc)/presrc_<sc##>.jpg 몽타주. 채팅 첨부해 확인.
#   # 불 섞인 sc## 를 골라: os.environ['PRESRC_DROP']='sc10,sc12' 후 colab_realtest_eval.py 재실행.
#
# ★ 안전: OUT_DIR 은 읽기만(테스트셋) · 출력은 INSP_DIR 전용 · rmtree 없음(makedirs exist_ok).
# 프레임 파일명 규칙(build): {sid}_{t:06.1f}s.jpg (예: sc10_0000.5s.jpg).

import os, glob, re, json
from PIL import Image, ImageDraw, ImageFont

OUT  = os.environ.get('OUT_DIR', '/content/oilfire_realtest')   # 읽기 전용(테스트셋 경로)
INSP = os.environ.get('INSP_DIR', '/content/inspect_presrc')    # 출력 전용(OUT_DIR 와 분리)
PREDIR = f'{OUT}/nofire_presrc'
os.makedirs(INSP, exist_ok=True)                                # rmtree 안 함 — 기존 파일 보존

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except Exception:
    F = ImageFont.load_default()

imgs = sorted(glob.glob(f'{PREDIR}/*.jpg'))
assert imgs, f'발화전 음성 없음: {PREDIR} — colab_build_firetest.py(빌드 모드) 먼저 실행.'

# 장면(sc##)별 그룹
scenes = {}
for p in imgs:
    sid = os.path.basename(p).split('_', 1)[0]
    scenes.setdefault(sid, []).append(p)

# manifest 로 원본 파일명 매핑(있으면 라벨에 붙임)
mani = {}
mp = f'{OUT}/realtest_manifest.json'
if os.path.exists(mp):
    try:
        mani = json.load(open(mp)).get('accepted', {})
    except Exception:
        mani = {}

print(f'발화전 음성 검증 몽타주 — {len(imgs)}프레임 / {len(scenes)}장면\n')
for sid in sorted(scenes):
    fs = sorted(scenes[sid])
    cols = 8; cw = 200
    im0 = Image.open(fs[0]); ch = round(cw * im0.height / im0.width)
    rows = (len(fs) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * (ch + 18) + 20), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    src = mani.get(sid, '')[:60]
    dr.text((3, 2), f'{sid}  ({len(fs)} presrc)   {src}', fill=(120, 220, 255), font=F)
    for j, p in enumerate(fs):
        x = Image.open(p).convert('RGB').resize((cw, ch)); c, r = j % cols, j // cols
        sh.paste(x, (c * cw, 20 + r * (ch + 18)))
        m = re.search(r'_(\d+\.\d)s', os.path.basename(p))
        lab = f'{float(m.group(1)):.0f}s' if m else os.path.basename(p)[:8]
        dr.text((c * cw + 2, 20 + r * (ch + 18) + 1), lab, fill=(255, 210, 0), font=F)
    out = f'{INSP}/presrc_{sid}.jpg'
    sh.save(out, quality=84)
    print(f'  {sid:<6} {len(fs):>3} presrc -> {out}   {src}')

print('\n→ "발화 전(불 없음)" 인데 실제 불꽃이 보이는 장면 확인(=오라벨).')
print('  오라벨 있으면: os.environ["PRESRC_DROP"]="sc10,..." 후 colab_realtest_eval.py 재실행 → fpr_발화전 정직화.')
print('  경계: fpr_발화전은 부차지표(배포 지표는 fpr_급식실) — 이 정리는 정직성용, 배포 결론 불변.')
