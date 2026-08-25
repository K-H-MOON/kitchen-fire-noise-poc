# ===== 0. 조리영상 주방 수 확인 — 영상당 대표 프레임 몽타주 · Colab =====
#
# 목적: 28 조리영상이 '몇 개 distinct 주방'에서 왔나를 눈으로 세기 위해,
#   영상당 1프레임(중간 지점)을 ck##·파일명 라벨로 펼친다.
#   → 여러 주방이면 leave-kitchen-out(다른 급식실 일반화) 가능 · ⑤ held-out 이 교차-주방인지도 판정.
#   ck## 순번은 colab_build_cook_negs.py / colab_diag_cook_fp.py 와 동일(진단 fpr·⑤ split 과 일치).
#
# env: COOK_DIR(조리영상) · INSP_DIR(몽타주 출력)

import os, glob, json, subprocess
from PIL import Image, ImageDraw, ImageFont

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DRIVE = '/content/drive/MyDrive'
COOK  = os.environ.get('COOK_DIR', f'{DRIVE}/조리 데이터 영상')
INSP  = os.environ.get('INSP_DIR', '/content/inspect')
os.makedirs(INSP, exist_ok=True)

cvids = sorted(p for e in ('mp4', 'mkv', 'mov', 'MOV', 'avi', 'MP4') for p in glob.glob(f'{COOK}/*.{e}'))
assert cvids, f'조리영상 없음: {COOK}'
ids = {f'ck{i:02d}': v for i, v in enumerate(cvids)}


def duration(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', p],
                       capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return 0.0


def grab(v, t, op):
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', v, '-frames:v', '1', '-q:v', '3', op],
                   check=False)
    return os.path.exists(op)


print(f'조리영상 {len(cvids)}개 (ck## ↔ 파일명):')
frames = []
for cid, v in ids.items():
    d = duration(v)
    op = f'{INSP}/_kitchen_{cid}.jpg'
    ok = grab(v, max(0.5, d * 0.5), op)               # 중간 지점 1프레임
    frames.append((cid, os.path.basename(v), op if ok else None))
    print(f'  {cid}  {os.path.basename(v)}')

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

valid = [(c, n, p) for c, n, p in frames if p]
cols = 4; cw = 380
im0 = Image.open(valid[0][2]); ch = round(cw * im0.height / im0.width)
rows = (len(valid) + cols - 1) // cols
sh = Image.new('RGB', (cols * cw, rows * (ch + 24) + 4), (16, 16, 16))
dr = ImageDraw.Draw(sh)
for j, (cid, name, p) in enumerate(valid):
    im = Image.open(p).convert('RGB').resize((cw, ch))
    r, c = divmod(j, cols); y = r * (ch + 24)
    sh.paste(im, (c * cw, y))
    dr.text((c * cw + 3, y + ch + 2), f'{cid}  {name[:34]}', fill=(120, 220, 255), font=F)
out = f'{INSP}/cook_kitchens.jpg'
sh.save(out, quality=86)
print(f'\n-> {out}')
print('  → 몽타주에서 distinct 주방(배경·레이아웃·카메라 시점) 몇 개인지 눈으로 세기.')
print('  ⑤ held-out = ck02,ck09,ck11,ck13,ck16,ck18,ck20,ck25 · train = 나머지.')
print('  held-out 주방이 train 과 다르면 = 교차-주방 일반화(강함), 같으면 = 동일주방 내(약함).')
