# ===== 7단계 보조: 실제화재 영상 구간 주석용 컨택트 시트 (Colab, CPU) =====
#
# real_fire.json 의 fire_shots·nofire_shots 를 손으로 채우기 전에, 각 영상을 일정
# 간격(STRIDE_SEC)으로 썸네일 격자로 펼치고 **각 썸네일에 타임스탬프(mm:ss)를 박아**
# 준다. 스크럽 없이 한눈에 "불꽃이 몇 초에 나타나 몇 초까지 있는지" 를 대략 집는다.
#
#   쓰는 법(정확도 우선): 이 시트(B)로 **대략 위치**를 잡고 → Drive 웹 플레이어(A)로
#   경계 초를 다듬는다. 둘이 어긋나면 = 초 적기 실수이거나 간헐적 불꽃.
#
# GPU 불필요. ultralytics 불필요(ffmpeg + PIL 만). **별도 CPU 런타임에서 돌리면
# 학습 GPU 런타임과 무관.** 각 썸네일은 그 초로 정확히 seek 해서 뽑으므로 라벨=실제 초.
#
# 결과: {SHEETS_OUT}/{key}_sheetN.jpg 로 저장 + 노트북에 인라인 표시.

import os, glob, json, shutil, subprocess, unicodedata, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
REPO       = '/content/kitchen-fire-noise-poc/scripts'
FIRE       = '/content/drive/MyDrive/fire_frames'
SHEETS_OUT = f'{FIRE}/realfire_sheets'
STRIDE_SEC = 2.0          # 썸네일 간격(초) — 촘촘히 보려면 1.0
COLS       = 6            # 격자 열 수
THUMB_W    = 300          # 썸네일 폭(px)
ROWS_PER_SHEET = 12       # 시트당 최대 행(넘으면 여러 장으로 나눔)
ONLY       = None         # 특정 key 만: 예 ['jikken_douga','simulation'] · None=전부

drive.mount('/content/drive')
inv = json.load(open(f'{REPO}/real_fire.json', encoding='utf-8'))
SRC_DIR = inv['src_dir']
allf = glob.glob(f'{SRC_DIR}/*')


def norm(s):
    return unicodedata.normalize('NFC', s)


def match_file(name):
    hit = [p for p in allf if norm(name) in norm(os.path.basename(p))]
    return hit[0] if len(hit) == 1 else (None if not hit else hit)


def duration(src):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'json', src], capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return None


def grab(src, t):
    """정확히 t 초로 seek 해서 프레임 1장 → PIL(RGB). 실패 시 None."""
    tmp = '/content/_thumb.jpg'
    if os.path.exists(tmp):
        os.remove(tmp)
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', src,
                    '-frames:v', '1', '-q:v', '3', tmp], check=False)
    if not os.path.exists(tmp):
        return None
    im = Image.open(tmp).convert('RGB')
    return im


try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except Exception:
    F = ImageFont.load_default()


def label(im, txt):
    """좌상단에 mm:ss — 검은 외곽선 + 노란 글자(격자에서도 읽힘)."""
    d = ImageDraw.Draw(im)
    for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 0)]:
        fill = (0, 0, 0) if (dx or dy) else (255, 230, 0)
        d.text((6 + dx, 4 + dy), txt, fill=fill, font=F)
    return im


def mmss(t):
    return f'{int(t) // 60:d}:{int(t) % 60:02d}'


# ---------------------------------------------------------------------------
# 영상별 시트 생성
# ---------------------------------------------------------------------------
os.makedirs(SHEETS_OUT, exist_ok=True)
sources = [s for s in inv['sources'] if (ONLY is None or s['key'] in ONLY)]

print('=' * 70)
print(f'컨택트 시트 — {STRIDE_SEC}초 간격 · {COLS}열 · src={SRC_DIR}')
print('=' * 70)

try:
    from IPython.display import display
    inline = True
except Exception:
    inline = False

for s in sources:
    src = match_file(s['file'])
    if src is None:
        print(f'  [건너뜀] {s["key"]}: "{s["file"]}" 매칭 0개'); continue
    if isinstance(src, list):
        print(f'  [건너뜀] {s["key"]}: 매칭 {len(src)}개(모호) — 파일명을 더 길게'); continue

    dur = duration(src)
    if not dur:
        print(f'  [건너뜀] {s["key"]}: 길이 확인 실패'); continue
    ts = [round(t, 1) for t in np.arange(0, dur, STRIDE_SEC)]
    print(f'\n  {s["key"]:<14} 길이 {mmss(dur)} · 썸네일 {len(ts)}장 뽑는 중 ...')

    tiles = []
    for t in ts:
        im = grab(src, t)
        if im is None:
            continue
        w, h = im.size
        im = im.resize((THUMB_W, round(THUMB_W * h / w)))
        tiles.append((t, label(im, mmss(t))))
    if not tiles:
        print(f'    프레임 못 뽑음 — 코덱/경로 확인'); continue

    tw, th = tiles[0][1].size
    per = COLS * ROWS_PER_SHEET
    n_sheets = math.ceil(len(tiles) / per)
    saved = []
    for k in range(n_sheets):
        chunk = tiles[k * per:(k + 1) * per]
        rows = math.ceil(len(chunk) / COLS)
        sheet = Image.new('RGB', (COLS * tw, rows * (th + 4)), (16, 16, 16))
        for j, (t, im) in enumerate(chunk):
            r, c = divmod(j, COLS)
            sheet.paste(im, (c * tw, r * (th + 4)))
        suffix = '' if n_sheets == 1 else f'_{k+1}'
        outp = f'{SHEETS_OUT}/{s["key"]}{suffix}.jpg'
        sheet.save(outp, quality=85)
        saved.append(outp)
        if inline:
            lo, hi = mmss(chunk[0][0]), mmss(chunk[-1][0])
            print(f'    [{s["key"]}{suffix}]  {lo} ~ {hi}')
            display(sheet)
    print(f'    -> ' + ' · '.join(saved))

print('\n' + '=' * 70)
print('읽는 법: 각 썸네일 좌상단이 그 프레임의 초(mm:ss).')
print('  불꽃이 뚜렷한 첫/끝 썸네일 → fire_shots [시작,끝](초).')
print('  발화 전(불꽃 없음, 연기·김만) 구간 → nofire_shots.')
print('  자막·타이머 띠가 불꽃색이면 crop=[위,아래] 비율로 잘라낼 것.')
print('  이 시트로 대략 잡고 → Drive 웹 플레이어로 경계 초를 다듬어 최종 확정.')
