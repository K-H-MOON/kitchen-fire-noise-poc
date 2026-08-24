# ===== 하드네거티브 test 구성 (Colab) =====
#
# 목적: 정상 조리·김·수증기·끓는 냄비(불 없음) 영상에서 프레임을 뽑아
#       "김·수증기를 불로 오인하나(헛불)"를 재는 하드네거티브 test 를 만든다.
#
# 입력: Drive 루트의 스톡 조리 영상(불 없음) mp4 들(200MB↓ · 'Kitchen_Fire'·'yt_' 제외).
# 산출: fire_frames/oilfire_hardneg/nofire/*.jpg (하드네거티브)
#        + fire_frames/oilfire_hardneg/fire/*.jpg (sanity: 파일럿 화염 소량 — 모델이 불은 여전히 잡나 확인)
#        + fire_frames/inspect/verify_hardneg.jpg (전 프레임 몽타주 — 불꽃 혼입 최종 육안 검증용)
#
# 검증: 각 영상은 사전 16프레임 시트로 불꽃 없음 확인됨. 여기서 전 추출 프레임 몽타주를 다시 만들어
#       (특히 가스레인지 팬볶음) 불꽃 프레임이 없는지 최종 확인 후 평가로 넘어갈 것.
# 환경: STEP(추출 간격 초, 기본 1.5) · CAP(영상당 최대 프레임, 기본 12) · N_SANITY(기본 10)

import os, glob, shutil, subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = '/content/drive/MyDrive'
FIRE = f'{ROOT}/fire_frames'
OUTD = f'{FIRE}/oilfire_hardneg'
INSP = f'{FIRE}/inspect'
STEP = float(os.environ.get('STEP', '1.5'))
CAP  = int(os.environ.get('CAP', '12'))
N_SANITY = int(os.environ.get('N_SANITY', '10'))

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

if os.path.isdir(OUTD):
    shutil.rmtree(OUTD)
os.makedirs(f'{OUTD}/nofire', exist_ok=True)
os.makedirs(f'{OUTD}/fire', exist_ok=True)
os.makedirs(INSP, exist_ok=True)

def duration(p):
    import json
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'json', p], capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return 20.0

# --- 하드네거티브 프레임 추출 (루트 스톡 조리 영상) ---
# 제외: 화재 원본·yt_·부적합(주방/조리 아님). HARDNEG_EXCLUDE 로 추가 지정(쉼표구분 substring).
EXCLUDE = ['Kitchen_Fire', '7409223']  # 7409223 = 테킬라 정물(주방 아님)
EXCLUDE += [s for s in os.environ.get('HARDNEG_EXCLUDE', '').split(',') if s]
vids = [p for p in sorted(glob.glob(f'{ROOT}/*.mp4'))
        if os.path.getsize(p) < 200e6
        and not os.path.basename(p).startswith('yt_')
        and not any(x in os.path.basename(p) for x in EXCLUDE)]
print(f'하드네거티브 소스 영상 {len(vids)}개:')
for v in vids:
    print('  ', os.path.basename(v))

for v in vids:
    tag = os.path.splitext(os.path.basename(v))[0][:20]
    d = duration(v); t = STEP * 0.5; n = 0
    while t < d and n < CAP:
        op = f'{OUTD}/nofire/{tag}_{t:05.1f}s.jpg'
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', v,
                        '-frames:v', '1', '-q:v', '3', op], check=False)
        if os.path.exists(op):
            n += 1
        t += STEP
    print(f'  {tag}: {n} 프레임')

# --- sanity fire (파일럿 화염 소량 복사) ---
src = None
for cand in (f'{FIRE}/oilfire_pilot/fire', f'{FIRE}/oilfire_early/fire'):
    if glob.glob(f'{cand}/*.jpg'):
        src = cand; break
if src:
    fs = sorted(glob.glob(f'{src}/*.jpg'))
    pick = fs[::max(1, len(fs) // N_SANITY)][:N_SANITY]
    for p in pick:
        shutil.copy(p, f'{OUTD}/fire/{os.path.basename(p)}')
    print(f'sanity fire {len(pick)}장 복사 ({src})')
else:
    print('sanity fire 소스 없음 — recall 열은 무시')

nof = sorted(glob.glob(f'{OUTD}/nofire/*.jpg'))
print(f'\n하드네거티브 nofire {len(nof)} · sanity fire {len(glob.glob(f"{OUTD}/fire/*.jpg"))}')

# --- 전 프레임 검증 몽타주 (불꽃 혼입 최종 육안 확인) ---
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
except Exception:
    F = ImageFont.load_default()
cols, cw = 8, 200
im0 = Image.open(nof[0]); ch = round(cw * im0.height / im0.width)
rows = (len(nof) + cols - 1) // cols
sh = Image.new('RGB', (cols * cw, rows * (ch + 16)), (16, 16, 16)); dr = ImageDraw.Draw(sh)
for j, p in enumerate(nof):
    x = Image.open(p).convert('RGB').resize((cw, ch)); c, r = j % cols, j // cols; y = r * (ch + 16)
    dr.text((c * cw + 2, y + 1), os.path.basename(p)[:20], fill=(255, 120, 120), font=F)
    sh.paste(x, (c * cw, y + 16))
sh.save(f'{INSP}/verify_hardneg.jpg', quality=82)
print(f'-> 검증 몽타주 {INSP}/verify_hardneg.jpg  (불꽃 혼입 육안 확인 후 평가)')
print('평가: EVAL_SET=oilfire_hardneg 로 colab_oilfire_eval.py 실행')
