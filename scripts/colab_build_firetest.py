# ===== frame-level 실화재 test 구성 (A안 ④ 측정용) · Colab =====
#
# 목적: 채택 화재영상에서 '깨끗한 불 프레임'(사람이 지정한 시간구간)을 양성으로,
#       음성은 ① 급식실 조리 CCTV(배포-대표) + ② 발화 전 프레임(동일소스 하드네거)으로
#       frame-level test 를 구성한다. 박스 라벨 없음(현재 모델 재측정용).
#
# 2모드:
#   (A) 몽타주 모드 — RANGES 미정의 시: 채택 장면별 '시간표시 몽타주'를 만들고 종료.
#        → 사람이 몽타주 보고 "깨끗한 불 구간(초)"을 읽어 RANGES 딕트를 채운다.
#   (B) 빌드 모드 — RANGES 정의 시: 그 구간에서 불 프레임 추출 + 음성 구성 → test 생성.
#
# 사용:
#   # 1) 몽타주 먼저
#   %run scripts/colab_build_firetest.py
#   # 2) 몽타주 보고 RANGES 채운 뒤(초 단위 (시작,끝) 리스트):
#   RANGES = {'Chip pan': [(20, 34)], '774563476': [(0, 7)], 'NIST_Cooktop Reignition': [(30,120)], ...}
#   %run scripts/colab_build_firetest.py
#
# 환경: FIRE_STEP(불 프레임 간격 초, 기본 1.0) · COOK_STEP(조리 CCTV 샘플 간격, 기본 6) ·
#        COOK_CAP(조리 영상당 최대 음성 프레임, 기본 8) · PRE_STEP(발화 전 간격, 기본 2.0)

import os, glob, json, subprocess
from PIL import Image, ImageDraw, ImageFont

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DRIVE = '/content/drive/MyDrive'
FIRE  = f'{DRIVE}/fire_frames'
# 작업 폴더는 env 로 재지정 가능(Drive 쓰기 유실 회피 → 로컬 /content 권장). 소스는 Drive 에서 읽음.
RAW   = os.environ.get('RAW_DIR', f'{FIRE}/oilfire_raw')     # 화재영상 소스
COOK  = os.environ.get('COOK_DIR', f'{DRIVE}/조리 데이터 영상')  # 급식실 조리 CCTV(음성)
OUT   = os.environ.get('OUT_DIR', f'{FIRE}/oilfire_realtest')  # test 출력
INSP  = os.environ.get('INSP_DIR', f'{FIRE}/inspect')         # 몽타주 출력
FIRE_STEP = float(os.environ.get('FIRE_STEP', '1.0'))
COOK_STEP = float(os.environ.get('COOK_STEP', '6'))
COOK_CAP  = int(os.environ.get('COOK_CAP', '8'))
PRE_STEP  = float(os.environ.get('PRE_STEP', '2.0'))
os.makedirs(INSP, exist_ok=True)

# 채택 16장면(test·train 겸용) — 각 토큰은 oilfire_raw 에서 정확히 1파일과 매칭돼야 함.
ACCEPTED = [
    'How to Prevent', 'Chip pan', 'Cooking Fire Safety', 'Kitchen Grease Fire Safety',
    '2 東京防災', 'IHこんろ「4', '発生', 'シミュレーション', '恐怖',
    '1637681405', '401469436', '774563476', '32125355803', '34938882503',
    'NIST_Cooktop Reignition', 'NIST_Cooktop ignition',
]


def resolve(tok):
    m = [p for p in glob.glob(f'{RAW}/*') if tok in os.path.basename(p)]
    assert len(m) == 1, f'토큰 "{tok}" 이 {len(m)}개 매칭: {[os.path.basename(x) for x in m]} (정확히 1개여야)'
    return m[0]


def duration(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'json', p], capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return 0.0


def grab(video, t, outpath):
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', video,
                    '-frames:v', '1', '-q:v', '3', outpath], check=False)
    return os.path.exists(outpath)


paths = {tok: resolve(tok) for tok in ACCEPTED}     # 매칭 검증(실패 시 여기서 중단)

try:
    RANGES
except NameError:
    RANGES = {}

# ---------------------------------------------------------------------------
# (A) 몽타주 모드 — 불 구간 읽기용. 채택 장면별 조밀(≈4s) 시간표시 몽타주.
# ---------------------------------------------------------------------------
if not RANGES:
    try:
        F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
    except Exception:
        F = ImageFont.load_default()
    print('몽타주 모드 — 채택 장면별 시간표시 몽타주 생성(불 구간 읽기용):\n')
    WORK = '/content/firetest_montage'
    os.makedirs(WORK, exist_ok=True)
    for tok in ACCEPTED:
        v = paths[tok]; dur = duration(v)
        step = max(2.0, dur / 24)                    # 최대 ~24프레임
        ts = [t for t in [step * (i + 0.5) for i in range(24)] if t < dur]
        fs = []
        for i, t in enumerate(ts):
            op = f'{WORK}/{tok[:10].strip().replace(" ", "_")}_{i:02d}.jpg'
            if grab(v, t, op):
                fs.append((t, op))
        if not fs:
            print(f'  [빈영상?] {tok}'); continue
        cols = 8; cw = 220
        im0 = Image.open(fs[0][1]); ch = round(cw * im0.height / im0.width)
        rows = (len(fs) + cols - 1) // cols
        sh = Image.new('RGB', (cols * cw, rows * (ch + 18) + 18), (16, 16, 16))
        dr = ImageDraw.Draw(sh)
        dr.text((3, 1), f'{tok}   |   {os.path.basename(v)[:70]}   ({dur:.0f}s)', fill=(120, 220, 255), font=F)
        for j, (t, p) in enumerate(fs):
            x = Image.open(p).convert('RGB').resize((cw, ch)); c, r = j % cols, j // cols
            sh.paste(x, (c * cw, 18 + r * (ch + 18)))
            dr.text((c * cw + 2, 18 + r * (ch + 18) + 1), f'{t:.0f}s', fill=(255, 210, 0), font=F)
        out = f'{INSP}/firetest_{tok[:14].strip().replace(" ", "_")}.jpg'
        sh.save(out, quality=84)
        print(f'  {tok:<26} {dur:5.0f}s -> {out}')
    print('\n→ 몽타주를 보고 "깨끗한 불 구간(초)"을 읽어 아래 형식으로 RANGES 를 채운 뒤 재실행:')
    print("RANGES = {")
    for tok in ACCEPTED:
        print(f"    {tok!r}: [(0, 0)],")
    print("}")
    print("# 각 값 = [(시작초, 끝초), ...]. 불이 처음부터 끝까지면 (0, 길이). 불 구간 없으면 [] (양성 제외).")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# (B) 빌드 모드 — RANGES 로 test 구성
# ---------------------------------------------------------------------------
for sub in ('fire', 'nofire_kitchen', 'nofire_presrc'):
    d = f'{OUT}/{sub}'
    if os.path.isdir(d):
        import shutil; shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

manifest = {}
n_fire = n_pre = 0
for idx, tok in enumerate(ACCEPTED):
    v = paths[tok]; sid = f'sc{idx:02d}'
    manifest[sid] = os.path.basename(v)
    rngs = RANGES.get(tok, [])
    if not rngs:
        print(f'  [불구간 없음·양성 제외] {sid} {tok}')
        continue
    # 양성: 불 구간에서 FIRE_STEP 간격
    for (s, e) in rngs:
        t = float(s)
        while t <= float(e):
            if grab(v, t, f'{OUT}/fire/{sid}_{t:06.1f}s.jpg'):
                n_fire += 1
            t += FIRE_STEP
    # 동일소스 음성: 첫 불 시작 이전 [0.5, start-2] 구간(있으면)
    first = min(float(s) for (s, e) in rngs)
    t = 0.5
    while t < first - 2.0:
        if grab(v, t, f'{OUT}/nofire_presrc/{sid}_{t:06.1f}s.jpg'):
            n_pre += 1
        t += PRE_STEP

# 배포-대표 음성: 급식실 조리 CCTV(불 없음)
n_cook = 0
if os.path.isdir(COOK):
    cvids = sorted(p for e in ('mp4', 'mkv', 'mov', 'MOV', 'avi', 'MP4') for p in glob.glob(f'{COOK}/*.{e}'))
    for ci, cv in enumerate(cvids):
        cid = f'ck{ci:02d}'; dur = duration(cv); t = COOK_STEP * 0.5; n = 0
        while t < dur and n < COOK_CAP:
            if grab(cv, t, f'{OUT}/nofire_kitchen/{cid}_{t:06.1f}s.jpg'):
                n += 1; n_cook += 1
            t += COOK_STEP
    print(f'\n조리 CCTV(음성) {len(cvids)}영상 → {n_cook}프레임')
else:
    print(f'\n[주의] 조리 CCTV 폴더 없음: {COOK} — 배포-대표 음성 생략(발화전 음성만)')

json.dump({'accepted': manifest, 'ranges': {k: v for k, v in RANGES.items()},
           'n_fire': n_fire, 'n_nofire_kitchen': n_cook, 'n_nofire_presrc': n_pre,
           'fire_step': FIRE_STEP},
          open(f'{OUT}/realtest_manifest.json', 'w'), ensure_ascii=False, indent=1)

print(f'\ntest 구성 완료 -> {OUT}')
print(f'  fire(양성) {n_fire} · nofire_kitchen(급식실조리) {n_cook} · nofire_presrc(발화전) {n_pre}')
print(f'  manifest {OUT}/realtest_manifest.json')
print('  다음: %run scripts/colab_realtest_eval.py 로 현재 모델 측정.')
print('  ⚠ 양성은 사람이 지정한 불 구간 기반(모델 독립) · 음성 조리CCTV는 불 없음 확인된 정상 조리.')
