# ===== 7단계: 실제 화재 검증 (honesty check, Colab) =====
#
# 우리 test 는 전부 합성(배경+얹은 불꽃)이라, 결과는 엄밀히 "합성된 불꽃을 검출한다".
# 이 단계는 **소재로 쓰지 않은 실제 유류화재 영상**의 전체 프레임(합성 아님)에서
# 학습된 모델이 진짜 불을 찾는지 잰다. 이 PoC 의 최종 정직성 검증.
#
# 주의(해석) — 실제 화재 영상은 급식실이 아니라 실험실/가정 주방이다. 그래서 이
# 검증은 '실제 불꽃' + '다른 장면' 을 함께 본다. 실패해도 어느 쪽 탓인지 딱 분리되진
# 않는다. **"실제 불을 아예 찾기라도 하나" 의 홀리스틱 검증**으로 읽는다.
#
# 재는 값 (이미지 단위):
#   real_flame_rate = fire_shots 프레임에서 검출되는 비율   (높아야 함)
#   real_fp_rate    = nofire_shots(발화 전) 프레임에서 검출 (낮아야 함)
#
# GPU 권장. best.pt(5단계)와 real_fire.json 의 shots 가 채워져 있어야 한다.

import os, glob, json, shutil, subprocess, unicodedata
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

REPO = '/content/kitchen-fire-noise-poc/scripts'
FIRE = '/content/drive/MyDrive/fire_frames'
# 깨끗한 Phase B baseline 으로 검증 (구 runs/fire_s 는 konro LED 오염 시절 산출물).
# 다른 모델로 보려면 BEST_MODEL 만 바꾼다 (예: 'v8_modelA_s1').
BEST_MODEL = 'v8_baseline_s1'
BEST = f'{FIRE}/runs_phaseB/{BEST_MODEL}/best.pt'
OUT  = f'{FIRE}/realfire'
CONF = 0.25

drive.mount('/content/drive')
if not os.path.exists(BEST):
    raise SystemExit(f'{BEST} 가 없음 — 5단계를 먼저 돌릴 것')
model = YOLO(BEST)
inv = json.load(open(f'{REPO}/real_fire.json', encoding='utf-8'))
SRC, FPS = inv['src_dir'], inv['fps']


def norm(s):
    return unicodedata.normalize('NFC', s)

allf = glob.glob(f'{SRC}/*')

def frames_of(src, a, b, crop):
    tmp = '/content/_rf'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    vf = f'fps={FPS}'
    if crop:
        vf = f'crop=iw:ih*{1 - crop[0] - crop[1]}:0:ih*{crop[0]},' + vf
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                    '-t', str(round(b - a + 1, 2)), '-vf', vf, '-q:v', '2',
                    f'{tmp}/%05d.jpg'], check=False)
    return sorted(glob.glob(f'{tmp}/*.jpg'))

def detected(path):
    r = model.predict(path, conf=CONF, verbose=False)[0]
    return (len(r.boxes) > 0), (float(r.boxes.conf.max()) if len(r.boxes) else 0.0)

# ---------------------------------------------------------------------------
# shots 채워진 출처만
# ---------------------------------------------------------------------------
ready = [s for s in inv['sources'] if s.get('fire_shots')]
if not ready:
    raise SystemExit('real_fire.json 에 fire_shots 가 채워진 출처가 없음 — 먼저 채울 것')

os.makedirs(OUT, exist_ok=True)
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
except Exception:
    F = ImageFont.load_default()

print('=' * 66)
print('실제 화재 검증 (합성 아님 · 소재 미사용 영상)')
print('=' * 66)

fire_det = fire_tot = nof_det = nof_tot = 0
KEEP = '/content/_rf_keep'                        # 시트용 프레임 영속 보관 (frames_of 의 tmp 는 매 shot 삭제됨)
shutil.rmtree(KEEP, ignore_errors=True); os.makedirs(KEEP)
sheets = []                                       # (kept_path, conf, kind, key) — kind: miss(놓친 불꽃)·fp(헛불)·hit(검출)
def keep(p):
    kp = f'{KEEP}/{len(sheets):02d}.jpg'; shutil.copy(p, kp); return kp
def n_kind(k):
    return len([x for x in sheets if x[2] == k])
for s in ready:
    hit = [p for p in allf if norm(s['file']) in norm(os.path.basename(p))]
    if len(hit) != 1:
        print(f'  [건너뜀] {s["key"]}: "{s["file"]}" 매칭 {len(hit)}개'); continue
    src = hit[0]
    fd = ft = nd = nt = 0
    for a, b in s['fire_shots']:
        for p in frames_of(src, a, b, s.get('crop')):
            ok, cf = detected(p); fd += ok; ft += 1
            if not ok and n_kind('miss') < 9:          # 놓친 불꽃 우선 — 형태 간극 진단
                sheets.append((keep(p), cf, 'miss', s['key']))
            elif ok and n_kind('hit') < 3:             # 검출 성공 몇 장 (대조용)
                sheets.append((keep(p), cf, 'hit', s['key']))
    for a, b in s.get('nofire_shots', []):
        for p in frames_of(src, a, b, s.get('crop')):
            ok, cf = detected(p); nd += ok; nt += 1
            if ok and n_kind('fp') < 6:                # 실제 헛불(발화전 오탐)만
                sheets.append((keep(p), cf, 'fp', s['key']))
    fire_det += fd; fire_tot += ft; nof_det += nd; nof_tot += nt
    fr = fd / ft if ft else 0
    print(f'  {s["key"]:<14} 불꽃 {fd}/{ft} ({fr:.2f})' +
          (f'  ·  발화전 오탐 {nd}/{nt}' if nt else ''))

# ---------------------------------------------------------------------------
# 종합
# ---------------------------------------------------------------------------
real_flame_rate = fire_det / fire_tot if fire_tot else 0
real_fp_rate = nof_det / nof_tot if nof_tot else 0
print('\n' + '=' * 66)
print(f'real_flame_rate (실제 불→검출)   {real_flame_rate:.3f}  ({fire_det}/{fire_tot})  [높아야]')
if nof_tot:
    print(f'real_fp_rate    (발화전→검출)    {real_fp_rate:.3f}  ({nof_det}/{nof_tot})  [낮아야]')
print('=' * 66)
if real_flame_rate >= 0.6:
    print('판정: 실제 불을 상당히 잡음 — 합성 학습이 실전 불에 부분적으로라도 전이됨.')
elif real_flame_rate >= 0.3:
    print('판정: 절반 못 미침 — 합성↔실제 간극이 큼. 장면 차이/불꽃 차이 분석 필요.')
else:
    print('판정: 실제 불을 거의 못 잡음 — 합성 학습의 실전 전이가 약함(정직한 한계).')

json.dump({'conf': CONF, 'fps': FPS, 'real_flame_rate': real_flame_rate,
           'real_fp_rate': real_fp_rate, 'fire_tot': fire_tot, 'nofire_tot': nof_tot},
          open(f'{OUT}/realfire.json', 'w'), ensure_ascii=False, indent=1)

# 시트 — 놓친 불꽃(miss)·헛불(fp)·검출(hit) 을 박스 그려서. miss 는 박스 없음(놓쳤으므로).
LAB = {'miss': '놓침', 'fp': '헛불', 'hit': '검출'}
if sheets:
    CW = 380; rows = (len(sheets) + 2) // 3
    tiles = []
    for p, cf, kind, key in sheets:
        r = model.predict(p, conf=CONF, verbose=False)[0]
        im = Image.fromarray(r.plot()[..., ::-1]); d = ImageDraw.Draw(im)
        d.text((6, 6), f'{key} · {LAB[kind]} conf={cf:.2f}',
               fill=(0, 255, 0), font=F)
        tiles.append(im)
    h0, w0 = np.asarray(tiles[0]).shape[:2]; ch = round(CW * h0 / w0)
    sh = Image.new('RGB', (3 * CW, rows * (ch + 8)), (16, 16, 16))
    for j, im in enumerate(tiles):
        r, c = divmod(j, 3)
        sh.paste(im.resize((CW, ch)), (c * CW, r * (ch + 8)))
    sh.save(f'{OUT}/_realfire.jpg', quality=88)
    print(f'\n시트 -> {OUT}/_realfire.jpg')
