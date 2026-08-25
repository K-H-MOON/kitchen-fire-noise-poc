# ===== B: 시간축/이벤트-level 평가 (지속성 필터) · Colab =====
#
# 목적: 프레임 판정(2ck: 급식실 실불꽃 0.69·헛불 0.03)을 '화재 이벤트' 경보로 바꿔,
#   지속성 로직(K-of-N)이 recall 을 회복(성장하는 불)하고 헛불을 줄이나(순간 깜빡 제거) 측정.
#   진짜 불=여러 프레임 지속·성장 vs 조리 헛불=순간 → 외형이 못 가르는 걸 시간축이 가름.
#
# 데이터: 불=교정 RANGES 구간 밀집프레임(=불 이벤트) · 조리=held-out 8영상 전체 밀집(=비-이벤트).
#   ⚠️ %run -i 로 실행(RANGES 파이썬변수 필요). 조리 held-out=⑤ held-out과 동일(누수 통제 일관).
#
# env: RAW_DIR(불영상)·COOK_DIR(조리영상)·MODEL(기본 real_only_grouped_ck)·STEP(밀집간격초,기본0.5)·
#      CONF(기본0.25)·HELDOUT(조리 ck##)
# 지표: 불 이벤트 recall(경보 뜬 이벤트 비율)·지연(첫 경보 초)·조리 헛경보율(경보 뜬 영상 비율). K 스윕.

import os, glob, json, subprocess, sys
import numpy as np
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE  = '/content/drive/MyDrive/fire_frames'
IFRUN = f'{FIRE}/runs_if'
RAW   = os.environ.get('RAW_DIR', '/content/oilfire_raw')
COOK  = os.environ.get('COOK_DIR', '/content/drive/MyDrive/조리 데이터 영상')
MODEL = os.environ.get('MODEL', 'real_only_grouped_ck')
STEP  = float(os.environ.get('STEP', '0.5'))
CONF  = float(os.environ.get('CONF', '0.25'))
WORK  = '/content/temporal_frames'
HELDOUT = [t.strip() for t in os.environ.get('HELDOUT', 'ck02,ck09,ck11,ck13,ck16,ck18,ck20,ck25').split(',') if t.strip()]
drive.mount('/content/drive')

ACCEPTED = [
    'How to Prevent', 'Chip pan', 'Cooking Fire Safety', 'Kitchen Grease Fire Safety',
    '2 東京防災', 'IHこんろ「4', '発生', 'シミュレーション', '恐怖',
    '1637681405', '401469436', '774563476', '32125355803', '34938882503',
    'NIST_Cooktop Reignition', 'NIST_Cooktop ignition',
]
try:
    RANGES
except NameError:
    raise SystemExit('RANGES 미정의 — %run -i 로 실행하고 RANGES 셀(교정본)에서 정의할 것.')


def resolve(tok, root):
    m = [p for p in glob.glob(f'{root}/*') if tok in os.path.basename(p)]
    assert len(m) == 1, f'"{tok}" {len(m)}개 매칭 in {root}'
    return m[0]


def duration(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', p],
                       capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return 0.0


def grab_seq(v, s, e, tag):
    d = f'{WORK}/{tag}'
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(f'{d}/*.jpg'):
        os.remove(f)
    t = float(s); i = 0
    while t <= float(e):
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', v, '-frames:v', '1',
                        '-q:v', '3', f'{d}/{i:04d}.jpg'], check=False)
        t += STEP; i += 1
    return sorted(glob.glob(f'{d}/*.jpg'))


m = YOLO(f'{IFRUN}/{MODEL}/weights/best.pt')


def detect_seq(paths):
    flags = []
    for i in range(0, len(paths), 64):
        for r in m.predict(paths[i:i + 64], conf=CONF, verbose=False):
            flags.append(int(len(r.boxes) > 0))
    return np.array(flags)


def alarmed(flags, K, WIN):
    """최근 WIN 프레임 중 K 이상 검출 시 경보. 반환 (경보여부, 첫경보 인덱스)."""
    for i in range(len(flags)):
        if flags[max(0, i - WIN + 1):i + 1].sum() >= K:
            return True, i
    return False, -1


# ---- 불 이벤트 시퀀스 (구간별) ----
print('불 이벤트 밀집추출·추론...')
fire_seqs = {}
for tok in ACCEPTED:
    rngs = RANGES.get(tok, [])
    if not rngs:
        continue
    v = resolve(tok, RAW)
    for ri, (s, e) in enumerate(rngs):
        seq = detect_seq(grab_seq(v, s, e, f'fire_{tok[:8].strip().replace(" ", "_")}_{ri}'))
        if len(seq):
            fire_seqs[f'{tok[:12].strip()}#{ri}'] = seq
print(f'  불 이벤트 {len(fire_seqs)}개 (구간별)')

# ---- 조리 비-이벤트 (held-out 영상 전체) ----
print('조리 held-out 밀집추출·추론...')
cvids = sorted(p for e in ('mp4', 'mkv', 'mov', 'MOV', 'avi', 'MP4') for p in glob.glob(f'{COOK}/*.{e}'))
cook_ids = {f'ck{i:02d}': v for i, v in enumerate(cvids)}
cook_seqs = {}
for tok in HELDOUT:
    if tok not in cook_ids:
        print(f'  [스킵] {tok} 없음'); continue
    v = cook_ids[tok]
    seq = detect_seq(grab_seq(v, 0, duration(v), f'cook_{tok}'))
    if len(seq):
        cook_seqs[tok] = seq
print(f'  조리 비-이벤트 {len(cook_seqs)}개 (held-out 영상)')

# ---- K-of-N 스윕 ----
GRID = [(1, 1), (2, 3), (3, 5), (4, 7), (5, 10)]
print('\n' + '=' * 74)
print(f'B 시간축 — 이벤트-level (model {MODEL} · STEP {STEP}s · conf {CONF})')
print('=' * 74)
print(f'{"규칙(K-of-N)":<14}{"불이벤트 recall":>16}{"조리 헛경보율":>14}{"지연(초,중앙)":>14}')
rows = []
for (K, WIN) in GRID:
    fa = [alarmed(s, K, WIN) for s in fire_seqs.values()]
    frec = float(np.mean([a for a, _ in fa]))
    lats = [idx * STEP for (a, idx) in fa if a]
    lat = float(np.median(lats)) if lats else None
    cfa = float(np.mean([alarmed(s, K, WIN)[0] for s in cook_seqs.values()])) if cook_seqs else None
    rows.append(dict(K=K, WIN=WIN, fire_event_recall=frec, cook_fa_rate=cfa, latency_med=lat))
    lc = f'{lat:.1f}' if lat is not None else '  -  '
    cc = f'{cfa:.3f}' if cfa is not None else '  -  '
    tag = ' (=프레임단위)' if (K, WIN) == (1, 1) else ''
    print(f'{f"{K}-of-{WIN}":<14}{frec:>16.3f}{cc:>14}{lc:>14}{tag}')

print('\n판독: K=1(프레임단위) 대비 K-of-N 이 ① 불이벤트 recall 유지/회복 ② 조리 헛경보율↓ 이면 시간축 유효.')
print('  지연=불 구간 시작~첫 경보(초). 낮을수록 조기경보. 경계: 구간 시작=불 시작 근사·held-out N 작음.')
json.dump({'model': MODEL, 'step': STEP, 'conf': CONF,
           'n_fire_events': len(fire_seqs), 'n_cook': len(cook_seqs), 'grid': rows},
          open('/content/temporal_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('\n-> /content/temporal_eval.json')
