# ===== 수집 화재영상 검수 (오염·중복·누수·사용성) · Colab =====
#
# 목적: seochorobotics Drive 에 업로드한 유류/주방 화재 영상들을 학습·평가에 넣기 전에
#   ① 사용성(포맷): 길이·해상도·프레임수·세로/360/짧음/제작물 플래그
#   ② 중복·누수: 영상쌍별 근접중복 프레임 '개수'(dHash) = 재업로드·같은 시연 · 기존 test셋과 겹침
#   ③ 오염: 영상별 프레임 몽타주 → 오버레이(자막·타이틀·만화불)·비주방 육안 검증
# 를 한 번에 검수한다. (오염의 최종 판정은 몽타주 육안 — Drive 커넥터는 픽셀 못 봄, 채팅 첨부 필요.)
#
# 중복/누수 판정 견고화:
#   - 저분산 프레임(암전·타이틀카드·단색 자막배경 등)은 해시에서 제외 → 공통 blank 프레임 오탐 방지.
#   - union-find 연쇄 대신 '영상쌍별 매칭 프레임 개수'를 세서 MIN_MATCH(기본 2) 이상일 때만 중복/누수로 판정.
#
# 환경:
#   FIRE_VIDEOS  검수할 폴더(예 '/content/drive/MyDrive/oilfire_raw'). 미지정 시 MyDrive 영상 후보만 나열 후 종료.
#   STEP(추출 간격 초, 기본 2.0) · CAP(영상당 최대 프레임, 기본 16) · HAM(근접중복 Hamming, 기본 8)
#   MIN_MATCH(중복/누수로 판정할 최소 매칭 프레임 수, 기본 2) · VAR_MIN(해시 제외 저분산 임계, 기본 12)

import os, sys, glob, json, subprocess, shutil
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageStat

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass
try:
    import imagehash
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imagehash'], check=True)
    import imagehash

ROOT = '/content/drive/MyDrive'
FIRE = f'{ROOT}/fire_frames'
INSP = f'{FIRE}/inspect'
WORK = '/content/firevid_inspect'
FOLDER = os.environ.get('FIRE_VIDEOS', '')
STEP = float(os.environ.get('STEP', '2.0'))
CAP  = int(os.environ.get('CAP', '16'))
HAM  = int(os.environ.get('HAM', '8'))
MIN_MATCH = int(os.environ.get('MIN_MATCH', '2'))
VAR_MIN = float(os.environ.get('VAR_MIN', '12'))
EXTS = ('mp4', 'MP4', 'mkv', 'mov', 'MOV', 'webm', 'avi', 'flv', 'm4v', 'ts')
os.makedirs(INSP, exist_ok=True)


def find_videos(folder):
    return sorted({p for e in EXTS for p in glob.glob(f'{folder}/**/*.{e}', recursive=True)})


if not FOLDER:
    print('FIRE_VIDEOS 미지정 — MyDrive 에서 발견한 영상 후보(상위 2단계):')
    cands = sorted({p for e in EXTS for lvl in ('*', '*/*') for p in glob.glob(f'{ROOT}/{lvl}.{e}')})
    for p in cands:
        print(f'  {os.path.getsize(p)/1e6:7.1f}MB  {p.replace(ROOT + "/", "")}')
    print(f'\n총 {len(cands)}개. → os.environ["FIRE_VIDEOS"]="/content/drive/MyDrive/<폴더>" 지정 후 재실행')
    raise SystemExit(0)

vids = find_videos(FOLDER)
assert vids, f'영상 없음: {FOLDER}'
print(f'검수 대상 영상 {len(vids)}개 @ {FOLDER}  (HAM≤{HAM}, 매칭 {MIN_MATCH}장 이상만 중복/누수 판정)\n')


def probe(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
                        '-of', 'json', p], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout); s = d['streams'][0]
        w, h = int(s.get('width', 0)), int(s.get('height', 0))
        dur = float(d.get('format', {}).get('duration') or 0)
        num, den = (s.get('r_frame_rate', '0/1').split('/') + ['1'])[:2]
        fps = float(num) / float(den) if float(den) else 0
        return w, h, dur, fps
    except Exception:
        return 0, 0, 0, 0


if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
meta = []
montage_frames = {}          # i -> [모든 샘플 프레임경로] (몽타주용)
allframes = []               # (video_idx, frame_path, dhash) — 저분산 제외(해시/중복/누수용)

print(f'{"tag":<5}{"영상":<44}{"해상도":>11}{"길이":>7}{"프레임":>6}{"해시":>5}  플래그')
for i, v in enumerate(vids):
    w, h, dur, fps = probe(v)
    tag = f'v{i:02d}'
    d = f'{WORK}/{tag}'; os.makedirs(d, exist_ok=True)
    t = STEP * 0.5; n = 0; frames = []
    while t < dur and n < CAP:
        op = f'{d}/{n:02d}_{t:05.1f}s.jpg'
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', v,
                        '-frames:v', '1', '-q:v', '3', op], check=False)
        if os.path.exists(op):
            frames.append(op); n += 1
        t += STEP
    montage_frames[i] = frames
    nhash = 0
    for f in frames:
        try:
            if ImageStat.Stat(Image.open(f).convert('L')).stddev[0] < VAR_MIN:
                continue                                   # 암전·타이틀·단색 → 오탐 유발, 해시 제외
            allframes.append((i, f, imagehash.dhash(Image.open(f).convert('RGB'))))
            nhash += 1
        except Exception:
            pass
    ratio = (w / h) if h else 0
    flags = []
    if w and h > w:            flags.append('세로')
    if ratio >= 1.9:           flags.append('가로2:1(360?)')
    if dur and dur < 10:       flags.append('짧음<10s')
    if dur and dur > 300:      flags.append('긺>5m(제작물?)')
    if w and max(w, h) < 640:  flags.append('저해상')
    meta.append(dict(idx=i, name=os.path.basename(v), w=w, h=h, dur=round(dur, 1),
                     fps=round(fps, 1), nframes=len(frames), nhash=nhash, flags=flags))
    print(f'{tag:<5}{os.path.basename(v)[:42]:<44}{w}x{h:>9}{dur:>6.1f}s{len(frames):>6}{nhash:>5}  {" ".join(flags)}')

# --- ② 영상쌍별 매칭 프레임 개수 (연쇄 없이 pair 단위) ---
paircount = defaultdict(int)
N = len(allframes)
for a in range(N):
    va, _, ha = allframes[a]
    for b in range(a + 1, N):
        vb, _, hb = allframes[b]
        if va != vb and (ha - hb) <= HAM:
            paircount[(min(va, vb), max(va, vb))] += 1
dup = sorted([(k, c) for k, c in paircount.items() if c >= MIN_MATCH], key=lambda x: -x[1])

print('\n' + '=' * 64)
print(f'② 영상 간 중복 (매칭 {MIN_MATCH}장 이상 · Hamming≤{HAM}) — 같은 시연/재업로드/화질변형')
print('=' * 64)
if dup:
    for (a, b), c in dup:
        print(f'  ⚠️ v{a:02d} ↔ v{b:02d}  매칭 {c}장  | {meta[a]["name"][:26]}  ↔  {meta[b]["name"][:26]}')
    print('  → 매칭 많을수록 확실. 같은 장면이면 하나만 쓰거나 train/test 에 갈라 넣지 말 것(누수).')
else:
    print('  ✅ 영상 간 실질 중복 없음.')

# --- ②b 기존 test셋과 겹침 (매칭 개수 기준) ---
EXIST = []
for base in ('oilfire_pilot', 'oilfire_early', 'oilfire_hardneg', 'oilfire_hardneg_test'):
    for f in glob.glob(f'{FIRE}/{base}/**/*.jpg', recursive=True):
        try:
            if ImageStat.Stat(Image.open(f).convert('L')).stddev[0] < VAR_MIN:
                continue
            EXIST.append((base, imagehash.dhash(Image.open(f).convert('RGB'))))
        except Exception:
            pass
leakcount = defaultdict(lambda: defaultdict(int))
for i, f, hh in allframes:
    for base, eh in EXIST:
        if (hh - eh) <= HAM:
            leakcount[i][base] += 1
leak = {i: {b: c for b, c in bc.items() if c >= MIN_MATCH} for i, bc in leakcount.items()}
leak = {i: bc for i, bc in leak.items() if bc}

print('\n' + '=' * 64)
print(f'②b 기존 test셋과 겹침 (매칭 {MIN_MATCH}장 이상) — 있으면 학습에 넣으면 누수')
print('=' * 64)
if leak:
    for i, bc in sorted(leak.items()):
        det = ' · '.join(f'{b}:{c}장' for b, c in bc.items())
        print(f'  ⚠️ v{i:02d}({meta[i]["name"][:30]}) ↔ {det}')
    print('  → 이 영상들은 기존 test와 실질 겹침 = 학습셋에서 제외(또는 그 test 재검토).')
else:
    print('  ✅ 기존 셋과 실질 겹침 없음.')

# --- ③ 영상별 몽타주 (모든 샘플 프레임 · 오염 육안) ---
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
except Exception:
    F = ImageFont.load_default()
sheets = []
for i in sorted(montage_frames):
    fs = montage_frames[i]
    if not fs:
        continue
    cols = min(8, len(fs)); cw = 200
    im0 = Image.open(fs[0]); ch = round(cw * im0.height / im0.width)
    rows = (len(fs) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * (ch + 16) + 18), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    dr.text((3, 1), f'v{i:02d}  {meta[i]["name"][:60]}  {meta[i]["w"]}x{meta[i]["h"]} '
                    f'{meta[i]["dur"]}s  {" ".join(meta[i]["flags"])}', fill=(120, 220, 255), font=F)
    for j, f in enumerate(fs):
        x = Image.open(f).convert('RGB').resize((cw, ch))
        c, r = j % cols, j // cols
        sh.paste(x, (c * cw, 18 + r * (ch + 16)))
        dr.text((c * cw + 2, 18 + r * (ch + 16) + 1), os.path.basename(f)[:14], fill=(255, 200, 0), font=F)
    out = f'{INSP}/firevid_v{i:02d}.jpg'
    sh.save(out, quality=82)
    sheets.append(out)

print('\n' + '=' * 64)
print('③ 오염 육안용 몽타주 저장(오버레이·자막·만화불·비주방 확인 — 채팅 첨부 필요)')
print('=' * 64)
print(f'  {len(sheets)}개 -> {INSP}/firevid_v00.jpg … v{len(sheets)-1:02d}.jpg')

json.dump(dict(folder=FOLDER, step=STEP, cap=CAP, ham=HAM, min_match=MIN_MATCH, var_min=VAR_MIN,
               videos=meta,
               dup_pairs=[{'a': a, 'b': b, 'match': c} for (a, b), c in dup],
               leak_existing={str(i): bc for i, bc in leak.items()}),
          open(f'{FIRE}/firevideo_inspect.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n-> 요약 json {FIRE}/firevideo_inspect.json')
print('판정: 세로/360/저해상=포맷부적합 · 긺=제작물(오염위험) · ②중복=하나만 · ②b누수=학습제외 ·'
      ' 몽타주에서 자막이 불꽃 덮으면 오염 배제.')
