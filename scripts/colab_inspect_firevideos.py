# ===== 수집 화재영상 검수 (오염·중복·누수·사용성) · Colab =====
#
# 목적: seochorobotics Drive 에 업로드한 유류/주방 화재 영상들을 학습·평가에 넣기 전에
#   ① 사용성(포맷): 길이·해상도·프레임수·세로/360/짧음/제작물 플래그
#   ② 중복·누수: 영상 간 근접중복(dHash) 클러스터 = 재업로드·같은 시연 · 기존 test셋과 겹침
#   ③ 오염: 영상별 프레임 몽타주 → 오버레이(자막·타이틀·만화불)·비주방 육안 검증
# 를 한 번에 검수한다. (오염의 최종 판정은 몽타주 육안 — Drive 커넥터는 픽셀 못 봄, 채팅 첨부 필요.)
#
# 환경:
#   FIRE_VIDEOS  검수할 폴더(예 '/content/drive/MyDrive/oilfire_raw'). 미지정 시 MyDrive 에서 영상 후보만 나열하고 종료.
#   STEP(추출 간격 초, 기본 2.0) · CAP(영상당 최대 프레임, 기본 16) · HAM(근접중복 Hamming, 기본 8)

import os, sys, glob, json, subprocess, shutil
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

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
EXTS = ('mp4', 'MP4', 'mkv', 'mov', 'MOV', 'webm', 'avi', 'flv', 'm4v', 'ts')
os.makedirs(INSP, exist_ok=True)


def find_videos(folder):
    return sorted({p for e in EXTS for p in glob.glob(f'{folder}/**/*.{e}', recursive=True)})


# --- 폴더 미지정: 영상 후보 나열하고 종료(사용자가 FIRE_VIDEOS 지정하도록) ---
if not FOLDER:
    print('FIRE_VIDEOS 미지정 — MyDrive 에서 발견한 영상 후보(상위 2단계):')
    cands = sorted({p for e in EXTS for lvl in ('*', '*/*') for p in glob.glob(f'{ROOT}/{lvl}.{e}')})
    for p in cands:
        sz = os.path.getsize(p) / 1e6
        print(f'  {sz:7.1f}MB  {p.replace(ROOT + "/", "")}')
    print(f'\n총 {len(cands)}개. → os.environ["FIRE_VIDEOS"]="/content/drive/MyDrive/<폴더>" 지정 후 재실행')
    raise SystemExit(0)

vids = find_videos(FOLDER)
assert vids, f'영상 없음: {FOLDER}'
print(f'검수 대상 영상 {len(vids)}개 @ {FOLDER}\n')


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


# --- 프레임 추출 + 메타 + dHash ---
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
meta = []
allframes = []      # (video_idx, frame_path, dhash)

print(f'{"tag":<5}{"영상":<46}{"해상도":>11}{"길이":>7}{"프레임":>6}  플래그')
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
    ratio = (w / h) if h else 0
    flags = []
    if w and h > w:            flags.append('세로')
    if ratio >= 1.9:           flags.append('가로2:1(360?)')
    if dur and dur < 10:       flags.append('짧음<10s')
    if dur and dur > 300:      flags.append('긺>5m(제작물?)')
    if w and max(w, h) < 640:  flags.append('저해상')
    for f in frames:
        try:
            allframes.append((i, f, imagehash.dhash(Image.open(f).convert('RGB'))))
        except Exception:
            pass
    meta.append(dict(idx=i, name=os.path.basename(v), w=w, h=h, dur=round(dur, 1),
                     fps=round(fps, 1), nframes=len(frames), flags=flags))
    print(f'{tag:<5}{os.path.basename(v)[:44]:<46}{w}x{h:>10}{dur:>6.1f}s{len(frames):>6}  {" ".join(flags)}')

# --- 영상 간 근접중복 클러스터 (재업로드·같은 시연 = 중복/누수) ---
N = len(allframes)
parent = list(range(N))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
for a in range(N):
    for b in range(a + 1, N):
        if (allframes[a][2] - allframes[b][2]) <= HAM:      # imagehash '-' = Hamming
            union(a, b)
clu = defaultdict(set)
for k in range(N):
    clu[find(k)].add(allframes[k][0])
cross = sorted([sorted(vs) for vs in clu.values() if len(vs) > 1])

print('\n' + '=' * 60)
print(f'② 영상 간 근접중복(Hamming≤{HAM}) — 있으면 같은 시연/재업로드 = 중복·누수 위험')
print('=' * 60)
if cross:
    seen = set()
    for vs in cross:
        key = tuple(vs)
        if key in seen:
            continue
        seen.add(key)
        print(f'  ⚠️ 중복 의심: ' + ' ↔ '.join(f'v{j:02d}({meta[j]["name"][:24]})' for j in vs))
    print('  → 같은 장면이면 하나만 쓰거나, train/test 에 갈라 넣지 말 것(누수).')
else:
    print('  ✅ 영상 간 근접중복 없음(각 영상 독립 장면).')

# --- 기존 test셋과 겹침(누수) ---
EXIST = []
for base in ('oilfire_pilot', 'oilfire_early', 'oilfire_hardneg', 'oilfire_hardneg_test'):
    for f in glob.glob(f'{FIRE}/{base}/**/*.jpg', recursive=True):
        try:
            EXIST.append((base, imagehash.dhash(Image.open(f).convert('RGB'))))
        except Exception:
            pass
leakvids = defaultdict(set)
for i, f, hh in allframes:
    for base, eh in EXIST:
        if (hh - eh) <= HAM:
            leakvids[i].add(base); break

print('\n' + '=' * 60)
print(f'②b 기존 셋과 겹침(누수) — 새 영상이 기존 pilot/early/hardneg 와 근접중복?')
print('=' * 60)
if leakvids:
    for i, bases in sorted(leakvids.items()):
        print(f'  ⚠️ v{i:02d}({meta[i]["name"][:28]}) ↔ 기존 {sorted(bases)} — 기존 test와 겹침(누수 주의)')
else:
    print('  ✅ 기존 셋과 겹침 없음.')

# --- 영상별 몽타주(오염 육안) ---
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
except Exception:
    F = ImageFont.load_default()
frames_by_v = defaultdict(list)
for i, f, hh in allframes:
    frames_by_v[i].append(f)
sheets = []
for i in sorted(frames_by_v):
    fs = sorted(frames_by_v[i])
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

print('\n' + '=' * 60)
print('③ 오염 육안용 몽타주 저장(오버레이·자막·만화불·비주방 확인 — 채팅 첨부 필요)')
print('=' * 60)
for s in sheets:
    print('  ', s)

json.dump(dict(folder=FOLDER, step=STEP, cap=CAP, ham=HAM, videos=meta,
               cross_dup=cross, leak_existing={str(k): sorted(v) for k, v in leakvids.items()}),
          open(f'{FIRE}/firevideo_inspect.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n-> 요약 json {FIRE}/firevideo_inspect.json')
print('판정 가이드: 세로/360/저해상=포맷 부적합 · 짧음=프레임 적음 · 긺=제작물(오염 위험) ·'
      ' 중복/누수=장면 병합 or 배제 · 몽타주에서 자막이 불꽃 덮으면 오염 배제.')
