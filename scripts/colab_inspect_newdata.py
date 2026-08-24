# ===== 새 화재 데이터 적합성 검사 (Colab) =====
#
# 목적: 새로 확보한 mp4/zip 이 도메인 이동 평가(급식실/유류 화재 proxy test)에 쓸 만한지
#       '눈으로' 판정하기 위한 컨택트 시트 생성. (파일명·출처만으론 판단 금지 — 오염 교훈)
#
# 하는 일:
#   - mp4: ffprobe(길이·fps·해상도) + 균등 간격 프레임 N장 추출 → 타임스탬프 라벨 컨택트 시트
#   - zip: 내용 목록(namelist) 요약(폴더 구조·확장자별 개수·샘플 경로) +
#          이미지 있으면 샘플 시트 / 동영상 있으면 가장 작은 것 프레임 샘플
#   - 산출은 fire_frames/inspect/ 에 저장(= blessmoonkh 와 공유되는 폴더 → 커넥터로 열람 가능)
#
# 환경: INSPECT_MP4(기본 자동탐색 '*.mp4' 루트) · INSPECT_ZIP(기본 '*archive*.zip')
#       N_FRAMES(기본 16) ·판정 포인트는 로그 마지막 참조.

import os, glob, json, subprocess, zipfile, shutil, tempfile
from collections import Counter
from google.colab import drive
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    subprocess.run(['pip', 'install', '-q', 'Pillow'], check=True)
    from PIL import Image, ImageDraw, ImageFont

drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive'
OUT  = f'{ROOT}/fire_frames/inspect'
os.makedirs(OUT, exist_ok=True)
N_FRAMES = int(os.environ.get('N_FRAMES', '16'))

def find_one(env, patterns):
    v = os.environ.get(env)
    if v:
        return v if os.path.isabs(v) else f'{ROOT}/{v}'
    for pat in patterns:
        hit = sorted(glob.glob(f'{ROOT}/{pat}'), key=lambda p: -os.path.getsize(p))
        if hit:
            return hit[0]
    return None

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
except Exception:
    F = ImageFont.load_default()

def sheet(frame_paths, labels, out, cols=4, cw=360):
    if not frame_paths:
        print('  (프레임 없음)'); return
    im0 = Image.open(frame_paths[0]).convert('RGB')
    ch = round(cw * im0.height / im0.width)
    rows = (len(frame_paths) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * (ch + 22)), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    for j, (p, lab) in enumerate(zip(frame_paths, labels)):
        try:
            im = Image.open(p).convert('RGB').resize((cw, ch))
        except Exception:
            continue
        c, r = j % cols, j // cols; y = r * (ch + 22)
        dr.text((c * cw + 4, y + 2), lab, fill=(0, 255, 0), font=F)
        sh.paste(im, (c * cw, y + 22))
    sh.save(out, quality=85)
    print(f'  -> {out}  ({len(frame_paths)}컷)')

def ffprobe(path):
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height,r_frame_rate,duration',
                            '-of', 'json', path], capture_output=True, text=True)
        s = json.loads(r.stdout)['streams'][0]
        num, den = s.get('r_frame_rate', '0/1').split('/')
        fps = float(num) / float(den) if float(den) else 0
        dur = float(s.get('duration', 0) or 0)
        return s.get('width'), s.get('height'), round(fps, 1), round(dur, 1)
    except Exception as e:
        print('  ffprobe 실패:', e); return None, None, None, None

def sample_video(path, tag):
    w, h, fps, dur = ffprobe(path)
    print(f'  {os.path.basename(path)} · {w}x{h} · {fps}fps · {dur}s')
    if not dur or dur <= 0:
        dur = 60
    tmp = tempfile.mkdtemp()
    times = [dur * (i + 0.5) / N_FRAMES for i in range(N_FRAMES)]
    fps_paths, labels = [], []
    for i, t in enumerate(times):
        op = f'{tmp}/{i:02d}.jpg'
        subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', path,
                        '-frames:v', '1', '-q:v', '3', op], check=False)
        if os.path.exists(op):
            fps_paths.append(op); labels.append(f'{tag} t={t:.0f}s')
    sheet(fps_paths, labels, f'{OUT}/inspect_{tag}.jpg')

# ---------------------------------------------------------------------------
# 1) mp4 (여러 개 지원 — yt_*.mp4 우선, 각각 개별 시트)
# ---------------------------------------------------------------------------
print('=' * 66); print('1) 동영상 검사'); print('=' * 66)
v = os.environ.get('INSPECT_MP4')
if v:
    mp4s = [v if os.path.isabs(v) else f'{ROOT}/{v}']
elif os.environ.get('INSPECT_ALL') == '1':
    # 루트의 모든 mp4(거대 파일 200MB 초과는 제외 — 360 원본 등)
    mp4s = [p for p in sorted(glob.glob(f'{ROOT}/*.mp4')) if os.path.getsize(p) < 200e6]
else:
    mp4s = (sorted(glob.glob(f'{ROOT}/yt_*.mp4'))
            or sorted(glob.glob(f'{ROOT}/*Kitchen_Fire*.mp4'))
            or sorted(glob.glob(f'{ROOT}/*.mp4')))
if not mp4s:
    print('  루트에서 mp4 못 찾음 (INSPECT_MP4 로 지정)')
for mp4 in mp4s:
    tag = os.path.splitext(os.path.basename(mp4))[0][:24]
    sample_video(mp4, tag)

if os.environ.get('INSPECT_ALL') == '1':
    print('\nINSPECT_ALL → zip 검사 건너뜀. 시트: fire_frames/inspect/inspect_*.jpg')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 2) zip
# ---------------------------------------------------------------------------
print('\n' + '=' * 66); print('2) zip 검사'); print('=' * 66)
zp = find_one('INSPECT_ZIP', ['*archive*.zip', '*.zip'])
if not zp:
    print('  루트에서 zip 못 찾음 (INSPECT_ZIP 로 지정)')
else:
    print(f'  {os.path.basename(zp)} ({os.path.getsize(zp)/1e9:.2f} GB)')
    zf = zipfile.ZipFile(zp)
    names = [n for n in zf.namelist() if not n.endswith('/')]
    ext = Counter(os.path.splitext(n)[1].lower() for n in names)
    tops = Counter(n.split('/')[0] for n in names)
    print(f'  총 항목 {len(names)}개')
    print('  확장자별:', dict(ext.most_common(12)))
    print('  최상위 항목/폴더:', dict(tops.most_common(10)))
    print('  샘플 경로:')
    for n in names[:8]:
        print('   ', n)

    IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    VID = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    imgs = [n for n in names if os.path.splitext(n)[1].lower() in IMG]
    vids = [n for n in names if os.path.splitext(n)[1].lower() in VID]
    txts = [n for n in names if os.path.splitext(n)[1].lower() == '.txt']
    jsons = [n for n in names if os.path.splitext(n)[1].lower() == '.json']
    print(f'\n  이미지 {len(imgs)} · 동영상 {len(vids)} · txt {len(txts)} · json {len(jsons)}')

    tmp = tempfile.mkdtemp()
    if imgs:
        pick = [imgs[i] for i in range(0, len(imgs), max(1, len(imgs) // N_FRAMES))][:N_FRAMES]
        paths, labels = [], []
        for i, n in enumerate(pick):
            try:
                zf.extract(n, tmp); paths.append(f'{tmp}/{n}')
                labels.append(os.path.basename(n)[:22])
            except Exception:
                pass
        sheet(paths, labels, f'{OUT}/inspect_zip_images.jpg')
        # 라벨(txt) 샘플 내용 — 클래스 분포 감 잡기
        if txts:
            print('  txt 라벨 샘플(첫 3개):')
            for n in txts[:3]:
                try:
                    print(f'   [{n}]', zf.read(n).decode('utf-8', 'ignore')[:120].replace('\n', ' | '))
                except Exception:
                    pass
    elif vids:
        v = min(vids, key=lambda n: zf.getinfo(n).file_size)
        print(f'  가장 작은 동영상 추출: {v} ({zf.getinfo(v).file_size/1e6:.1f} MB)')
        zf.extract(v, tmp)
        sample_video(f'{tmp}/{v}', 'zipvideo')
    else:
        print('  이미지/동영상 없음 — 확장자 목록 참고.')

print('\n' + '=' * 66)
print('판정 포인트 (컨택트 시트 육안):')
print(' - 실제 불이 프레임에 보이나? (특히 기름/튀김유 화재)')
print(' - 편집 오염? 자막·타이틀 카드·PIP·만화 불·워터마크 — 있으면 배포와 다른 인공물')
print(' - 급식실/주방 장면 근접도 · 해상도/화질 · 독립 장면 수(같은 컷 반복인지)')
print(f'산출: {OUT}/inspect_*.jpg  (fire_frames 공유 폴더 → 커넥터로 열람 가능)')
