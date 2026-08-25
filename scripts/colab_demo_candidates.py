# ===== 데모용 화재 영상 후보 점검 — 해상도 + 불 구간 프레임 · Colab =====
#
# 목적: 발표 데모(2ck + 2-of-3)에 쓸 화재 영상을 고르기 위해,
#   채택 16장면의 ① 해상도/fps/길이 표 ② 후보 장면의 '불 구간' 프레임 몽타주
#   (선명도·자막/오버레이·불꽃 크기 육안 확인)를 뽑는다. 누수는 무관(2ck는 화재영상 미학습).
#
# ⚠️ %run -i 로 실행(RANGES 파이썬변수 필요 — 오염 없는 불 구간만 샘플).
# env: RAW_DIR(화재영상) · INSP_DIR(몽타주) · CANDS(몽타주할 sc## 또는 토큰, 쉼표; 기본 HD후보) · NPER(장면당 프레임 수)

import os, glob, json, subprocess
from PIL import Image, ImageDraw, ImageFont

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

RAW  = os.environ.get('RAW_DIR', '/content/oilfire_raw')
INSP = os.environ.get('INSP_DIR', '/content/inspect')
NPER = int(os.environ.get('NPER', '6'))
os.makedirs(INSP, exist_ok=True)

ACCEPTED = [
    'How to Prevent', 'Chip pan', 'Cooking Fire Safety', 'Kitchen Grease Fire Safety',
    '2 東京防災', 'IHこんろ「4', '発生', 'シミュレーション', '恐怖',
    '1637681405', '401469436', '774563476', '32125355803', '34938882503',
    'NIST_Cooktop Reignition', 'NIST_Cooktop ignition',
]
# 기본 HD 후보(비-급식실 소방/교육 데모). 원하면 CANDS 로 덮어씀.
CANDS = [t.strip() for t in os.environ.get(
    'CANDS', 'How to Prevent,Kitchen Grease Fire Safety,2 東京防災,発生,シミュレーション').split(',') if t.strip()]

try:
    RANGES
except NameError:
    RANGES = {}


def resolve(tok):
    m = [p for p in glob.glob(f'{RAW}/*') if tok in os.path.basename(p)]
    assert len(m) == 1, f'"{tok}" {len(m)}개 매칭'
    return m[0]


def probe(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
                        '-of', 'json', p], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout); s = j['streams'][0]
        num, den = (s.get('r_frame_rate', '0/1').split('/') + ['1'])[:2]
        fps = float(num) / float(den) if float(den) else 0.0
        return int(s['width']), int(s['height']), fps, float(j['format']['duration'])
    except Exception:
        return 0, 0, 0.0, 0.0


def grab(v, t, op):
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', v, '-frames:v', '1', '-q:v', '2', op],
                   check=False)
    return os.path.exists(op)


# ---- 해상도 표 (전 16장면) ----
print(f'{"sc":<5}{"WxH":>12}{"fps":>7}{"길이":>8}  파일')
info = {}
for i, tok in enumerate(ACCEPTED):
    v = resolve(tok); w, h, fps, dur = probe(v)
    info[tok] = (w, h, fps, dur, v)
    hd = 'HD+' if h >= 720 else ('SD' if h >= 480 else 'low')
    print(f'sc{i:02d}  {f"{w}x{h}":>12}{fps:>7.1f}{dur:>7.0f}s  [{hd}] {os.path.basename(v)[:44]}')

# ---- 후보 몽타주 (불 구간 프레임) ----
try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
except Exception:
    F = ImageFont.load_default()

print('\n후보 불 구간 몽타주 생성...')
for tok in CANDS:
    key = next((k for k in ACCEPTED if tok in k or tok == k), None) or \
          next((k for i, k in enumerate(ACCEPTED) if f'sc{i:02d}' == tok), None)
    if key is None:
        print(f'  [스킵] "{tok}" 매칭 실패'); continue
    w, h, fps, dur, v = info[key]
    rngs = RANGES.get(key, [])
    # RANGES 있으면 그 구간, 없으면 영상 중반 근처 균등
    ts = []
    if rngs:
        span = [(float(s), float(e)) for (s, e) in rngs]
        total = sum(e - s for s, e in span)
        for k in range(NPER):
            x = total * (k + 0.5) / NPER; acc = 0
            for s, e in span:
                if acc + (e - s) >= x:
                    ts.append(s + (x - acc)); break
                acc += e - s
    else:
        ts = [dur * (k + 0.5) / NPER for k in range(NPER)]
    fs = []
    for k, t in enumerate(ts):
        op = f'{INSP}/_demo_{key[:8].strip().replace(" ", "_")}_{k}.jpg'
        if grab(v, t, op):
            fs.append((t, op))
    if not fs:
        continue
    cols = min(NPER, 6); cw = 300
    im0 = Image.open(fs[0][1]); ch = round(cw * im0.height / im0.width)
    rows = (len(fs) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * ch + 22), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    dr.text((3, 2), f'{key}   {w}x{h}   {os.path.basename(v)[:50]}', fill=(120, 220, 255), font=F)
    for j, (t, p) in enumerate(fs):
        x = Image.open(p).convert('RGB').resize((cw, ch)); c, r = j % cols, j // cols
        sh.paste(x, (c * cw, 22 + r * ch))
        d = ImageDraw.Draw(sh); d.text((c * cw + 2, 22 + r * ch + 1), f'{t:.0f}s', fill=(255, 210, 0), font=F)
    out = f'{INSP}/demo_cand_{key[:12].strip().replace(" ", "_")}.jpg'
    sh.save(out, quality=90)
    print(f'  {key:<26} {w}x{h} -> {out}')

print('\n→ 해상도 표에서 HD+(720p↑) 확인 · 몽타주에서 선명도·자막/오버레이·불꽃 크기 육안 확인 후 데모용 1개 확정.')
