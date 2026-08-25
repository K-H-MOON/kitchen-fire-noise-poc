# ===== D-Fire(21k) 실데이터 다운로드 → fire-only 변환 (배찬우 제안 실데이터 보강) · Colab =====
#
# 목적: 배찬우 리서치가 제안한 D-Fire(실사 fire·smoke 21k)를 받아, 우리 파이프라인(fire 단일클래스)
#   에 넣을 수 있게 fire-only 로 변환한다. → colab_indoorfire_split_audit.py 에 DFIRE_DIR 로 주입.
#   "비-합성(실데이터)으로 도메인 갭 좁히기"가 실제로 recall 을 올리나 falsifiable 하게 검증.
#
# ★ 안전장치: fire 클래스 인덱스를 하드코딩하지 않고 data.yaml 의 '이름'에서 자동 매핑(0/1 뒤바뀜 방지).
#   변환 후 매핑·카운트를 출력하니 눈으로 확인할 것.
#
# 선행: Kaggle API 토큰 필요 — https://www.kaggle.com/settings → 'Create New Token' → kaggle.json.
#   Colab 에 업로드(/content/kaggle.json) 하거나 os.environ['KAGGLE_USERNAME']·['KAGGLE_KEY'] 설정.
#
# env(선택): DFIRE_CAP(총 이미지 상한·fire양성 우선, 기본 8000; 0=전체 21k) · DFIRE_OUT(기본 /content/dfire_fireonly)
#   KAGGLE_SLUG(기본 sayedgamal99/smoke-fire-detection-yolo)
#
# 산출: {OUT_DIR}/images/*.jpg (심링크) · {OUT_DIR}/labels/*.txt (fire=0 만, 없으면 빈=음성)

import os, glob, shutil, subprocess, sys, random

CAP  = int(os.environ.get('DFIRE_CAP', '8000'))
OUT  = os.environ.get('DFIRE_OUT', '/content/dfire_fireonly')   # ★ OUT_DIR 아님(테스트셋 경로와 충돌 방지)
SLUG = os.environ.get('KAGGLE_SLUG', 'sayedgamal99/smoke-fire-detection-yolo')
RAW  = '/content/dfire_raw'

try:
    import yaml
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyyaml'], check=True)
    import yaml

# --- Kaggle 인증 (구 kaggle.json[username/key] · 신 access_token[KGAT_...] 둘 다 지원) ---
KDIR = os.path.expanduser('~/.kaggle'); os.makedirs(KDIR, exist_ok=True)
_tok = os.environ.get('KAGGLE_API_TOKEN')          # 신형: KGAT_... (권장)
if _tok:
    open(f'{KDIR}/access_token', 'w').write(_tok.strip()); os.chmod(f'{KDIR}/access_token', 0o600)
if os.path.exists('/content/kaggle.json'):          # 구형: 업로드한 kaggle.json
    shutil.copy('/content/kaggle.json', f'{KDIR}/kaggle.json'); os.chmod(f'{KDIR}/kaggle.json', 0o600)
_have = (os.path.exists(f'{KDIR}/access_token') or os.path.exists(f'{KDIR}/kaggle.json')
         or (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')))
if not _have:
    print('⚠️ Kaggle 인증 없음. 셋 중 하나:')
    print("   (권장) os.environ['KAGGLE_API_TOKEN']='KGAT_...'  ← Settings→API→Create New Token")
    print('   또는 /content/kaggle.json 업로드 · 또는 KAGGLE_USERNAME·KAGGLE_KEY env')
    raise SystemExit(1)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)

# --- 다운로드(이미 있으면 스킵) ---
if not os.path.isdir(RAW) or not os.listdir(RAW):
    os.makedirs(RAW, exist_ok=True)
    print(f'D-Fire 다운로드: {SLUG} (수 분 소요)...')
    r = subprocess.run(['kaggle', 'datasets', 'download', '-d', SLUG, '-p', RAW, '--unzip'])
    if r.returncode != 0:
        print('⚠️ 다운로드 실패 — 슬러그/인증/네트워크 확인. KAGGLE_SLUG 로 다른 미러 지정 가능.')
        raise SystemExit(1)

# --- data.yaml 에서 fire 인덱스 자동 매핑 ---
yamls = glob.glob(f'{RAW}/**/*.yaml', recursive=True) + glob.glob(f'{RAW}/**/*.yml', recursive=True)
fire_idx = None; names = None
for y in yamls:
    try:
        d = yaml.safe_load(open(y))
        nm = d.get('names') if isinstance(d, dict) else None
        if nm:
            names = nm if isinstance(nm, dict) else dict(enumerate(nm))
            for i, n in names.items():
                if 'fire' in str(n).lower() and 'smoke' not in str(n).lower():
                    fire_idx = int(i)
            if fire_idx is not None:
                print(f'data.yaml: {y}\n  names={names} → fire_idx={fire_idx} ({names[fire_idx]})')
                break
    except Exception:
        continue
assert fire_idx is not None, ('data.yaml 에서 fire 클래스 못 찾음 — 수동 확인 필요. '
                              f'발견된 yaml: {yamls}')

# --- 이미지·라벨 수집 (fire 양성 여부 판정) ---
imgs = []
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
    imgs += glob.glob(f'{RAW}/**/{ext}', recursive=True)
imgs = sorted(set(imgs))


def label_of(imgp):
    # 표준 YOLO: .../images/x.jpg → .../labels/x.txt
    if f'{os.sep}images{os.sep}' in imgp:
        lp = imgp.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')
    else:
        lp = imgp
    return os.path.splitext(lp)[0] + '.txt'


recs = []   # (imgp, [fire_lines_remapped])
for p in imgs:
    lp = label_of(p)
    fire_lines = []
    if os.path.exists(lp) and os.path.getsize(lp) > 0:
        for l in open(lp):
            t = l.split()
            if t and t[0].isdigit() and int(t[0]) == fire_idx:
                fire_lines.append('0 ' + ' '.join(t[1:]) + '\n')   # fire → class 0 로 remap
    recs.append((p, fire_lines))

pos = [r for r in recs if r[1]]      # fire 양성
neg = [r for r in recs if not r[1]]  # 음성(smoke/neither)
print(f'D-Fire 수집: 총 {len(recs)}장 · fire양성 {len(pos)} · 음성 {len(neg)}')

# --- CAP: fire 양성 우선 유지, 나머지를 음성으로 채움 ---
random.Random(0).shuffle(neg)
if CAP and len(recs) > CAP:
    keep = pos + neg[:max(0, CAP - len(pos))]
    print(f'DFIRE_CAP={CAP} → 양성 {len(pos)} + 음성 {max(0, CAP-len(pos))} = {len(keep)}장 사용')
else:
    keep = pos + neg
    print(f'전체 {len(keep)}장 사용 (CAP 미적용)')

# --- fire-only 디렉터리 작성(심링크 + 라벨) ---
if os.path.isdir(OUT):
    # 안전장치: 출력 경로가 테스트셋 등 남의 데이터로 보이면 삭제 거부(오염 방지)
    if any(os.path.isdir(f'{OUT}/{d}') for d in ('fire', 'nofire_kitchen', 'nofire_presrc')):
        print(f'⚠️ {OUT} 가 테스트셋으로 보임(fire/·nofire_* 존재) — 삭제 거부. DFIRE_OUT를 다른 경로로 지정하세요.')
        raise SystemExit(1)
    shutil.rmtree(OUT)
os.makedirs(f'{OUT}/images', exist_ok=True); os.makedirs(f'{OUT}/labels', exist_ok=True)
for i, (p, lines) in enumerate(keep):
    name = f'df{i:06d}{os.path.splitext(p)[1].lower()}'
    stem = os.path.splitext(name)[0]
    dst = f'{OUT}/images/{name}'
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(p), dst)
    open(f'{OUT}/labels/{stem}.txt', 'w').writelines(lines)

npos = sum(1 for _, l in keep if l)
print(f'\n-> {OUT}/images ({len(keep)}장) · fire양성 {npos} · 음성 {len(keep)-npos}')
print('  다음: colab_indoorfire_split_audit.py 에 DFIRE_DIR 로 주입 → real_only_grouped_df 학습.')
print(f"    os.environ['DFIRE_DIR']='{OUT}'  후  %run ...colab_indoorfire_split_audit.py")
print('  ★ 위 fire_idx 매핑이 맞는지 눈으로 확인(연기를 불로 학습하면 실험 오염).')
