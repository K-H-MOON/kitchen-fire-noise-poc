# ===== ⑤ 급식실 조리 하드네거 빌드 (장면=영상 단위 train/held-out split) · Colab =====
#
# 목적: 급식실 조리 CCTV(불 없음)를 학습 네거티브로 쓰기 위해, '영상 단위'로
#   train(학습 주입) / held-out(fpr 측정)로 가른다(같은 영상이 양쪽에 안 걸침 = 누수 0).
#   → fpr 개선이 '기억'이 아니라 '안 본 조리영상으로의 일반화'임을 증명.
#
# ⚠ 이 조리영상들은 현재 oilfire_realtest/nofire_kitchen(fpr 테스트)에도 쓰였음.
#   재학습 후엔 반드시 held-out(cook_neg_test)으로만 fpr 측정해야 정직(train 영상 test 금지).
#
# 2모드:
#   (A) 목록 모드 — HELDOUT 미지정: ck## ↔ 파일명 목록 출력 후 종료(어느 영상 held-out 할지 고르기).
#   (B) 빌드 모드 — HELDOUT 지정: train 영상 조밀추출 → cook_neg_train, held-out → cook_neg_test.
#       ck## 순번은 colab_build_firetest.py 의 조리영상 정렬과 동일(진단 fpr 의 ck## 와 일치).
#
# env: COOK_DIR(조리영상) · HELDOUT(ck## 쉼표) · TRAIN_STEP(기본4s)·TRAIN_CAP(기본20) ·
#      TEST_STEP(기본6s)·TEST_CAP(기본8) · COOK_TRAIN_OUT·COOK_TEST_OUT(출력)
#
# 다음: HARDNEG=1 HARDNEG_TRAIN_DIR=<train> HN_TAG=ck  %run colab_indoorfire_split_audit.py
#       COOK_TEST_DIR=<test> OUT_DIR=/content/oilfire_realtest  %run colab_realtest_eval.py

import os, glob, json, shutil, subprocess

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DRIVE = '/content/drive/MyDrive'
COOK  = os.environ.get('COOK_DIR', f'{DRIVE}/조리 데이터 영상')
TR    = os.environ.get('COOK_TRAIN_OUT', '/content/cook_neg_train/nofire')
TE    = os.environ.get('COOK_TEST_OUT',  '/content/cook_neg_test/nofire')
TRAIN_STEP = float(os.environ.get('TRAIN_STEP', '4'))
TRAIN_CAP  = int(os.environ.get('TRAIN_CAP', '20'))
TEST_STEP  = float(os.environ.get('TEST_STEP', '6'))
TEST_CAP   = int(os.environ.get('TEST_CAP', '8'))
HELDOUT = [t.strip() for t in os.environ.get('HELDOUT', '').split(',') if t.strip()]

cvids = sorted(p for e in ('mp4', 'mkv', 'mov', 'MOV', 'avi', 'MP4') for p in glob.glob(f'{COOK}/*.{e}'))
assert cvids, f'조리영상 없음: {COOK}'
ids = {f'ck{i:02d}': v for i, v in enumerate(cvids)}   # build_firetest 와 동일 정렬 → ck## 일치

# ---------------------------------------------------------------------------
# (A) 목록 모드
# ---------------------------------------------------------------------------
if not HELDOUT:
    print(f'조리영상 {len(cvids)}개 (ck## ↔ 파일명):')
    for cid, v in ids.items():
        print(f'  {cid}  {os.path.basename(v)}')
    print('\n→ HELDOUT 에 held-out(fpr 테스트용) ck## 토큰을 쉼표로 지정 후 재실행.')
    print('  권장: 헛불 몰린 ck02/03/07/08 중 "일부"를 held-out 에 넣어 일반화 증명(§F 설계).')
    print('  예: HELDOUT=ck02,ck09,ck11,ck13,ck16,ck18,ck20,ck25')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# (B) 빌드 모드
# ---------------------------------------------------------------------------
held = []
for tok in HELDOUT:
    if tok not in ids:
        raise SystemExit(f'[중단] HELDOUT 토큰 "{tok}" 이 목록에 없음 (ck00~ck{len(cvids)-1:02d}). 위 목록 확인.')
    held.append(tok)
held = sorted(set(held))
train_ids = sorted(c for c in ids if c not in held)


def duration(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'json', p], capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)['format']['duration'])
    except Exception:
        return 0.0


def grab(v, t, op):
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', v,
                    '-frames:v', '1', '-q:v', '3', op], check=False)
    return os.path.exists(op)


def extract(cids, dst, step, cap):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for cid in cids:
        v = ids[cid]; d = duration(v); t = step * 0.5; k = 0
        while t < d and k < cap:
            if grab(v, t, f'{dst}/{cid}_{t:06.1f}s.jpg'):
                k += 1; n += 1
            t += step
    return n


ntr = extract(train_ids, TR, TRAIN_STEP, TRAIN_CAP)
nte = extract(held, TE, TEST_STEP, TEST_CAP)
print(f'train-neg   {ntr}장 ({len(train_ids)}영상) -> {TR}')
print(f'held-out-neg {nte}장 ({len(held)}영상) -> {TE}')
print(f'  held-out: {held}')
print(f'  train:    {train_ids}')

json.dump(dict(heldout=held, train_ids=train_ids, n_train=ntr, n_test=nte,
               train_step=TRAIN_STEP, train_cap=TRAIN_CAP,
               ids={k: os.path.basename(v) for k, v in ids.items()}),
          open('/content/cook_negs_split.json', 'w'), ensure_ascii=False, indent=1)
print('\n다음:')
print(f'  1) HARDNEG=1 HARDNEG_TRAIN_DIR={TR} HN_TAG=ck  %run colab_indoorfire_split_audit.py')
print(f'  2) COOK_TEST_DIR={TE} OUT_DIR=/content/oilfire_realtest EVAL_OUT=/content  %run colab_realtest_eval.py')
