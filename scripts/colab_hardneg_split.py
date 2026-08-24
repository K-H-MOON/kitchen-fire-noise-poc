# ===== 하드네거 장면 단위 train / held-out 분리 (Colab, GPU 불필요) =====
#
# 목적: §E 헛불 하드네거 130장(16장면)을 '장면(=원본 영상) 단위'로 train/held-out 으로 가른다.
#   - train-hardneg → 학습 주입용(헛불 고치기: 빈 라벨 음성으로 재학습).
#   - held-out      → fpr 측정용. 같은 영상 프레임이 train·test 에 겹치지 않음(프레임 누수 0).
#     → fpr 개선이 '기억'이 아니라 '안 본 영상으로의 일반화'임을 증명.
#
# held-out(4장면, 기본): 13476222(주황조명 test) · 8094275(어두운 스팀) · 945875270 · 267
#   설계 의도: 주황조명 일반화 증명 = 13578888(train) → 13476222(test).
#   ⚠ 경계: 유형당 장면 수가 적음 → held-out 이 모든 헛불 모드를 대표 못 할 수 있음
#            (1장면뿐인 유형 존재). 이 한계는 결과 해석 시 명시할 것.
#
# 산출(Drive):
#   fire_frames/oilfire_hardneg_train/nofire/*.jpg  (train 주입용 · symlink)
#   fire_frames/oilfire_hardneg_test/nofire/*.jpg   (held-out fpr · symlink)
#   fire_frames/oilfire_hardneg_test/fire/*.jpg     (sanity: 파일럿 화염 소량 — recall 회귀 확인용)
#   fire_frames/hardneg_split.json                  (manifest)
#
# 환경: HELDOUT (쉼표구분 토큰; 각 토큰은 정확히 1장면과 매칭돼야 함)
#
# 다음: HARDNEG=1 로 colab_indoorfire_train.py / colab_indoorfire_split_audit.py 재학습 →
#       EVAL_SET=oilfire_hardneg_test(fpr) · oilfire_pilot · oilfire_early(recall 회귀) 로 colab_oilfire_eval.py

import os, glob, json, shutil

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

FIRE = '/content/drive/MyDrive/fire_frames'
SRC  = f'{FIRE}/oilfire_hardneg/nofire'      # colab_build_hardneg.py 산출(하드네거 130장)
SANI = f'{FIRE}/oilfire_hardneg/fire'        # sanity 화염(파일럿)
TR   = f'{FIRE}/oilfire_hardneg_train'
TE   = f'{FIRE}/oilfire_hardneg_test'
HELDOUT = [t for t in os.environ.get('HELDOUT', '13476222,8094275,94587527,267').split(',') if t]


def scene(p):                                # 장면 = 파일명에서 시간접미사(_NNN.Ns) 앞 부분
    return os.path.basename(p).rsplit('_', 1)[0]


imgs = sorted(glob.glob(f'{SRC}/*.jpg'))
assert imgs, f'하드네거 소스 없음: {SRC}  (먼저 colab_build_hardneg.py 실행)'
scenes = {}
for p in imgs:
    scenes.setdefault(scene(p), []).append(p)
print(f'하드네거 {len(imgs)}장 · {len(scenes)}장면')
for s in sorted(scenes):
    print(f'  {s:<26} {len(scenes[s])}장')

# --- held-out 토큰 → 장면 매칭 (각 토큰이 정확히 1장면과 매칭돼야 함; 아니면 중단) ---
sc_names = list(scenes)
test_scenes = []
for tok in HELDOUT:
    matched = [s for s in sc_names if tok in s]
    if len(matched) != 1:
        raise SystemExit(f'[중단] held-out 토큰 "{tok}" 이 {len(matched)}개 장면과 매칭: {matched}\n'
                         f'       (정확히 1개여야 함 — 위 장면 목록 보고 HELDOUT 토큰 수정)')
    test_scenes.append(matched[0])
test_scenes = sorted(set(test_scenes))
train_scenes = sorted(s for s in sc_names if s not in test_scenes)
print(f'\nheld-out(test) {len(test_scenes)}장면:')
for s in test_scenes:
    print(f'  [TEST]  {s:<26} {len(scenes[s])}장')
print(f'train {len(train_scenes)}장면:')
for s in train_scenes:
    print(f'  [TRAIN] {s:<26} {len(scenes[s])}장')


def stage(dst_root, scene_list, sub='nofire'):
    d = f'{dst_root}/{sub}'
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    n = 0
    for s in scene_list:
        for p in scenes[s]:
            shutil.copy(p, f'{d}/{os.path.basename(p)}')   # Drive(FUSE)는 symlink 미지원 → 복사
            n += 1
    return n


ntr = stage(TR, train_scenes)
nte = stage(TE, test_scenes)

# sanity 화염 → test/fire (recall 회귀 확인 + oilfire_eval 압축해제 가드 통과)
df = f'{TE}/fire'
if os.path.isdir(df):
    shutil.rmtree(df)
os.makedirs(df, exist_ok=True)
sf = sorted(glob.glob(f'{SANI}/*.jpg'))
for p in sf:
    shutil.copy(p, f'{df}/{os.path.basename(p)}')   # Drive(FUSE) symlink 미지원 → 복사

print(f'\ntrain-hardneg   {ntr}장 -> {TR}/nofire')
print(f'held-out-hardneg {nte}장 -> {TE}/nofire · sanity fire {len(sf)}장 -> {TE}/fire')

json.dump(dict(heldout_tokens=HELDOUT, test_scenes=test_scenes, train_scenes=train_scenes,
               n_train=ntr, n_test=nte, n_sanity_fire=len(sf),
               scene_counts={s: len(v) for s, v in sorted(scenes.items())}),
          open(f'{FIRE}/hardneg_split.json', 'w'), ensure_ascii=False, indent=1)
print(f'-> manifest {FIRE}/hardneg_split.json')

print('\n다음 단계:')
print('  1) HARDNEG=1 RUN=both  %run colab_indoorfire_train.py        # real_only_hn · mixed_hn')
print('  2) HARDNEG=1           %run colab_indoorfire_split_audit.py  # real_only_grouped_hn')
print('  3) EVAL_SET=oilfire_hardneg_test %run colab_oilfire_eval.py  # held-out fpr (핵심)')
print('     EVAL_SET=oilfire_pilot / oilfire_early 로도 실행           # recall 회귀 확인')
