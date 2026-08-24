# ===== 도메인 이동 평가 (파일럿): 기름/튀김유 화재 test =====
#
# 목적: 일반 실화재(Indoor)로 학습한 모델이 '진짜 기름불(급식실 근접 도메인)'을 잡나?
#   = 실험 A/누수검증 모델들을 오염 검증된 소규모 기름불 test 에 돌려 frame-level 성능 측정.
#
# test: oilfire_pilot (유튜브/빌리빌리 소방 시연 튀김유·기름솥 화재에서 오염 검증 후 큐레이션)
#   fire  65장(4개 독립 장면: NITE·삿포로물투입·Bilibili·삿포로저해상)
#   nofire 14장(2개 장면: 발화 전 스토브·연기만)
#   ※ 프레임 다수지만 독립 장면 4개 → 장면(scene) 단위로도 함께 보고(유효 N 작음, CI 큼).
#
# 준비: seochorobotics Drive 루트에 oilfire_pilot.zip 업로드(스크립트가 자동 압축해제).
# 모델(Drive): runs_phaseB/v8_C0_s1(합성) · runs_if/{real_only,mixed,real_only_grouped}.
# 경계: 시연 영상이라 자막·워터마크 오버레이 잔존(불은 미가림) · 급식실 아닌 시연 세트 · N(장면)=4.

import os, glob, json, zipfile, subprocess, sys
import numpy as np
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO

FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'
IFRUN = f'{FIRE}/runs_if'
NAME = os.environ.get('EVAL_SET', 'oilfire_pilot')   # oilfire_pilot(큰불) / oilfire_early(초기·작은 불)
ZIP  = f'/content/drive/MyDrive/{NAME}.zip'
TEST = f'{FIRE}/{NAME}'
OUT  = f'{FIRE}/indoorfire_eval'
CONF = 0.25

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(TEST) or not glob.glob(f'{TEST}/fire/*.jpg'):
    os.makedirs(TEST, exist_ok=True)
    print('oilfire_pilot 압축 해제...')
    zipfile.ZipFile(ZIP).extractall(TEST)

fire_imgs = sorted(glob.glob(f'{TEST}/fire/*.jpg'))
nof_imgs  = sorted(glob.glob(f'{TEST}/nofire/*.jpg'))
print(f'test: fire {len(fire_imgs)} · nofire {len(nof_imgs)}')

def source(p):                      # 파일명 prefix = 장면(영상) 단위
    b = os.path.basename(p)
    return b.rsplit('_', 1)[0]

models = {
    '1_synth_only (v8_C0_s1)': f'{RUNS}/v8_C0_s1/best.pt',
    '2_real_only':             f'{IFRUN}/real_only/weights/best.pt',
    '3_mixed':                 f'{IFRUN}/mixed/weights/best.pt',
    '2g_real_only_grouped':    f'{IFRUN}/real_only_grouped/weights/best.pt',
}

def detected(model, paths):
    """이미지별 검출여부(bool) 리스트."""
    flags = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=CONF, verbose=False):
            flags.append(len(r.boxes) > 0)
    return flags

rows = {}
for key, best in models.items():
    if not os.path.exists(best):
        print(f'  [없음] {key}: {best}'); continue
    m = YOLO(best)
    fdet = detected(m, fire_imgs)      # 화염 이미지 검출여부
    ndet = detected(m, nof_imgs)       # 무화염 이미지 검출여부(=헛불)
    fd, nd = int(sum(fdet)), int(sum(ndet))
    recall = fd / len(fire_imgs) if fire_imgs else 0
    prec = fd / (fd + nd) if (fd + nd) else 0
    fpr = nd / len(nof_imgs) if nof_imgs else 0
    # 장면(scene) 단위 recall: 각 소스별 프레임 recall → 소스 평균 ± std (유효 N=소스수)
    by_src = {}
    for p, f in zip(fire_imgs, fdet):
        by_src.setdefault(source(p), []).append(f)
    src_rec = {s: float(np.mean(v)) for s, v in by_src.items()}
    scene_mean = float(np.mean(list(src_rec.values())))
    scene_std = float(np.std(list(src_rec.values())))
    rows[key] = dict(recall=recall, precision=prec, fpr=fpr, fire_det=fd, nof_det=nd,
                     scene_recall=src_rec, scene_mean=scene_mean, scene_std=scene_std)

print('\n' + '=' * 74)
print(f'도메인 이동 평가 [{NAME}] · frame-level (conf 0.25)')
print('=' * 74)
print(f'{"모델":<26}{"recall":>8}{"prec":>7}{"fpr":>7}{"scene_mean±std":>18}')
for k, r in rows.items():
    print(f'{k:<26}{r["recall"]:>8.3f}{r["precision"]:>7.3f}{r["fpr"]:>7.3f}'
          f'{r["scene_mean"]:>11.3f}±{r["scene_std"]:.3f}')

print('\n장면별 recall (유효 N=4 장면):')
for k, r in rows.items():
    print(f'  {k}')
    for s, v in r['scene_recall'].items():
        print(f'      {s:<22} {v:.3f}')

print('\n해석: ② 실/③ 혼합/②g 실-그룹 이 ① 합성보다 기름불 recall↑면 "실데이터가 목표 도메인에도 전이".')
print('경계: 시연 오버레이 잔존 · 급식실 아닌 시연 세트 · 장면 N=4(수치 큰 CI).')
json.dump({'conf': CONF, 'n_fire': len(fire_imgs), 'n_nof': len(nof_imgs), 'rows': rows},
          open(f'{OUT}/{NAME}_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/{NAME}_eval.json')
