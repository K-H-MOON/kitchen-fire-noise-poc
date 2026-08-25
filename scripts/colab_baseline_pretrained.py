# ===== 공개 사전학습 fire·smoke 모델 baseline (배찬우 리서치 검증) · Colab =====
#
# 목적: "공개 fire·smoke YOLO 를 바로 돌려 우리 문제를 풀 수 있나"를,
#   우리가 이미 우리 모델을 측정한 그 clean test set(oilfire_realtest)에
#   공개 모델을 그대로 얹어 **같은 지표(recall·fpr_급식실)** 로 직접 대조.
#   → 발표에서 "공개 모델 vs 우리 모델" 한 표 + 실패 케이스 몽타주.
#
# ▶ 선행조건: colab_build_firetest.py 로 oilfire_realtest 가 빌드돼 있어야 함
#   (fire/ · nofire_kitchen/ · nofire_presrc/). HANDOFF "재현 레시피" (1)(2) 먼저.
#
# env(선택): OUT_DIR(테스트셋 경로·build 와 동일)·CONF(0.25, 우리와 동일)·
#   EVAL_OUT(json)·PRETRAINED_REPO(HF repo)·PRETRAINED_FILE(가중치 파일명)·
#   INSP_DIR(몽타주 출력)·HF_TOKEN(게이트 repo 인 경우만)
#
# 산출: 콘솔 대조표 + baseline_pretrained.json + 실패 몽타주 3종(_strip.jpg, 채팅 첨부용)

import os, glob, json, subprocess, sys
import numpy as np
from PIL import Image
from google.colab import drive
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'], check=True)
    from ultralytics import YOLO
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
    from huggingface_hub import hf_hub_download

FIRE  = '/content/drive/MyDrive/fire_frames'
IFRUN = f'{FIRE}/runs_if'
RUNS  = f'{FIRE}/runs_phaseB'
TEST  = os.environ.get('OUT_DIR', f'{FIRE}/oilfire_realtest')
OUT   = os.environ.get('EVAL_OUT', f'{FIRE}/indoorfire_eval')
CONF  = float(os.environ.get('CONF', '0.25'))
INSP  = os.environ.get('INSP_DIR', '/content/inspect_baseline')
REPO  = os.environ.get('PRETRAINED_REPO', 'TommyNgx/YOLOv10-Fire-and-Smoke-Detection')
WFILE = os.environ.get('PRETRAINED_FILE', 'best.pt')

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)
os.makedirs(INSP, exist_ok=True)

fire_imgs = sorted(glob.glob(f'{TEST}/fire/*.jpg'))
cook_imgs = sorted(glob.glob(f'{TEST}/nofire_kitchen/*.jpg'))
pre_imgs  = sorted(glob.glob(f'{TEST}/nofire_presrc/*.jpg'))
assert fire_imgs, (f'양성 없음: {TEST}/fire — colab_build_firetest.py(빌드 모드) 먼저.\n'
                   f'  HANDOFF "A안 ④→⑤ 재현 레시피" (1)재clone+oilfire_raw · (2)test 빌드 실행 후 이 스크립트.')
print(f'test: fire {len(fire_imgs)} · nofire_kitchen {len(cook_imgs)} · nofire_presrc {len(pre_imgs)}')


def source(p):
    return os.path.basename(p).rsplit('_', 1)[0]


# --- 공개 모델 다운로드 ---
def load_pretrained():
    tok = os.environ.get('HF_TOKEN') or None
    try:
        path = hf_hub_download(repo_id=REPO, filename=WFILE, token=tok)
        print(f'공개 모델 다운로드 OK: {REPO}/{WFILE} -> {path}')
        return YOLO(path)
    except Exception as e:
        print(f'⚠️ 공개 모델 다운로드 실패: {e}')
        print('   게이트 repo 면 https://huggingface.co/settings/tokens 에서 read 토큰 만들어')
        print("   os.environ['HF_TOKEN']='hf_...' 넣고 재실행. (PRETRAINED_REPO/FILE 로 다른 모델 지정도 가능)")
        raise


# --- 클래스 이름 → fire/smoke id 매핑(모델마다 인덱스 다름 → 이름 기준) ---
def class_ids(model):
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    fire_ids  = {i for i, n in names.items() if 'fire'  in str(n).lower()}
    smoke_ids = {i for i, n in names.items() if 'smoke' in str(n).lower()}
    return names, fire_ids, smoke_ids


# --- 프레임별 fire/smoke 검출 여부 ---
def detect(model, paths, fire_ids, smoke_ids):
    fires, smokes = [], []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], conf=CONF, verbose=False):
            if len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                fires.append(bool(np.isin(cls, list(fire_ids)).any())  if fire_ids  else False)
                smokes.append(bool(np.isin(cls, list(smoke_ids)).any()) if smoke_ids else False)
            else:
                fires.append(False); smokes.append(False)
    return np.array(fires), np.array(smokes)


def scene_recall(paths, flags):
    d = {}
    for p, f in zip(paths, flags):
        d.setdefault(source(p), []).append(bool(f))
    return {s: float(np.mean(v)) for s, v in d.items()}


def montage(paths_mask, title, tag):
    """검출/실패 프레임 균등 최대 10장 → _strip.jpg (채팅 첨부용)."""
    picks = [p for p, m in paths_mask if m]
    if not picks:
        print(f'   [{tag}] 해당 프레임 0장 — 몽타주 생략')
        return
    NP = min(10, len(picks))
    sel = [picks[int(round(k * (len(picks) - 1) / max(1, NP - 1)))] for k in range(NP)]
    cols, cw = 5, 300
    im0 = Image.open(sel[0]); ch = round(cw * im0.height / im0.width)
    rows = (len(sel) + cols - 1) // cols
    sh = Image.new('RGB', (cols * cw, rows * ch), (16, 16, 16))
    for j, p in enumerate(sel):
        sh.paste(Image.open(p).convert('RGB').resize((cw, ch)), ((j % cols) * cw, (j // cols) * ch))
    outp = f'{INSP}/baseline_{tag}_strip.jpg'
    sh.save(outp, quality=88)
    print(f'   [{tag}] {title}: {len(picks)}장 중 {NP}장 -> {outp}')


# ---- 평가할 모델들: 공개 모델 + 우리 배포후보 (같은 test·같은 conf) ----
pub = load_pretrained()
pub_names, pub_fire, pub_smoke = class_ids(pub)
print(f'공개 모델 클래스: {pub_names} · fire_ids={pub_fire} · smoke_ids={pub_smoke}')

ours = {
    '2g_real_only_grouped':     f'{IFRUN}/real_only_grouped/weights/best.pt',
    '2ck_real_only_grouped_ck': f'{IFRUN}/real_only_grouped_ck/weights/best.pt',
}

rows = {}

# 공개 모델: fire-only / smoke / any(fire|smoke) 세 관점
pf_fire, pf_smoke = detect(pub, fire_imgs, pub_fire, pub_smoke)
pc_fire, pc_smoke = detect(pub, cook_imgs, pub_fire, pub_smoke) if cook_imgs else (np.array([]), np.array([]))
pp_fire, pp_smoke = detect(pub, pre_imgs,  pub_fire, pub_smoke) if pre_imgs  else (np.array([]), np.array([]))
pub_scene = scene_recall(fire_imgs, pf_fire)
rows[f'PUB {REPO.split("/")[-1]}'] = dict(
    recall_fire=float(pf_fire.mean()),
    scene_mean=float(np.mean(list(pub_scene.values()))), scene_std=float(np.std(list(pub_scene.values()))),
    n_scene=len(pub_scene),
    fpr_kitchen_fire=float(pc_fire.mean()) if len(pc_fire) else None,
    fpr_presrc_fire=float(pp_fire.mean()) if len(pp_fire) else None,
    # 수증기-강건성 관점(배찬우 §3): 조리 프레임을 'smoke' 로 부르나 / 불이든 연기든 경보 뜨나
    smoke_on_cook=float(pc_smoke.mean()) if len(pc_smoke) else None,
    anyalarm_on_cook=float((pc_fire | pc_smoke).mean()) if len(pc_fire) else None,
    recall_any=float((pf_fire | pf_smoke).mean()),
    scene_recall=pub_scene,
)

# 우리 모델(fire 단일 클래스)
for key, best in ours.items():
    if not os.path.exists(best):
        print(f'  [없음] {key}: {best}'); continue
    m = YOLO(best)
    _, fid, sid = class_ids(m)
    fid = fid or {0}
    of, _ = detect(m, fire_imgs, fid, set())
    oc, _ = detect(m, cook_imgs, fid, set()) if cook_imgs else (np.array([]), np.array([]))
    op, _ = detect(m, pre_imgs,  fid, set()) if pre_imgs  else (np.array([]), np.array([]))
    sc = scene_recall(fire_imgs, of)
    rows[key] = dict(recall_fire=float(of.mean()),
                     scene_mean=float(np.mean(list(sc.values()))), scene_std=float(np.std(list(sc.values()))),
                     n_scene=len(sc),
                     fpr_kitchen_fire=float(oc.mean()) if len(oc) else None,
                     fpr_presrc_fire=float(op.mean()) if len(op) else None,
                     smoke_on_cook=None, anyalarm_on_cook=None, recall_any=float(of.mean()),
                     scene_recall=sc)

# ---- 대조표 ----
print('\n' + '=' * 92)
print(f'공개 사전학습 baseline vs 우리 모델 · 같은 clean test · frame-level (conf {CONF})')
print('=' * 92)
hd = f'{"모델":<34}{"recall":>8}{"scene±std":>15}{"fpr_급식실":>11}{"fpr_발화전":>11}'
print(hd)
for k, r in rows.items():
    fk = f'{r["fpr_kitchen_fire"]:.3f}' if r["fpr_kitchen_fire"] is not None else '  -  '
    fp = f'{r["fpr_presrc_fire"]:.3f}'  if r["fpr_presrc_fire"]  is not None else '  -  '
    print(f'{k:<34}{r["recall_fire"]:>8.3f}{r["scene_mean"]:>9.3f}±{r["scene_std"]:.3f}{fk:>11}{fp:>11}')

pr = rows[f'PUB {REPO.split("/")[-1]}']
print(f'\n[공개 모델 수증기-강건성 관점]')
print(f'  조리 프레임을 smoke 로 검출:  {pr["smoke_on_cook"]}')
print(f'  조리 프레임에 (불|연기) 경보:  {pr["anyalarm_on_cook"]}   (fire-only fpr={pr["fpr_kitchen_fire"]})')
print(f'  불 프레임 recall (불|연기):    {pr["recall_any"]:.3f}   (fire-only={pr["recall_fire"]:.3f})')

print(f'\n장면별 recall (공개 모델, 유효 N=장면수 {pr["n_scene"]}):')
for s, v in sorted(pr['scene_recall'].items()):
    print(f'      {s:<8} {v:.3f}')

# ---- 실패 케이스 몽타주(공개 모델) ----
print('\n[공개 모델 실패 케이스 몽타주 — 채팅 첨부용]')
montage(list(zip(fire_imgs, ~pf_fire)), '놓친 불(fire 미검출)', 'missed_fire')
if len(pc_fire):
    montage(list(zip(cook_imgs, pc_fire)),  '조리 헛불(fire 오검출)', 'cook_falsefire')
    montage(list(zip(cook_imgs, pc_smoke)), '수증기→smoke 오검출', 'cook_smoke')

print('\n해석: recall↑=실화재 잘 잡음 · fpr_급식실↓=실제 조리에 헛불 적음(배포 핵심).')
print('  공개 모델이 recall 높아도 fpr_급식실 높으면 → "그냥 갖다 쓰면 헛불" = 우리 ⑤ 조리 하드네거의 가치.')
print('  공개 모델 recall 도 낮으면 → 우리 실데이터 학습의 가치. (어느 쪽이든 우리 서사 강화)')
print('경계: frame-level(위치 무시·관대) · 장면 N 작음 · 저해상 다수 · 우리 conf 0.25 로 통일(공개 모델 최적 conf 아닐 수 있음).')

json.dump({'conf': CONF, 'pretrained_repo': REPO, 'pretrained_file': WFILE,
           'pretrained_classes': {int(k): v for k, v in pub_names.items()},
           'n_fire': len(fire_imgs), 'n_cook': len(cook_imgs), 'n_pre': len(pre_imgs),
           'rows': rows}, open(f'{OUT}/baseline_pretrained.json', 'w'),
          ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/baseline_pretrained.json')
