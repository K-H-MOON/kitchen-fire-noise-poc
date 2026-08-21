# ===== v3 Phase 0: 고정특징 전이 프로브 — DINOv3 vs YOLO (Colab, GPU 권장) =====
#
# pre-reg v3 §2~§6 — "실제-프리트레인 파운데이션 ViT(DINOv3) 표현이 합성→실제 전이가
# YOLO보다 나은가?"를 **백본 학습 없이** 단독 검증(단일변수=표현).
#
# 절차:
#   1) 데이터 — fit: synth_C0/train(라벨=양성 / 빈라벨=음성) · sanity: synth_C0/val
#      전이시험: realfire(real_fire.json → ffmpeg 프레임, fire/nofire, 영상단위)
#   2) 백본별(전부 고정) 특징 추출 → L2정규화
#      - DINOv3(폴백 DINOv2) · YOLO(v8_C0_s1 합성) · YOLO(COCO yolov8s) · ResNet50(ImageNet)
#   3) 선형 프로브(로지스틱) + kNN 을 synth train 으로 fit
#   4) 평가 — synth-val AUROC(in-dist 게이트) · realfire AUROC(영상단위±군집CI, 주지표)
#      + synth-val 보정 임계값에서 realfire recall/fpr(YOLO flame_rate 비교용)
#   5) 판정 — pre-reg v3 §6 (ΔAUROC = DINO − YOLO_synth)
#
# 선행: colab_v2_train.py(v8_C0_s1) · synth_C0 생성 · real_fire.json 채워짐.
# 환경: PROBE_BACKBONE(콤마구분 선택) · DINO_HUB(강제 dinov3/dinov2) · MAX_PER_CLASS(기본 1500)

import os, glob, json, subprocess, sys, unicodedata, random
import numpy as np

def pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=False)

try:
    import torch
except ImportError:
    pip('torch'); import torch
from PIL import Image
from google.colab import drive
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
except ImportError:
    pip('scikit-learn')
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score

REPO = '/content/kitchen-fire-noise-poc/scripts'
FIRE = '/content/drive/MyDrive/fire_frames'
RUNS = f'{FIRE}/runs_phaseB'
OUT  = f'{FIRE}/v3_probe'
COND = 'C0'                                   # v1 기준선 합성으로 fit (§4)
SYN_TRAIN = f'{FIRE}/synth_{COND}/train'
SYN_VAL   = f'{FIRE}/synth_{COND}/val'
YOLO_SYNTH = f'{RUNS}/v8_C0_s1/best.pt'       # 전이에 실패한 그 표현(핵심 대조)
CACHE = '/content/_rf_v3'
CONF  = 0.25                                  # 부지표 임계값 보정 목표 FPR 대용 아님(아래 THR_FPR)
THR_FPR = 0.10                                # synth-val 음성 기준 목표 FPR로 임계값 보정
MAX_PER_CLASS = int(os.environ.get('MAX_PER_CLASS', '1500'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(0); np.random.seed(0)

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

def norm(s):
    return unicodedata.normalize('NFC', s)

# ---------------------------------------------------------------------------
# 1) 데이터 수집
# ---------------------------------------------------------------------------
def synth_split(root):
    """synth 디렉터리 → (양성 경로, 음성 경로). 라벨 빈 파일 = 음성 (ablation과 동일 규칙)."""
    pos, neg = [], []
    for p in sorted(glob.glob(f'{root}/images/*.jpg')):
        lab = f'{root}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
        (pos if os.path.exists(lab) and os.path.getsize(lab) > 0 else neg).append(p)
    return pos, neg

def cap(lst, n):
    if len(lst) <= n:
        return lst
    random.shuffle(lst)
    return sorted(lst[:n])

tr_pos, tr_neg = synth_split(SYN_TRAIN)
va_pos, va_neg = synth_split(SYN_VAL)
tr_pos, tr_neg = cap(tr_pos, MAX_PER_CLASS), cap(tr_neg, MAX_PER_CLASS)
print(f'synth train  양성 {len(tr_pos)} · 음성 {len(tr_neg)}   (cap {MAX_PER_CLASS})')
print(f'synth val    양성 {len(va_pos)} · 음성 {len(va_neg)}')

def extract_realfire():
    """real_fire.json → 영상별 fire/nofire 프레임 (colab_v2_eval 과 동일 로직·캐시)."""
    inv = json.load(open(f'{REPO}/real_fire.json', encoding='utf-8'))
    SRC, FPS = inv['src_dir'], inv['fps']
    allf = glob.glob(f'{SRC}/*')
    frames = {}
    for s in inv['sources']:
        if not s.get('fire_shots'):
            continue
        hit = [p for p in allf if norm(s['file']) in norm(os.path.basename(p))]
        if len(hit) != 1:
            print(f'  [건너뜀] {s["key"]}: 영상 매칭 {len(hit)}개'); continue
        src = hit[0]; frames[s['key']] = {'fire': [], 'nofire': []}
        for kind, shots in (('fire', s['fire_shots']), ('nofire', s.get('nofire_shots', []))):
            for a, b in shots:
                d = f'{CACHE}/{s["key"]}/{kind}/{a}_{b}'
                os.makedirs(d, exist_ok=True)
                if not glob.glob(f'{d}/*.jpg'):
                    vf = f'fps={FPS}'
                    cr = s.get('crop')
                    if cr:
                        vf = f'crop=iw:ih*{1 - cr[0] - cr[1]}:0:ih*{cr[0]},' + vf
                    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(a), '-i', src,
                                    '-t', str(round(b - a + 1, 2)), '-vf', vf, '-q:v', '2',
                                    f'{d}/%05d.jpg'], check=False)
                frames[s['key']][kind] += sorted(glob.glob(f'{d}/*.jpg'))
        print(f'  {s["key"]:<14} fire {len(frames[s["key"]]["fire"]):>3} · '
              f'nofire {len(frames[s["key"]]["nofire"]):>3}')
    return frames

print('realfire 프레임 추출(캐시)...')
RF = extract_realfire()

# --- (B) realfire 확장 — 추가 실제 화재 이미지로 검정력↑ (Roboflow 등, 학습 0) ---
# REALFIRE_EXTRA 폴더 구조:  <dir>/fire/<source>/*.jpg · <dir>/nofire/<source>/*.jpg
#   각 <source> 하위폴더 = 독립 '씬'(군집 CI 단위). 하위폴더 없으면 통째로 1 씬(extra).
# 이렇게 하면 pooled 표본↑ + 군집 수↑ → v3 1·2차의 N=5 거대 CI 를 좁힘.
EXTRA = os.environ.get('REALFIRE_EXTRA', '').strip()
if EXTRA and os.path.isdir(EXTRA):
    def _scan(kind):
        base = f'{EXTRA}/{kind}'; out = {}
        if not os.path.isdir(base):
            return out
        subs = [d for d in sorted(glob.glob(f'{base}/*')) if os.path.isdir(d)]
        if subs:
            for d in subs:
                out[os.path.basename(d)] = sorted(glob.glob(f'{d}/*.jpg') + glob.glob(f'{d}/*.png') +
                                                  glob.glob(f'{d}/*.jpeg'))
        else:
            flat = sorted(glob.glob(f'{base}/*.jpg') + glob.glob(f'{base}/*.png') + glob.glob(f'{base}/*.jpeg'))
            if flat:
                out['_flat'] = flat
        return out
    fm, nm = _scan('fire'), _scan('nofire')
    print(f'realfire 확장(REALFIRE_EXTRA={EXTRA})...')
    for sc in sorted(set(fm) | set(nm)):
        key = f'extra_{sc}'
        RF[key] = {'fire': fm.get(sc, []), 'nofire': nm.get(sc, [])}
        print(f'  {key:<20} fire {len(RF[key]["fire"]):>4} · nofire {len(RF[key]["nofire"]):>4}')

VIDEOS = list(RF.keys())
if not VIDEOS:
    raise SystemExit('realfire 영상이 없음 — real_fire.json 확인')

# ---------------------------------------------------------------------------
# 2) 백본 로더 — 각각 embed(paths)->np[N,D] 반환 (전부 고정)
# ---------------------------------------------------------------------------
from torchvision import transforms as T
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 작은 불꽃 보존 — ViT 는 고해상(448) + 패치토큰 mean/max 풀링, CNN 은 conv맵 avg/max 풀링.
# 전역 CLS/avgpool 하나만 쓰면 큰 프레임 속 작은 불꽃이 뭉개짐(v3 1차 sanity 실패 원인).
RES_VIT = int(os.environ.get('RES_VIT', '448'))
RES_CNN = int(os.environ.get('RES_CNN', '224'))
def _tf(r):
    return T.Compose([T.Resize((r, r)), T.ToTensor(), T.Normalize(*IMAGENET)])

def _load_batch(paths, r):
    return torch.stack([_tf(r)(Image.open(p).convert('RGB')) for p in paths]).to(DEV)

def load_dino():
    """DINOv3 우선, 실패 시 DINOv2 폴백. (used_name, embed_fn) 반환.
    특징 = concat[CLS, mean(patch), max(patch)] (작은 불꽃 국소신호 보존)."""
    pip('torchmetrics')                       # dinov3 hub 의존성(없으면 로드 실패했었음)

    # --- (신규·우선) HF DINOv3 — 게이트 모델. HF_TOKEN 필요(사용자가 라이선스 동의+토큰) ---
    hf_repo = os.environ.get('DINOV3_HF', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if hf_repo and hf_repo.lower() != 'skip':
        for attempt in range(2):
            try:
                from transformers import AutoModel
                m = AutoModel.from_pretrained(hf_repo, token=token).eval().to(DEV)
                nreg = int(getattr(m.config, 'num_register_tokens', 0) or 0)
                print(f'  [DINO] HF {hf_repo} 로드됨 (res {RES_VIT} · register {nreg})')
                def embed(paths, _m=m, _nreg=nreg):
                    out = []
                    with torch.no_grad():
                        for i in range(0, len(paths), 32):
                            h = _m(_load_batch(paths[i:i+32], RES_VIT)).last_hidden_state
                            cls, pat = h[:, 0], h[:, 1 + _nreg:]     # CLS · (레지스터 건너뜀) 패치
                            out.append(torch.cat([cls, pat.mean(1), pat.amax(1)], 1).float().cpu().numpy())
                    return np.concatenate(out, 0)
                return hf_repo.split('/')[-1], embed
            except Exception as e:
                msg = str(e)
                if attempt == 0 and ('dinov3' in msg.lower() or 'model type' in msg.lower() or 'Unrecognized' in msg):
                    print('  [DINO] transformers 업그레이드 후 재시도...'); pip('-U', 'transformers'); continue
                gated = '401' in msg or '403' in msg or 'gated' in msg.lower() or 'awaiting' in msg.lower()
                hint = ' → HF 라이선스 동의+HF_TOKEN 설정 필요(아래 안내)' if gated else ''
                print(f'  [DINO] HF {hf_repo} 실패 — {msg[:140]}{hint}')
                break

    # --- torch.hub 경로 (DINOv3 가중치는 403일 수 있음 → DINOv2로 폴백) ---
    forced = os.environ.get('DINO_HUB', '').strip()
    if forced:
        trials = [(forced.split('/')[-1], forced.rsplit('/', 1)[0] if '/' in forced else 'facebookresearch/dinov3')]
    else:
        trials = [('dinov2_vitb14', 'facebookresearch/dinov2')]
    for entry, repo in trials:
        try:
            m = torch.hub.load(repo, entry, trust_repo=True).eval().to(DEV)
            print(f'  [DINO] torch.hub {repo}:{entry} 로드됨 (res {RES_VIT})')
            def embed(paths, _m=m):
                out = []
                with torch.no_grad():
                    for i in range(0, len(paths), 32):
                        x = _load_batch(paths[i:i+32], RES_VIT)
                        f = _m.forward_features(x)              # dict: x_norm_clstoken · x_norm_patchtokens
                        cls, pat = f['x_norm_clstoken'], f['x_norm_patchtokens']
                        emb = torch.cat([cls, pat.mean(1), pat.amax(1)], 1)
                        out.append(emb.float().cpu().numpy())
                return np.concatenate(out, 0)
            return entry, embed
        except Exception as e:
            print(f'  [DINO] {repo}:{entry} 실패 — {str(e)[:120]}')
    # HF transformers 폴백 (DINOv2 base — 게이트 없음)
    try:
        pip('transformers')
        from transformers import AutoModel
        m = AutoModel.from_pretrained('facebook/dinov2-base').eval().to(DEV)
        print('  [DINO] HF facebook/dinov2-base 로드됨(폴백)')
        def hf_embed(paths, _m=m):
            out = []
            with torch.no_grad():
                for i in range(0, len(paths), 32):
                    h = _m(_load_batch(paths[i:i+32], RES_VIT)).last_hidden_state
                    cls, pat = h[:, 0], h[:, 1:]
                    emb = torch.cat([cls, pat.mean(1), pat.amax(1)], 1)
                    out.append(emb.float().cpu().numpy())
            return np.concatenate(out, 0)
        return 'dinov2-base(hf)', hf_embed
    except Exception as e:
        raise SystemExit(f'DINO 로드 전부 실패: {e}')

def load_yolo_backbone(path, tag, allow_download=False):
    try:
        from ultralytics import YOLO
    except ImportError:
        pip('ultralytics'); from ultralytics import YOLO
    if not allow_download and not os.path.exists(path):
        print(f'  [{tag}] {path} 없음 — 건너뜀'); return None
    try:
        m = YOLO(path)                          # COCO 가중치명이면 자동 다운로드
    except Exception as e:
        print(f'  [{tag}] 로드 실패 — {str(e)[:100]}'); return None
    def embed(paths, _m=m):
        out = []
        for i in range(0, len(paths), 64):
            es = _m.embed(paths[i:i+64], verbose=False)     # list of 1D tensors
            out.append(np.stack([e.float().cpu().numpy().ravel() for e in es]))
        return np.concatenate(out, 0)
    return embed

def load_resnet():
    """conv 특징맵(layer4, [B,2048,H,W]) → concat[avgpool, maxpool] (작은 불꽃 국소신호 보존)."""
    from torchvision import models
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(DEV)
    feat = torch.nn.Sequential(*list(base.children())[:-2])   # avgpool·fc 제거 → 특징맵
    def embed(paths):
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                fm = feat(_load_batch(paths[i:i+64], RES_CNN))   # [B,2048,H,W]
                emb = torch.cat([fm.mean((2, 3)), fm.amax((2, 3))], 1)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out, 0)
    return embed

# 백본 등록 (순서 = 표에 나올 순서)
BACKBONES = {}
want = os.environ.get('PROBE_BACKBONE', 'dino,yolo_synth,yolo_coco,resnet').split(',')
dino_used = None
if 'dino' in want:
    dino_used, fn = load_dino(); BACKBONES[f'DINO({dino_used})'] = fn
if 'yolo_synth' in want:
    fn = load_yolo_backbone(YOLO_SYNTH, 'YOLO_synth');  BACKBONES['YOLO(합성 v8_C0_s1)'] = fn if fn else None
if 'yolo_coco' in want:
    fn = load_yolo_backbone('yolov8s.pt', 'YOLO_coco', allow_download=True); BACKBONES['YOLO(COCO)'] = fn if fn else None
if 'resnet' in want:
    BACKBONES['ResNet50(ImageNet)'] = load_resnet()
BACKBONES = {k: v for k, v in BACKBONES.items() if v is not None}
KEY_DINO  = next((k for k in BACKBONES if k.startswith('DINO')), None)
KEY_YSYN  = next((k for k in BACKBONES if k.startswith('YOLO(합성')), None)

# ---------------------------------------------------------------------------
# 3~4) 백본별: 특징 추출 → 프로브 fit → 평가
# ---------------------------------------------------------------------------
def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def tcrit(df):
    return {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
            6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(df, 1.96)

def cluster_ci(pervid):
    pv = np.array(pervid, float)
    if len(pv) <= 1:
        return float(pv.mean()) if len(pv) else None, 0.0
    return float(pv.mean()), float(tcrit(len(pv) - 1) * pv.std(ddof=1) / np.sqrt(len(pv)))

results = {}
for name, embed in BACKBONES.items():
    print(f'\n=== {name} — 특징 추출 ===')
    Xtr = l2(np.concatenate([embed(tr_pos), embed(tr_neg)], 0))
    ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(tr_neg))]
    Xvp, Xvn = l2(embed(va_pos)), l2(embed(va_neg))

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(Xtr, ytr)
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(Xtr, ytr)

    def score(clf_or_knn, X):
        return clf_or_knn.predict_proba(X)[:, 1] if len(X) else np.array([])

    # synth-val in-dist AUROC (sanity 게이트)
    sv_s = np.r_[score(clf, Xvp), score(clf, Xvn)]
    sv_y = np.r_[np.ones(len(Xvp)), np.zeros(len(Xvn))]
    synth_auc = float(roc_auc_score(sv_y, sv_s)) if len(set(sv_y)) > 1 else None
    # synth-val 로 임계값 보정(목표 FPR)
    thr = float(np.quantile(score(clf, Xvn), 1 - THR_FPR)) if len(Xvn) else 0.5

    # realfire 전이 — 영상단위 AUROC(주지표) + recall/fpr(부지표)
    rf_feats = {v: {'fire': l2(embed(RF[v]['fire'])) if RF[v]['fire'] else np.empty((0, Xtr.shape[1])),
                    'nofire': l2(embed(RF[v]['nofire'])) if RF[v]['nofire'] else np.empty((0, Xtr.shape[1]))}
                for v in VIDEOS}
    per_auc_lin, per_auc_knn, per_recall, all_s, all_y = [], [], [], [], []
    for v in VIDEOS:
        fp, nf = rf_feats[v]['fire'], rf_feats[v]['nofire']
        sf, sn = score(clf, fp), score(clf, nf)
        kf, kn = score(knn, fp), score(knn, nf)
        if len(fp) and len(nf):
            per_auc_lin.append(float(roc_auc_score(np.r_[np.ones(len(fp)), np.zeros(len(nf))], np.r_[sf, sn])))
            per_auc_knn.append(float(roc_auc_score(np.r_[np.ones(len(fp)), np.zeros(len(nf))], np.r_[kf, kn])))
        if len(fp):
            per_recall.append(float((sf >= thr).mean()))
        all_s += list(sf) + list(sn); all_y += [1]*len(fp) + [0]*len(nf)

    pooled_auc = float(roc_auc_score(all_y, all_s)) if len(set(all_y)) > 1 else None
    auc_m, auc_ci = cluster_ci(per_auc_lin)
    knn_m, _ = cluster_ci(per_auc_knn)
    rec_m, rec_ci = cluster_ci(per_recall)
    fpr_all = float((np.array([s for s, y in zip(all_s, all_y) if y == 0]) >= thr).mean()) if all_y.count(0) else None

    results[name] = {'synth_val_auc': synth_auc, 'thr': thr,
                     'realfire_auc_lin': auc_m, 'realfire_auc_lin_ci': auc_ci,
                     'realfire_auc_lin_pervid': per_auc_lin,
                     'realfire_auc_knn': knn_m, 'realfire_auc_pooled': pooled_auc,
                     'realfire_recall': rec_m, 'realfire_recall_ci': rec_ci,
                     'realfire_fpr': fpr_all}
    print(f'  synth-val AUROC {synth_auc}  · realfire AUROC(영상단위) {auc_m:.3f} ± {auc_ci:.3f} '
          f'(kNN {knn_m:.3f}) · recall@thr {rec_m:.3f} · fpr {fpr_all}')

# ---------------------------------------------------------------------------
# 5) 표 + 판정 (pre-reg v3 §6)
# ---------------------------------------------------------------------------
print('\n' + '=' * 74)
print(f'v3 Phase 0 프로브 — realfire 전이 (영상 {len(VIDEOS)}개 · 고정특징 + 선형 프로브)')
print('=' * 74)
print(f'{"백본":<22}{"synth-val":>10}{"realfire AUROC":>18}{"kNN":>7}{"recall":>8}{"fpr":>7}')
for name, r in results.items():
    sa = f'{r["synth_val_auc"]:.3f}' if r['synth_val_auc'] is not None else '  -  '
    print(f'{name:<22}{sa:>10}{r["realfire_auc_lin"]:>11.3f} ±{r["realfire_auc_lin_ci"]:.3f}'
          f'{r["realfire_auc_knn"]:>7.3f}{r["realfire_recall"]:>8.3f}'
          f'{(r["realfire_fpr"] if r["realfire_fpr"] is not None else 0):>7.3f}')

print('\n' + '-' * 74)
verdict = {'used_dino': dino_used}
if KEY_DINO and KEY_YSYN:
    d, y = results[KEY_DINO], results[KEY_YSYN]
    delta = d['realfire_auc_lin'] - y['realfire_auc_lin']
    # per-video 방향 (짝 지어 비교 가능한 영상 수 기준)
    n = min(len(d['realfire_auc_lin_pervid']), len(y['realfire_auc_lin_pervid']))
    win = sum(1 for i in range(n) if d['realfire_auc_lin_pervid'][i] > y['realfire_auc_lin_pervid'][i])
    ci_no_overlap = (d['realfire_auc_lin'] - d['realfire_auc_lin_ci']) > (y['realfire_auc_lin'] + y['realfire_auc_lin_ci'])
    sanity_ok = (d['synth_val_auc'] or 0) >= 0.90
    verdict.update({'delta_auc': delta, 'pervid_win': f'{win}/{n}',
                    'ci_no_overlap': bool(ci_no_overlap), 'sanity_ok': bool(sanity_ok)})
    print(f'ΔAUROC (DINO − YOLO_synth) = {delta:+.3f}  · per-video {win}/{n} DINO 우세 · '
          f'CI 비겹침 {ci_no_overlap} · sanity(≥0.90) {sanity_ok}')
    if not sanity_ok:
        v = '판정 보류 — synth-val sanity 게이트 미달(특징 추출/데이터 점검)'
    elif delta >= 0.10 and (win >= 0.8 * n or ci_no_overlap):
        v = 'GO — 표현 축 유력 → (A) 고정백본 탐지기 착수 정당화 (pre-reg §6 GO)'
    elif abs(delta) < 0.05:
        v = 'NO-GO — 표현 축 아님 → 배경·씬·데이터·temporal 피벗 (pre-reg §6 NO-GO)'
    else:
        v = '애매 — 방향 지표로만, 팀과 판단(추가 검증·n↑) (pre-reg §6 애매)'
    verdict['verdict'] = v
    print(f'판정: {v}')
else:
    print('판정 불가 — DINO 또는 YOLO(합성) 백본 결과 없음')

print('\n※ 경계(§7): 이는 분류 전이 proxy지 최종 탐지 성능 아님 · realfire 5영상 CI 큼 · '
      'DINOv3 배포 무거움(이득이 비용 정당화하는지 별도).')

json.dump({'cond': COND, 'videos': VIDEOS, 'thr_fpr': THR_FPR,
           'max_per_class': MAX_PER_CLASS, 'results': results, 'verdict': verdict},
          open(f'{OUT}/v3_probe.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(f'\n-> {OUT}/v3_probe.json  (문서·시각화용)')

# 간단 막대그림(미팅용) — realfire AUROC ± CI
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(results.keys())
    vals = [results[n]['realfire_auc_lin'] for n in names]
    cis  = [results[n]['realfire_auc_lin_ci'] for n in names]
    plt.figure(figsize=(7, 3.6))
    plt.bar(range(len(names)), vals, yerr=cis, capsize=5,
            color=['#d1495b' if n == KEY_DINO else '#8d99ae' for n in names])
    plt.axhline(0.5, ls='--', c='k', lw=0.8, label='chance 0.5')
    labels_en = [n.replace('합성', 'synth').replace('영상단위', '') for n in names]
    plt.xticks(range(len(names)), labels_en, rotation=20, ha='right', fontsize=8)
    plt.ylabel('realfire AUROC (per-video +/- CI)'); plt.ylim(0, 1)
    plt.title('v3 Phase 0 - frozen-feature synth->real transfer'); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT}/v3_probe.png', dpi=130)
    print(f'-> {OUT}/v3_probe.png  (미팅용 막대그림)')
except Exception as e:
    print(f'  [그림 건너뜀] {e}')
