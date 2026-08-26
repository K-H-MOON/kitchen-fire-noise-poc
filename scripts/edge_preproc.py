# ===== 엣지/소벨 전처리 — 공유 단일 소스 (#3) =====
#
# 목적(가설): 급식실 헛불의 상당수가 '색 기반' 오검(금색 튀김음식·스테인리스 반사·스팀
#   = 불색이지만 구조는 불꽃 아님)이다. 이미지에 엣지(소벨) 구조 정보를 주입하면
#   색이 아닌 '형태/윤곽'으로도 판별하게 되어 색-헛불이 줄 수 있다.
#   (사전확률 낮음 — 색을 빼면 '불색' 정의가 손실돼 recall 리스크. 낮은 사전확률은
#    스킵 사유 아님: 도메인갭 지도 완성용 falsifiable 실험. [[no-premature-conclusions]])
#
# ★핵심 설계 원칙: **train·Indoor·oilfire_realtest·조리네거 전부 '완전히 동일한' 전처리**를
#   거쳐야 비교가 유효. 그래서 변환 함수를 이 한 파일에 두고 학습(colab_indoorfire_split_audit.py)과
#   평가(colab_realtest_eval.py)가 둘 다 여기서 import 한다(코드 중복→드리프트=실험 오염 방지).
#
# 4ch 수술(모델 입력 채널 추가)은 회피 — 3채널을 유지하며 아래 세 모드로 엣지를 섞는다:
#   sobelb   : B(채널0, BGR) ← 소벨 엣지. R·G(불의 난색)는 보존 → recall 리스크 최저. **1순위.**
#   blend    : RGB 전부 보존 + 엣지를 alpha 로 덧입힘(구조 강조, 색도 유지). 2순위.
#   edgegray : 순수 소벨(색 완전 제거·3채널 복제). recall 리스크 최대 — 도메인갭 지도 완성용.
#
# env(두 스크립트 공통): EDGE_MODE(''=off)·EDGE_GAIN(소벨 스케일, 기본 0.5)·EDGE_ALPHA(blend, 기본 0.4)

import cv2
import numpy as np

MODES = ('sobelb', 'blend', 'edgegray')


def edge_suffix(mode):
    """모델명/파일명 접미사(학습·평가가 같은 이름을 쓰도록 단일 소스). 예: sobelb → '_edge_sobelb'."""
    return f'_edge_{mode}'


def _imread(path):
    """유니코드 경로 안전 읽기(cv2.imread 는 비-ASCII 경로에 취약)."""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path, img, quality=95):
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        buf.tofile(path)
    return ok


def _sobel_mag(gray, gain):
    """소벨 그래디언트 크기 → uint8. gain 은 고정 스케일(프레임마다 min-max 하면
    밝기가 프레임 의존이 돼 train/eval 불일치 → 고정 스케일 + clip 로 일관성 유지)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) * float(gain)
    return np.clip(mag, 0, 255).astype(np.uint8)


def edge_transform(bgr, mode='sobelb', gain=0.5, alpha=0.4):
    """BGR uint8 → 엣지 전처리된 BGR uint8(3채널 유지). 학습·평가 공통 진입점."""
    if bgr is None:
        raise ValueError('edge_transform: 입력 이미지가 None (읽기 실패)')
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mag = _sobel_mag(gray, gain)
    if mode == 'sobelb':                       # B 채널만 엣지로 교체(R·G=불색 보존)
        out = bgr.copy()
        out[:, :, 0] = mag
    elif mode == 'blend':                      # 색 전부 보존 + 엣지 덧입힘
        edge3 = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR).astype(np.float32)
        out = np.clip((1.0 - alpha) * bgr.astype(np.float32) + alpha * edge3, 0, 255).astype(np.uint8)
    elif mode == 'edgegray':                   # 순수 구조(색 제거)
        out = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f'알 수 없는 EDGE_MODE={mode!r} (택1: {MODES})')
    return out


def edge_transform_file(src, dst, mode='sobelb', gain=0.5, alpha=0.4, quality=95):
    """파일 → 엣지 전처리 → 파일(jpg). 학습이 디스크에 쓰고 평가도 같은 jpg 파이프라인을
    거치게 해 jpeg 재압축까지 동일 조건으로 맞춘다. 성공 여부 bool 반환."""
    img = _imread(src)
    if img is None:
        return False
    return _imwrite(dst, edge_transform(img, mode, gain, alpha), quality)
