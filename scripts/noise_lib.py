# ===== 공유 노이즈 라이브러리 =====
# Phase A(테스트 손상)와 Phase B(학습 증강)가 **같은 구현**을 써야 "노이즈 X 로
# 학습하면 X 에 강해지나" 가 깨끗하게 성립한다. 그래서 한 곳에 모은다.
# 모든 함수 시그니처는 (img_rgb_uint8, severity, rng) — rng 안 쓰는 것도 받기만 함.

import numpy as np, cv2

SEVS = [0, 1, 2, 3, 4, 5]


def n_gaussian(img, s, rng):
    std = [0, 8, 16, 28, 44, 64][s]
    return img if std == 0 else np.clip(img + rng.normal(0, std, img.shape), 0, 255).astype(np.uint8)


def n_jpeg(img, s, rng):
    q = [100, 40, 25, 15, 9, 5][s]
    if q >= 100:
        return img
    ok, enc = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]


def n_motion(img, s, rng):
    k = [0, 5, 9, 13, 19, 27][s]
    if k <= 1:
        return img
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker)


def n_defocus(img, s, rng):
    sig = [0, 1, 2, 3.5, 5, 7][s]
    return img if sig == 0 else cv2.GaussianBlur(img, (0, 0), sigmaX=sig)


def n_lowlight(img, s, rng):
    f = [1.0, 0.7, 0.5, 0.35, 0.22, 0.13][s]
    return img if f >= 1.0 else np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def n_contrast(img, s, rng):
    c = [1.0, 0.75, 0.58, 0.42, 0.30, 0.20][s]
    if c >= 1.0:
        return img
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)


def n_grayscale(img, s, rng):
    f = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0][s]
    if f == 0:
        return img
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
    return np.clip(img * (1 - f) + g * f, 0, 255).astype(np.uint8)


def n_steam(img, s, rng):
    if s == 0:
        return img
    nb, alpha = [(0, 0), (2, 0.20), (3, 0.30), (4, 0.42), (5, 0.55), (6, 0.68)][s]
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.float32)
    for _ in range(nb):
        cx, cy = rng.randint(0, W), rng.randint(int(H * 0.3), H)
        r = rng.randint(int(min(H, W) * 0.15), int(min(H, W) * 0.4))
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(H, W) * 0.06) * alpha
    white = np.full_like(img, 235)
    return np.clip(img * (1 - mask[..., None]) + white * mask[..., None], 0, 255).astype(np.uint8)


def n_erasing(img, s, rng):
    if s == 0:
        return img
    nb, frac = [(0, 0), (1, 0.10), (2, 0.15), (3, 0.20), (4, 0.28), (5, 0.35)][s]
    H, W = img.shape[:2]
    out = img.copy()
    for _ in range(nb):
        bw, bh = int(W * frac), int(H * frac)
        x, y = rng.randint(0, max(1, W - bw)), rng.randint(0, max(1, H - bh))
        out[y:y + bh, x:x + bw] = 128
    return out


NOISE = {'gaussian': n_gaussian, 'jpeg': n_jpeg, 'motion_blur': n_motion, 'defocus': n_defocus,
         'low_light': n_lowlight, 'contrast': n_contrast, 'steam': n_steam,
         'grayscale': n_grayscale, 'random_erasing': n_erasing}

ALL9 = ['gaussian', 'jpeg', 'motion_blur', 'defocus', 'low_light', 'contrast',
        'steam', 'grayscale', 'random_erasing']
QUALITY = ['gaussian', 'jpeg', 'motion_blur', 'defocus', 'low_light', 'contrast']  # model_B 학습군
HELDOUT = ['steam', 'grayscale', 'random_erasing']                                  # model_B held-out


def apply_random(img, names, rng):
    """무작위 노이즈 하나를 강도 1~5 로 적용 (Phase B 오프라인 증강용)."""
    name = names[rng.randint(len(names))]
    sev = rng.randint(1, 6)
    return NOISE[name](img, sev, rng)
