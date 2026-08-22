# ===== v2 예시 그림 — C0(플랫) vs C3(발광) 같은 배경 비교 시트 (Colab) =====
#
# 목적: 팀 설명용. 같은 배경·같은 불꽃에 C0(v1 알파 오버) vs C3(v2 발광 합성)을
#       나란히 붙여 "발광 합성이 뭘 바꿨나"를 한눈에 보여준다.
# 방법: 이미 생성된 synth_C0 / synth_C3 의 **같은 파일명(=같은 배경·불꽃)** 한 쌍을 골라
#       라벨 붙여 저장(재합성 아님 → 실제 학습에 쓴 이미지 그대로).
# 산출: Drive fire_frames/v2_examples/v2_c0_c3.jpg
#       → 다운로드해 repo docs/img/v2_c0_c3.jpg 로 커밋하면 SUMMARY 에 표시됨.

import os, glob
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive

FIRE = '/content/drive/MyDrive/fire_frames'
OUT  = f'{FIRE}/v2_examples'
SPLIT = os.environ.get('EX_SPLIT', 'test')          # test/val/train
PICK  = os.environ.get('EX_NAME', '')               # 특정 파일명 강제(선택)

drive.mount('/content/drive')
os.makedirs(OUT, exist_ok=True)

def index(cond):
    return {os.path.basename(p): p
            for p in glob.glob(f'{FIRE}/synth_{cond}/{SPLIT}/images/*.jpg')}

c0, c3 = index('C0'), index('C3')
common = sorted(set(c0) & set(c3))
if not common:
    raise SystemExit(f'synth_C0 / synth_C3 의 {SPLIT} 공통 파일명이 없음 — 경로/생성 확인')

def positive(n):                                    # 불꽃 있는(양성) 이미지 우선
    lab = f'{FIRE}/synth_C0/{SPLIT}/labels/{os.path.splitext(n)[0]}.txt'
    return os.path.exists(lab) and os.path.getsize(lab) > 0

pos = [n for n in common if positive(n)]
pick = PICK if (PICK in common) else (pos[0] if pos else common[0])
print(f'선택: {pick}  (공통 {len(common)} · 양성 {len(pos)})')
print('다른 이미지로 바꾸려면 EX_NAME 환경변수로 파일명 지정.')

def panel(path, label):
    im = Image.open(path).convert('RGB'); W = 500; h = round(W * im.height / im.width)
    im = im.resize((W, h))
    c = Image.new('RGB', (W, h + 36), (20, 20, 20)); c.paste(im, (0, 36))
    try:
        F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    except Exception:
        F = ImageFont.load_default()
    ImageDraw.Draw(c).text((8, 7), label, fill=(255, 255, 255), font=F)   # 한글 미지원 → 영문
    return c

A = panel(c0[pick], 'C0  (v1: flat alpha-over)')
B = panel(c3[pick], 'C3  (v2: glow = screen + core-bloom)')
G = 12
sheet = Image.new('RGB', (A.width + B.width + G, max(A.height, B.height)), (20, 20, 20))
sheet.paste(A, (0, 0)); sheet.paste(B, (A.width + G, 0))
dst = f'{OUT}/v2_c0_c3.jpg'
sheet.save(dst, quality=90)
print(f'-> {dst}')
print('  이 파일을 다운로드해 repo의 docs/img/v2_c0_c3.jpg 로 커밋하세요.')
