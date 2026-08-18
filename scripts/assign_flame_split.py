# ===== 불꽃 출처를 train / test 풀로 배정 (사전 등록) =====
#
# 누수 규칙 3: 같은 불꽃 스프라이트가 train 과 test 에 동시에 있으면 모델이 그
# 불꽃 모양을 외운다. 그래서 **출처 단위**로 풀을 나눈다 — 한 출처에서 나온 매트는
# 전부 같은 풀로만 간다.
#
# 왜 무작위가 아니라 지정 배정인가 — 통과한 불꽃 출처가 7개뿐이라 시드 무작위는
# 쉽게 치우친다(실제로 시드 7 은 가장 약한 두 소재를 test 에 몰았다). 배경 사이트를
# 지정 배정한 것과 같은 방식으로, test 에 **잘 형성됐고 train 과 구별되는** 불꽃을
# 넣는다. 쉬운 것을 고르는 게 아니라 *구별되는* 것을 고른다 — 일반화를 재기 위함.
#
# 배정 근거:
#   test  clean_pan   — 어두운 벽 배경의 강한 세로 불꽃. 형태가 뚜렷.
#         grease_prev — 구리팬·붉은 전열코일 세팅. train 의 냄비·가스레인지와 구별됨.
#   train 나머지 4개 (tempura01·reproduce·low_oil·dirty_pan). konro_ignite 는 오염 탈락.
#
# shots 가 빈(=제외된) 출처는 배정하지 않는다. 표준 라이브러리만.

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FLAMES = os.path.join(HERE, 'flames.json')
OUT = os.path.join(HERE, 'flame_split.json')

# 지정 배정 — 통과한 6개 출처만. 근거는 위 주석.
# konro_ignite 는 빨간 LED 타이머 오염(불꽃까지 빨강이라 색 구별 불가)으로 탈락 —
# 자세한 사유는 flames.json 의 konro_ignite _excluded 와 docs/TIMELINE.md.
POOL = {
    'clean_pan':    'test',
    'grease_prev':  'test',
    'tempura01':    'train',
    'reproduce':    'train',
    'low_oil':      'train',
    'dirty_pan':    'train',
}

inv = json.load(open(FLAMES, encoding='utf-8'))
ready = sorted(s['key'] for s in inv['sources'] if s.get('shots'))
excluded = sorted(s['key'] for s in inv['sources'] if not s.get('shots'))

# ---------------------------------------------------------------------------
# 검산 — 배정이 '통과한 출처' 와 정확히 맞물리는가
# ---------------------------------------------------------------------------
assert set(POOL) == set(ready), (
    f'배정과 통과 출처가 다름\n'
    f'  통과인데 배정 없음: {set(ready) - set(POOL)}\n'
    f'  배정했는데 통과 아님: {set(POOL) - set(ready)}')

train = sorted(k for k, v in POOL.items() if v == 'train')
test = sorted(k for k, v in POOL.items() if v == 'test')
assert set(train) & set(test) == set(), '풀이 겹침'

print('=' * 66)
print('불꽃 출처 → train / test 풀 배정 (출처 단위, 지정)')
print('=' * 66)
if excluded:
    print(f'  [제외] shots 비어 배정 안 함 {len(excluded)}개: {" · ".join(excluded)}')
print(f'  통과 출처 {len(ready)}개')
print(f'\n  train 풀 {len(train)}개  {" · ".join(train)}')
print(f'  test  풀 {len(test)}개  {" · ".join(test)}')

out = {
    'unit': 'flame_source',
    'note': '불꽃 출처를 풀로 분리(지정). 같은 출처의 매트는 한 풀로만. 합성 때 배경 세트와 풀을 맞춘다.',
    'pools': {'train': train, 'test': test},
    'excluded': excluded,
}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n-> {os.path.basename(OUT)}')
