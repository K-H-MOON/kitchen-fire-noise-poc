# ===== 불꽃 출처를 train / test 풀로 배정 (사전 등록) =====
#
# 누수 규칙 3: 같은 불꽃 스프라이트가 train 과 test 에 동시에 있으면 모델이 그
# 불꽃 모양을 외운다. 그래서 **출처 단위**로 풀을 나눈다 — 한 출처에서 나온 매트는
# 전부 같은 풀로만 간다.
#
# 배경(주방) 분할과 다른 축이다. 최종 합성은 (배경 세트) × (불꽃 풀) 로 조합되며,
# train 합성 = train 배경 + train 불꽃, test 합성 = test 배경 + test 불꽃 이 되도록
# 합성 스크립트에서 맞춘다.
#
# 시드 고정 단순 무작위. 표준 라이브러리만. val 은 따로 두지 않는다 — 불꽃 소재 수가
# 적어 train/test 두 풀로만 나눈다(임계값 튜닝은 배경 val 로 한다).
#
# 주의 — flames.json 의 shots 가 채워진(=실제로 쓸) 출처만 배정한다. shots 가 빈
# 출처는 아직 불꽃 구간을 확인하지 않은 것이므로 제외하고, 경고로 알린다.

import json
import os
import sys
from random import Random

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FLAMES = os.path.join(HERE, 'flames.json')
OUT = os.path.join(HERE, 'flame_split.json')

TEST_FRAC = 0.30    # 출처의 약 30% 를 test 풀로
SEED = 7

inv = json.load(open(FLAMES, encoding='utf-8'))
ready = [s['key'] for s in inv['sources'] if s.get('shots')]      # shots 채워진 것만
pending = [s['key'] for s in inv['sources'] if not s.get('shots')]

print('=' * 66)
print('불꽃 출처 → train / test 풀 배정 (출처 단위)')
print('=' * 66)
if pending:
    print(f'  [대기] shots 미확인 {len(pending)}개 — 배정에서 제외:')
    print(f'         {" · ".join(pending)}')
print(f'  배정 대상(shots 확인됨) {len(ready)}개')

if not ready:
    print('\n  아직 불꽃 구간이 확정된 출처가 없음 — flames.json 의 shots 를 먼저 채울 것.')
    print('  (지금은 배정할 것이 없어 flame_split.json 을 쓰지 않음)')
    raise SystemExit(0)

n_test = max(1, round(len(ready) * TEST_FRAC))
test = sorted(Random(SEED).sample(ready, n_test))
train = sorted(set(ready) - set(test))

# 검산 — 겹치지 않고 합이 전체
assert set(train) & set(test) == set(), '풀이 겹침'
assert set(train) | set(test) == set(ready), '빠진 출처가 있음'

print(f'\n  시드 {SEED} · test 비율 {TEST_FRAC:.0%}')
print(f'  train 풀 {len(train)}개  {" · ".join(train)}')
print(f'  test  풀 {len(test)}개  {" · ".join(test)}')

out = {
    'unit': 'flame_source',
    'seed': SEED,
    'test_frac': TEST_FRAC,
    'note': '불꽃 출처를 풀로 분리. 같은 출처의 매트는 한 풀로만. 합성 때 배경 세트와 풀을 맞춘다.',
    'pools': {'train': train, 'test': test},
    'pending': pending,
}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n-> {os.path.basename(OUT)}')
