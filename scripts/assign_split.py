# ===== 사이트 단위 train / val / test 배정 (사전 등록) =====
#
# 이 스크립트는 **프레임을 한 장도 뽑기 전에** 돌린다. 배정이 장수에 의존하지
# 않기 때문이다 — 사이트(주방) 단위로만 나눈다. 프레임을 보기 전에 분할을 확정하는
# 것이 사전 등록의 요점이다.
#
# 왜 사이트 단위인가 — 데이터 누수를 막기 위해서다. 같은 주방의 프레임이 train 과
# test 에 동시에 있으면 모델이 배경·조명·각도를 외운다. 최종 합성 이미지가 서로
# 달라 보여도 누수다. 그래서 재료(주방) 단위로 먼저 나누고, 그다음에 각 세트 안에서
# 합성한다.
#
# 왜 무작위가 아니라 지정 배정인가 — 전 프로젝트(연기)는 "출처가 달랐으면 결론이
# 뒤집혔을까"를 재려고 시드 무작위를 썼다. 이 프로젝트의 목적은 다르다. 우리는
# **가장 배포에 가까운 환경(CCTV)에서 일반화되는가**를 재려 한다. 그래서 그 조건을
# test 에 의도적으로 넣는다. 배정 근거를 아래에 전부 적는다.
#
# 파이썬 표준 라이브러리만 쓴다. 어디서 돌려도 같은 결과가 나와야 한다.

import json
import os
import sys

# Windows 콘솔(cp949)에서도 한글·em-dash 출력이 깨지지 않게. Colab 은 원래 utf-8.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(HERE, 'videos.json')
OUT = os.path.join(HERE, 'split.json')

# ---------------------------------------------------------------------------
# 배정 — 사이트를 세 세트 중 하나에 넣는다. 근거를 값 옆에 적는다.
#
# test  개원중  — 유일한 CCTV. 실제 배포 환경에 가장 가깝다. 학습에서 한 번도
#                안 본 주방으로 두어 "진짜 처음 보는 급식실"의 성능을 잰다.
#       내곡중  — 국탕. test 에 튀김(개원중) 말고 다른 조리 종류를 하나 넣어
#                종류 편중을 줄인다.
# val   로봇고  — 튀김 영상 3개로 프레임이 넉넉하다. 임계값·하이퍼파라미터 튜닝용.
#       인화여중 — 볶음 3개. val 에 튀김(로봇고) 말고 볶음을 넣어 종류를 벌린다.
# train 나머지 9개 사이트.
#
# 종류 균형은 강제하지 않는다. 국탕·볶음 사이트가 각 3개뿐이라 세 세트에 다 넣기
# 어렵다. **사이트 분리가 1순위**, 종류 균형은 차선이다. 대신 아래 표로 어느 세트에
# 어느 종류가 비었는지 전부 찍는다 — 편중을 숨기지 않고 드러낸다. 평가 때는 조리
# 종류별로 쪼갠 지표(per-dish)를 함께 본다.
# ---------------------------------------------------------------------------
ASSIGN = {
    '개원중':     'test',
    '내곡중':     'test',
    '로봇고':     'val',
    '인화여중':   'val',
    '숭곡중':     'train',
    '울산현대차': 'train',
    '영동중':     'train',
    '금정초':     'train',
    '진선여고':   'train',
    '논현중':     'train',
    '남일고':     'train',
    '부산체고':   'train',
    '원촌중':     'train',
}

SETS = ('train', 'val', 'test')

# ---------------------------------------------------------------------------
# 인벤토리 적재
# ---------------------------------------------------------------------------
inv = json.load(open(VIDEOS, encoding='utf-8'))
vids = inv['videos']

sites = sorted({v['site'] for v in vids})

# ---------------------------------------------------------------------------
# 검산 — 배정이 인벤토리와 정확히 맞물리는가
# ---------------------------------------------------------------------------
assert set(ASSIGN) == set(sites), (
    f'배정과 인벤토리의 사이트가 다름\n'
    f'  인벤토리에만: {set(sites) - set(ASSIGN)}\n'
    f'  배정에만:    {set(ASSIGN) - set(sites)}')
assert set(ASSIGN.values()) <= set(SETS), '세트 이름 오타'
# 한 사이트가 두 세트에 들어가는 일은 dict 구조상 불가능 — 그래도 명시
assert len(ASSIGN) == len(sites) == inv['n_sites'], '사이트 수 불일치'

# ---------------------------------------------------------------------------
# 세트별 사이트·영상 모으기
# ---------------------------------------------------------------------------
split = {s: {'sites': [], 'videos': []} for s in SETS}
for site in sites:
    split[ASSIGN[site]]['sites'].append(site)
for v in vids:
    split[ASSIGN[v['site']]]['videos'].append(v)
for s in SETS:
    split[s]['sites'].sort()

# 교차 검산 — 어떤 영상도 두 세트에 있지 않고, 합이 전체와 같다
seen = [v['file'] for s in SETS for v in split[s]['videos']]
assert len(seen) == len(set(seen)) == len(vids), '영상이 새거나 겹침'

# ---------------------------------------------------------------------------
# 출력 리포트
# ---------------------------------------------------------------------------
dishes = sorted({v['dish'] for v in vids})


def dish_counts(videos):
    return {d: sum(1 for v in videos if v['dish'] == d) for d in dishes}


print('=' * 72)
print('사이트 단위 배정 — 프레임 추출 전 사전 등록')
print('=' * 72)
print(f'  사이트 {len(sites)}개 · 영상 {len(vids)}개')
print(f'  train {len(split["train"]["sites"])} : '
      f'val {len(split["val"]["sites"])} : '
      f'test {len(split["test"]["sites"])} (사이트 수)\n')

print(f'{"세트":<7}{"사이트수":>6}{"영상수":>6}   {"조리종류(영상수)":<28}{"CCTV":>6}')
print('-' * 72)
for s in SETS:
    v = split[s]['videos']
    dc = dish_counts(v)
    dc_str = ' '.join(f'{d}{n}' for d, n in dc.items() if n)
    ncctv = sum(1 for x in v if x['cctv'])
    print(f'{s:<7}{len(split[s]["sites"]):>6}{len(v):>6}   {dc_str:<28}{ncctv:>6}')

print('\n세트별 사이트')
for s in SETS:
    print(f'  {s:<6} {" · ".join(split[s]["sites"])}')

# 조리 종류 커버리지 — 어느 세트에 어느 종류가 비었는지 드러냄
print('\n조리 종류 커버리지 (세트 × 종류, 영상 수)')
print(f'  {"":<7}' + ''.join(f'{d:>6}' for d in dishes))
for s in SETS:
    dc = dish_counts(split[s]['videos'])
    row = ''.join(f'{(dc[d] or "·"):>6}' for d in dishes)
    print(f'  {s:<7}{row}')
print('  · = 그 세트에 그 종류 없음. 사이트 분리를 위해 감수한 공백 —')
print('    평가는 per-dish 로 쪼개 보고 이 공백을 숨기지 않는다.')

# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------
out = {
    'unit': 'site',
    'note': '사이트 단위 배정. 같은 주방은 한 세트에만. 프레임 추출 전 확정.',
    'assign': ASSIGN,
    'sites': {s: split[s]['sites'] for s in SETS},
    'videos': {s: [v['file'] for v in split[s]['videos']] for s in SETS},
    'counts': {s: {'sites': len(split[s]['sites']),
                   'videos': len(split[s]['videos']),
                   'dishes': dish_counts(split[s]['videos'])}
               for s in SETS},
}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n-> {os.path.basename(OUT)}')
