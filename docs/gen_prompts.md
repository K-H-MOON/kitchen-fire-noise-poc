# 생성형 합성(#6) — Nano Banana Pro 프롬프트 세트 (바로 붙여넣기용)

> 도구: Google AI Studio · Nano Banana Pro (Gemini image). 목표 = **급식실/상업주방 + CCTV 각도 + 유류불** 도메인 통합 이미지(“예쁜 불꽃” 아님).
> 라벨: 생성 후 Roboflow에서 **불꽃 박스 → YOLO export (class0=fire)**. 저장: `gen_fire/p01`~`p10` 하위폴더.
> 각 프롬프트를 **블록째 복사**해 넣으면 됨(공통 앵커가 각 프롬프트에 이미 인라인돼 있음).

---

## 공통 설계 (왜 이렇게 쓰나)

**도메인 앵커(전이의 핵심 — 실 배포 급식실 CCTV를 닮게):**
- `low-quality indoor security CCTV / surveillance still` · `grainy, noisy, low dynamic range, compression artifacts, mild motion blur`
- `cool fluorescent lighting` · `small white timestamp overlaid in a corner` · `no people`
- `mundane surveillance footage, NOT cinematic / dramatic / movie still`

**다양성 축(프롬프트마다 다르게 — 한 유형 과대표집 방지):**
불 종류·크기(초기 소형 ↔ 큰불) · 용기(웍·튀김기·큰솥·프라이팬) · 주방(학교급식·병원·단체급식·상업·산업) · 각도(천장 하향·측면·코너 어안·눈높이·광각) · 조명(주간 형광 ↔ 야간) · 헷갈림요소(스팀·젖은바닥 반사·불색 음식).

**⑦(조리로봇)이 실 배포 도메인에 가장 근접** — 여유 되면 비중 조금 더.

**피해야 할 것(각 프롬프트 끝 `Avoid:` 참고):** cinematic/dramatic, DSLR 고해상, 푸드 포토그래피, illustration/cartoon, 불꽃만 매크로 클로즈업, 사람/손.

---

## ★ 필드 템플릿 (권장 · 슬롯 채워넣기)

아래 p01~p10은 고정 예시고, **실제로는 이 템플릿의 대괄호 슬롯만 매번 바꿔** 생성하는 게 낫다(다양성 관리 쉬움). **대괄호 = 변주, 나머지 = 고정(도메인 앵커).**

```
Photorealistic CCTV surveillance still, [CAMERA], [FRAMING] of [KITCHEN + BACKGROUND], [VESSEL] with cooking oil on fire — [FIRE]. [LIGHTING]. Grainy, noisy, low-resolution, compression artifacts, small white timestamp "[TIME]" in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: cinematic or dramatic lighting, DSLR sharpness, professional food photography, illustration, cartoon, people.
```

**고정(절대 안 바꿈)** = `Photorealistic CCTV surveillance still` … `Grainy, noisy, low-resolution, compression artifacts` … `no people. Mundane surveillance footage, not cinematic. Avoid: …` → 도메인 매칭 상수. CCTV/grainy 톤·유류불 판별 문구는 **다양화하지 않는다**.

### 슬롯 메뉴

| 슬롯 | 옵션 |
|---|---|
| **[CAMERA]** | `wide fisheye view from a ceiling corner` · `standard non-fisheye ceiling camera looking down` · `high side angle from a wall-mounted camera` · `overhead camera looking straight down` · `eye-level camera across the kitchen line` |
| **[FRAMING]** | `a wide view`(주방 전체) · `a medium-close view`(스토브 하나 — 불 커지고 다양성↑) |
| **[KITCHEN]** | `an industrial school cafeteria kitchen` · `an institutional hospital kitchen` · `a large catering canteen kitchen` · `a commercial restaurant kitchen` · `a hospital kitchen with an industrial robotic cooking arm`(⑦ 배포근접) |
| **[BACKGROUND]** ⓝ | 아래 "[BACKGROUND] 메뉴" — [KITCHEN] 뒤에 붙여 배경 다양화 |
| **[VESSEL]** | `a large stainless steel wok on a gas range` · `a stainless steel deep fryer` · `a large stainless steel stockpot on a range` · `a frying pan with food` |
| **[FIRE]** ⚖️ | **소형40%**: `a small but clearly visible oil fire, low orange flames on the oil surface with thin smoke`(웍·팬: `at the oil surface` / 솥: `along the rim` / 튀김기: `around the fryer basket`) · **중간40%**: `an uncontrolled oil/grease fire, medium orange-yellow flames rising from the oil surface with smoke` · **큰불20%**: `a large uncontrolled grease fire, tall orange flames and dark smoke rising from the pan` |
| **[LIGHTING]** | `Cool white fluorescent lighting` · `Warm tungsten-tinted lighting` · `Bright daylight through windows` · `Dim low light at night` |
| **[TIME]** | `2023-11-02 20:14:08` — **매번 날짜·시각 바꾸기**(near-dupe 방지) |

### ⓝ [BACKGROUND] 메뉴 — **배포 도메인(상업/급식 주방) 안에서만**

> ⚠️ 우리 핵심 교훈 = 도메인 매칭. 배경 다양화는 **상업/급식 주방 계열 안에서만.** 가정집·야외·산업용광로로 나가면 도메인 매칭이 깨져 역효과(공개모델·D-Fire 실패 방향). 스테인리스가 "똑같아" 보여도 그게 배포 도메인이다 — 아래 축으로 **계열 안에서** 변주.

- **벽/바닥 재질**: `white subway-tiled walls and a worn tiled floor` · `pale green ceramic-tiled walls and a wet concrete floor` · `stainless steel wall panels with a grated floor drain` · `painted concrete block walls` · `cream-colored tiled walls`
- **연식/청결(가장 큰 시각 레버)**: `an older, worn kitchen with stained walls and grease marks` · `a newer, clean kitchen with a polished floor`
- **규모/배치**: `a large canteen with rows of equipment` · `a compact, cluttered kitchen with wall shelves full of pots and utensils` · `a spacious, mostly empty prep kitchen`
- **지역/스타일**: `an Asian-style kitchen with a row of wok burners` · `a sterile hospital kitchen with white walls`

### 완성 예시 (템플릿 적용)

**중간 · 웍 · 직선천장 · 낡은 초록타일:**
```
Photorealistic CCTV surveillance still, standard non-fisheye ceiling camera looking down, a medium-close view of an institutional catering kitchen with pale green ceramic-tiled walls and a wet concrete floor, aged and grimy with grease marks, a large stainless steel wok with cooking oil on fire — an uncontrolled oil/grease fire, medium orange-yellow flames rising from the oil surface with smoke. Cool white fluorescent lighting. Grainy, noisy, low-resolution, compression artifacts, small white timestamp "2023-11-15 18:32:10" in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: cinematic or dramatic lighting, DSLR sharpness, professional food photography, illustration, cartoon, people.
```

**소형 · 솥 · 어안 · 흰 서브웨이타일 어수선한 소형주방:**
```
Photorealistic CCTV surveillance still, wide fisheye view from a ceiling corner, a wide view of a commercial restaurant kitchen with white subway-tiled walls, compact and cluttered with wall-mounted shelves full of pots and utensils, a large stainless steel stockpot with cooking oil on fire — a small but clearly visible oil fire, low orange flames along the rim with thin smoke. Warm tungsten-tinted lighting. Grainy, noisy, low-resolution, compression artifacts, small white timestamp "2024-03-06 21:10:55" in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: cinematic or dramatic lighting, DSLR sharpness, professional food photography, illustration, cartoon, people.
```

### 1×2 그리드로 뽑을 때
[CAMERA] 뒤에 `shown as two separate camera views side by side` 추가. 한 패널만 불 있어도 됨(나머지=하드네거). 저장 후 [`slice_grid.py`](../scripts/slice_grid.py)로 슬라이스 → 개별 패널 라벨링.

---

## p01 — 학교급식실 · 웍 · 중간불 · 천장 하향

```
Photorealistic still frame from a low-quality indoor security CCTV camera, high angle looking straight down from the kitchen ceiling. Industrial school cafeteria kitchen with stainless steel counters and tiled walls. A large stainless steel wok sits on a commercial gas range; the cooking oil in it has caught fire with medium orange and yellow flames rising about 30 cm, light smoke drifting toward the overhead stainless exhaust hood. Grainy, noisy, low dynamic range, compression artifacts, cool fluorescent lighting, small white timestamp overlaid in the top-right corner, no people. Mundane surveillance footage, not cinematic. Avoid: dramatic lighting, DSLR, food photography, illustration, people.
```

## p02 — 병원주방 · 튀김기 · 큰불 · 측면각

```
Photorealistic still frame from a low-quality security CCTV camera, high side angle. Hospital institutional kitchen with plain stainless steel surfaces and fluorescent ceiling lights. A stainless steel deep fryer is on fire with large intense orange flames and dark smoke rising toward the vent hood above. Grainy, noisy, low dynamic range, compression artifacts, small white timestamp in the corner, no people. Mundane surveillance footage, not cinematic. Avoid: dramatic movie lighting, DSLR, illustration, cartoon, people, hands.
```

## p03 — 단체급식 · 큰솥 · 초기 작은불 · 코너 어안

```
Photorealistic wide fisheye CCTV view from a ceiling corner of an institutional catering kitchen, several large stainless steel pots on a range. In one large cooking pot the oil is just beginning to ignite — small orange flames along the rim and a light haze rising. Grainy, noisy, low-resolution, compression artifacts, cool fluorescent lighting, small white timestamp overlaid in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: big dramatic fire, DSLR sharpness, food photography, illustration, people.
```

## p04 — 상업레스토랑 · 프라이팬 · 중간불 · 눈높이 + 음식 + 스팀

```
Photorealistic still from a low-quality restaurant security camera, eye-level angle across a commercial restaurant kitchen line. A frying pan containing food and cooking oil is on fire with medium orange flames; a pot of water steaming nearby releases visible white vapor, and stainless steel surfaces reflect the firelight. Grainy, noisy, low dynamic range, compression artifacts, fluorescent lighting, small white timestamp in the corner, no people. Mundane surveillance footage, not cinematic. Avoid: dramatic lighting, DSLR, professional food photography, illustration, people, hands.
```

## p05 — 산업구내식당 · 웍 · 큰불 · 천장 + 연기

```
Photorealistic still frame from a ceiling-mounted security CCTV camera looking down at a large industrial canteen kitchen. A big stainless steel wok is on fire with large orange flames and thick dark smoke rising toward a wide stainless steel exhaust hood. Grainy, noisy, low dynamic range, compression artifacts, cool fluorescent lighting, small white timestamp overlaid in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: cinematic smoke, dramatic lighting, DSLR, illustration, people.
```

## p06 — 학교급식실 · 튀김바구니 · 초기불 + 스팀 · 높은각

```
Photorealistic still from a low-quality kitchen CCTV camera, high angle. School cafeteria kitchen. A stainless steel deep-fryer basket of cooking oil is just catching fire — small orange flames flickering at the surface — with rising steam and vapor around it that partly resembles smoke. Grainy, noisy, low-resolution, compression artifacts, fluorescent lighting, small white timestamp in the corner, no people. Mundane surveillance footage, not cinematic. Avoid: big flames, dramatic lighting, DSLR, illustration, people.
```

## p07 — 병원주방 · 조리로봇 + 큰솥 · 중간불 · 어안 (★배포 도메인 근접)

```
Photorealistic wide fisheye CCTV view of an institutional hospital kitchen. An industrial robotic cooking arm is positioned next to a large stainless steel pot that is on fire with medium orange flames; light smoke rises toward the ceiling. Grainy, noisy, low-resolution, compression artifacts, cool fluorescent lighting, small white timestamp overlaid in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: dramatic lighting, DSLR, sci-fi rendering, illustration, cartoon, people.
```

## p08 — 상업주방 · 웍 · 큰불 · 측면 · 야간

```
Photorealistic still from a low-quality security CCTV camera at night, high side angle. A dimly lit commercial kitchen. A stainless steel wok is on fire with large bright orange flames that light up the dark kitchen and cast reflections on nearby steel surfaces. Grainy, noisy, low dynamic range, heavy sensor noise in the shadows, compression artifacts, small white timestamp in the corner, no people. Mundane surveillance footage, not cinematic. Avoid: cinematic night photography, dramatic lighting, DSLR, illustration, people.
```

## p09 — 단체급식 · 프라이팬 · 발화 순간 · 천장 + 기름/음식

```
Photorealistic still frame from a ceiling security CCTV camera looking down at an institutional catering kitchen. A large frying pan full of cooking oil and food is just igniting — small orange flames spreading across the oil surface, thin smoke starting to rise. Grainy, noisy, low dynamic range, compression artifacts, cool fluorescent lighting, small white timestamp overlaid in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: big dramatic fire, DSLR, food photography, illustration, people, hands.
```

## p10 — 산업주방 · 튀김기 · 중간불 · 광각 · 젖은바닥 반사

```
Photorealistic wide-angle CCTV view of an industrial kitchen with a wet tiled floor reflecting the light. A stainless steel deep fryer is on fire with medium orange flames; reflections of the flames appear on the wet floor and on stainless steel counters. Grainy, noisy, low dynamic range, compression artifacts, cool fluorescent lighting, small white timestamp overlaid in a corner, no people. Mundane surveillance footage, not cinematic. Avoid: dramatic lighting, DSLR, illustration, glossy render, people.
```

---

## 폴더당 30~40장 다양성 만드는 법 (같은 프롬프트 반복 = 미세변형만 → 금지)

각 p## 안에서 아래 **노브를 바꿔** 문장을 조금씩 다르게 재생성:
- **타임스탬프 값**: `2024-03-12 13:42:07` → 날짜·시각 바꾸기(과대표집 방지 겸 텍스트 다양화).
- **불 크기 단계**: `just beginning to ignite (small)` → `medium` → `large intense`. 한 폴더 안에서도 크기 스펙트럼 섞기.
- **화염 색**: `orange` / `orange and yellow` / `orange with some bluish base near the oil`.
- **각도 미세조정**: `high angle` ↔ `slightly tilted overhead` ↔ `corner fisheye`.
- **주방 디테일**: 타일 색·솥 개수·후드 모양·바닥 상태(건조/젖음)·창문 유무.
- **헷갈림요소 on/off**: 스팀·불색 튀김음식·스테인리스 반사를 넣었다 뺐다(현실성 + 라벨은 여전히 fire).

## 수량·구성 권장 (미확정 — 전이는 측정할 것)

- **총 ~300~500장**(10폴더 × 30~40). fine-tune 아닌 **사전학습 풀**이라 양이 좀 있어야 신호.
- **불 크기 분포**: 초기 소형 ~40% · 중간 ~40% · 큰불 ~20% (실 test 약점이 창백 초기불 sc14라 초기불 비중 확보).
- **각도 분포**: 천장 하향·어안 코너를 다수(실 급식실 CCTV 대표) · 눈높이/측면은 소수.
- ⑦(조리로봇) 폴더는 되도록 채우기(배포 도메인 최근접).

> ⚠️ 이미지 도메인 매칭이 좋다 = **필요조건**일 뿐. 커리큘럼(gen 사전학습 → 실 파인튜닝)이 실제로 전이를 **개선하는지는 미확정 → 학습·eval로 측정**(`2gencurr` vs `2g`). 좋게 나오든 아니든 발표 지도의 한 칸.

---

## (선택·보너스) 하드네거 gen 프롬프트 — 불 없는 헷갈림 장면

사전학습에 소량의 **음성(불 없음)**을 섞으면 “불색이면 다 불”이라는 편향을 줄일 수 있음(라벨 = 빈 박스/음성). 원하면:

```
Photorealistic low-quality kitchen CCTV still, high angle. A commercial kitchen with a large wok of golden deep-fried food (crispy orange-colored) and rising steam; stainless steel surfaces reflecting the fluorescent light. NO fire, NO flames — just steam, hot oil shimmer, and orange-colored food. Grainy, noisy, compression artifacts, small white timestamp in the corner, no people. Avoid: any flame, dramatic lighting, DSLR, illustration.
```

> 헷갈림요소만 있고 불은 없는 장면(스팀·금색 튀김음식·스테인리스 반사). 커리큘럼에 넣을지는 빌더 단계에서 결정(넣으면 gen 폴더 `n01` 등 별도).
