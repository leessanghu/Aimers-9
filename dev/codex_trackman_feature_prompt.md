# Codex 프롬프트 — Trackman 파생 피처 후보 명세 (안 건드린 지점만)

## 배경

같은 대회, `data/trackman_history.csv`(179만행×30컬럼, 338MB)가 보조 물리 데이터로
제공돼. 지금까지 이 파일에서 우리가 실제로 쓴 건 일부뿐이고, 뭘 썼고 뭘 안 썼는지
아래에 정확히 정리했어. **이미 실패로 결론난 항목은 절대 다시 제안하지 마.**

## trackman_history.csv 전체 컬럼 (30개)

```
trackman_id, season, game_date, game_month, game_dayofweek, trackman_game_id,
pitch_no, inning, top_bottom, balls_before, strikes_before, outs_before,
pitch_of_pa, pitcher_trackman_id, batter_trackman_id, pitcher_hand, batter_hand,
pitcher_team, batter_team, tagged_pitch_type, auto_pitch_type, pitch_type_group,
rel_speed, spin_rate, induced_vert_break, horz_break, extension, rel_height,
rel_side, zone_speed
```

`pitcher_id`(train.csv)와 `pitcher_trackman_id`(trackman)를 잇는 매칭 테이블이
`dev/pitcher_map.csv`에 이미 있음(유사도 기반 매칭, 정밀도 99.5~99.7% 검증됨).
**`batter_id` <-> `batter_trackman_id` 매칭 테이블은 존재하지 않음** — 이게 첫 번째
막힌 지점이야.

## 이미 시도해서 확정적으로 실패한 것 (다시 제안 금지)

1. **tagged_pitch_type vs auto_pitch_type 불일치율**(`tm_disagree_raw`) — 비편향
   split-half 스크리너에서 기각선을 못 넘음.
2. **zone_speed 기반 velo_loss = rel_speed - zone_speed의 평균/표준편차**
   (`tm_velo_loss_mean`, `tm_velo_loss_sd`) — 마찬가지로 기각.
3. 구종매칭(`pitch_type_group`)을 이용한 투수별 구종 조건부 제구력 추정
   (`pitchtype.py`) — 이미 채택돼서 모델에 들어있음(재제안 불필요).
4. (투수 x 구종) **내부** 릴리스포인트 반복성 SD, 무브먼트 SD, 등판내 구속감쇠,
   압박시 릴리스산포(`trackman_profile.py`, PROFILE_COLS 16개) — 이미 채택돼서
   모델에 들어있음.

## 완전히 안 써본 지점 (여기서만 제안해줘)

### A. 타자 관점 Trackman 프로파일 (batter_trackman_id, 0% 사용)

`batter_id` <-> `batter_trackman_id` 매칭이 없어서 지금까지 타자 시점 물리데이터를
전혀 못 썼어. 매칭 방법 제안부터 해줘(pitcher_map.csv가 어떤 유사도 기준으로
매칭했는지 이 프롬프트만으론 모르니, 일반적으로 쓸 수 있는 접근 — 예를 들어
같은 경기/이닝/카운트에서 나타나는 (pitcher_trackman_id, batter_trackman_id) 쌍의
빈도 패턴을 train.csv의 (pitcher_id, batter_id) 쌍의 빈도 패턴과 대조하는 방식,
또는 시즌별 타자 출현 횟수 분포 매칭 등). 매칭이 되면:
- 타자가 실제로 보는 구속/무브먼트 분포(투수가 그 타자 상대로 실제 던진 공)
- 타자의 상대 투수 유형별 트릭맨 노출 이력(빠른공 상대 노출 빈도 등)
- (타자, 구종) 내부 트릭맨 물리량의 노출 분산 — "이 타자가 다양한 구질을 자주
  보는가" 같은 것

### B. 구종 간(cross-pitch-type) 신호 — 지금까지는 구종 "내부"만 봤음

기존 tm_release_sd 등은 반드시 같은 구종 내부에서만 계산했어(레퍼토리 폭과 제구를
구분하려고). 이번엔 반대로 **구종 간** 차이/유사성 자체를 신호로 만들어줘:
- 이 투수의 직구 릴리스포인트와 변화구 릴리스포인트 사이의 거리(디셉션 신호 —
  타자가 구종을 릴리스만 보고 구분하기 어려운 정도)
- 주 구종(가장 많이 던지는 구종) 대비 부 구종들의 물리량 상대적 편차
- 시즌 내 구종 레퍼토리 자체의 인원별 물리적 다양성(같은 구종그룹 안에서도
  세부 변형이 있는지, 예: 같은 breaking이라도 induced_vert_break 분산이 큰지)

### C. 투구 시퀀스 델타(pitch_of_pa 활용, 지금까지 라벨복원에서만 씀)

같은 타석(같은 pitch_of_pa 시퀀스) 안에서 **연속 투구 간** 물리량 변화:
- 이번 투구의 rel_speed - 직전 투구의 rel_speed (같은 타석 내)
- 이번 투구의 릴리스 위치가 직전 투구 대비 얼마나 이동했는지(rel_height, rel_side
  변화량) — 타자에게 다른 각도를 보여주려는 의도적 변화 vs 메커닉 불안정 구분은
  안 해도 됨, 일단 원시 델타로
- 시즌 누적: "이 투수가 평균적으로 타석 내 투구 간 물리량을 얼마나 바꾸는가"를
  (pitcher, season-1) 프로파일로 만들어 규칙 위반 없이 조회

### D. 카운트/이닝 조건부 Trackman 물리량 분포 (target-free, 개인 아님)

개인 단위 아니라 **전체 표본** 유지하는 형태로:
- (game_type, balls_before, strikes_before) 별 rel_speed/spin_rate/induced_vert_break
  분포(mean/std/quantile) — 카운트 압박에 따른 리그 전체 구속/스핀 변화 패턴
- (inning, top_bottom) 별 물리량 분포 — 경기 후반 리그 전체 구위 저하 패턴

## 규칙 (기존과 동일)

- target-free/OOF-expanding 구분 명시, leakage 없이(각 행은 그 투수/타자의
  '직전 시즌까지' 또는 '같은 타석 내 이전 투구까지'만 참조)
- trackman은 2019~2024만 존재, 2025는 없음(원래 train.csv에도 trackman 원본은
  없고 우리가 미리 계산한 프로파일 테이블만 저장해서 씀) — 이 구조 유지
- 개인 단위(pitcher_id x batter_id x pitch_type_group처럼 3중 이상 개인교차)로
  잘게 쪼개는 조합은 셀 크기 부족으로 이미 여러 번 실패 확인됨 — 만들 땐 축소
  강도를 크게 잡거나 표본 큰 집계 단위로 유지
- 후보 각각: id, category(target-free/oof-expanding), 어떤 컬럼/그룹핑, 근거
  1줄, cell_size_concern(low/medium/high) 명시
- A(타자 매칭)는 실현 가능성이 불확실하니, 매칭 방법론 제안 + 매칭 성공 시
  만들 피처 후보를 분리해서 제시해줘. 매칭 자체가 막히면 B/C/D만으로도 충분

## 출력에 포함

- 전체 후보 개수, 카테고리(A/B/C/D)별 개수
- A(타자 매칭)의 예상 매칭 정밀도나 리스크에 대한 판단
