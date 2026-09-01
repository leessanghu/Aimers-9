# Codex 프롬프트 — 파생 피처 후보 명세 생성 (v43 Feature Factory)

## 배경

LG Aimers 대회, 투구 제구 성공(`control_success`, 이진) 예측 문제야. 현재 실측 최고
1065.92(Brier Skill Score 기반 리더보드 점수, 100등 컷은 계속 올라 현재 ~1088 근처).
지금 162개 피처 + Hurdle/MultiRes/Ordinal 등 여러 앙상블 멤버로 v44까지 왔어.

**이미 확인된 사실(매우 중요, 반드시 반영)**: 이 데이터의 "새로운 선형/비선형 피처"
경로는 최근 세션에서 비편향 split-half 스크리너로 12개 후보를 검증한 결과 전부
기각됐고, 결론이 "신호가 근본적으로 저차원(투수 실력이라는 잠재변수 1개가 지배적)
이라 남은 정보가 거의 없다"였어. 그럼에도 **자동 대량생성(300~1000개)으로 우리가
수동으로 안 떠올린 조합을 커버할 여지가 있어 재시도**하는 거야 — 즉 "뻔한 조합"
말고 우리가 놓쳤을 법한 조합을 우선해줘.

## 데이터 스키마 (train.csv, 약 147만 행)

- 식별자: `row_id`, `pitcher_id`, `batter_id`, `season`(2019~2024), `game_type`
- 상황: `inning`, `balls_before`, `strikes_before`, `pitcher_hand`, `batter_hand`
- 주최측 제공 누적통계(전부 "이 투구 이전까지"의 as-of 값, 즉 이미 leakage-safe):
  `asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`,
  `asof_pitcher_middle_rate`, `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate`,
  `asof_batter_n`, `asof_batter_success_rate`, 유사하게 batter도 있음,
  `asof_pitcher_fastball_rate/breaking_rate/offspeed_rate`(레퍼토리 비중),
  `asof_pitcher_prev{1,3,5}_game_success_rate/middle_rate`(최근 등판 성적)
- 타겟: `control_success` (0/1, 평균 약 0.52)

## 반드시 지킬 leakage 규칙

1. **target-free 피처**: `control_success`를 전혀 안 쓰는 그룹별 통계(count, entropy,
   std, 최근값과의 차이 등)는 같은 행 기준으로 자유롭게 만들어도 됨. 단, 그룹 집계에
   "미래 행"이 섞이면 안 됨 — 반드시 "이 행 이전 시점까지"만 누적.
2. **target 포함 피처(OOF/expanding)**: `control_success`를 쓰는 통계는 반드시
   **그 투수/타자 자신의 과거 시즌 데이터까지만**(직전 시즌 끝 시점, 또는 현재 시즌
   내 그 행 이전까지) 누적해서 만들어야 함. 같은 시즌의 "미래 행"이나 "동시점 다른
   행"을 참조하면 절대 안 됨(이게 제일 흔한 leakage 실수).
3. 기존 코드 패턴 예시(참고용, 그대로 베낄 필요는 없고 leakage 처리 스타일만 참고):
   - `dev/inseason.py`: 그 투수의 "직전 시즌 끝 시점"까지 누적 → 현재 시즌 진행 중
     rate를 K-스무딩(`(n*raw + K*prior)/(n+K)`)해서 만듦
   - `dev/platoon.py`: (투수, 타자손) 조건부 성공률을 투수 자신 marginal 대비 편차로
   - `dev/count_split.py`: (투수, count_state) 조건부

## 요청사항

**target-free 그룹별 통계**와 **OOF/expanding target encoding** 두 카테고리로
나눠서, 각각 최소 150개씩(총 300개 이상) 후보를 만들어줘. 코드가 아니라 **명세
리스트**로 줘(아래 형식). 실제 구현/leakage 검증/스크리닝은 우리가 직접 함.

### 우선순위(이걸 우선해줘)

1. **3중 이상 교차 조합**: (투수, 타자손, count_state) 같은 이미 시도한 2중 조합보다
   한 단계 더 세분화된 조합. 단, 표본이 너무 작아지는 조합(예: 투수x타자x구종x카운트)은
   후순위 — 셀 크기가 크게 유지되는(예: 리그 전체 x count_state, 팀 x inning) 쪽을
   더 우선해줘. **[이 세션의 핵심 교훈: 투수/타자 개인 단위로 잘게 쪼개는 조합은 셀
   크기가 이론상 필요한 표본의 20~40배 부족해서 정보가 거의 안 산다는 게 이미
   정량적으로 증명됨 — 그러니 개인 단위 잘게 쪼개기보다 "전체표본 유지하며 타겟을
   다른 각도로 재구성"하는 쪽을 우선해줘]**
2. **비선형 결합**: 기존 rate 컬럼들의 비율/차/엔트로피가 아니라, 순위(percentile),
   구간화(binning) 후 재조합, 시계열 변화율(가속도: 최근 변화의 변화) 등 트리가
   스스로 근사하기 어려운 형태
3. **Trackman/구종 관련 미탐색 조합**: 기존엔 구종별 제구력만 봤는데, 구종 전환
   패턴(직전 투구 구종 대비 이번 구종의 변화), 레퍼토리 예측불가능성(entropy의
   시계열 변화) 등

### 각 후보 항목 형식

```
- id: F0001
  category: target-free | oof-expanding
  group_keys: [pitcher_id, batter_hand]   # 그룹핑 기준 컬럼
  agg: mean | std | count | entropy | pct_rank | ...
  source_col: asof_pitcher_success_rate   # oof-expanding이면 반드시 명시
  window: "직전 시즌까지" | "현재 시즌 내 이전 행까지" | "전체 과거"
  rationale: "한 줄 근거 — 왜 이 조합이 성공률과 관련 있을 것으로 예상하는지"
  cell_size_concern: low | medium | high   # 셀이 얼마나 잘게 쪼개지는지 자체 평가
```

### 최종 출력에 반드시 포함

- 전체 후보 개수 (정확한 숫자, 우리가 다중검정 보정 임계값 계산에 씀)
- category별 개수
- cell_size_concern=high로 표시한 항목 개수와 이유

## 참고: 지금까지 시도해서 기각된 것 (중복 방지용)

- pitcher x count_state, batter x count_state, pitcher x batter_hand,
  pitcher x inning (전부 Bayes 축소 후 정보량 2.7~13.1%만 회수, 기각)
- SHAP 상위 피처 기반 파생(예: ability x pressure 교차) — 이미 crosses.py에 포함
- 구종 전환/레퍼토리 JS divergence 일부(arsenal_js) — 이미 있음, 완전히 새로운
  각도만 제안해줘(예: 그냥 레퍼토리 비중 말고 "전환 패턴" 자체)
- 리그/시즌 레벨 전역 통계는 이미 많음(inseason_full, lastyear) — 개인 단위
  세분화보다 "다른 방식의 재구성"을 우선해줘
