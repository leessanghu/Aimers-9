# 야구 제구 성공확률 예측 — 963.8점에서 1100점으로 가는 아이디어를 찾습니다

## 0. 요청

아래는 제가 지금까지 한 모든 것과, **측정해서 기각한 것들**입니다.
이미 시도/기각한 것을 다시 제안하지 마시고, **아직 안 본 정보원이나 접근**을 제안해 주세요.
특히 "왜 그게 통할 것 같은지"를 **검증 가능한 형태(측정 방법 포함)**로 제시해 주시면 좋겠습니다.

---

## 1. 문제

- **타깃**: `control_success` (0/1). 투구가 의도한 제구에 성공했는가.
- **평가**: `score = max(0, 100000 × (1 − BS / (r(1−r))))`, r = test 실제 성공률. 즉 **Brier Skill Score × 100000**.
- **train**: 1,475,092행, 2019~2024 시즌. **test**: 2025 시즌 (라벨 비공개).
- 현재 우리 점수 **963.796** (BSS ≈ 0.0096). 등수 142위. 100위권이 ~1000점, **1위가 ~1200점**.

## 2. 대회 규칙 제약 (중요)

- **test.csv의 각 행은 독립적으로 예측해야 함.** test 행 간 참조 금지.
- 명시적 금지 예시: "test.csv 행 순서 기반 rolling / expanding feature".
- → test 전체를 집계해서 통계를 내는 것도 불가. (실제로 이걸 어겨서 0점 처리된 적 있음)
- train.csv 내부의 과거 정보로 (pitcher, season) 같은 조회 테이블을 만들어 test에서 **조회만** 하는 것은 허용.

## 3. 주어진 데이터

### train.csv / test.csv 컬럼
- 상황: `season, game_month, game_dayofweek, inning, top_bottom, game_type, balls_before, strikes_before, outs_before, run_top_before, run_bot_before, run_total_before, score_diff_home, score_diff_pitcher_team, runner_on_1b/2b/3b, num_runners_on, base_state, home_win_expectancy, away_win_expectancy, li`
- 식별: `pitcher_id, batter_id, pitcher_hand, batter_hand, pitcher_team_id, batter_team_id`
- **투수 as-of 누적(그 투구 직전까지)**: `asof_pitcher_n`, `asof_pitcher_success_rate / reverse_rate / middle_rate / ball_rate / strike_rate`
- **투수 직전 등판**: `asof_pitcher_prev1/prev3/prev5_game_success_rate`, 동 `_middle_rate`
- **타자 as-of**: `asof_batter_n, asof_batter_success_rate, asof_batter_middle_rate`
- **구종 믹스 as-of**: `asof_pitcher_pitchmix_n, asof_pitcher_fastball_rate / breaking_rate / offspeed_rate`
- 주의: 현재 투구의 **구종/구속/코스는 주어지지 않음**(결과 누출 방지).

### trackman_history.csv (별도 제공, 투구 단위 2019~2024)
`trackman_id, season, game_date, game_month, game_dayofweek, trackman_game_id, pitch_no, inning, top_bottom, balls_before, strikes_before, outs_before, pitch_of_pa, pitcher_trackman_id, batter_trackman_id, pitcher_hand, batter_hand, pitcher_team, batter_team, tagged_pitch_type, auto_pitch_type, pitch_type_group, rel_speed, spin_rate, induced_vert_break, horz_break, extension, rel_height, rel_side, zone_speed`
- train.csv와 **직접 조인 키가 없음**. 우리가 (팀 스케줄 지문 상관 + 헝가리안) → (투수별 등판 패턴 코사인 유사도)로 **604명 매핑 완료**(손잡이 일치 100%, 시즌간 일관성 99.25%).

---

## 4. 현재 모델 (963.796점 구성)

**81 피처, HistGradientBoosting + CatBoost 50:50 블렌드**

1. base 58개: 원시 상황 변수 + as-of rate들을 n으로 스무딩(K=20) + team/id count encoding + team target encoding
2. **in-season 5개** ← 최대 승자(+114점)
   - 트릭: `round(asof_rate × asof_n)`으로 누적 성공 횟수를 정확히 복원 → `(현재 누적) − (직전 시즌 종료 시점 누적)` = **이번 시즌 한정** 성적
   - 즉 주최측은 커리어 누적만 줬는데, 우리가 시즌 한정으로 분해함
3. **platoon 2개** (+14점): `(투수, 타자손)` 조건부 성공률의 그 투수 자신 대비 편차. 직전 시즌까지 누적을 조회
4. **inning 2개** (+9점): `(투수, 이닝)` 조건부. 전역 이닝 주효과는 prior에서 빼서 순수 개인 상호작용만
5. **교차항 14개** (+5.7점): 비율(x/y), 여러 항의 합, 실력×상황 곱

## 5. 실제 리더보드 제출 이력 (전부 실측)

| 버전 | 구성 | 점수 |
|---|---|---|
| v4 | 58 base + in-season, RF+HGB | 925.908 |
| v7b | + platoon, HGB 단독 | 939.681 |
| v7c | + inning | **948.970** |
| v8 | v9 + 칼만 + CatBoost | 939.875 |
| v9 | v7c − batter_asof 4개 | **922.415** |
| v10 | v7c + 칼만교체 + CatBoost | 950.813 |
| v11 | v10 + 교차항 | 956.493 |
| v12 | in-season 원본 + 교차항 + CatBoost | **963.796** ← 현재 |

---

## 6. 측정해서 **기각**한 것들 (재제안 불필요)

### 6-1. 조건부 분할 (2승 6패)
노이즈 제거 분산으로 상한을 재고, 재현상관(직전시즌 편차 → 다음시즌 같은 편차)까지 측정함.
- 채택: 투수×타자손(진짜SD 0.0438, 재현상관 **0.328**), 투수×이닝(0.0209)
- **기각**: 투수×볼카운트(재현상관 0.121) −12.4, 타자×투수손(0.257) −3.5, 투수×월(0.079) −31.3, 투수×상대팀(0.054) −4.8, 투수×주자유무(진짜SD **0.000**), 투수×2루주자(**0.000**), 투수×아웃카운트(**0.000**), 투수×점수차(0.026, CI에 0 포함), 투수×레버리지(0.0087)
- 진단: 재현상관 0.05~0.12대 신호는 **난수보다 해로움**. 난수 4개 추가 = −0.9인데 이들은 −6~−31. 구조화된 노이즈라 트리가 시즌 특정 패턴을 외움.

### 6-2. Trackman (전패)
- 시즌 평균 물리값 20개 추가: −14.6 ~ −28.1
- 구속 변화(직전시즌) → 다음시즌 제구: r = **−0.052**, 95%CI [−0.134, +0.043] (0 포함)
- **릴리스 포인트 산포**(rel_height/rel_side, 구종 내 편차): 다음시즌 제구와 r=+0.060 (CI에 0 포함). **제구 일관성의 물리적 지표라 기대했으나 신호 없음**
- 익스텐션 산포: 제구 **수준**과는 r=−0.149(유의)이나 제구 **변화**와는 r=−0.028(CI에 0 포함) → 이미 아는 실력의 노이즈 버전

### 6-3. in-season 메커니즘 확장 (전패)
- 타자 in-season(−6.4), 피치믹스 in-season(−6.5), 둘 다(−7.0)
- 투수 middle/strike in-season, season_trend, K sweep — 이전 라운드에서 전부 실패

### 6-4. 기타 기각
- **batter_asof 4개 제거**: 로컬 ablation **+33.0** → 실제 **−26.6** (부호 반전)
- workload/form (prev1/3/5에서 투구수 복원 + 폼 편차): 로컬 +17 → 실제 **−4.2**
- 칼만 필터(상태공간 실력추정, in-season 대체): v11 956.5 vs v12 963.8 → **in-season 원본이 더 나음**
- 시대 보정(리그 수준 대비 상대화): 다음시즌 예측 R² 0.2705 → 0.2778 (+2.7%), 두 변수 상관 0.969 → 무의미
- 모델 용량 증가: leaves 63/127, depth 8 → **−106 ~ −143**. 용량 축소도 −5 ~ −43. **현재 설정이 양방향 최적점**
- 엔티티 임베딩 NN: 임베딩 있으면 과적합(등장 투수 BSS 0.0044 < 미등장 0.0057), 순수 MLP가 오히려 나음. GBM과 상관 0.75로 블렌드 이득은 있으나 GBM 앙상블 도입 후 +5.4로 축소

---

## 7. 상한 분석 (2024 폴드, 정답 컨닝)

| | 로컬 점수 |
|---|---|
| 상수(리그평균) 예측 | 0 |
| 우리 모델 | ~814 |
| **투수의 2024 실제 성공률(축소판) 안다고 가정** | **946.5** |
| **위 + 볼카운트 주효과** | **981.0** |
| (투수×볼카운트) 셀 실제평균 [과적합 상한] | 2740.9 |

- 로컬 → 실제 환산비 ≈ **1.20배** (v12: 로컬 805.8 → 실제 963.8)
- 즉 **투수 실력을 완벽히 알아도 실제 환산 ~1177점**. 1100점은 로컬 ~920점.
- 투수 실력 진짜 개인차 SD = **0.0555**(노이즈 제거) → BSS 상한 0.0123 = **1235점**

**해석**: 남은 헤드룸이 거의 전부 "투수의 현재 실력을 얼마나 정확히 추정하느냐"에 있고,
새로운 피처 카테고리(조건부, 물리, 타자)는 측정상 전부 소진됨.

---

## 8. 우리가 부딪힌 방법론적 한계

- 2024 폴드 검증의 **시드 노이즈 SD ≈ 11점** (같은 설정, 시드만 바꿔 804.3 ~ 826.1).
- 그런데 최근 실제 개선폭은 +2 ~ +9점. **측정 대상이 측정 오차보다 작음.**
- 실제로 로컬이 부호를 틀린 사례 3건(위 6-4 참조).
- → 지금은 "폴드와 무관한 독립 측정 / 여러 조건 일관성 / 이론적 보장" 중 하나가 없으면 채택 불가.

---

## 9. 질문

1. **위 상한 분석이 맞다면**, 투수 실력 추정 정밀도를 크게 올릴 방법이 있을까요?
   (현재: 커리어 누적 + 작년 한 시즌 + 이번 시즌 부분 관측을 각각 스무딩해서 피처로 제공.
    스무딩 K는 in-season 60, platoon 2500 등으로 재조정 중)
2. **우리가 못 본 정보원**이 있을까요? 특히:
   - test 행 독립성을 지키면서 게임 상태(inning/outs/runners/score)로부터 추출 가능한 것
   - Trackman을 평균/산포가 아닌 다른 방식으로 쓰는 법
   - as-of 컬럼들에서 아직 복원 안 한 정보
3. 1위가 1200점(BSS 0.012)인데, 우리 상한 분석상 투수 실력만으로 1235가 최대입니다.
   **이들이 우리와 다르게 보고 있을 만한 것**이 무엇일지 추측해 주세요.
4. BSS 0.01 수준의 **극도로 미약한 신호** 문제에서, 모델링 측면에서
   (GBM 하이퍼파라미터가 아니라 **구조적으로**) 더 나은 접근이 있을까요?
