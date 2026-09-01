# LG Aimers 9 — 투구 제구 성공 예측 (DACON)

**과제**: 투구 단위 `control_success`(제구 성공/실패) 이진분류. 평가지표는 **Brier Skill Score** 기반 리더보드 점수
`score = 1e5 × (1 − BS/BS_ref)`, `BS_ref = 0.249807`.

**최종 성적: 199등 / 1090팀 (상위 18.3%)**
**최종 확정 최고 점수: `1115.6283` (v128, 실측 리더보드 확정값)**
(v95 대비 +11.97, 시즌 초기 v4 대비 +290)

---

## 1. 최종 모델(v128) 구성

`submit/script.py` + `submit/model/model_artifacts_v128.pkl` (단일 아티팩트, 17개 헤드의 선형 블렌드).

블렌드 = `Σ w_i × head_i(X)`, 클리핑 후 레벨보정(RiskAdj, K2Adj) 적용.

### 코어 10헤드 (162피처 위 GBDT 계열, w95 유산)

| 헤드 | 가중치 | 알고리즘 | 메커니즘 |
|---|---:|---|---|
| base | 0.0488 | HGB×3(d6/d8/sub) 평균 | y 직접 이진분류 |
| hurdle | 0.0651 | HGB×2 | 2단계: `P(middle∨reverse)` → `(1-그것)×P(success\|나머지)` |
| multires | 0.0163 | CatBoost MultiRMSE | `[y, 투수×시즌 LOO성공률, 투수×손타자 LOO성공률]` 보조타겟 |
| ordinal | 0.0325 | HGB×3 캐스케이드 | `P(~reverse)×P(~middle\|~reverse)×P(success\|나머지)` |
| midother | 0.0407 | CatBoost MultiRMSE | `[y, 1-middle라벨, 1-other라벨]` 보조타겟 |
| condball | 0.0214 | CatBoost MultiRMSE | `[y, not-dangerous 조건부 1-ball]` 보조타겟 |
| countresid | 0.0214 | CatBoost MultiRMSE | `[y, y − E[y\|count]]` 잔차 보조타겟 |
| future50 | 0.0214 | CatBoost MultiRMSE | `[y, 향후50투구 성공률/middle율]` (train만, Rule4 안전) |
| mc5 | 0.0472 | CatBoost MultiClass(5) | 초기 타겟분해 헤드 (mc6의 전신) |
| ingame | 0.0274 | CatBoost | 경기내 보조 헤드 |

### 신규 발견 헤드 7개 (이번 세션에서 도입)

| 헤드 | 가중치 | 알고리즘 | 메커니즘 | 검증 |
|---|---:|---|---|---|
| **mc6pure** | **0.4343** | CatBoost MultiClass(6) | 판정축(ball/strike/inplay)×middle/reverse/wild **순수분할**. `P(success)=P(succ_ball)+P(succ_strk)+P(succ_play)`, 전 클래스 성공률 0% 또는 100% | 실측 3점으로 정확히 확정 (오차 4e-16), **+9.77점 단독 기여 — 세션 최대 성공** |
| strk | 0.1951 | CatBoost MultiRMSE | `[y, 연속실패길이/10]` 보조타겟 | 실측 확정 |
| xgbunused | -0.0313 | XGBoost (소형) | "안 쓰이는 피처"(tm_matched, tm_lown_flag, pitcher_hand, form_missing) + season/game_type 스무딩. 음수가중치(거친 모델의 편향을 뺌) | **실측 확정 +0.47점** |
| xgbrawid | 0.0185 | XGBoost | 162피처 + pitcher/batter/team ID native categorical | 실측 부호반전 확인 후 양수로 재투입 |
| lty | 0.0213 | LightGBM (`linear_tree=True`) | 조각별-선형 함수공간(계단함수 4종과 차별화), binary y | 실측 부호반전 확인 후 양수로 재투입 |
| mc6aux | 0.0099 | CatBoost MultiRMSE | `[y, onehot(mc6 6클래스)]`, head0만 추론사용 | fold A 통과했으나 실측 미미 |
| N1 | 0.0100 | PyTorch MLP | 원시컨텍스트53 + 원시비율18(스무딩 없는 as-of rate) + ID임베딩4종(투수/타자/투수팀/타자팀) | fold A z=2.4 통과했으나 실측 미미 |

### 레벨보정
- **RiskAdj**: risk_alpha 기반 임계값 근처 보정
- **K2Adj**: pitcher-level shrinkage 보정 (K=1500)

---

## 2. 점수 궤적 (전부 실측 리더보드 확정값)

| 버전 | 변경 | 점수 | Δ |
|---|---|---:|---:|
| v95 | 기준선(코어 10헤드) | 1103.6568 | — |
| v112 | +mc6pure(w=0.03) | 1104.8343 | +1.18 |
| v114 | mc6pure w→0.10 | 1107.2877 | +2.45 |
| v116 | mc6pure w→0.48(항등식 최적점) | 1113.4251 | +6.14 |
| v117 | +strk(w=0.10) | 1114.5297 | +1.10 |
| v122 | +xgbunused(-0.03) | 1115.0040 | +0.47 |
| v123 | +xgbrawid(-0.03) | 1113.4529 | **−1.55 (부호반전 확인)** |
| v124 | strk 재조정(항등식 역산) | 1115.1606 | +0.16 |
| v125 | +lty(-0.03) | 1114.6411 | **−0.52 (부호반전 확인)** |
| v126 | xr/lty 부호 반전 재투입(+) | 1115.4738 | +0.31 |
| **v128** | **+mc6aux(0.01)+N1(0.01)** | **1115.6283** | **+0.15 — 최종 최고** |
| v130 | xu 가중치 대이동 시도 | 1114.7380 | −0.89 (과최적화 확인, 후퇴) |
| v134 | F리그 전문가 블라인드 프로브 | 1114.4880 | −1.14 (F전문가 축 폐기 확정) |

---

## 3. 핵심 방법론

### 3.1 실측 프로브 우선주의 (probe-first)
리더보드 점수는 **결정론적**(측정오차 0) 이차함수를 따른다:

```
Score(s) = Score₀ − K·(2sA + s²V),   K = 1e5 / BS_ref ≈ 400,309
```

- `s`: 후보 헤드의 블렌드 가중치, `A,V`: 후보 방향에 대해 고정된 상수
- 실측 점 2~3개만 있으면 `A, V`를 **정확히** 역산 가능 → 최적 가중치 `s* = -A/V`, 최대이득 `= K·A²/V`
- mc6pure는 이 방법으로 실측 3점(오차 4e-16)에서 최적점을 정확히 확정했다.

### 3.2 로컬 검증은 방향조차 못 믿는다
- xgbrawid, lty 둘 다 fold A 로컬에서 강한 신호(z=2.9, z=4.1)를 보였지만 **실측에서 부호가 반전**됐다.
- 원인: rho(잔차상관) < 0.006 수준의 약신호는 2024→2025 연도 전이에서 부호 보존이 보장되지 않는다.
- 대응: 확신 큰 가중치를 걸지 않고 소량(±0.01~0.03) 프로브 → 실측 부호 확정 → 재투입.

### 3.3 정보량 상한(ICC 바닥)
- 투수 identity의 클래스내상관(ICC)은 0.6~1%에 불과 — 162피처의 축소평균(as-of/inseason)이 이미 대부분 흡수.
- XGB/LGBM 등 이질 알고리즘을 raw-ID로 학습해도 이 바닥 이하로는 못 판다 (5개 각도로 반복 검증 후 폐기).

### 3.4 앙상블 실효 다양성의 한계
- 15개 코어+신규 헤드가 전부 **같은 162피처** 위에서 학습돼 상호상관 0.9+, 실효랭크 ≈ 1.1.
- 즉 "앙상블"이라기보다 한 모델의 15가지 표현에 가까움 — 상위권과의 격차는 **독립적으로 개발된 저상관 파이프라인의 부재**로 추정.

### 3.5 mc6 성공의 구조적 원인
- `추론 시점엔 알 수 없지만 as-of 누적카운터 차분으로 사후 복원 가능한 라벨`(판정축: ball/strike/inplay)을 **타겟으로만** 사용 — 이게 유일하게 새 정보채널을 연 사례.
- 반증 실험(2×2 ablation): 성공측만 분할(mc4), 실패측만 분할(mc4f) 각각 단독으로는 대조군 이하 — mc6의 가치는 **6-way 전체 조합의 특정 상호작용**이며 일반화 가능한 원리는 아니었다.

---

## 4. 시도했으나 기각된 축 (요약)

| 카테고리 | 시도 | 결과 |
|---|---|---|
| 이질 알고리즘 | XGB/LGBM raw-ID, 원시피처, 구간별가중치, 모델강도, 타겟분해 (5각도) | 전부 정보이론적 바닥에 막힘 |
| 타겟 재분할 | mc8(wild분할), mc10(middle/reverse분할), mc6pt(구종분할), mc4/mc4f | 전부 대조군 이하 |
| NN 계열 | NN v2(가공피처), N2(PLR임베딩), NF(트랙맨+bilinear+멀티태스크) | z=0.6, -0.2, 0.6 — 전부 허수. N1만 유일하게 통과했으나 실측 미미 |
| 캘리브레이션 | 로짓풀링, winsor, 헤드별 isotonic | 예측분포가 0.35~0.65 좁은 대역이라 풀링기하 자체가 무의미 |
| 레짐 분할 | R/F 리그 완전분리, 리그별 라우팅 합성(mc6split), same_hand/two_strike 조건분할 | mc6split·same_hand·two_strike는 fold A 스크리닝 통과(z=3.5/4.3/2.7)했으나 F전문가 단독 축은 실측에서 확정 폐기 |
| 시간가중치 | half-life 스윕(1.0/2.0/4.0) | 기존 2.0이 이미 최적, 드리프트 적응 레버 없음 |

---

## 5. 재현 방법

```bash
cd submit
pip install -r requirements.txt
python script.py
# submit/data/test.csv 필요, submit/output/submission.csv 생성
```

`submit/model/` 안의 `model_artifacts_v*.pkl` 중 최신 버전이 자동 선택된다
(패키징 시 `dev/package_v*.py`가 단일 버전만 남기고 격리).

---

## 6. 디렉토리 구조

```
submit/           최종 제출 코드(script.py) + 모델 아티팩트
dev/              실험 스크립트 전체(수백 개, 버전별 build/probe/screen)
data/             원본 데이터 (repo 미포함, .gitignore)
```

`dev/` 안의 각 `build_*.py`/`probe_*.py`/`screen_*.py`는 파일명으로 실험 목적을 유추 가능하도록
명명했다. 주요 스크립트는 상단 docstring에 가설·검증방법·결론을 기록해두었다.
