# Claude에게 요청: 1200점 근처로 가기 위한 다음 아이디어 점검

## 0. 한 줄 결론

지금 가장 큰 병목은 모델 용량이나 구종 확장이 아니라 **투수의 현재 잠재 제구 상태(current command ability)를 얼마나 잘 추정하느냐**로 보입니다.

단순히 in-season rate 컬럼을 더 많이 GBM에 넣는 방식은 이미 여러 번 실패했습니다. 대신 in-season 관측치를 이용해 별도의 **Current Ability Teacher**를 만들고, 그 출력값을 최종 모델의 상태 피처 또는 additive prior로 쓰는 방향을 우선 검토해 주세요.

---

## 1. 현재 상황 요약

현재 최고 제출은 v14b입니다.

```text
v12  = 963.796   (in-season 원본 + 교차항 + CatBoost/HGB 50:50)
v14a = 970.493   (v12 + 구종 주변화, 50:50)
v14b = 970.546   (v14a에서 HGB:CatBoost = 20:80)
```

구종 아이디어는 작동했지만 +6.7점에 그쳤습니다. 반면 지금까지 제일 크게 오른 피처군은 in-season 원본입니다.

```text
base 811.9 -> in-season 포함 925.9, 약 +114점
```

상한 분석상 현재 모델은 로컬 기준 약 821점이고, 투수의 2024 실제 성공률을 안다고 가정하면 946.5점, 거기에 볼카운트 주효과까지 알면 981.0점입니다. 실제 LB 환산을 생각하면 1100~1200점대의 대부분은 **투수 현재 실력 추정 정밀도**에 남아 있는 것으로 보입니다.

---

## 2. 평가 산식과 우리가 실제로 최적화해야 하는 값

대회 점수는 Brier Skill Score입니다.

```math
\mathrm{BS} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
```

```math
\bar{y}=\frac{1}{N}\sum_{i=1}^{N}y_i
```

```math
\mathrm{BS}_{base}=\bar{y}(1-\bar{y})
```

```math
\mathrm{Score}=100000\left(1-\frac{\mathrm{BS}}{\mathrm{BS}_{base}}\right)
```

따라서 새 피처의 가치는 raw correlation이나 재현상관보다, 기존 모델 잔차를 얼마나 줄이는지가 핵심입니다.

현재 예측을 `p`, 정답을 `y`, 후보 피처를 `z`라 하면 잔차는:

```math
e_i = y_i - p_i
```

후보 피처 `z`에서 기존 예측/기존 피처로 설명되는 부분을 제거한 값을 `z_\perp`라고 할 때, 선형으로 얻을 수 있는 이상적 Brier 개선량은 대략:

```math
\Delta \mathrm{BS}
=
\frac{\operatorname{Cov}(e,z_\perp)^2}{\operatorname{Var}(z_\perp)}
```

즉 앞으로는 후보 피처마다 아래를 봐야 합니다.

- raw SD
- 재현상관
- 기존 예측 `p`와의 상관
- 잔차와의 상관 `corr(z_\perp, y-p)`
- 최적 선형 계수
- paired Brier delta
- pitcher-season block bootstrap CI

이 기준으로 보면 구종은 신호 진폭과 재현상관은 좋지만, 기존 모델 잔차에 남은 독립 정보가 생각보다 작았을 가능성이 큽니다.

---

## 3. 구종 아이디어에서 배운 점

구종 주변화 구현은 다음과 같습니다.

```math
\hat{p}_{pt}
=
\sum_{t} P(t \mid \mathrm{pitcher}, \mathrm{count}) \cdot
\mathrm{ctrl}(\mathrm{pitcher}, t)
```

```math
pt\_dev
=
\hat{p}_{pt}
-
\mathrm{prior}(\mathrm{pitcher})
```

실측:

```text
매칭 정밀도: 99.5% / 99.7%
커버리지: 61.5%
투수×구종 true SD: 0.0271
재현상관: r = 0.475
pt_dev SD: 0.0122
LB 개선: +6.7
```

분산 분해:

```math
\operatorname{SD}(pt\_dev)=0.01217
```

```text
between pitcher SD = 0.00681, 분산의 31.4%
within pitcher SD  = 0.01008, 분산의 68.6%
```

처음에는 주변화 때문에 신호가 희석됐다고 봤지만, 순수 신규 신호의 SD가 0.01008로 충분히 큽니다. 그런데 platoon은 비슷한 진폭에서 +14점, 구종은 +6.7점입니다.

따라서 앞으로는 “진폭 + 재현상관”만으로는 부족하고, 반드시 기존 모델 잔차 기준의 증분 가치를 측정해야 합니다.

---

## 4. 모델 블렌딩에 대한 중요한 정정

v14a와 v14b 결과:

```text
v14a: HGB 50%, CatBoost 50% -> 970.493
v14b: HGB 20%, CatBoost 80% -> 970.546
```

이 결과를 “HGB와 CatBoost가 완전히 동등하다”로 해석하면 안 됩니다. Brier score는 블렌딩 weight에 대해 2차식입니다.

```math
p(w)=w p_{HGB}+(1-w)p_{Cat}
```

```math
\mathrm{BS}(w)
=
\frac{1}{N}\sum_i
\left(w p_{HGB,i}+(1-w)p_{Cat,i}-y_i\right)^2
=Aw^2+Bw+C
```

`w=0.5`와 `w=0.2`가 거의 같다면, 최적점은 대략 중간인:

```math
w^\* \approx \frac{0.5+0.2}{2}=0.35
```

일 가능성이 있습니다. 즉 `HGB:CatBoost = 35:65`가 논리적으로는 다음 후보입니다. 다만 v14a/v14b 차이가 0.05점뿐이라 기대 이득은 매우 작습니다. 1200점을 위한 주력 방향은 아닙니다.

---

## 5. 핵심 제안: Current Ability Teacher

최종 모델이 바로 noisy in-season 피처를 먹는 대신, 별도 모델이 “현재 투수 상태”를 먼저 추정하게 합니다.

개념적으로는:

```math
\theta_{p,s,t}
=
\text{pitcher }p\text{의 season }s,\text{ 시점 }t\text{에서의 잠재 제구력}
```

최종 예측은 아래처럼 분해합니다.

```math
\operatorname{logit}(P(y_i=1))
=
\theta_{p_i,s_i,t_i}
+
\delta(x_i)
```

또는 확률 스케일에서 단순하게:

```math
\hat{p}_i
=
\operatorname{clip}
\left(
\hat{\theta}_{p_i,s_i,t_i}
+
\widehat{\delta}(x_i),
\epsilon,
1-\epsilon
\right)
```

여기서:

- `theta`: 투수의 현재 상태, 느리게 변하는 ability
- `delta(x)`: count, inning, runner, batter hand, pitchtype 같은 row context 효과

즉 최종 모델을 “상태 tower + 문맥 tower”로 나눠 생각합니다.

---

## 6. Teacher의 학습 타깃 후보

각 투수-시즌의 시점 snapshot을 만듭니다. 예를 들어 시즌 내 누적 투구 수가 25, 75, 150, 300개를 넘는 시점에서 snapshot을 생성합니다.

snapshot 시점까지의 정보만 feature로 쓰고, target은 그 이후 미래 구간의 성공률로 둡니다. 이 미래값은 학습용 auxiliary label일 뿐이고, test 예측 때는 사용하지 않습니다.

### 타깃 1: remainder-of-season ability

```math
y^{EOS}_{p,s,t}
=
\frac{
\sum_{j \in \mathcal{F}_{p,s,t}} y_j
}{
|\mathcal{F}_{p,s,t}|
}
```

여기서:

```math
\mathcal{F}_{p,s,t}
=
\{j: pitcher_j=p,\ season_j=s,\ time_j>t\}
```

표본 수가 작으면 empirical Bayes shrinkage를 적용합니다.

```math
\tilde{y}^{EOS}_{p,s,t}
=
\frac{
n_{future}y^{EOS}_{p,s,t}+K\mu_s
}{
n_{future}+K
}
```

### 타깃 2: next 100 pitches

```math
y^{100}_{p,s,t}
=
\frac{1}{100}
\sum_{j \in \mathrm{next100}(p,s,t)} y_j
```

실제로는 100개 미만이면 `n_future` 가중치와 shrinkage를 적용합니다.

### 타깃 3: next 3 games

```math
y^{3G}_{p,s,t}
=
\frac{
\sum_{j \in \mathrm{next3games}(p,s,t)} y_j
}{
|\mathrm{next3games}(p,s,t)|
}
```

LG AI 연구원 자료에서 강조한 “최근 경기 흐름, 컨디션, 상황별 제구 안정성”과 가장 잘 맞는 타깃은 `next3games`입니다. 다만 noisy할 수 있으므로 `EOS`, `next100`, `next3games`를 모두 만들고 OOF residual Brier로 비교해야 합니다.

---

## 7. Teacher 입력 피처 후보

Teacher에는 row-level context보다 pitcher-state 관련 피처를 넣습니다.

우선 후보:

- career/asof success, reverse, middle, ball, strike rate
- career/asof sample count
- current season success, reverse, middle, ball, strike rate
- current season sample count
- prev1/3/5 game success, middle, ball, reverse
- month, season, season progress
- previous season final rate
- career vs current-season delta
- current-season posterior variance
- cold-start flags: 작년 등판 여부, 마지막 등판 시즌 gap, career n, previous season n

중요한 점은 `success` 하나만 보지 말고, success를 구성하는 atomic indicator를 상태 관측으로 넣되 강한 shrinkage/teacher 내부에서만 쓰는 것입니다.

우리가 누적값으로 복원한 atomic indicator를 `S,R,M,B,K`라고 하면:

```math
S_i=\mathbf{1}(\mathrm{success})
```

```math
R_i=\mathbf{1}(\mathrm{reverse})
```

```math
M_i=\mathbf{1}(\mathrm{middle})
```

```math
B_i=\mathbf{1}(\mathrm{ball})
```

```math
K_i=\mathbf{1}(\mathrm{strike})
```

파생축 후보:

```math
zone = K - B
```

```math
neutral = 1-K-B
```

```math
bad\_contact = R + M
```

```math
command = S - R - M
```

다만 half-season multivariate test에서 `S/R/M/B/K`를 단순 추가하면 2023/2024에서 악화됐으므로, 최종 GBM에 직접 투입하기보다 Teacher 안에서 denoising 용도로 쓰는 게 맞아 보입니다.

---

## 8. 새 in-season 계열: prev1/3/5를 disjoint block으로 분해

현재 `prev1/3/5_game_*`는 겹치는 window입니다. 겹치는 평균을 그대로 넣으면 트리가 이미 알고 있는 정보와 중복되고, form/workload 실험처럼 LB에서 깨질 가능성이 큽니다.

대신 가능한 경우 `prev1`, `prev3`, `prev5`를 서로 겹치지 않는 block으로 복원합니다.

투수 `p`, 시점 `t`에서 최근 1/3/5경기 누적 투구 수를:

```math
n_1,\ n_3,\ n_5
```

성공률을:

```math
r_1,\ r_3,\ r_5
```

라고 하면, 정수 성공 수는:

```math
c_1 = round(n_1r_1)
```

```math
c_3 = round(n_3r_3)
```

```math
c_5 = round(n_5r_5)
```

겹치지 않는 블록은:

```math
c_{2:3}=c_3-c_1,\quad n_{2:3}=n_3-n_1
```

```math
c_{4:5}=c_5-c_3,\quad n_{4:5}=n_5-n_3
```

```math
r_{2:3}=\frac{c_{2:3}}{n_{2:3}},\quad
r_{4:5}=\frac{c_{4:5}}{n_{4:5}}
```

이걸로 trend를 만듭니다.

```math
trend_{short}=r_1-r_{2:3}
```

```math
trend_{long}=r_{2:3}-r_{4:5}
```

```math
accel=(r_1-r_{2:3})-(r_{2:3}-r_{4:5})
```

주의: `prev3/5`가 pooled pitch rate가 아니라 game-rate 평균이면 위 복원은 틀립니다. 먼저 `n_1 \le n_3 \le n_5`와 정수 일관성을 검증해야 합니다.

이 block 피처도 최종 모델에 바로 넣기보다 Teacher 입력으로 넣는 것을 권장합니다.

---

## 9. 리그 시즌 shock과 개인 form을 분리

in-season이 컸던 이유 중 하나는 시즌별 리그 평균 성공률이 크게 변했기 때문입니다. 따라서 current-season delta를 하나의 개인 form으로만 보면 안 되고, 공통 시즌 shock과 개인 deviation으로 분리해야 합니다.

```math
\Delta_{p,s,t}
=
r^{cur}_{p,s,t}
-
r^{prior}_{p,s}
```

이를:

```math
\Delta_{p,s,t}
=
\mu_{s,t}+u_{p,s,t}+\epsilon_{p,s,t}
```

로 분해합니다.

- `mu_{s,t}`: 해당 시즌/월의 공통 리그 shock
- `u_{p,s,t}`: 투수 개인의 현재 form 변화
- `epsilon`: 표본 noise

test에서는 행 간 집계가 금지되어 있으므로 test row들을 모아 `mu`를 추정하면 안 됩니다. 대신 과거 시즌에서 학습한 mapping으로, row-local 정보만 이용해 `mu`와 `u`의 posterior를 예측해야 합니다.

---

## 10. 누수 방지 설계

Teacher는 미래 성공률을 label로 쓰므로 cross-fitting이 필수입니다.

학습 fold가 season 기준일 때:

```math
\hat{\theta}^{OOF}_{i}
=
f_{-season(i)}(X_i)
```

즉 어떤 행의 Teacher prediction도 같은 season의 미래 label을 보고 학습된 모델에서 나오면 안 됩니다.

최종 모델 학습:

```math
X^{final}_i
=
[X^{v14}_i,\ \hat{\theta}^{OOF}_i]
```

2025 test 예측:

```math
\hat{\theta}^{2025}_i
=
f_{2019:2024}(X^{2025}_i)
```

최종 제출:

```math
\hat{p}^{2025}_i
=
g_{2019:2024}
\left(
X^{v14,2025}_i,
\hat{\theta}^{2025}_i
\right)
```

---

## 11. 이미 한 간이 진단 결과

snapshot 기반 Teacher 진단을 작게 돌렸습니다.

설정:

- pitcher-season snapshot: 시즌 내 25/75/150/300 pitches 지점
- 총 5,751 snapshots
- 입력: season/month, career rate/n, prev1/3/5 success+middle, current inseason success/ball/reverse/n, threshold
- target: remainder-of-season actual success rate

결과:

```text
2022 baseline current inseason MSE .006554
2022 Ridge .003630
2022 HGB .002700

2023 baseline .003910
2023 Ridge .002447
2023 HGB .002716

2024 baseline .003979
2024 Ridge .001963
2024 HGB .001940
```

season mean 제거 후:

```text
2022 base .006247 -> aux .003008, -51.8%
2023 base .003873 -> aux .002394, -38.2%
2024 base .003907 -> aux .001922, -50.8%
```

다만 correlation은 항상 좋아지지 않았습니다.

```text
2022 corr base .463 -> aux .533
2023 corr base .412 -> aux .195
2024 corr base .394 -> aux .345
```

해석: Teacher가 MSE를 크게 줄인 것은 강한 shrinkage와 global calibration 덕분일 수 있습니다. 그래도 “현재 상태를 별도 denoising 모델로 추정한다”는 방향은 가장 강한 후보입니다. 다음 검증은 반드시 최종 v14 residual Brier 기준으로 해야 합니다.

---

## 12. Claude에게 묻고 싶은 구체 질문

1. 위 `Current Ability Teacher` 설계가 맞다고 보시나요? 아니라면 `theta + context_delta` 구조에서 어떤 부분을 바꾸는 게 좋을까요?

2. Teacher target은 `EOS`, `next100`, `next3games` 중 무엇을 우선해야 할까요? LG AI 연구원 문제 설명은 최근 경기 흐름/컨디션/상황별 안정성을 강조하는데, 실제 Brier 최적화 관점에서는 어느 horizon이 맞을지 궁금합니다.

3. Teacher의 출력값을 최종 모델에 넣는 방식은 어떤 게 좋을까요?

```math
\hat{p}_{final}=g(X,\hat{\theta})
```

또는:

```math
\hat{p}_{final}
=
\hat{p}_{v14}
+
\alpha \cdot
\left(
\hat{\theta}
-
\mathrm{prior}
\right)
```

혹은 logit additive:

```math
\operatorname{logit}(\hat{p}_{final})
=
\operatorname{logit}(\hat{p}_{v14})
+
\alpha z_{\theta}
```

어느 쪽이 Brier와 calibration에 더 안전할까요?

4. disjoint prev-game block 복원이 유망해 보이나요? 유망하다면 success뿐 아니라 reverse/middle/ball/strike에도 같은 방식으로 적용할 가치가 있을까요?

5. in-season이 큰 이유를 `league shock + individual form`으로 분해하는 가설이 맞을까요? 맞다면 test row 간 집계 없이 row-local하게 `mu_{s,t}`를 추정하는 좋은 방법이 있을까요?

6. 구종 확장은 여기서 멈추는 게 맞을까요? sequence alignment로 coverage를 61.5%에서 90%로 올리면 단순 비례상 +3점 정도밖에 기대하기 어렵습니다. 세부 구종/타자손/이닝 교차를 파는 게 residual 기준으로 정말 가치가 있을지 판단 기준이 필요합니다.

7. 로컬 fold가 자주 부호를 틀립니다. 이 상황에서 후보 아이디어를 채택하기 위한 최소 검증 프로토콜을 어떻게 잡는 게 좋을까요?

---

## 13. 제가 생각하는 우선순위

1. v14 OOF prediction을 기준으로 residual을 만든다.
2. `EOS`, `next100`, `next3games` Teacher를 season-wise OOF로 만든다.
3. 각 Teacher 출력의 residual Brier contribution을 계산한다.
4. `n_season` 구간별 gate를 둔다.

```math
g(n)=
\begin{cases}
0 & n=0\\
g_1 & 1\le n<50\\
g_2 & 50\le n<150\\
g_3 & n\ge150
\end{cases}
```

5. disjoint prev1/3/5 block을 Teacher 입력에 추가한다.
6. Teacher 출력은 먼저 단순 residual additive로 붙이고, 그 다음 최종 GBM feature로 넣는 방식을 비교한다.
7. 마지막으로만 HGB:CatBoost 35:65나 구종 coverage 확장을 확인한다.

---

## 14. 핵심 부탁

새로운 피처를 그냥 “추가”하는 아이디어보다, 아래 질문에 답하는 방식으로 봐주세요.

```math
\text{Can this feature estimate } \theta_{p,s,t}
\text{ better than current in-season rate?}
```

그리고:

```math
\text{Does it reduce } \operatorname{Cov}(y-\hat{p}_{v14}, z_\perp)?
```

즉 1200점으로 가려면 “더 많은 상황 피처”보다 **현재 투수 상태를 더 깨끗하게 추정하는 장치**가 필요하다고 봅니다.
