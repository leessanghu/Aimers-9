TabM, Yandex Research, ICLR 2025
지금 제일 현실적인 최신 NN 후보는 이거예요. Yandex가 공개한 tabular DL 모델이고, “효율적인 MLP ensemble” 느낌입니다. TabNet보다 훨씬 먼저 봐야 합니다. GitHub README에서도 Kaggle 우승/상위권 사례와 큰 데이터 학습 사례를 언급합니다.
우리에게는 embedding MLP 다음 단계 또는 대체재로 적합합니다. Colab A100에서 학습하고, CPU 추론 속도만 확인하면 됨.
출처: https://github.com/yandex-research/tabm

AutoGluon Tabular, AWS/Amazon 계열
기업 공개 오픈소스 예측기 중 가장 실전형입니다. AutoGluon 1.5는 TabArena에서 새 포트폴리오와 foundation model들을 추가했고, stacked ensemble/OOF 기반으로 강합니다.
단, 제출 서버에 통째로 넣기엔 무겁습니다. 우리한테는 Colab에서 후보 모델 발굴/OOF 예측 생성용으로 쓰는 게 맞아요. 최종 제출은 그중 이긴 모델만 가볍게 재구현/저장.
출처: https://auto.gluon.ai/dev/whats_new/v1.5.0.html

TabPrep-LightGBM, AutoGluon 1.5
이건 이름이 중요해요. 그냥 LGBM이 아니라 custom preprocessing + target mean encoding + feature crossing이 붙은 LightGBM입니다. 우리 상황이랑 매우 닮았어요.
즉 “최신 모델”이라기보다, 우리가 해야 할 방향을 말해줍니다: 모델을 더 화려하게 하기보다 전처리/교차피처/OOF ensemble을 정교하게.
출처: https://auto.gluon.ai/dev/whats_new/v1.5.0.html

TabPFN / RealTabPFN, Prior Labs
최신 tabular foundation model 대표입니다. Nature 2025 논문도 있고, GitHub도 활발합니다. 하지만 우리 데이터에는 큰 제약이 있어요. TabPFN은 GPU 권장이고, 큰 데이터는 별도 large-data mode나 제약이 있습니다. README도 기본적으로 작은/중간 데이터에 강하다고 말합니다.
더 큰 문제: pretrained checkpoint가 외부 데이터/합성 데이터 기반이라 대회 규칙상 외부 데이터 사용으로 해석될 위험이 있습니다. 특히 real-data finetuned checkpoint는 더 위험합니다.
출처: https://github.com/PriorLabs/TabPFN

TabICL / TabICLv2, INRIA
TabPFN류 in-context tabular foundation model입니다. 2025/2026 계열이라 최신성은 좋고, “large data에서 TabPFN보다 빠를 수 있다”는 주장이 있습니다.
그래도 pretrained checkpoint 사용 문제와 제출 추론 시간 리스크가 큽니다. 연구 실험 후보지, 안전 제출 후보는 아닙니다.
출처: https://github.com/soda-inria/tabicl

TabDPT, TD Bank Layer6
“real data로 학습한 tabular foundation model” 계열입니다. AutoGluon 1.5에도 들어왔고, HuggingFace 모델도 있습니다.
그런데 real-data pretrained라 대회 규칙상 더 민감합니다. 우리 대회가 “공식 train/test/trackman만”이면 이건 사용 전 문의 없이는 위험합니다.
출처: https://huggingface.co/Layer6/TabDPT


# 투수 내부의 시점 별 폼 변동
살아있는 레버: 투수 내부의 시점별 폼 변동
BSS = Var(예측)/0.25 이므로 (우리 모델은 이미 캘리브레이션 완료라 이 등식이 정확히 성립):

필요 Var	필요 pred_std
현재 926점	0.00232	0.0481
목표 1300점	0.00325	0.0570
그중 투수 실력 몫	0.00207	—
부족분 (폼·상황에서 와야 함)	0.00118	—
그리고 폼 신호는 실재합니다. 각 투구에서 그 투수 자신의 통산 기준선 대비 편차의 표준편차:

피처	자기 베이스라인 대비 SD
prev1_game (직전 1경기)	0.1228
prev3_game	0.0716
prev5_game	0.0579
prev5의 0.0579는 투수 실력 진짜 SD(0.0455)보다 큽니다. 표본 노이즈를 걷어내도(5경기 ≈ 250구 → 노이즈 SD ≈ 0.032) 진짜 폼 변동은 SD 0.048 수준으로 추정됩니다. 우리가 찾는 0.00118을 채우기에 충분한 크기입니다.

핵심: 이 prev1/3/5 피처들은 2025 test에서 순수 2025 신호입니다(2025 경기로 계산됨). in-season과 정확히 같은 성질인데, 우리는 이걸 원본 절대값 그대로 넣고 결측은 전체 평균으로 채우고 있습니다. in-season이 +114점을 낸 이유가 "잘 가공된 요약 통계 하나"였다는 걸 생각하면, 여기가 그대로 방치돼 있는 겁니다.

# 투수 피로 존재
ML 말고 "투수가 왜 공을 놓치는가"로 보면 물리적 메커니즘이 있고, 데이터로 확인됩니다.

피로는 실재합니다 — 이닝별 성공률:


1회 0.4957 → 5회 0.4849 → 8회 0.4751 → 9회 0.4830(마무리 투입)
1회에서 8회까지 -2.1%p 단조 하락, 9회에 반등(다른 투수가 나오니까). 깔끔한 피로 곡선입니다.

시즌 누적 피로도 실재합니다 — 월별:


3월 0.5026 → 7월 0.4799 → 9월 0.4757
inning과 game_month는 이미 원본 피처로 들어가 있지만, 결정적으로 빠진 게 있습니다: 선발/불펜 구분. 7회의 선발은 100구 넘게 던진 상태고, 7회의 불펜은 방금 나온 상태입니다. 같은 inning=7인데 피로도가 정반대예요. 트리는 이걸 구분할 방법이 없습니다.

train에서 투수별 등판 이닝 중앙값·경기당 평균 투구 수를 뽑으면 선발/불펜 점수가 나오고(정적 매핑이라 규칙 준수), 그러면 트리가 inning × 역할을 쓸 수 있습니다.


# 아이디어 1: 지연 공개된 정답으로 최근 투구 상태 복원
현재 in-season은 시즌 시작부터 현재까지를 하나로 평균냅니다. 이를 최근 25/50/100/200구 단위로 쪼갭니다.
같은 투수의 현재 행 i와 과거 행 j에 대해:
N_i = asof_pitcher_n_i
S_i = round(N_i × asof_pitcher_success_rate_i)

최근 성공 횟수 = S_i - S_j
최근 투구 수   = N_i - N_j
최근 성공률   = (S_i - S_j) / (N_i - N_j)
이게 단순 가설이 아닌 이유는 실제 데이터를 전수 확인했을 때:
같은 투수의 연속 행에서 N_i - N_j = 1: 147만 건 전부
S_i - S_j = 직전 control_success: 99.946% 일치
최근 50구 창 계산 가능 행: 92.8%
최근 100구 창 계산 가능 행: 86.7%
즉 asof 컬럼 안에 직전 결과가 거의 정확하게 지연 공개되고 있습니다.
추천 피처는 많이 만들 필요 없이 다음 정도입니다.
recent_success_50_beta
recent_success_200_beta
recent_ball_50_beta
recent_reverse_50_beta
recent_vs_inseason_success
recent_reliability
성공률은 그대로 쓰지 말고 Beta smoothing을 적용합니다.
p_recent = (recent_success + k × p_inseason) / (recent_n + k)
k는 20, 50, 100 정도를 3개 시계열 폴드로 선택합니다. 진짜 중요한 값은 절대 최근률보다 다음 변화량입니다.
recent_form =
logit(p_recent_50) - logit(p_inseason)
이 피처는 “2025 리그가 내려갔다”가 아니라 이 투수가 현재 시즌 평균보다 최근에 더 내려갔는지를 잡습니다.

# 아이디어 2: 온라인 리그 상태를 직접 관측하는 계층적 사전확률
in-season 피처는 투수별로만 만들어지므로, 표본이 적은 투수는 여전히 2024까지의 과거 환경에 강하게 끌립니다.
하지만 아이디어 1의 누적 횟수 차이를 이용하면, 2025 행을 시간순으로 처리하면서 이미 끝난 투구 결과를 복원할 수 있습니다. 이를 모든 투수에 대해 합치면 현재 시즌의 리그 제구 환경을 실시간으로 추정할 수 있습니다.
A_t = 시점 t까지 복원된 성공 횟수
M_t = 시점 t까지 복원된 투구 수

league_state_t =
(A_t + κ × league_2024) / (M_t + κ)
여기서 매우 중요한 제약은 전체 test를 미리 집계하면 안 된다는 것입니다. 반드시 row_id 순서로 처리하며, 현재 행에서 확인된 누적 변화까지만 상태에 반영해야 합니다.
그다음 투수 확률을 계층적으로 분해합니다.
최종 상태 =
현재 리그 상태
+ 투수의 현재 시즌 상대 편차
+ 투수의 최근 50구 상대 편차
실제 피처 형태는 다음이면 충분합니다.
online_league_success
inseason_minus_online_league
recent50_minus_online_league
online_league_n

## 아이디어 9 — 반복구간 스냅샷 평균 (거의 공짜 분산감소)

배경: 오늘(idea1~8) 확인된 것 — 우리 앙상블 멤버 간 상관이 0.86~0.94 밑으로 안 내려간다.
seed/depth/rsm/행드롭 전부 "무작위로 조금씩 흔들기"라 결국 같은 정보 경로로 수렴한다.

아이디어: CatBoost는 `predict(ntree_end=k)`로 **학습 도중 임의 시점의 예측**을 뽑을 수 있다.
한 번 학습해서 iteration 250/350/450/550 시점 예측을 평균내면, **추가 학습 비용 0**으로
분산이 줄어든다. 각 시점은 (a) 서로 다른 만큼의 노이즈를 학습했고 (b) 핵심 신호는 공유하므로
전형적인 분산감소 앙상블이다.

구현 스케치:
```python
cb.fit(X_tr, y_tr, eval_set=(X_es, y_es), early_stopping_rounds=50)
best = cb.best_iteration_
snap_iters = [int(best*0.5), int(best*0.7), int(best*0.85), best]
preds = [cb.predict_proba(X_va, ntree_end=k)[:,1] for k in snap_iters]
p_snapshot_avg = np.mean(preds, axis=0)
```

검증 시 주의: v38/v39 교훈 반영 — 반드시 (1) 시드 반복으로 노이즈 바닥 먼저 측정,
(2) fold A/C 클린 폴드 기준으로 판정, fold B(regime단절)의 큰 수는 신뢹불가.
스냅샷 간 상관이 기존 seed/rsm 조합보다 낮게 나오는지가 채택 여부의 핵심 지표.

## 아이디어 10 — 음의 상관 학습 (다양성을 명시적으로 강제)

배경: 지금은 다양성이 "우연히 생기길" 바라는 구조다(seed/rsm/블록드롭 등). 대신 앙상블에
이미 있는 모델이 틀리는 지점에 새 모델이 강제로 집중하게 만들 수 있다(Negative Correlation
Learning 계열).

핵심 식: 이미 학습된 모델 p_1이 있을 때, 두 번째 모델의 타겟을 이렇게 바꾼다.
```
y' = y + λ * (y - p_1)      # p_1이 틀린 방향으로 잔차를 증폭
```
λ는 증폭 강도(0.3~1.0 범위 탐색). 부스팅 자체의 잔차학습(같은 모델 내부, 이전 트리의 오차를
다음 트리가 보완)과 다른 점은, 이건 **독립된 완전한 모델**을 만드는 것이라 그대로 앙상블
멤버로 쓸 수 있다는 것이다. p_1이 이미 학습한 것과 반대 방향으로 편향되게 유도되므로
상관이 구조적으로 낮아질 가능성이 있다.

리스크: y'이 [0,1] 밖으로 나갈 수 있어 클리핑 필요, λ가 크면 노이즈까지 증폭할 위험 있음
(특히 우리처럼 rho≈0.10인 노이즈 지배 타겟에서는 특히 조심). 작은 λ부터 스윕 필요.

검증도 동일 원칙: 시드반복 + fold A/C 우선 판정 + fold B 결과는 참고만.
season_progress
이 방식의 장점은 콜드스타트 투수에게 특히 큽니다. 자기 시즌 표본이 5구뿐이어도 이미 리그 전체에서 수만 구가 관측됐다면, 2025의 공통 드리프트는 바로 반영할 수 있습니다.