"""v79 = v78(11-class 구종) + ingame(경기내 컨디션) aux head.

발견 경위:
  투수-경기 오라클 Resolution 0.015388 vs 투수-시즌 0.003878 (4배).
  경기간 컨디션 자기상관은 0.0474로 죽었지만, 경기내는 0.1908로 살아있다.
  즉 "그날 컨디션"은 경기를 넘어 이어지지 않지만 경기 안에서는 지속된다.
  우리는 지금까지 prev1/3/5_game(경기간)만 썼고 경기내는 한 번도 안 썼다.

경기 경계는 asof_pitcher_prev1_game_success_rate가 갱신되는 시점으로 복원한다
(경기당 median 19~22투구, 45,121경기).

aux 타깃 = 현재 경기에서 '직전 투구까지'의 누적 성공률(자기 자신 제외, K=8 축소).
  - 자기 자신을 빼므로 leakage 없음
  - 경기 첫 투구는 NaN -> MultiRMSEWithMissingValues
  - 커버리지 96.9%, 10분위 격차 19.73%p(단조), corr(y)=+0.1147

Rule 4 준수: train에서만 이 타깃을 계산해 학습하고, test에서는 각 행이 자기
피처(asof/inning/role 등)로 모델이 추정한다. test의 다른 행을 참조하지 않는다.
이는 v49 formcast(실측 +4), v64 future50(실측 +2.16)과 동일한 검증된 패턴이다.
"""
src = open('train_final_v78b_mc11.py', encoding='utf-8').read()

# 1) 11-class 라벨 뒤에 ingame aux 타깃 구성 추가
anchor = 'fit_mask = cls >= 0'
assert anchor in src
new = '''fit_mask = cls >= 0

# ---- 경기내 컨디션 aux 타깃 ----
p1g = df["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
p1o = p1g[order]
same_prev = np.zeros(len(df), dtype=bool)
same_prev[1:] = pid[order][1:] == pid[order][:-1]
chg = np.zeros(len(df), dtype=bool)
chg[1:] = (p1o[1:] != p1o[:-1]) & ~(np.isnan(p1o[1:]) & np.isnan(p1o[:-1]))
newgame = (~same_prev) | (chg & same_prev)
gid_o = np.cumsum(newgame)
y_o = y.astype(np.float64)[order]
cum = pd.Series(y_o).groupby(gid_o).cumsum().to_numpy() - y_o   # 자기 자신 제외
kk = pd.Series(gid_o).groupby(gid_o).cumcount().to_numpy()
K_ING = 8.0
ing_o = np.where(kk > 0, (cum + K_ING * g) / (kk + K_ING), np.nan)
head_ingame = np.empty(len(df))
head_ingame[order] = ing_o
log(f"  경기내컨디션 aux: 경기수={gid_o.max():,} 커버리지={np.isfinite(head_ingame).mean()*100:.1f}%")'''
src = src.replace(anchor, new, 1)

# 2) 학습부: MultiClass -> 별도 ingame 회귀 head를 갖는 두 번째 모델 추가
old_fit = src[src.index('log("11-class softmax CatBoost training'):src.index('log(f"  done best_iter=')]
add = '''
# ---- ingame aux head 모델 (y + 경기내컨디션 2-head 공유트리) ----
log("ingame aux head 학습 (y / 경기내 누적성공률 2-head)...")
from catboost import CatBoostRegressor
Ying = np.column_stack([y.astype(np.float64), head_ingame])
ing_ok = np.isfinite(head_ingame)
ing_idx = np.where(ing_ok)[0]
n_es2 = int(len(ing_idx) * 0.92)
ti2, ei2 = ing_idx[:n_es2], ing_idx[n_es2:]
ts2 = time.time()
ingame_model = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                                 verbose=False, random_seed=42,
                                 loss_function="MultiRMSEWithMissingValues",
                                 early_stopping_rounds=50)
ingame_model.fit(X.iloc[ti2], Ying[ti2], sample_weight=w[ti2],
                 eval_set=(X.iloc[ei2], Ying[ei2]))
log(f"  ingame done best_iter={ingame_model.best_iteration_} ({time.time()-ts2:.0f}s)")
strip_rng(ingame_model)

'''
src = src.replace('log(f"  done best_iter=', add + 'log(f"  done best_iter=', 1)

# 3) 아티팩트 저장: v78 위에 ingame 추가, mc5 0.15 유지 + ingame 0.08
old_common = '''common = dict(v66)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - MC_WEIGHT)
common["mc5_model"] = mc_model
common["mc5_succ"] = succ_by_cls
common["mc5_weight"] = MC_WEIGHT'''
new_common = '''ING_WEIGHT = 0.08
common = dict(v66)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - MC_WEIGHT - ING_WEIGHT)
common["mc5_model"] = mc_model
common["mc5_succ"] = succ_by_cls
common["mc5_weight"] = MC_WEIGHT
common["ingame_model"] = ingame_model
common["ingame_weight"] = ING_WEIGHT'''
assert old_common in src
src = src.replace(old_common, new_common, 1)
src = src.replace('log(f"weights: mc5={MC_WEIGHT:.3f} sum={s:.6f}")',
                  'log(f"weights: mc5={MC_WEIGHT:.3f} ingame={ING_WEIGHT:.3f} sum={s:.6f}")')
src = src.replace('model_artifacts_v78.pkl', 'model_artifacts_v79.pkl')

open('train_final_v79_ingame.py', 'w', encoding='utf-8').write(src)
print("train_final_v79_ingame.py 생성")
