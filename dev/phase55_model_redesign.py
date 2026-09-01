"""모델 설계 재검토 — v15 피처셋 고정, 세 가지를 한번에 검증한다.

1. team_te: shuffled KFold(seed 의존) vs expanding(시간순, seed 무관) — 여러 seed에 걸친
   점수 분산이 줄어드는지 확인. features.py에 이미 "시간 구조와 안 맞을 수 있음" 주석이
   있던 부분을 실제로 고쳐서 비교한다.
2. HGB max_leaf_nodes(현재 31 = 사실상 depth 5 수준) 스윕 — 깊이가 seed 분산을 늘리는지.
3. 고정 0.5:0.5 블렌딩 vs OOF 로지스틱 회귀 스태킹.

baseline = v15 정확 재현(2023->2024 폴드, seed=42, shuffled_kfold, HGB max_leaf_nodes=31).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
SEEDS = [42, 7, 10, 100]

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
gr = build_global_rates(df)
dly = transform_lastyear(df, build_lastyear_table(df), gr, sr, k=30.0)
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
yva_check = df.loc[va, "control_success"].to_numpy()
print(f"베이스 피처(공통, seed 무관) 준비 완료 ({time.time()-t0:.0f}s)", flush=True)


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


def run_once(seed, team_te_mode, hgb_leaves=31, hgb_depth=6, cat_depth=6):
    fold = build_fold(df, 2023, 2024, extra_features=None, seed=seed, include_team_te=True,
                      team_te_mode=team_te_mode)
    ytr, yva = fold["y_train"], fold["y_valid"]
    Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
    ti, ei = time_split_es(len(Xtr))

    h = HistGradientBoostingClassifier(max_depth=hgb_depth, max_leaf_nodes=hgb_leaves, max_iter=500,
                                       learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=20,
                                       random_state=seed).fit(Xtr, ytr)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=cat_depth, l2_leaf_reg=5.0,
                            random_seed=seed, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))

    p_h = h.predict_proba(Xva)[:, 1]
    p_c = cb.predict_proba(Xva)[:, 1]
    p_blend = 0.5 * p_h + 0.5 * p_c
    s_blend = max(0, evaluate(yva, p_blend)["bss"] * 1e5)
    s_h = max(0, evaluate(yva, p_h)["bss"] * 1e5)
    s_c = max(0, evaluate(yva, p_c)["bss"] * 1e5)

    # 스태킹용: train fold 내부를 다시 시간분할해서 OOF 스타일 검증셋 하나 더 만든다
    # (Xtr의 마지막 10%를 held-out으로 써서 HGB/CatBoost 예측을 메타러너 입력으로)
    p_h_tr_ho = h.predict_proba(Xtr.iloc[ei])[:, 1]
    p_c_tr_ho = cb.predict_proba(Xtr.iloc[ei])[:, 1]
    meta = LogisticRegression().fit(np.column_stack([p_h_tr_ho, p_c_tr_ho]), ytr[ei])
    p_stack = meta.predict_proba(np.column_stack([p_h, p_c]))[:, 1]
    s_stack = max(0, evaluate(yva, p_stack)["bss"] * 1e5)
    stack_w = meta.coef_[0]

    return dict(seed=seed, team_te_mode=team_te_mode, hgb_leaves=hgb_leaves, hgb_depth=hgb_depth,
                cat_depth=cat_depth, s_hgb=s_h, s_cat=s_c, s_blend=s_blend, s_stack=s_stack,
                stack_w_hgb=stack_w[0], stack_w_cat=stack_w[1])


print("\n=== 1) team_te 모드별 seed 분산 비교 (HGB leaves=31, CatBoost depth=6, 기본 v15 설정) ===", flush=True)
results = []
for mode in ["shuffled_kfold", "expanding"]:
    for seed in SEEDS:
        r = run_once(seed, mode)
        results.append(r)
        print(f"  mode={mode:15s} seed={seed:4d}  hgb={r['s_hgb']:.1f}  cat={r['s_cat']:.1f}  "
              f"blend={r['s_blend']:.1f}  stack={r['s_stack']:.1f} (w_hgb={r['stack_w_hgb']:+.2f} w_cat={r['stack_w_cat']:+.2f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

rdf = pd.DataFrame(results)
print("\n--- 요약: mode별 blend 점수 평균/표준편차 (4 seed) ---", flush=True)
summ = rdf.groupby("team_te_mode")["s_blend"].agg(["mean", "std", "min", "max"])
print(summ, flush=True)
print("\n--- 요약: mode별 stack 점수 평균/표준편차 (4 seed) ---", flush=True)
summ2 = rdf.groupby("team_te_mode")["s_stack"].agg(["mean", "std", "min", "max"])
print(summ2, flush=True)

best_mode = summ["std"].idxmin()
print(f"\n분산이 더 작은 team_te 모드: {best_mode}", flush=True)

print(f"\n=== 2) HGB leaves/depth 스윕 (team_te_mode={best_mode}, seed 3개로 축소) ===", flush=True)
sweep_results = []
SWEEP_SEEDS = SEEDS[:3]
for hgb_leaves, cat_depth in [(31, 6), (63, 6), (31, 8)]:
    for seed in SWEEP_SEEDS:
        r = run_once(seed, best_mode, hgb_leaves=hgb_leaves, cat_depth=cat_depth)
        sweep_results.append(r)
        print(f"  leaves={hgb_leaves:4d} cat_depth={cat_depth}  seed={seed:4d}  blend={r['s_blend']:.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

sdf = pd.DataFrame(sweep_results)
print("\n--- 요약: (leaves,cat_depth)별 blend 평균/표준편차 ---", flush=True)
print(sdf.groupby(["hgb_leaves", "cat_depth"])["s_blend"].agg(["mean", "std", "min", "max"]), flush=True)

print(f"\n총 {time.time()-t0:.0f}s", flush=True)
