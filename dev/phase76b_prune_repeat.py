"""phase76b — phase76 결과 신뢰도 검증 (재학습 노이즈 vs 진짜 신호).

phase76 결과가 단조롭지 않고 요동쳤다 (K=100 +15.71, 바로 옆 K=80 -37.88).
편차 폭이 phase64에서 측정한 재학습-비교 노이즈 SD(7~24점)와 같은 범위라
K=100이 진짜 최적인지 단일 실행 노이즈인지 구분이 안 된다.

세 지점(162/100/65)만 골라 random_state 5개로 반복해 노이즈 바닥을 직접 재고,
관측된 K간 차이가 그 SD보다 충분히 큰지(>=2시그마)로 판정한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import batter_split as bsplit
from batterform import K_BATTER, build_batter_table, transform_batter
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from count_split import K_COUNT, build_count_table, transform_count
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from inseason_full import build_global_priors, build_season_end_table_full, transform_inseason_full
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
SHAP_CSV = "phase75_v28_shap_magnitude.csv"
VALID_SEASON = 2024
SEEDS = [42, 7, 2024, 123, 999]
Ks = [162, 100, 65]

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


def score(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


log("데이터 로드 + 피처 재구성 (v28 162개, phase76과 동일)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

fb = FeatureBuilder(seed=42, include_raw_rates=False, team_te_mode="expanding").fit(df)
X_base = fb.transform_train_oof(df).reset_index(drop=True)
se = build_season_end_table(df)
X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
X_plt = transform_platoon(df, build_platoon_table(df), prior, sr, k=K_PLATOON).reset_index(drop=True)
it, io = build_inning_table(df), build_inning_offset(df)
X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
X_cnt = transform_count(df, build_count_table(df), prior, sr, k=K_COUNT).reset_index(drop=True)
X_pt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), prior, g, sr).reset_index(drop=True)
X_ly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0).reset_index(drop=True)
X_vol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL).reset_index(drop=True)
role_tbl = build_role_table(df)
X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                        base_middle).reset_index(drop=True)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)
lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)
X_bat = transform_batter(df, build_batter_table(df), sr, g, k=K_BATTER).reset_index(drop=True)
n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
X_isf = transform_inseason_full(df, build_season_end_table_full(df), build_global_priors(df), sr,
                                n_end_row, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
X_bmid = bsplit.transform_batter_middle(df, bsplit.build_batter_middle_table(df), sr, g_bmid).reset_index(drop=True)
bmarg = bsplit.build_batter_marginal(df)
b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
X_bplat = bsplit.transform_bplatoon(df, bsplit.build_bplatoon_table(df), b_prior, sr,
                                    k=bsplit.K_BPLATOON).reset_index(drop=True)

X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
C = add_crosses(X)
X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
               X_isf, X_bmid, X_bplat], axis=1).astype(np.float64)
log(f"피처 {X.shape[1]}개")

rank = pd.read_csv(SHAP_CSV, index_col=0)["magnitude"].sort_values(ascending=False)
order = [c for c in rank.index if c in X.columns]

seasons = df["season"].to_numpy(np.float64)
tr_m = seasons <= VALID_SEASON - 1
va_m = seasons == VALID_SEASON
yv = y[va_m].astype(np.float64)
w = recency_weight(seasons[tr_m], half_life=2.0)
log(f"train={tr_m.sum():,}  valid={va_m.sum():,}  seeds={SEEDS}")

rows = []
for K in Ks:
    cols = order[:K]
    scores = []
    for s in SEEDS:
        ts = time.time()
        params = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                     l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                     n_iter_no_change=20, random_state=s)
        m = HistGradientBoostingClassifier(**params).fit(X.loc[tr_m, cols], y[tr_m], sample_weight=w)
        p = m.predict_proba(X.loc[va_m, cols])[:, 1]
        sc = score(yv, p)
        scores.append(sc)
        log(f"  K={K} seed={s}  score={sc:.2f}  ({time.time()-ts:.0f}s)")
    scores = np.array(scores)
    rows.append(dict(K=K, mean=scores.mean(), sd=scores.std(ddof=1), scores=list(scores)))

t = pd.DataFrame(rows)
print()
print("=" * 60)
print(f"{'K':>5} {'mean':>10} {'sd':>8}   scores")
print("-" * 60)
for _, r in t.iterrows():
    print(f"{r.K:5d} {r['mean']:10.2f} {r.sd:8.2f}   {[round(x,1) for x in r.scores]}")

base_mean, base_sd = t.iloc[0]["mean"], t.iloc[0]["sd"]
print()
for _, r in t.iloc[1:].iterrows():
    diff = r["mean"] - base_mean
    pooled_sd = np.sqrt(base_sd**2 + r.sd**2) / np.sqrt(len(SEEDS))
    z = diff / pooled_sd if pooled_sd > 0 else float("nan")
    print(f"K={r.K} vs K=162: 차이={diff:+.2f}  풀링SD={pooled_sd:.2f}  z={z:+.1f}")

t.to_csv("phase76b_prune_repeat.csv", index=False)
log(f"총 {time.time()-t0:.0f}s")
