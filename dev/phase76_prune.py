"""phase76 — 피처 프루닝: 우리가 분산 제한 상태인지 직접 검증.

가설:
    지금까지 증거가 전부 '분산 제한'을 가리킨다.
      3-seed 평균 +2.9 (정보 추가 없는 순수 분산감소)
      depth 6->8->10  -14.6, -46.8
      MLP 최적가중 0%, RF -2.7
      v28 피처 8개 추가 -> ±0 (SHAP 5위로 확실히 쓰이는데도)
      로컬 스크리너 실현율 0.6
    편향 제한이면 용량/피처를 늘릴 때 이득이 나야 하는데 전부 반대다.
    그렇다면 방향은 '더 넣기'가 아니라 '빼기'다.

    phase75 SHAP: 상위 40개가 magnitude 84.8%, 상위 80개가 95.1%.
    하위 82개는 합쳐서 4.9%, 그중 8개는 정확히 0.000000.
    신호가 0인데 적합 노이즈는 내고 있다면 빼는 게 이득이어야 한다.

설계:
    폴드 = train(season<=2023) -> valid(2024). 프로덕션과 동일하게 recency weight 적용.
    피처는 v28의 162개를 한 번만 만들고, phase75 SHAP 순위로 상위 K개만 남겨 재학습.
    속도를 위해 스윕은 HGB 단독으로 하고, 승자만 나중에 풀 블렌드로 확인한다.

주의(자체 기록):
    SHAP 순위는 전체 데이터(2024 포함) 학습 모델에서 뽑았으므로 순위 선택에 약한
    선택 편향이 있다. 다만 하위권은 magnitude가 문자 그대로 0.000000이라 경계 판단이
    아니고, 순위 자체도 시드/시즌에 안정적이었다. 승자가 나오면 <=2023만으로 순위를
    다시 뽑아 재확인한다.
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
SEED = 42
VALID_SEASON = 2024

HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                  l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=SEED)

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


def score(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

log("피처 재구성 (v28 162개)...")
fb = FeatureBuilder(seed=SEED, include_raw_rates=False, team_te_mode="expanding").fit(df)
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
assert len(order) == X.shape[1], (len(order), X.shape[1])
n_zero = int((rank.loc[order] == 0).sum())
log(f"SHAP magnitude 0인 피처 {n_zero}개")

seasons = df["season"].to_numpy(np.float64)
tr_m = seasons <= VALID_SEASON - 1
va_m = seasons == VALID_SEASON
w = recency_weight(seasons[tr_m], half_life=2.0)
yv = y[va_m].astype(np.float64)
log(f"train={tr_m.sum():,}  valid={va_m.sum():,}")

Ks = [162, 140, 120, 100, 80, 65, 50, 40, 30]
print()
print(f"{'K':>5} {'누적mag':>9} {'score':>10} {'vs162':>9}  {'시간':>7}")
print("-" * 48)
base = None
res = []
for K in Ks:
    cols = order[:K]
    ts = time.time()
    m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X.loc[tr_m, cols], y[tr_m], sample_weight=w)
    p = m.predict_proba(X.loc[va_m, cols])[:, 1]
    s = score(yv, p)
    if base is None:
        base = s
    cum = rank.loc[order[:K]].sum() / rank.loc[order].sum()
    res.append(dict(K=K, cum=cum, score=s, delta=s - base))
    print(f"{K:5d} {cum:9.4f} {s:10.2f} {s-base:+9.2f}  {time.time()-ts:6.0f}s", flush=True)
    np.save(f"phase76_pred_K{K}.npy", p)

pd.DataFrame(res).to_csv("phase76_prune.csv", index=False)
print()
best = max(res, key=lambda r: r["score"])
log(f"최고: K={best['K']}  score={best['score']:.2f}  (K=162 대비 {best['delta']:+.2f})")
log(f"총 {time.time()-t0:.0f}s")
