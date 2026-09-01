"""v44 = v43b(d6만 시간순ES) + d8, sub도 시간순ES로 교체 (base3 전체 완료).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
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
from phase2_common import time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
ITER_GRID = [150, 200, 300]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("v43b 아티팩트 로드 (d6 이미 시간순ES 완료)...")
v43b = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v43b.pkl"))
log(f"  hgbs={len(v43b['hgbs'])} cats={len(v43b['cats'])}")

log("데이터 로드 + 피처 재구성...")
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
X = X[v43b["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)
tr_i, es_i = time_split_es(len(X))
log(f"시간순 검증셋: train={len(tr_i)}  es={len(es_i)}")

HGB_CONFIGS = [
    ("hgb_d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("hgb_sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]

new_extra_hgbs = []
for name, extra in HGB_CONFIGS:
    log(f"[{name}] iter 그리드 스윕 {ITER_GRID}...")
    best_score, best_iter = -1e18, ITER_GRID[0]
    for it in ITER_GRID:
        ts = time.time()
        params = dict(max_iter=it, learning_rate=0.03, l2_regularization=5.0, early_stopping=False)
        params.update(extra)
        m = HistGradientBoostingClassifier(**params).fit(X.iloc[tr_i], y[tr_i], sample_weight=w[tr_i])
        p_es = m.predict_proba(X.iloc[es_i])[:, 1]
        yv_es = y[es_i]
        r_es = yv_es.mean(); bs_es = r_es * (1 - r_es)
        s_es = 1e5 * (1 - np.mean((p_es - yv_es) ** 2) / bs_es)
        log(f"    iter={it}: 시간순검증={s_es:.2f} ({time.time()-ts:.0f}s)")
        if s_es > best_score:
            best_score, best_iter = s_es, it
    log(f"  [{name}] 최적 iter={best_iter} (검증점수={best_score:.2f})")

    ts = time.time()
    params_full = dict(max_iter=best_iter, learning_rate=0.03, l2_regularization=5.0, early_stopping=False)
    params_full.update(extra)
    m_full = HistGradientBoostingClassifier(**params_full).fit(X, y, sample_weight=w)
    strip_rng(m_full)
    new_extra_hgbs.append(m_full)
    log(f"  [{name}] 전체데이터 최종학습 완료 ({time.time()-ts:.0f}s)")

new_hgbs = [v43b["hgbs"][0], new_extra_hgbs[0], new_extra_hgbs[1]]  # d6(기존 시간순ES), d8, sub(신규)
common = dict(v43b)
common["hgbs"] = new_hgbs
out = os.path.join(OUT_DIR, "model_artifacts_v44.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
