"""v53 = v50 + base CatBoost(cats, d6/d8/rsm) refit-closure.
v29 이후 cats 3개는 early_stopping_rounds=50 + eval_set(마지막 8%) 방식 그대로였음
-> 가장 최근(2025와 분포가 가장 가까운) 데이터 8%가 트리 분할에 한 번도 안 쓰임.
hgb(v44)/hurdle(v45)은 이미 refit-closure 완료됐지만 cats만 누락돼 있었다.
ES로 iteration 확정 후 early_stopping 끄고 전체데이터로 고정반복 재학습.
fold A +6.75 / fold C +24.50 (idea35_catrefit.py). "ES방식 변경류"라 세션 규칙상
편향 거의 없음(±0.7) -> fold A 델타를 거의 그대로 신뢰.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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

CAT_CONFIGS = [
    ("cat_d6", dict(depth=6, random_seed=42)),
    ("cat_d8", dict(depth=8, l2_leaf_reg=10.0, random_seed=7)),
    ("cat_rsm", dict(depth=6, rsm=0.6, random_seed=2024)),
]


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


log("v50 아티팩트 로드...")
v50 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v50.pkl"))
log(f"  hgbs={len(v50['hgbs'])} cats={len(v50['cats'])}")

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
X = X[v50["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)
tr_i, es_i = time_split_es(len(X))
log(f"ES홀드아웃 {len(es_i):,}행 (전체의 마지막 8%, 트리분할엔 refit단계에서 포함)")

log("CatBoost 3변종 refit-closure (ES로 iter 확정 -> 고정 후 전체데이터 재학습)...")
new_cats = []
for name, extra in CAT_CONFIGS:
    params = dict(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                 early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    params.update(extra)
    ts = time.time()
    m_es = CatBoostClassifier(**params)
    m_es.fit(X.iloc[tr_i], y[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
    best_iter = max(m_es.best_iteration_, 1)
    log(f"  [{name}] ES best_iter={best_iter} ({time.time()-ts:.0f}s)")

    ts = time.time()
    params_fixed = dict(params)
    params_fixed.pop("early_stopping_rounds")
    params_fixed["iterations"] = best_iter
    m_refit = CatBoostClassifier(**params_fixed)
    m_refit.fit(X, y, sample_weight=w)
    strip_rng(m_refit)
    new_cats.append(m_refit)
    log(f"  [{name}] 전체데이터 refit 완료 ({time.time()-ts:.0f}s)")

common = dict(v50)
common["cats"] = new_cats

out = os.path.join(OUT_DIR, "model_artifacts_v53.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
