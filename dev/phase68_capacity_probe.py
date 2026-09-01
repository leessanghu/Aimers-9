"""모델 capacity가 병목인지 직접 테스트 — CatBoost depth/규제를 확 풀어서 점수가 오르는지 본다.

논리: 지금 CatBoost는 depth=6, min_data_in_leaf=200으로 상당히 보수적이다. 만약 '모델이
너무 단순해서' 132개 feature의 정보를 다 못 쓰고 있는 거라면, depth를 늘리고 규제를
풀었을 때 (같은 feature, 같은 데이터) 점수가 뚜렷이 올라야 한다.
반대로 안 오르거나 오히려 떨어지면(과적합), 지금 모델은 이미 이 feature들이 주는 정보를
포화 상태로 뽑아내고 있다는 뜻이고 -> 병목은 feature 정보량이지 모델 표현력이 아니다.

phase67과 동일 폴드/feature(132개, train<=2023->valid=2024)를 그대로 재사용한다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from count_split import build_count_table, transform_count, K_COUNT
from crosses import add_crosses
from career_volatility import build_volatility_table, transform_volatility, K_VOL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import build_trackman_profile, transform_trackman
import os

SEED = 42
TRAIN_MAX, VALID_SEASON = 2023, 2024
HALF_LIFE = 2.0
TM_CACHE = "phase64_trackman_profile.parquet"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


log("피처 구성 (v26과 동일 132개)...")
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
dcnt = transform_count(df, build_count_table(df), pp, sr, k=K_COUNT)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
dvol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL)
role_tbl = build_role_table(df)
drole = transform_role(df, role_tbl, sr)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
dform = transform_form(df, drole, dins["inseason_success_smooth"].to_numpy(np.float64), base_middle)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
dtm = transform_trackman(df, prof, sr)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]


def stack(i, base_frame):
    X = pd.concat([base_frame.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    parts = [X, add_crosses(X), dly.loc[i].reset_index(drop=True), dcnt.loc[i].reset_index(drop=True),
             dvol.loc[i].reset_index(drop=True), drole.loc[i].reset_index(drop=True),
             dform.loc[i].reset_index(drop=True), dtm.loc[i].reset_index(drop=True)]
    return pd.concat(parts, axis=1)


tr_i = df[df.season <= TRAIN_MAX].index
va_i = df[df.season == VALID_SEASON].index
fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED,
                  include_team_te=True, team_te_mode="expanding")
y_tr, y_va = fold["y_train"], fold["y_valid"]
X_tr = stack(tr_i, fold["X_train"])
X_va = stack(va_i, fold["X_valid"])
log(f"피처 {X_tr.shape[1]}개  train={len(X_tr):,}  valid={len(X_va):,}")


def recency_weight(seasons, half_life=HALF_LIFE):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


w_tr = recency_weight(df.loc[tr_i, "season"].to_numpy(np.float64))
w_tr = w_tr / w_tr.mean()
ti, ei = time_split_es(len(X_tr))


def run(tag, **params):
    t1 = time.time()
    cb = CatBoostClassifier(iterations=params.pop("iterations", 3000), random_seed=SEED, verbose=0,
                            early_stopping_rounds=50, loss_function="Logloss", **params)
    cb.fit(X_tr.iloc[ti], y_tr[ti], sample_weight=w_tr[ti], eval_set=(X_tr.iloc[ei], y_tr[ei]))
    p = cb.predict_proba(X_va)[:, 1]
    sc = max(0, evaluate(y_va, p)["bss"] * 1e5)
    n_leaves = getattr(cb, "tree_count_", None)
    log(f"[{tag}] score={sc:8.1f}  best_iter={cb.best_iteration_}  trees={n_leaves}  ({time.time()-t1:.0f}s)")
    return sc


log("\n=== capacity sweep (CatBoost 단독, HGB 없이 순수 트리 용량만 비교) ===")
results = {}
results["baseline(depth6,minleaf200,L2=5)"] = run("baseline(depth6,minleaf200,L2=5)",
    depth=6, min_data_in_leaf=200, l2_leaf_reg=5.0, learning_rate=0.03)
results["depth8,minleaf50,L2=3"] = run("depth8,minleaf50,L2=3",
    depth=8, min_data_in_leaf=50, l2_leaf_reg=3.0, learning_rate=0.03)
results["depth10,minleaf20,L2=1"] = run("depth10,minleaf20,L2=1",
    depth=10, min_data_in_leaf=20, l2_leaf_reg=1.0, learning_rate=0.02)
results["depth6,minleaf50,L2=3(약한완화)"] = run("depth6,minleaf50,L2=3(약한완화)",
    depth=6, min_data_in_leaf=50, l2_leaf_reg=3.0, learning_rate=0.03)
results["depth8,minleaf200,L2=5(depth만)"] = run("depth8,minleaf200,L2=5(depth만)",
    depth=8, min_data_in_leaf=200, l2_leaf_reg=5.0, learning_rate=0.03)

log("\n" + "=" * 70)
log("요약")
log("=" * 70)
base = results["baseline(depth6,minleaf200,L2=5)"]
for k, v in results.items():
    log(f"  {k:<38} {v:8.1f}   (기준 대비 {v-base:+.1f})")

log("\n읽는 법: capacity를 확 풀었는데도 점수가 안 오르거나 떨어지면(과적합) -> 병목은 모델이 아니라 feature 정보량.")
log(f"\n총 {time.time()-t0:.0f}s")
