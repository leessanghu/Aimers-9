"""v79 = v78(11-class 구종, 이미 학습됨) + ingame(경기내 컨디션) aux head 추가.

v78의 mc_model을 재학습 없이 그대로 재사용한다(train_final_v79_ingame.py의 실수 수정 --
그 스크립트는 v66부터 11-class를 처음부터 다시 학습해 2.5시간을 중복 낭비했다).
여기서는 ingame CatBoost 모델 하나만 새로 학습하고 v78 위에 얹는다.

경기 경계는 asof_pitcher_prev1_game_success_rate 갱신 시점으로 복원(45,121경기).
aux 타깃 = 현재 경기에서 직전 투구까지의 누적 성공률(자기 제외, K=8 축소).
10분위 성공률 43.6%~63.3%로 완전 단조, corr(y)=+0.1147, 커버리지 96.9%.
Rule 4 준수: train에서만 계산, test는 각 행 자기 피처로 모델이 추정.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

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
OUT_DIR = "../submit/model"
t0 = time.time()
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
ING_WEIGHT = 0.08


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


log("v78 아티팩트 로드 (mc_model 재사용)...")
v78 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v78.pkl"))
log(f"  hgbs={len(v78['hgbs'])} cats={len(v78['cats'])} mc5_weight={v78.get('mc5_weight')}")

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
X = X[v78["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("경기내 컨디션 aux 타깃 구성...")
pid = df["pitcher_id"].to_numpy()
order = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
p1g = df["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
p1o = p1g[order]
same_prev = np.zeros(len(df), dtype=bool)
same_prev[1:] = pid[order][1:] == pid[order][:-1]
chg = np.zeros(len(df), dtype=bool)
chg[1:] = (p1o[1:] != p1o[:-1]) & ~(np.isnan(p1o[1:]) & np.isnan(p1o[:-1]))
newgame = (~same_prev) | (chg & same_prev)
gid_o = np.cumsum(newgame)
y_o = y.astype(np.float64)[order]
cum = pd.Series(y_o).groupby(gid_o).cumsum().to_numpy() - y_o
kk = pd.Series(gid_o).groupby(gid_o).cumcount().to_numpy()
K_ING = 8.0
ing_o = np.where(kk > 0, (cum + K_ING * g) / (kk + K_ING), np.nan)
head_ingame = np.empty(len(df))
head_ingame[order] = ing_o
log(f"  경기수={gid_o.max():,}  커버리지={np.isfinite(head_ingame).mean()*100:.1f}%")

log("ingame aux head 학습 (y / 경기내 누적성공률 2-head 공유트리, 전체데이터)...")
Ying = np.column_stack([y.astype(np.float64), head_ingame])
ing_ok = np.isfinite(head_ingame)
ing_idx = np.where(ing_ok)[0]
n_es2 = int(len(ing_idx) * 0.92)
ti2, ei2 = ing_idx[:n_es2], ing_idx[n_es2:]
ts2 = time.time()
ingame_model = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                                 verbose=100, random_seed=42,
                                 loss_function="MultiRMSEWithMissingValues",
                                 early_stopping_rounds=50)
ingame_model.fit(X.iloc[ti2], Ying[ti2], sample_weight=w[ti2],
                 eval_set=(X.iloc[ei2], Ying[ei2]))
log(f"  done best_iter={ingame_model.best_iteration_} ({time.time()-ts2:.0f}s)")
strip_rng(ingame_model)

common = dict(v78)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - ING_WEIGHT)
common["ingame_model"] = ingame_model
common["ingame_weight"] = ING_WEIGHT
s = sum(float(v) for k, v in common.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
log(f"weights: ingame={ING_WEIGHT:.3f} sum={s:.6f}")

out = os.path.join(OUT_DIR, "model_artifacts_v79.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
