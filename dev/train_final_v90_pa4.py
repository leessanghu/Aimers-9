"""v90용 pa4 = PA-event 4-class 멤버 (continue-ball/continue-strike/2스트라이크파울/PA종료).

라벨은 같은 투수+같은 경기의 연속 행에서 카운트 전이로 100% 복원(codex idea74 수치와
소수점까지 일치: 34.48/32.79/6.87/25.85%, 성공률 42.63/59.36/58.20/54.97%).
오라클 증분 +0.000649로 구종(+0.000595)과 동급. 전이법칙상 정보축은 전이됨.

mc5 패턴 그대로: 4-class CatBoost -> P(event) @ E[y|event] 디코딩, head0만.
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
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
PA4_WEIGHT = 0.10


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


log("v89 아티팩트 로드...")
v89 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v89.pkl"))
log(f"  hgbs={len(v89['hgbs'])} cats={len(v89['cats'])} mc5_weight={v89.get('mc5_weight')}")

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
X = X[v89["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("PA-event 4-class 라벨 로드...")
ev = np.load("paevent_labels.npy")
fit_mask = ev >= 0
succ_by_ev = np.array([y[fit_mask & (ev == c)].mean() for c in range(4)])
log(f"  유효 {fit_mask.sum():,}/{len(df):,}  E[y|event]={np.round(succ_by_ev,5)}")
for c, nm in enumerate(["cont-ball", "cont-strike", "2s-foul", "PA-end"]):
    m = ev == c
    log(f"    {nm:12s} n={m.sum():,} ({m.mean()*100:.2f}%) succ={y[m].mean()*100:.3f}%")

log("PA-event 4-class CatBoost 학습 (전체데이터)...")
fit_idx = np.where(fit_mask)[0]
n_es = int(len(fit_idx) * 0.92)
ti, ei = fit_idx[:n_es], fit_idx[n_es:]


class ProgressCallback:
    def __init__(self, period=20):
        self.period = period
        self.t0 = time.time()
        self.best = None
        self.best_iter = 0

    def after_iteration(self, info):
        it = info.iteration
        loss = info.metrics["validation"]["MultiClass"][-1]
        if self.best is None or loss < self.best:
            self.best, self.best_iter = loss, it
        if it % self.period == 0 or it < 3:
            el = time.time() - self.t0
            log(f"    iter {it:4d}  loss={loss:.6f}  best={self.best:.6f}@{self.best_iter}  경과={el/60:.1f}분")
        return True


ts2 = time.time()
pa4_model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
                               verbose=False, random_seed=42, loss_function="MultiClass",
                               classes_count=4, early_stopping_rounds=40)
pa4_model.fit(X.iloc[ti], ev[ti], sample_weight=w[ti],
              eval_set=(X.iloc[ei], ev[ei]), callbacks=[ProgressCallback()])
log(f"  done best_iter={pa4_model.best_iteration_} ({time.time()-ts2:.0f}s)")
strip_rng(pa4_model)

common = dict(v89)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - PA4_WEIGHT)
common["pa4_model"] = pa4_model
common["pa4_succ"] = succ_by_ev
common["pa4_weight"] = PA4_WEIGHT
s = sum(float(v) for k, v in common.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
log(f"weights: pa4={PA4_WEIGHT:.3f} sum={s:.6f}")

out = os.path.join(OUT_DIR, "model_artifacts_v90.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
