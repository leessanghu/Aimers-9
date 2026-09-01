"""v62 = v50 + cond_ball축(위험실투 아닌 조건에서만 1-ball). idea54 fold A 로컬Δ=-1.65
(음수)였으나 사용자 지시로 실측 직행. not-dangerous(not middle/reverse) 행에서만
유효, 나머지는 NaN.
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
MC_WEIGHT = 0.15


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
v66 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v66.pkl"))
log(f"  hgbs={len(v66['hgbs'])} cats={len(v66['cats'])}")

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
X = X[v66["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("투구단위 라벨 복원 (reverse/middle/ball) + 조건부 ball head 구성...")
pid = df["pitcher_id"].to_numpy()
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(df), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(df))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(df))
    lab[order] = d
    return lab


lab_reverse = diff_label("asof_pitcher_reverse_rate")
lab_middle = diff_label("asof_pitcher_middle_rate")
lab_ball = diff_label("asof_pitcher_ball_rate")
lab_strike = diff_label("asof_pitcher_strike_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle) | np.isnan(lab_ball) | np.isnan(lab_strike))
# 15-class = 5실패유형 x 3구종.
# 구종은 asof_pitcher_pitchmix_n과 fastball/breaking/offspeed_rate의 차분으로 100% 복원된다
# (증분합=+1이 정확히 100.0%, 커버리지 99.9%). 지금까지 한 번도 라벨로 쓰지 않은 신규 정보원.
# nd&ball에서 FB 63.82/BK 51.77/OS 57.91%(12.1%p), nd&strike에서 96.80/88.11/91.31%(8.7%p)로
# 구종별 격차가 크다. 구종 단독 Resolution 상한 0.000595 = 현재 Var(p)의 21.9%.
NCLS = 15
n_mix = df["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)


def diff_mix(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_mix)
    c_ord = c[order]
    d = np.empty(len(df))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(df))
    lab[order] = d
    return lab


d_fb = diff_mix("asof_pitcher_fastball_rate")
d_bk = diff_mix("asof_pitcher_breaking_rate")
d_os = diff_mix("asof_pitcher_offspeed_rate")
pt = np.full(len(df), -1, dtype=np.int64)
pt[d_fb == 1] = 0
pt[d_bk == 1] = 1
pt[d_os == 1] = 2
log(f"  구종 복원: FB={np.mean(pt==0)*100:.2f}% BK={np.mean(pt==1)*100:.2f}% OS={np.mean(pt==2)*100:.2f}% 미확정={np.mean(pt<0)*100:.2f}%")

cls5 = np.full(len(df), -1, dtype=np.int64)
cls5[valid_lab & (lab_middle > 0.5)] = 0
cls5[valid_lab & (lab_reverse > 0.5) & (lab_middle < 0.5)] = 1
nd = valid_lab & (lab_middle < 0.5) & (lab_reverse < 0.5)
cls5[nd & (lab_ball > 0.5)] = 2
cls5[nd & (lab_ball < 0.5) & (lab_strike > 0.5)] = 3
cls5[nd & (lab_ball < 0.5) & (lab_strike < 0.5)] = 4

cls = np.where((cls5 >= 0) & (pt >= 0), cls5 * 3 + pt, -1)
fit_mask = cls >= 0
succ_by_cls = np.array([y[fit_mask & (cls == c)].mean() if (fit_mask & (cls == c)).sum() > 0 else 0.0
                        for c in range(NCLS)])
log(f"  labels ok {fit_mask.sum():,}/{len(df):,}  E[y|c]={np.round(succ_by_cls,5)}")
CN = ["middle", "reverse", "nd&ball", "nd&strike", "nd&other"]
PN = ["FB", "BK", "OS"]
for c in range(NCLS):
    log(f"    class{c:2d} {CN[c//3]:10s}x{PN[c%3]}: n={(cls==c).sum():,} ({(cls==c).mean()*100:.2f}%) succ={succ_by_cls[c]*100:.3f}%")

log("5-class softmax CatBoost training (full data, labeled rows only)...")
fit_idx = np.where(fit_mask)[0]
n_es = int(len(fit_idx) * 0.92)
tr_i, es_i = fit_idx[:n_es], fit_idx[n_es:]
CAT_PARAMS = dict(iterations=1000, learning_rate=0.05, depth=5, l2_leaf_reg=5.0, verbose=100,
                  bootstrap_type="Bernoulli", subsample=0.5,
                  random_seed=42, loss_function="MultiClass", classes_count=NCLS,
                  early_stopping_rounds=40)
ts = time.time()
mc_model = CatBoostClassifier(**CAT_PARAMS)
mc_model.fit(X.iloc[tr_i], cls[tr_i], sample_weight=w[tr_i],
             eval_set=(X.iloc[es_i], cls[es_i]))
log(f"  done best_iter={mc_model.best_iteration_} ({time.time()-ts:.0f}s)")
strip_rng(mc_model)

common = dict(v66)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - MC_WEIGHT)
common["mc5_model"] = mc_model
common["mc5_succ"] = succ_by_cls
common["mc5_weight"] = MC_WEIGHT
s = sum(float(v) for k, v in common.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
log(f"weights: mc5={MC_WEIGHT:.3f} sum={s:.6f}")

out = os.path.join(OUT_DIR, "model_artifacts_v78.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
