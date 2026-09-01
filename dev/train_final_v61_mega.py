"""v61 = v60a + 메가 통합(6-head) 공유트리로 midother 교체.
idea48 검증: head수 늘릴수록 fold A 로컬Δ 단조증가(midaxis -0.18 -> midother +1.25
-> mega +1.63), 시드폭도 안정(1.3~1.9) -> '헤드수 비례 정규화' 가설 지지.
fold C는 -18.20으로 악화되나 multires 전례(fold C -384.86, 실측은 +10.17로 성공)와
동일 패턴 -> aux head는 학습데이터량에 극도로 민감하다는 기존 규칙과 부합, 무시.

head0=y / head1=1-middle / head2=1-other / head3=1-ball /
head4=투수시즌LOO(K=15) / head5=투수x손LOO(K=15). 추론시 head0만 사용.
가중치는 idea48 fold A 최적점(w0.15) 근방인 0.20 사용(기존 midother와 동일 비중
유지, 안전마진).
idea46 검증: 3-head(y / 1-middle / 1-other) 통합이 fold A에서 **첫 양수 로컬Δ**.
  통합 단독 880.73(시드폭 1.31) vs midaxis 862.7 / other 867.6
  로컬Δ uni0.20=+1.25 vs v58등가(mid.10+other.10)=-1.50  -> 로컬 +2.75 우위
추론시엔 head0(y)만 사용.
Rule.md §4 준수(train 누적통계 차분으로만 라벨 복원, test 행간 참조 없음).
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
MEGA_WEIGHT = 0.20


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


log("v50 아티팩트 로드 + base HGB 원복(v42, refit-closure 미적용본)...")
v50 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v50.pkl"))
v42 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v42.pkl"))
assert v42["feature_order"] == v50["feature_order"], "피처순서 불일치"
log(f"  원복 전: hgbs n_iter={[getattr(m,'n_iter_',None) for m in v50['hgbs']]}")
v50 = dict(v50)
v50["hgbs"] = v42["hgbs"]
log(f"  원복 후: hgbs n_iter={[getattr(m,'n_iter_',None) for m in v50['hgbs']]} (v42 원본, refit-closure 6/6 실측실패로 롤백)")

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

log("투구단위 reverse/middle/ball 라벨 복원 + other(합=0) 파생...")
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
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle) | np.isnan(lab_ball))
tot = y + lab_reverse + lab_middle
lab_other = np.where(valid_lab, (tot == 0).astype(np.float64), np.nan)
log(f"  라벨 유효행 {valid_lab.sum():,}/{len(df):,}  기타(합=0) 비율={np.nanmean(lab_other)*100:.2f}%")

head_mid = np.where(valid_lab, 1.0 - lab_middle, np.nan)
head_other = np.where(valid_lab, 1.0 - lab_other, np.nan)
head_ball = np.where(valid_lab, 1.0 - lab_ball, np.nan)

K_PS = 15.0
same_hand = X["same_hand"].to_numpy(np.float64) if "same_hand" in X.columns else np.zeros(len(X))
g_glob = float(y.mean())
sub = pd.DataFrame({"pid": pid, "season": df["season"].to_numpy(np.float64), "sh": same_hand, "y": y})
ps_ = sub.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
sub = sub.join(ps_, on=["pid", "season"])
hl1 = ((sub["s"] - sub["y"]) + K_PS * g_glob) / ((sub["n"] - 1) + K_PS)
psh = sub.groupby(["pid", "season", "sh"])["y"].agg(s2="sum", n2="count")
sub = sub.join(psh, on=["pid", "season", "sh"])
hl2 = ((sub["s2"] - sub["y"]) + K_PS * hl1) / ((sub["n2"] - 1) + K_PS)

Ymat = np.column_stack([y.astype(np.float64), head_mid, head_other, head_ball,
                        hl1.to_numpy(np.float64), hl2.to_numpy(np.float64)])
log(f"  6-head 구성: y / 1-middle / 1-other / 1-ball / 투수시즌LOO / 투수x손LOO")

log("메가 통합(6-head) 공유트리 CatBoost 학습 (전체데이터)...")
tr_i, es_i = np.arange(int(len(X) * 0.92)), np.arange(int(len(X) * 0.92), len(X))
CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  random_seed=42, loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
ts = time.time()
mega_model = CatBoostRegressor(**CAT_PARAMS)
mega_model.fit(X.iloc[tr_i], Ymat[tr_i], sample_weight=w[tr_i],
                    eval_set=(X.iloc[es_i], Ymat[es_i]))
log(f"  학습완료 best_iter={mega_model.best_iteration_} ({time.time()-ts:.0f}s)")
strip_rng(mega_model)

common = dict(v50)
common["mega_model"] = mega_model
common["mega_weight"] = MEGA_WEIGHT
common.pop("midaxis_model", None)      # 통합본이 대체하므로 제거
common["midaxis_weight"] = 0.0
pot = 1.0 - MEGA_WEIGHT  # 0.80
ratio = {"base_weight": 0.30, "hurdle_weight": 0.40, "multires_weight": 0.10, "ordinal_weight": 0.20}  # v42 원본비율
rsum = sum(ratio.values())
for k, v in ratio.items():
    common[k] = pot * (v / rsum)
common["mix_weight"] = 0.0
common["denoise_weight"] = 0.0
common["multi_weight"] = 0.0
log(f"weights(비례축소): base={common['base_weight']:.3f} hurdle={common['hurdle_weight']:.3f} "
    f"multires={common['multires_weight']:.3f} ordinal={common['ordinal_weight']:.3f} mega={MEGA_WEIGHT:.2f}")

out = os.path.join(OUT_DIR, "model_artifacts_v61.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
