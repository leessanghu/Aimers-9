"""phase83 — Hurdle 인수분해 검증 (Codex 1순위).

core_fail = reverse OR middle (복원라벨). 확인됨: core_fail=1 -> success=1 0건 (완전
결정론적, 카테고리 배타성에서 나오는 항등식). 정보량 관점(오늘 죽인 등급타겟/의도축)과
다른 질문: 같은 정보를 2단계 분류기로 인수분해하면 단일모델과 다른 오차구조가 나오나.

    p_hurdle = (1 - p_core_fail) * p_success_given_no_core

리스크: p_hurdle이 v29 예측과 상관 0.95+ 면 앙상블 가치 없음 -> 이것부터 싸게 확인.
피처는 v28 162개 그대로(카테고리 라벨 자체는 피처로 안 씀, 타겟만 재구성 -> 누수 없음).
학습은 속도를 위해 HGB만(빠른 스크리닝), 살아남으면 CatBoost까지 확장.
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
VALID_SEASON = 2024
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

log("core_fail 타겟 복원 (reverse OR middle, 누적차분)...")
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


S_, R_, M_ = [cnt(c) for c in ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                                "asof_pitcher_middle_rate"]]
order = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[order]
n_o = n_[order]
step = np.zeros(len(df), dtype=bool)
step_body = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
step[order[:-1]] = step_body
r_diff = np.zeros(len(df)); m_diff = np.zeros(len(df))
r_diff[order[:-1]] = np.diff(R_[order])
m_diff[order[:-1]] = np.diff(M_[order])
core_fail = np.where(step, ((r_diff > 0) | (m_diff > 0)).astype(np.float64), np.nan)
log(f"  복원 {step.sum():,}행 ({100*step.mean():.2f}%)  core_fail 비율={np.nanmean(core_fail):.4f}")
# 결정론 검증
chk = step & (core_fail == 1)
assert (y[chk] == 0).all(), "core_fail=1인데 success=1이 있음 -> 라벨 복원 버그"
log("  결정론 검증 통과 (core_fail=1 -> success=0 100%)")

log("피처 재구성 (v28 162개)...")
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

seasons = df["season"].to_numpy(np.float64)
tr_m = (seasons <= VALID_SEASON - 1) & step
va_m = seasons == VALID_SEASON
yv = y[va_m].astype(np.float64)
r = yv.mean()
BSREF = r * (1 - r)


def score(p):
    return 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)


w_tr = recency_weight(seasons[tr_m], half_life=2.0)
log(f"train(복원성공)={tr_m.sum():,}  valid={va_m.sum():,}")

HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)

log("[1] core_fail 분류기 학습...")
m1 = HistGradientBoostingClassifier(**HGB).fit(X.loc[tr_m], core_fail[tr_m], sample_weight=w_tr)
p_core = m1.predict_proba(X.loc[va_m])[:, 1]
log(f"  core_fail 예측 완료  iters={m1.n_iter_}")

log("[2] success|no_core 분류기 학습 (core_fail=0 행만)...")
nc_m = tr_m & (core_fail == 0)
m2 = HistGradientBoostingClassifier(**HGB).fit(X.loc[nc_m], y[nc_m], sample_weight=w_tr[core_fail[tr_m] == 0])
p_succ_nc = m2.predict_proba(X.loc[va_m])[:, 1]
log(f"  success|no_core 예측 완료  iters={m2.n_iter_}  (학습행 {nc_m.sum():,})")

p_hurdle = (1 - p_core) * p_succ_nc
os.makedirs("phase80_cache", exist_ok=True)
np.save("phase80_cache/hurdle_p_core.npy", p_core)
np.save("phase80_cache/hurdle_p_succ_nc.npy", p_succ_nc)
np.save("phase80_cache/hurdle_p_hurdle.npy", p_hurdle)

log("[3] v29 직행 HGB(비교기준, 동일 파라미터/데이터)...")
m3 = HistGradientBoostingClassifier(**HGB).fit(X.loc[tr_m], y[tr_m], sample_weight=w_tr)
p_direct = m3.predict_proba(X.loc[va_m])[:, 1]
np.save("phase80_cache/hurdle_p_direct.npy", p_direct)

print()
print("=" * 56)
print(f"{'':<30}{'score':>10}")
print("-" * 56)
print(f"{'p_direct (기존 방식 동치)':<30}{score(p_direct):10.2f}")
print(f"{'p_hurdle (2단계 인수분해)':<30}{score(p_hurdle):10.2f}")
print(f"{'상관(p_hurdle, p_direct)':<30}{np.corrcoef(p_hurdle,p_direct)[0,1]:10.4f}")
print()
for w in [0.3, 0.5, 0.7]:
    blend = (1 - w) * p_direct + w * p_hurdle
    print(f"blend(direct={1-w:.1f}, hurdle={w:.1f})       {score(blend):10.2f}")

# v29 로컬 재구성 (phase80 캐시, 누수 없음 -- 프로덕션 v29 아티팩트는 전체데이터
# 학습이라 2024 폴드 평가에 쓰면 누수이므로 절대 사용하지 않는다)
v29 = None
try:
    CACHE_DIR = "phase80_cache"
    weights = {"hgb_d6": 9, "hgb_sub": 9, "cat_d6": 8, "hgb_d8": 8, "cat_d8": 5, "cat_rsm": 1}
    tot = sum(weights.values())
    v29 = np.zeros(len(yv))
    for k, wgt in weights.items():
        v29 += (wgt / tot) * np.load(f"{CACHE_DIR}/{k}.npy")
except Exception as e:
    log(f"  v29 로컬 재구성 스킵: {e}")
if v29 is not None:
    print()
    print(f"{'v29 로컬재구성(6개, 누수없음)':<30}{score(v29):10.2f}")
    print(f"{'상관(p_hurdle, v29로컬)':<30}{np.corrcoef(p_hurdle,v29)[0,1]:10.4f}")
    for w in [0.1, 0.15, 0.2, 0.3, 0.5]:
        blend = (1 - w) * v29 + w * p_hurdle
        print(f"v29+hurdle 블렌드(hurdle={w:.2f})        {score(blend):10.2f}  ({score(blend)-score(v29):+.2f})")

log(f"총 {time.time()-t0:.0f}s")
