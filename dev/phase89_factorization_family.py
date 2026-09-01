"""phase89 — 타겟 인수분해 계열 확장 (Hurdle 다양화 + 판정축 혼합분해).

근거: 최근 실측 성공 2건이 모두 '모델 구조' 변경이었다.
    v29 모델 다양화(HGB3+Cat3)  실측 +10.21
    v32 Hurdle 인수분해          실측 +14.98
반면 v28 이후 피처는 전패. 세분화는 수축산술(K=2486 -> 7.4%만 잔존)에 막히고,
재표현은 GBDT가 이미 아는 정보(cmd_index: SHAP 5위인데 증분 0)라 실패한다.
따라서 구조 축에 자원을 몰아준다.

이번에 검증하는 것 2가지:

[1] Hurdle 내부 다양화
    현재 core_fail / succ_nc 모델이 각각 HGB(depth=6) 1개뿐이다.
    v29에서 HGB를 d6/d8/sub 3변종으로 다양화해 +10.21을 얻었던 것과 같은 조작을
    Hurdle 내부에도 적용한다.

[2] 판정축 혼합분해 (신규 인수분해, Hurdle과 직교)
    Hurdle은 커맨드 축(core_fail=reverse or middle)으로 쪼갠다.
    판정 축(ball/strike/inplay)은 그와 직교하는 다른 분해다:
        P(success) = sum_c P(call=c|x) * P(success|call=c, x)
    같은 정보를 다른 축으로 쪼개므로 오차구조가 또 달라야 한다.
    Hurdle이 직행모델과 상관 0.87이었으니, 이건 둘 다와 낮은 상관일 가능성이 있다.

평가: fold A(train<=2023 -> valid=2024). v29 로컬 블렌드(893.68) 기준 한계이득과
      상호 상관을 본다. 오늘 확립한 규칙상 최종 채택 전 실측으로 확인한다.
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
CD = "phase80_cache"
VALID_SEASON = 2024
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("데이터 로드 + 피처 재구성 (v28 162개)...")
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

# ---- 행단위 라벨 복원: core_fail(커맨드축) + call(판정축) ----
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


R_, M_, B_, K_ = [cnt(c) for c in ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
                                    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]]
ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[ordr]
n_o = n_[ordr]
step = np.zeros(len(df), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
d_r = np.zeros(len(df)); d_m = np.zeros(len(df)); d_b = np.zeros(len(df)); d_k = np.zeros(len(df))
d_r[ordr[:-1]] = np.diff(R_[ordr]); d_m[ordr[:-1]] = np.diff(M_[ordr])
d_b[ordr[:-1]] = np.diff(B_[ordr]); d_k[ordr[:-1]] = np.diff(K_[ordr])

core_fail = np.where(step, ((d_r > 0) | (d_m > 0)).astype(np.float64), np.nan)
assert (y[step & (core_fail == 1)] == 0).all()
# 판정축: 0=ball, 1=strike, 2=inplay
call = np.full(len(df), np.nan)
call[step & (d_b > 0)] = 0
call[step & (d_k > 0)] = 1
call[step & (d_b == 0) & (d_k == 0)] = 2
log(f"복원 {step.sum():,}행  core_fail={np.nanmean(core_fail):.4f}  "
    f"call분포 ball={np.nanmean(call==0):.3f} strike={np.nanmean(call==1):.3f} inplay={np.nanmean(call==2):.3f}")

seasons = df["season"].to_numpy(np.float64)
tr_m = (seasons <= VALID_SEASON - 1) & step
va_m = seasons == VALID_SEASON
yv = y[va_m].astype(np.float64)
r = yv.mean(); BSREF = r * (1 - r)
w_all = recency_weight(seasons, 2.0)
Xva = X.loc[va_m]


def score(p):
    return 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)


VARIANTS = [
    ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)


def fit_predict(mask, target, variant_extra, tag):
    p = dict(BASE_HGB); p.update(variant_extra)
    ts = time.time()
    m = HistGradientBoostingClassifier(**p).fit(X.loc[mask], target[mask], sample_weight=w_all[mask])
    out = m.predict_proba(Xva)
    log(f"    {tag} 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
    return out


# ---------------- [1] Hurdle 다양화 ----------------
log("[1] Hurdle 3변종 학습...")
nc_m = tr_m & (core_fail == 0)
hurdle_preds = []
for name, extra in VARIANTS:
    f = f"{CD}/h89_core_{name}.npy"
    if os.path.exists(f):
        p_core = np.load(f)
    else:
        p_core = fit_predict(tr_m, core_fail, extra, f"core_{name}")[:, 1]
        np.save(f, p_core)
    f2 = f"{CD}/h89_snc_{name}.npy"
    if os.path.exists(f2):
        p_snc = np.load(f2)
    else:
        p_snc = fit_predict(nc_m, y.astype(np.float64), extra, f"snc_{name}")[:, 1]
        np.save(f2, p_snc)
    ph = (1 - p_core) * p_snc
    hurdle_preds.append(ph)
    log(f"  hurdle_{name} 단독 score={score(ph):.2f}")
p_hurdle_div = np.mean(hurdle_preds, axis=0)
log(f"  Hurdle 3변종 평균 score={score(p_hurdle_div):.2f}")

# ---------------- [2] 판정축 혼합분해 ----------------
log("[2] 판정축 혼합분해 (call 3-class + success|call 3개)...")
f = f"{CD}/h89_callprob.npy"
if os.path.exists(f):
    p_call = np.load(f)
else:
    p_call = fit_predict(tr_m, call, VARIANTS[0][1], "call_3class")
    np.save(f, p_call)
log(f"  call 예측 분포 mean={p_call.mean(axis=0).round(4)}")

p_succ_given = []
for c, cname in [(0, "ball"), (1, "strike"), (2, "inplay")]:
    f2 = f"{CD}/h89_succ_call{c}.npy"
    if os.path.exists(f2):
        ps = np.load(f2)
    else:
        m_c = tr_m & (call == c)
        ps = fit_predict(m_c, y.astype(np.float64), VARIANTS[0][1], f"succ|{cname}")[:, 1]
        np.save(f2, ps)
    p_succ_given.append(ps)
    log(f"  P(success|{cname}) 평균={ps.mean():.4f}")

p_callmix = sum(p_call[:, c] * p_succ_given[c] for c in range(3))
log(f"  판정축 혼합분해 단독 score={score(p_callmix):.2f}")

# ---------------- 평가 ----------------
weights = {"hgb_d6": 9, "hgb_sub": 9, "cat_d6": 8, "hgb_d8": 8, "cat_d8": 5, "cat_rsm": 1}
tot = sum(weights.values())
v29 = sum((wv / tot) * np.load(f"{CD}/{k}.npy") for k, wv in weights.items())
p_hurdle_old = np.load(f"{CD}/hurdle_p_hurdle.npy")

print()
print("=" * 66)
print(f"{'':<34}{'score':>10}{'v29상관':>10}")
print("-" * 66)
for tag, p in [("v29 로컬(6개)", v29), ("Hurdle 단일(v32에 들어간 것)", p_hurdle_old),
               ("Hurdle 3변종 평균", p_hurdle_div), ("판정축 혼합분해", p_callmix)]:
    print(f"{tag:<34}{score(p):10.2f}{np.corrcoef(p, v29)[0,1]:10.4f}")
print()
print("상호 상관:")
print(f"  Hurdle(단일) x Hurdle(3변종) = {np.corrcoef(p_hurdle_old, p_hurdle_div)[0,1]:.4f}")
print(f"  Hurdle(3변종) x 판정축       = {np.corrcoef(p_hurdle_div, p_callmix)[0,1]:.4f}")
print(f"  Hurdle(단일) x 판정축        = {np.corrcoef(p_hurdle_old, p_callmix)[0,1]:.4f}")

print()
print("블렌드 탐색 (v29 + a*Hurdle3변종 + b*판정축):")
best = (None, -9e9)
for a in np.arange(0.0, 0.55, 0.05):
    for b in np.arange(0.0, 0.35, 0.05):
        if a + b > 0.7:
            continue
        p = (1 - a - b) * v29 + a * p_hurdle_div + b * p_callmix
        s = score(p)
        if s > best[1]:
            best = ((round(a, 2), round(b, 2)), s)
print(f"  최적 (a_hurdle3, b_call) = {best[0]}  score={best[1]:.2f}")
print(f"  참고: v32 구성(단일hurdle w=0.30) = {score(0.7*v29 + 0.3*p_hurdle_old):.2f}")
print(f"        v33 구성(단일hurdle w=0.39) = {score(0.61*v29 + 0.39*p_hurdle_old):.2f}")
log(f"총 {time.time()-t0:.0f}s")
