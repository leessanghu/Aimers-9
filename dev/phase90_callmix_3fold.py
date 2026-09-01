"""phase90 — 판정축 혼합분해를 3폴드로 검증 (v35 기준선 위에 추가).

현재 최고 실측: v35 = 1051.40 (v29 앙상블 + Hurdle 2변종, w=0.45)
Hurdle은 '커맨드 축'(core_fail = reverse or middle)으로 타겟을 쪼갠다.
판정 축(ball/strike/inplay)은 그와 직교하는 다른 분해다:

    P(success) = sum_c P(call=c|x) * P(success|call=c, x),  c in {ball, strike, inplay}

같은 정보를 다른 축으로 쪼개므로 오차구조가 또 달라야 한다.

3폴드 구성 (F리그 2022->2023 단절 0.709->0.473을 고려):
    fold A: train<=2023 -> 2024   단절 이후, 깨끗
    fold B: train<=2022 -> 2023   단절을 가로지름. 기준선이 -1600까지 깨지는 폴드라
                                   절대점수는 신뢰 불가, 스트레스 테스트 용도
    fold C: train<=2021 -> 2022   단절 이전, 깨끗

채택 기준: 깨끗한 A와 C 둘 다에서 이득. B는 파국만 아니면 통과.
(phase85 교훈: 단일 폴드는 부호까지 뒤집힌다. CatBoost refit A +61.70 / B -456.04)

캐시: 폴드별로 모든 예측을 npy로 저장. 중간에 죽어도 재실행하면 이어서 진행.
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
CD = "phase90_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0, ref=None):
    r = ref if ref is not None else seasons.max()
    return 0.5 ** ((r - seasons) / half_life)


log("데이터 로드 + 피처 재구성 (v28/v29와 동일 162개)...")
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
assert (y[step & (core_fail == 1)] == 0).all(), "core_fail 라벨 복원 버그"
call = np.full(len(df), np.nan)
call[step & (d_b > 0)] = 0
call[step & (d_k > 0)] = 1
call[step & (d_b == 0) & (d_k == 0)] = 2
log(f"복원 {step.sum():,}행  core_fail={np.nanmean(core_fail):.4f}  "
    f"call: ball={np.nanmean(call==0):.3f} strike={np.nanmean(call==1):.3f} inplay={np.nanmean(call==2):.3f}")

seasons = df["season"].to_numpy(np.float64)
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)
HGB_VARIANTS = [
    ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]
HURDLE_VARIANTS = HGB_VARIANTS[:2]   # v35와 동일하게 d6, d8


def fit_cached(tag, mask, target, extra, Xva, w, multiclass=False):
    f = f"{CD}/{tag}.npy"
    if os.path.exists(f):
        log(f"    {tag} 캐시")
        return np.load(f)
    p = dict(BASE_HGB); p.update(extra)
    ts = time.time()
    m = HistGradientBoostingClassifier(**p).fit(X.loc[mask], target[mask], sample_weight=w[mask])
    out = m.predict_proba(Xva) if multiclass else m.predict_proba(Xva)[:, 1]
    np.save(f, out)
    log(f"    {tag} 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
    return out


def run_fold(train_upto, valid_season, tag):
    log(f"===== fold {tag}: train<={train_upto} -> valid={valid_season} =====")
    tr_m = (seasons <= train_upto) & step
    va_m = seasons == valid_season
    yv = y[va_m].astype(np.float64)
    r = yv.mean(); BS = r * (1 - r)
    Xva = X.loc[va_m]
    w = recency_weight(seasons, 2.0, ref=train_upto)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    log("  [1] 기준선 HGB 3변종...")
    base_preds = [fit_cached(f"{tag}_base_{n}", tr_m, y.astype(np.float64), e, Xva, w)
                  for n, e in HGB_VARIANTS]
    p_base = np.mean(base_preds, axis=0)
    log(f"    기준선 score={score(p_base):.2f}")

    log("  [2] Hurdle 2변종 (v35 구성)...")
    nc_m = tr_m & (core_fail == 0)
    hur = []
    for n, e in HURDLE_VARIANTS:
        pc = fit_cached(f"{tag}_core_{n}", tr_m, core_fail, e, Xva, w)
        ps = fit_cached(f"{tag}_snc_{n}", nc_m, y.astype(np.float64), e, Xva, w)
        hur.append((1 - pc) * ps)
    p_hur = np.mean(hur, axis=0)
    log(f"    Hurdle score={score(p_hur):.2f}")

    log("  [3] 판정축 혼합분해 (신규)...")
    p_call = fit_cached(f"{tag}_call3", tr_m, call, HGB_VARIANTS[0][1], Xva, w, multiclass=True)
    psg = []
    for c, cname in [(0, "ball"), (1, "strike"), (2, "inplay")]:
        psg.append(fit_cached(f"{tag}_succ_{cname}", tr_m & (call == c), y.astype(np.float64),
                              HGB_VARIANTS[0][1], Xva, w))
    p_mix = sum(p_call[:, c] * psg[c] for c in range(3))
    log(f"    판정축 score={score(p_mix):.2f}")

    # v35 구성 재현: base + hurdle(w=0.45)
    p_v35 = 0.55 * p_base + 0.45 * p_hur
    res = dict(
        base=score(p_base), hurdle=score(p_hur), mix=score(p_mix), v35=score(p_v35),
        corr_hur_base=np.corrcoef(p_hur, p_base)[0, 1],
        corr_mix_base=np.corrcoef(p_mix, p_base)[0, 1],
        corr_mix_hur=np.corrcoef(p_mix, p_hur)[0, 1],
    )
    best = (None, -9e9)
    for a in np.arange(0.0, 0.65, 0.05):
        for b in np.arange(0.0, 0.45, 0.05):
            if a + b > 0.8:
                continue
            s = score((1 - a - b) * p_base + a * p_hur + b * p_mix)
            if s > best[1]:
                best = ((round(a, 2), round(b, 2)), s)
    res["best_ab"] = best[0]
    res["best_score"] = best[1]
    res["mix_gain_over_v35"] = best[1] - res["v35"]
    log(f"    최적 (a_hurdle, b_mix)={best[0]}  score={best[1]:.2f}  "
        f"(v35구성 대비 {best[1]-res['v35']:+.2f})")
    return res


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    results[tag] = run_fold(upto, val, tag)

print()
print("=" * 78)
print(f"{'fold':<6}{'base':>10}{'hurdle':>10}{'판정축':>10}{'v35구성':>10}{'최적':>10}{'판정축이득':>12}")
print("-" * 78)
for tag in ["A", "C", "B"]:
    r = results[tag]
    print(f"{tag:<6}{r['base']:10.2f}{r['hurdle']:10.2f}{r['mix']:10.2f}{r['v35']:10.2f}"
          f"{r['best_score']:10.2f}{r['mix_gain_over_v35']:+12.2f}   {r['best_ab']}")
print()
print("상관 (판정축이 진짜 독립 축인지):")
for tag in ["A", "C", "B"]:
    r = results[tag]
    print(f"  fold {tag}: mix-base {r['corr_mix_base']:.4f}   mix-hurdle {r['corr_mix_hur']:.4f}   "
          f"(참고 hurdle-base {r['corr_hur_base']:.4f})")
print()
clean_ok = results["A"]["mix_gain_over_v35"] > 3 and results["C"]["mix_gain_over_v35"] > 3
b_ok = results["B"]["mix_gain_over_v35"] > -20
print(f"깨끗한 폴드(A,C) 둘 다 +3 이상: {clean_ok}")
print(f"스트레스 폴드(B) 파국 아님(> -20): {b_ok}")
print("=> 채택 검토" if (clean_ok and b_ok) else "=> 기각 또는 보류")
pd.DataFrame(results).T.to_csv("phase90_callmix_3fold.csv")
log(f"총 {time.time()-t0:.0f}s")
