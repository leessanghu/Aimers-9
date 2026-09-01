"""phase91 — R/F 전문가 모델, 3폴드 검증 + 증분/매그니튜드 분석.

배경: F리그는 전체의 11%(16.1만행)뿐이고 2022->2023 regime 단절(0.709->0.473)이 있다.
2023+만 쓰면 5.6만행으로 더 줄어든다. v29 국소분석(Codex): 2024에서 R점수 903.34,
F점수 475.71 -- F가 훨씬 어렵고, 모델다양화 이득도 R+12.93/F-6.43로 반대부호였다.

이번에 테스트하는 F 전문가 3가지:
    F_all    : F 전체이력(2019~2024), half_life=2.0 (기존과 동일한 감가)
    F_short  : F 전체이력, half_life=1.0 (regime 단절을 더 세게 할인)
    F_recent : F의 train_upto 기준 최근 2시즌만 (regime 순수, 표본 적음)
R 전문가는 1개(R만 학습, half_life=2.0 동일).

평가:
    1) 증분(partial_gain): 각 전문가 예측이 base(전역모델) 잔차에 대해 갖는 부분상관.
       R/F 서브셋 내에서만 계산 (해당 리그에서만 의미있는 신호인지 보려고).
    2) 매그니튜드(구간분해): base vs 전문가 라우팅/블렌드의 R서브셋/F서브셋 개별 점수.
    3) 3폴드(A:<=2023->2024, C:<=2021->2022, B:<=2022->2023) 전부 확인.

phase90 캐시(base_d6/d8/sub)를 전역모델로 재사용해 시간을 아낀다.
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
CD90 = "phase90_cache"
CD = "phase91_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0, ref=None):
    r = ref if ref is not None else seasons.max()
    return 0.5 ** ((r - seasons) / half_life)


def partial_gain(y, p, z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    if z.std() == 0 or len(y) < 30:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


log("데이터 로드 + 피처 재구성 (동일 162개)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())
gt = df["game_type"].to_numpy()

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
HGB_D6 = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)


def fit_cached(tag, mask, target, w, Xva):
    f = f"{CD}/{tag}.npy"
    if os.path.exists(f):
        log(f"    {tag} 캐시")
        return np.load(f)
    ts = time.time()
    m = HistGradientBoostingClassifier(**HGB_D6).fit(X.loc[mask], target[mask], sample_weight=w[mask])
    p = m.predict_proba(Xva)[:, 1]
    np.save(f, p)
    log(f"    {tag} 완료 iters={m.n_iter_} n={mask.sum():,} ({time.time()-ts:.0f}s)")
    return p


def run_fold(train_upto, valid_season, tag):
    log(f"===== fold {tag}: train<={train_upto} -> valid={valid_season} =====")
    tr_m = seasons <= train_upto
    va_m = seasons == valid_season
    yv = y[va_m].astype(np.float64)
    gt_va = gt[va_m]
    Xva = X.loc[va_m]
    r = yv.mean(); BS = r * (1 - r)

    def score(p, mask=None):
        if mask is None:
            mask = np.ones(len(yv), dtype=bool)
        yy = yv[mask]; pp = p[mask]
        rr = yy.mean(); bb = rr * (1 - rr)
        return 1e5 * (1 - np.mean((pp - yy) ** 2) / bb)

    base = np.mean([np.load(f"{CD90}/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)

    w2 = recency_weight(seasons, 2.0, ref=train_upto)
    w1 = recency_weight(seasons, 1.0, ref=train_upto)

    r_tr = tr_m & (gt == "R")
    f_tr_all = tr_m & (gt == "F")
    f_tr_recent = f_tr_all & (seasons >= train_upto - 1)

    p_rexp = fit_cached(f"{tag}_R_expert", r_tr, y.astype(np.float64), w2, Xva)
    p_fall = fit_cached(f"{tag}_F_all", f_tr_all, y.astype(np.float64), w2, Xva)
    p_fshort = fit_cached(f"{tag}_F_short", f_tr_all, y.astype(np.float64), w1, Xva)
    p_frecent = fit_cached(f"{tag}_F_recent", f_tr_recent, y.astype(np.float64), w2, Xva) \
        if f_tr_recent.sum() > 3000 else None

    is_R = gt_va == "R"; is_F = gt_va == "F"
    log(f"  valid: R={is_R.sum():,}  F={is_F.sum():,}")
    log(f"  [base]     전체={score(base):.2f}  R구간={score(base,is_R):.2f}  F구간={score(base,is_F):.2f}")
    log(f"  [R_expert] R구간={score(p_rexp,is_R):.2f}  (base대비 {score(p_rexp,is_R)-score(base,is_R):+.2f})")
    for nm, p in [("F_all", p_fall), ("F_short", p_fshort), ("F_recent", p_frecent)]:
        if p is None:
            log(f"  [{nm}] 표본부족으로 스킵")
            continue
        log(f"  [{nm}] F구간={score(p,is_F):.2f}  (base대비 {score(p,is_F)-score(base,is_F):+.2f})")

    # 증분: base 잔차에 대한 전문가 예측의 부분상관 (해당 서브셋 내에서만)
    gn_r, _ = partial_gain(yv[is_R], base[is_R], p_rexp[is_R])
    log(f"  증분(R_expert, R서브셋 partial_gain) = {gn_r:+.2f}")
    for nm, p in [("F_all", p_fall), ("F_short", p_fshort), ("F_recent", p_frecent)]:
        if p is None:
            continue
        gn, _ = partial_gain(yv[is_F], base[is_F], p[is_F])
        log(f"  증분({nm}, F서브셋 partial_gain) = {gn:+.2f}")

    # 최선의 F전문가 선택 (블렌드 그리드)
    def eval_blend(p_expert, mask, wf):
        out = base.copy()
        out[mask] = (1 - wf) * base[mask] + wf * p_expert[mask]
        return score(out, mask)

    grid = []
    for nm, p in [("F_all", p_fall), ("F_short", p_fshort), ("F_recent", p_frecent)]:
        if p is None:
            continue
        for wf in np.arange(0.0, 1.05, 0.1):
            grid.append((nm, round(wf, 2), eval_blend(p, is_F, wf)))
    grid.sort(key=lambda t: -t[2])
    log(f"  F 최적 블렌드: {grid[0]}  (base F구간 {score(base,is_F):.2f} 대비 {grid[0][2]-score(base,is_F):+.2f})")

    best_r = None
    r_grid = [(round(wr, 2), None) for wr in np.arange(0.0, 1.05, 0.1)]
    r_scores = []
    for wr in np.arange(0.0, 1.05, 0.1):
        out = base.copy()
        out[is_R] = (1 - wr) * base[is_R] + wr * p_rexp[is_R]
        r_scores.append((round(wr, 2), score(out, is_R)))
    r_scores.sort(key=lambda t: -t[1])
    log(f"  R 최적 블렌드: {r_scores[0]}  (base R구간 {score(base,is_R):.2f} 대비 "
        f"{r_scores[0][1]-score(base,is_R):+.2f})")

    # 최종 결합: R 최적 + F 최적 동시 적용, 전체 점수
    best_fname, best_fw, _ = grid[0]
    fmap = {"F_all": p_fall, "F_short": p_fshort, "F_recent": p_frecent}
    final = base.copy()
    final[is_R] = (1 - r_scores[0][0]) * base[is_R] + r_scores[0][0] * p_rexp[is_R]
    final[is_F] = (1 - best_fw) * base[is_F] + best_fw * fmap[best_fname][is_F]
    log(f"  [결합] 전체 score={score(final):.2f}  (base 전체 {score(base):.2f} 대비 {score(final)-score(base):+.2f})")

    return dict(base_total=score(base), final_total=score(final), gain=score(final) - score(base),
               r_best_w=r_scores[0][0], f_best=(best_fname, best_fw))


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    results[tag] = run_fold(upto, val, tag)

print()
print("=" * 70)
print(f"{'fold':<6}{'base':>10}{'결합':>10}{'이득':>10}   R가중   F선택")
print("-" * 70)
for tag in ["A", "C", "B"]:
    r = results[tag]
    print(f"{tag:<6}{r['base_total']:10.2f}{r['final_total']:10.2f}{r['gain']:+10.2f}   "
          f"{r['r_best_w']}   {r['f_best']}")
ok = all(results[t]["gain"] > 3 for t in ["A", "C"]) and results["B"]["gain"] > -20
print()
print("깨끗한 폴드(A,C) 둘 다 +3 이상, B 파국 아님:", ok)
log(f"총 {time.time()-t0:.0f}s")
