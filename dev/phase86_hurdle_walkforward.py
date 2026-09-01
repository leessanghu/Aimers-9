"""phase86 — Hurdle 인수분해를 2폴드로 검증 (phase85에서 확립된 필수 절차).

phase85 교훈: CatBoost refit은 fold A(2023->2024)에서 +61.70이었는데
fold B(2022->2023)에서 -456.04로 부호가 뒤집혔고, 실측도 -13.97(5시그마)로 fold B와
같은 방향이었다. 즉 '2024 폴드 하나'는 신뢰할 수 없다.

Hurdle(phase83)도 2024 폴드에서만 +27.06을 쟀으므로 동일하게 재검증한다.
    fold A: train<=2023 -> valid=2024  (phase83에서 이미 측정, 여기서 재현)
    fold B: train<=2022 -> valid=2023  (신규)
두 폴드 모두에서 개선되어야 채택한다.

주의: fold B는 F리그 체제단절(2022 0.709 -> 2023 0.473) 직후라 절대점수가 음수로
나올 수 있다(phase85에서 확인). 절대값이 아니라 '같은 폴드 내에서 hurdle을 섞었을 때
개선되는가'만 본다.
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
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0, ref=None):
    r = ref if ref is not None else seasons.max()
    return 0.5 ** ((r - seasons) / half_life)


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

# core_fail 라벨 복원
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


R_, M_ = [cnt(c) for c in ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"]]
ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[ordr]
n_o = n_[ordr]
step = np.zeros(len(df), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
r_diff = np.zeros(len(df)); m_diff = np.zeros(len(df))
r_diff[ordr[:-1]] = np.diff(R_[ordr])
m_diff[ordr[:-1]] = np.diff(M_[ordr])
core_fail = np.where(step, ((r_diff > 0) | (m_diff > 0)).astype(np.float64), np.nan)
assert (y[step & (core_fail == 1)] == 0).all()
log(f"core_fail 복원 {step.sum():,}행  비율={np.nanmean(core_fail):.4f}")

seasons = df["season"].to_numpy(np.float64)
HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)
HGB_VARIANTS = [
    ("hgb_d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("hgb_d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("hgb_sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]


def run_fold(train_upto, valid_season, tag):
    log(f"=== fold {tag}: train<={train_upto} -> valid={valid_season} ===")
    tr_m = (seasons <= train_upto) & step
    va_m = seasons == valid_season
    yv = y[va_m].astype(np.float64)
    r = yv.mean(); BS = r * (1 - r)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    w = recency_weight(seasons[tr_m], 2.0, ref=train_upto)
    Xva = X.loc[va_m]

    # 기준선: HGB 3변종 평균 (v29의 HGB 절반, CatBoost는 폴드마다 재학습 비싸서 제외)
    base_preds = []
    for name, extra in HGB_VARIANTS:
        p = dict(HGB); p.update(extra)
        m = HistGradientBoostingClassifier(**p).fit(X.loc[tr_m], y[tr_m], sample_weight=w)
        base_preds.append(m.predict_proba(Xva)[:, 1])
    p_base = np.mean(base_preds, axis=0)
    log(f"  기준선(HGB3 평균) score={score(p_base):.2f}")

    # hurdle
    m1 = HistGradientBoostingClassifier(**HGB).fit(X.loc[tr_m], core_fail[tr_m], sample_weight=w)
    p_core = m1.predict_proba(Xva)[:, 1]
    nc = tr_m & (core_fail == 0)
    w_nc = recency_weight(seasons[nc], 2.0, ref=train_upto)
    m2 = HistGradientBoostingClassifier(**HGB).fit(X.loc[nc], y[nc], sample_weight=w_nc)
    p_snc = m2.predict_proba(Xva)[:, 1]
    p_hur = (1 - p_core) * p_snc
    log(f"  hurdle 단독 score={score(p_hur):.2f}  상관={np.corrcoef(p_hur,p_base)[0,1]:.4f}")

    out = {}
    for wt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        out[wt] = score((1 - wt) * p_base + wt * p_hur)
    return out


res = {}
for upto, val, tag in [(2023, 2024, "A(2023->2024)"), (2022, 2023, "B(2022->2023)")]:
    res[tag] = run_fold(upto, val, tag)

print()
print("=" * 64)
print(f"{'w_hurdle':>10}" + "".join(f"{t:>26}" for t in res))
print("-" * 64)
for wt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    row = f"{wt:>10.1f}"
    for t in res:
        d = res[t][wt] - res[t][0.0]
        row += f"{res[t][wt]:>16.2f}({d:+7.2f})"
    print(row)
print()
ok = all(max(v.values()) > v[0.0] + 3 for v in res.values())
best = {t: max(v, key=v.get) for t, v in res.items()}
print(f"폴드별 최적 w: {best}")
print("=> 두 폴드 모두 개선: 채택 가능" if ok else "=> 한쪽 폴드에서 개선 없음: 기각 (phase85 CatBoost refit과 동일 패턴)")
log(f"총 {time.time()-t0:.0f}s")
