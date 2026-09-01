"""phase85 — CatBoost refit>ES가 2023->2024에만 있는 우연인지, 일반화되는 효과인지 검증.

v30 실측 실패(-13.97) 이후 재검토. phase84는 딱 한 쌍(train<=2023 -> valid=2024)만
봤다. 3개 config가 일치했다고 노이즈가 아니라고 판단했지만, 셋 다 '같은 폴드'를 봤으니
2023년 4분기 특유의 사정(리그 조건 변화 등)이었다면 셋 다 똑같이 낚였을 수 있다.

이번엔 완전히 다른 시대로 재현한다:
    fold A (phase84, 이미 확인): ES홀드아웃=<=2023의 마지막 8%(2023 Q4 근방) -> iter_A
                                 refit 전체<=2023 -> 2024 평가
    fold B (신규, 독립검증):    ES홀드아웃=<=2022의 마지막 8%(2022 Q4 근방) -> iter_B
                                 refit 전체<=2022 -> 2023 평가

fold A와 B가 둘 다 'refit > ES'면 일반화되는 효과 (v30 실패는 iteration/설정 문제).
fold B에서 효과가 사라지거나 반대면, phase84 결과 자체가 2023 특유의 우연이었다는 뜻
-> CatBoost refit 아이디어 자체를 접어야 한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

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
from phase2_common import time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0, ref_season=None):
    ref = ref_season if ref_season is not None else seasons.max()
    return 0.5 ** ((ref - seasons) / half_life)


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

seasons = df["season"].to_numpy(np.float64)
CONFIGS = [
    ("d6", dict(depth=6, random_seed=42)),
    ("d8", dict(depth=8, l2_leaf_reg=10.0, random_seed=7)),
    ("rsm", dict(depth=6, rsm=0.6, random_seed=2024)),
]


def run_fold(train_upto, valid_season, tag):
    log(f"=== fold {tag}: train<={train_upto} -> valid={valid_season} ===")
    tr_m = seasons <= train_upto
    va_m = seasons == valid_season
    yv = y[va_m].astype(np.float64)
    r = yv.mean(); BSREF = r * (1 - r)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)

    Xtr = X.loc[tr_m].reset_index(drop=True)
    ytr = y[tr_m]
    Xva = X.loc[va_m]
    w_tr = recency_weight(seasons[tr_m], half_life=2.0, ref_season=train_upto)
    tr_i, es_i = time_split_es(len(Xtr))
    log(f"  train={tr_m.sum():,}  valid={va_m.sum():,}  ES홀드아웃={len(es_i):,}행")

    results = {}
    for name, extra in CONFIGS:
        params = dict(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                     early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
        params.update(extra)
        ts = time.time()
        m_es = CatBoostClassifier(**params)
        m_es.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
        best_iter = m_es.best_iteration_
        p_es = m_es.predict_proba(Xva)[:, 1]
        s_es = score(p_es)

        params_fixed = dict(params); params_fixed.pop("early_stopping_rounds")
        params_fixed["iterations"] = max(best_iter, 1)
        m_refit = CatBoostClassifier(**params_fixed)
        m_refit.fit(Xtr, ytr, sample_weight=w_tr)
        p_refit = m_refit.predict_proba(Xva)[:, 1]
        s_refit = score(p_refit)

        log(f"  [{name}] iter={best_iter}  ES={s_es:.2f}  refit={s_refit:.2f}  "
           f"delta={s_refit-s_es:+.2f}  ({time.time()-ts:.0f}s)")
        results[name] = dict(iter=best_iter, es=s_es, refit=s_refit, delta=s_refit - s_es)
    return results


res_B = run_fold(2022, 2023, "B(신규,2022->2023)")

print()
print("=" * 60)
print("fold A (phase84, 이미 확인, train<=2023->valid=2024):")
print("  d6  iter=543  ES=844.69  refit=906.23  delta=+61.54")
print("  d8  iter=202  ES=828.51  refit=892.48  delta=+63.97")
print("  rsm iter=544  ES=835.98  refit=895.56  delta=+59.58")
print()
print("fold B (신규, train<=2022->valid=2023):")
for name, r in res_B.items():
    print(f"  {name:<4}iter={r['iter']:<5}ES={r['es']:.2f}  refit={r['refit']:.2f}  delta={r['delta']:+.2f}")
print()
avg_delta_B = np.mean([r["delta"] for r in res_B.values()])
print(f"fold B 평균 delta: {avg_delta_B:+.2f}  (fold A 평균: +61.70)")
if avg_delta_B > 20:
    print("=> 두 폴드 모두 강하게 재현됨. 일반화되는 효과일 가능성 높음.")
elif avg_delta_B > 0:
    print("=> 방향은 같으나 크기가 작음. 부분적으로만 일반화.")
else:
    print("=> fold B에서 사라지거나 반대. fold A는 2023 특유의 우연이었을 가능성 높음.")
log(f"총 {time.time()-t0:.0f}s")
