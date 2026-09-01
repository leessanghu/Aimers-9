"""phase84 — CatBoost early-stop 후 전체데이터 refit (Codex 1순위, 확정 버그).

확인된 사실: time_split_es(frac=0.08)는 전체 1,475,092행의 마지막 8%(118,008행)를
검증용으로 뗀다. 이 118,008행은 전부 2024년이고, 2024 전체(253,507행)의 46.55%다.
즉 v29의 CatBoost 3개는 2024년 후반 거의 절반을 한 번도 못 보고 학습됐다.

수정 방향 (직접 교체가 아니라 추가):
    Cat_ES    = 기존 방식 (시간 홀드아웃으로 ES, 이후 그 상태로 예측) — 시간적으로
                덜 과적합된 다양성 멤버로 유지
    Cat_refit = ES로 정한 iteration을 '고정'하고 전체 데이터로 재학습 (최신정보 반영)
    최종 후보 = 0.5 * Cat_refit + 0.5 * Cat_ES

로컬 검증 절차 (train<=2023 -> valid=2024 폴드, phase80/v29와 동일 프레임):
    1. train<=2023 내부에서 time_split_es로 ES -> best_iteration 확정 (= 현재 v29 방식,
       phase80의 cat_d6/d8/rsm과 동치)
    2. 같은 config, 같은 iteration 고정, train<=2023 '전체'(홀드아웃 없이)로 재학습
    3. 2024에서 Cat_ES vs Cat_refit vs 0.5/0.5 블렌드 비교
    4. v29 6개 앙상블에 Cat_refit 3개를 추가했을 때 한계이득도 측정
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
CACHE_DIR = "phase80_cache"
VALID_SEASON = 2024
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("데이터 로드 + 피처 재구성 (v28 162개, phase80과 동일)...")
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
tr_m = seasons <= VALID_SEASON - 1
va_m = seasons == VALID_SEASON
yv = y[va_m].astype(np.float64)
w_tr = recency_weight(seasons[tr_m], half_life=2.0)
Xtr = X.loc[tr_m].reset_index(drop=True)
ytr = y[tr_m]
Xva = X.loc[va_m]
r = yv.mean()
BSREF = r * (1 - r)


def score(p):
    return 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)


tr_i, es_i = time_split_es(len(Xtr))
log(f"train={tr_m.sum():,}  valid={va_m.sum():,}  (ES홀드아웃 {len(es_i):,}행 = train의 마지막 8%)")

CONFIGS = [
    ("d6", dict(depth=6, random_seed=42)),
    ("d8", dict(depth=8, l2_leaf_reg=10.0, random_seed=7)),
    ("rsm", dict(depth=6, rsm=0.6, random_seed=2024)),
]

results = {}
for name, extra in CONFIGS:
    params = dict(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                 early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    params.update(extra)

    log(f"[{name}] Cat_ES 학습 (ES로 iteration 확정)...")
    ts = time.time()
    m_es = CatBoostClassifier(**params)
    m_es.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
    best_iter = m_es.best_iteration_
    p_es = m_es.predict_proba(Xva)[:, 1]
    log(f"  best_iter={best_iter}  score={score(p_es):.2f}  ({time.time()-ts:.0f}s)")

    log(f"[{name}] Cat_refit (iteration={best_iter} 고정, train<=2023 전체로 재학습)...")
    ts = time.time()
    params_fixed = dict(params)
    params_fixed.pop("early_stopping_rounds")
    params_fixed["iterations"] = max(best_iter, 1)
    m_refit = CatBoostClassifier(**params_fixed)
    m_refit.fit(Xtr, ytr, sample_weight=w_tr)
    p_refit = m_refit.predict_proba(Xva)[:, 1]
    log(f"  score={score(p_refit):.2f}  ({time.time()-ts:.0f}s)")

    p_blend = 0.5 * p_es + 0.5 * p_refit
    corr = np.corrcoef(p_es, p_refit)[0, 1]
    log(f"  0.5/0.5 블렌드 score={score(p_blend):.2f}  상관(ES,refit)={corr:.4f}")

    np.save(f"{CACHE_DIR}/cat_{name}_ES.npy", p_es)
    np.save(f"{CACHE_DIR}/cat_{name}_refit.npy", p_refit)
    results[name] = dict(best_iter=best_iter, s_es=score(p_es), s_refit=score(p_refit),
                         s_blend=score(p_blend), corr=corr)

print()
print("=" * 70)
print(f"{'config':<8}{'iter':>6}{'ES':>10}{'refit':>10}{'blend':>10}{'corr':>8}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<8}{r['best_iter']:6d}{r['s_es']:10.2f}{r['s_refit']:10.2f}{r['s_blend']:10.2f}{r['corr']:8.4f}")

log("[전체] v29의 6개 로컬블렌드 재현 + Cat_refit 3개 추가 한계이득...")
weights = {"hgb_d6": 9, "hgb_sub": 9, "cat_d6": 8, "hgb_d8": 8, "cat_d8": 5, "cat_rsm": 1}
tot = sum(weights.values())
p_v29local = np.zeros(len(yv))
for k, w in weights.items():
    p_v29local += (w / tot) * np.load(f"{CACHE_DIR}/{k}.npy")
log(f"  v29 로컬(6개, phase80 캐시 재사용) score={score(p_v29local):.2f}")

p_es_all = np.mean([np.load(f"{CACHE_DIR}/cat_{n}_ES.npy") for n, _ in CONFIGS], axis=0)
p_refit_all = np.mean([np.load(f"{CACHE_DIR}/cat_{n}_refit.npy") for n, _ in CONFIGS], axis=0)
p_cat_new = 0.5 * p_es_all + 0.5 * p_refit_all
print()
print(f"{'조합':<42}{'score':>10}{'vs v29local':>12}")
print("-" * 66)
base = score(p_v29local)
print(f"{'v29 로컬(6개, 기존 Cat 3=ES만)':<42}{base:10.2f}{0:12.2f}")
# 6개 중 Cat 부분(d6/d8/rsm ES)을 refit들로 통째로 교체
p_hgb3 = np.mean([np.load(f"{CACHE_DIR}/hgb_d6.npy"), np.load(f"{CACHE_DIR}/hgb_sub.npy"),
                  np.load(f"{CACHE_DIR}/hgb_d8.npy")], axis=0)
cat6 = [np.load(f"{CACHE_DIR}/cat_d6_ES.npy"), np.load(f"{CACHE_DIR}/cat_d8_ES.npy"),
       np.load(f"{CACHE_DIR}/cat_rsm_ES.npy"), np.load(f"{CACHE_DIR}/cat_d6_refit.npy"),
       np.load(f"{CACHE_DIR}/cat_d8_refit.npy"), np.load(f"{CACHE_DIR}/cat_rsm_refit.npy")]
p_cat6avg = np.mean(cat6, axis=0)
p_9member = 0.5 * p_hgb3 + 0.5 * p_cat6avg
s9 = score(p_9member)
print(f"{'HGB3 + Cat6(ES3+refit3, 계열균형)':<42}{s9:10.2f}{s9-base:+12.2f}")

log(f"총 {time.time()-t0:.0f}s")
