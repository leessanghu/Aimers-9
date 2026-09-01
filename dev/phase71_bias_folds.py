"""편향이 재현되는가 — v27 설정으로 3개 폴드에서 forward bias를 각각 측정.

왜 필요한가:
  phase70에서 잰 편향 +0.01055는 2023->2024 폴드 하나뿐이다. 통보정을 태우려면 이 편향이
  '매 시즌 반복되는 구조적 현상'인지 '2024만의 우연'인지 알아야 한다.

  phase59가 잰 폴드별 편향은 -0.0058 / +0.0168 / +0.0105 로 부호까지 뒤집혔는데,
  그건 recency weighting이 없던 구버전(v23 계열) 설정이었다. recency weighting은 최근 시즌에
  가중을 줘서 드리프트 추종을 개선하므로 편향 구조가 달라졌을 수 있다. 지금 설정으로 다시 잰다.

측정 방식:
  각 폴드 (train<=Y, valid=Y+1)에서 v27과 동일 설정(154피처, recency half-life=2,
  HGB + CatBoost 50:50)으로 학습하고 valid에서 bias = mean(pred) - mean(y)를 잰다.
  편향 측정에는 시드 분산이 거의 영향 없으므로 CatBoost는 단일시드로 충분하다(시간 절약).

판정:
  세 폴드 모두 같은 부호이고 크기가 비슷하면 -> 구조적 편향. 통보정 정당.
  부호가 뒤집히면 -> 시즌마다 다른 현상. 통보정은 도박이고, 제출로 b를 역산하는 게 맞다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from batterform import K_BATTER, build_batter_table, transform_batter
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from count_split import K_COUNT, build_count_table, transform_count
from crosses import add_crosses
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

SEED = 42
HALF_LIFE = 2.0
TM_CACHE = "phase64_trackman_profile.parquet"
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


log("전체 피처 블록 준비 (v27과 동일 154개)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())

se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dcnt = transform_count(df, build_count_table(df), pp, sr, k=K_COUNT)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
dvol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL)
drole = transform_role(df, build_role_table(df), sr)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
dform = transform_form(df, drole, dins["inseason_success_smooth"].to_numpy(np.float64), base_middle)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
dtm = transform_trackman(df, prof, sr)
lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
dtmx = add_lown_interactions(dtm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr)
dbat = transform_batter(df, build_batter_table(df), sr, g, k=K_BATTER)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
log("블록 준비 완료")


def stack(i, base_frame):
    X = pd.concat([base_frame.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    parts = [X, add_crosses(X), dly.loc[i].reset_index(drop=True), dcnt.loc[i].reset_index(drop=True),
             dvol.loc[i].reset_index(drop=True), drole.loc[i].reset_index(drop=True),
             dform.loc[i].reset_index(drop=True), dtm.loc[i].reset_index(drop=True),
             dtmx.loc[i].reset_index(drop=True), dbat.loc[i].reset_index(drop=True)]
    return pd.concat(parts, axis=1)


def recency_weight(seasons, half_life=HALF_LIFE):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


results = []
for train_max, valid_season in FOLDS:
    log(f"\n=== 폴드 train<={train_max} -> valid={valid_season} ===")
    tr_i = df[df.season <= train_max].index
    va_i = df[df.season == valid_season].index
    fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED,
                      include_team_te=True, team_te_mode="expanding")
    y_tr, y_va = fold["y_train"], fold["y_valid"]
    X_tr = stack(tr_i, fold["X_train"])
    X_va = stack(va_i, fold["X_valid"])
    w = recency_weight(df.loc[tr_i, "season"].to_numpy(np.float64))
    w = w / w.mean()
    ti, ei = time_split_es(len(X_tr))

    h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=SEED)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                            verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    h.fit(X_tr, y_tr, sample_weight=w)
    cb.fit(X_tr.iloc[ti], y_tr[ti], sample_weight=w[ti], eval_set=(X_tr.iloc[ei], y_tr[ei]))
    p = 0.5 * h.predict_proba(X_va)[:, 1] + 0.5 * cb.predict_proba(X_va)[:, 1]

    r = y_va.mean()
    bsref = r * (1 - r)
    bias = float(p.mean() - r)
    cov = float(np.mean((p - p.mean()) * (y_va - r)))
    b_opt = cov / p.var()
    score = max(0, 1e5 * (1 - np.mean((p - y_va) ** 2) / bsref))
    # 통보정(상수 shift)을 적용했다면 얼마나 얻었을지
    p_shift = np.clip(p - bias, 1e-6, 1 - 1e-6)
    score_shift = max(0, 1e5 * (1 - np.mean((p_shift - y_va) ** 2) / bsref))
    results.append(dict(valid=valid_season, n=len(y_va), rate=r, score=score,
                        bias=bias, b_opt=b_opt, gain_full=score_shift - score))
    log(f"  score={score:.1f}  bias={bias:+.5f}  b_opt={b_opt:.4f}  통보정이득={score_shift-score:+.1f}")

log("\n" + "=" * 78)
log("폴드별 편향 (v27 설정, recency weighting 적용)")
log("=" * 78)
print(f"{'valid':>7}{'n':>10}{'성공률':>9}{'score':>9}{'bias':>10}{'b_opt':>8}{'통보정이득':>11}")
print("-" * 64)
for r_ in results:
    print(f"{r_['valid']:>7}{r_['n']:>10,}{r_['rate']:9.4f}{r_['score']:9.1f}"
          f"{r_['bias']:+10.5f}{r_['b_opt']:8.4f}{r_['gain_full']:+11.1f}")

biases = [r_["bias"] for r_ in results]
same_sign = all(b > 0 for b in biases) or all(b < 0 for b in biases)
print()
print(f"  부호 일치: {'예 -> 구조적 편향, 통보정 정당' if same_sign else '아니오 -> 시즌마다 다름, 제출로 b 역산이 안전'}")
print(f"  평균 편향: {np.mean(biases):+.5f}   표준편차: {np.std(biases):.5f}")
print(f"  참고: phase59(구버전, recency 없음) = -0.0058 / +0.0168 / +0.0105")

log(f"\n총 {time.time()-t0:.0f}s")
