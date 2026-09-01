"""블렌딩 가중치 재검토 — 고정 0.5:0.5 vs (a) 정규화 로지스틱 스태킹 (b) 직접 Brier 최소화
가중치 탐색. phase55의 스태킹이 극단적 가중치(+14/-10)로 깨졌던 건 HGB/CatBoost 예측이
너무 상관(~0.93)돼서, 무정규화(약한 정규화) 로지스틱이 두 예측의 미세한 공분산 차이를
과도하게 활용하려다 held-out(ei)엔 맞고 실제 valid(2024)엔 안 맞는 방향으로 터진 것으로 보인다.

여기선 (a) C를 촘촘히 스윕한 로지스틱 스태킹과 (b) [0,1] 제약된 blend weight를 직접
grid search로 Brier 최소화하는 방식을 같이 비교해서 고정 0.5:0.5 대비 실제 개선이 있는지 본다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
SEEDS = [42, 7, 10]

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
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
gr = build_global_rates(df)
dly = transform_lastyear(df, build_lastyear_table(df), gr, sr, k=30.0)
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
print(f"베이스 피처 준비 완료 ({time.time()-t0:.0f}s)", flush=True)


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def best_weight_search(y_ho, p_h_ho, p_c_ho):
    ws = np.linspace(0.0, 1.0, 101)
    briers = [brier(y_ho, w * p_h_ho + (1 - w) * p_c_ho) for w in ws]
    i = int(np.argmin(briers))
    return float(ws[i]), briers[i]


for seed in SEEDS:
    fold = build_fold(df, 2023, 2024, extra_features=None, seed=seed, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
    ti, ei = time_split_es(len(Xtr))

    h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=seed).fit(Xtr, ytr)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=seed, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))

    p_h_va, p_c_va = h.predict_proba(Xva)[:, 1], cb.predict_proba(Xva)[:, 1]
    p_h_ho, p_c_ho = h.predict_proba(Xtr.iloc[ei])[:, 1], cb.predict_proba(Xtr.iloc[ei])[:, 1]
    y_ho = ytr[ei]

    s_fixed = max(0, evaluate(yva, 0.5 * p_h_va + 0.5 * p_c_va)["bss"] * 1e5)

    w_best, br_ho = best_weight_search(y_ho, p_h_ho, p_c_ho)
    p_wsearch = w_best * p_h_va + (1 - w_best) * p_c_va
    s_wsearch = max(0, evaluate(yva, p_wsearch)["bss"] * 1e5)

    best_c, best_c_score, best_c_coef = None, -1, None
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        meta = LogisticRegression(C=C).fit(np.column_stack([p_h_ho, p_c_ho]), y_ho)
        p_stack = meta.predict_proba(np.column_stack([p_h_va, p_c_va]))[:, 1]
        s = max(0, evaluate(yva, p_stack)["bss"] * 1e5)
        if s > best_c_score:
            best_c, best_c_score, best_c_coef = C, s, meta.coef_[0]

    print(f"seed={seed:4d}  fixed(0.5/0.5)={s_fixed:.1f}  "
          f"weight_search(w_hgb={w_best:.2f})={s_wsearch:.1f}  "
          f"logistic(best C={best_c}, coef={best_c_coef})={best_c_score:.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

print(f"\n총 {time.time()-t0:.0f}s", flush=True)
