"""아이디어N(v46 검증) — Brier-native: HGBRegressor(squared_error), CatBoostRegressor(RMSE)를
기존 classifier(logloss)와 같은 조건(같은 피처, fold, seed)으로 직접 대조.
d6 설정 하나로 먼저 파일럿(시간순ES 적용), fold A/C 비교.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

CD = "idea25_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
ITER_GRID_HGB = [150, 200, 300]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    d6_cls = np.load(f"phase90_cache/{tag}_base_d6.npy")
    log(f"  v35local={sc(v35l):.2f}  d6(classifier,logloss)={sc(d6_cls):.2f}")

    X_tr, y_tr, w_tr = X.loc[tr_m], y[tr_m], w[tr_m]
    n = len(X_tr)
    cut = int(n * 0.92)
    tr_i, es_i = np.arange(cut), np.arange(cut, n)

    # --- HGB Regressor (squared_error), 시간순ES iter그리드 ---
    f_hgb = f"{CD}/{tag}_hgb_reg.npy"
    if os.path.exists(f_hgb):
        p_hgb_reg = np.load(f_hgb)
    else:
        best_score, best_iter = -1e18, ITER_GRID_HGB[0]
        for it in ITER_GRID_HGB:
            ts = time.time()
            m = HistGradientBoostingRegressor(max_depth=6, max_leaf_nodes=31, max_iter=it,
                                              learning_rate=0.03, l2_regularization=5.0,
                                              loss="squared_error", early_stopping=False, random_state=42)
            m.fit(X_tr.iloc[tr_i], y_tr[tr_i], sample_weight=w_tr[tr_i])
            p_es = np.clip(m.predict(X_tr.iloc[es_i]), 0, 1)
            yv_es = y_tr[es_i]
            r_es = yv_es.mean(); bs_es = r_es * (1 - r_es)
            s_es = 1e5 * (1 - np.mean((p_es - yv_es) ** 2) / bs_es)
            log(f"    hgb_reg iter={it}: 시간순검증={s_es:.2f} ({time.time()-ts:.0f}s)")
            if s_es > best_score:
                best_score, best_iter = s_es, it
        ts = time.time()
        m_full = HistGradientBoostingRegressor(max_depth=6, max_leaf_nodes=31, max_iter=best_iter,
                                               learning_rate=0.03, l2_regularization=5.0,
                                               loss="squared_error", early_stopping=False, random_state=42)
        m_full.fit(X_tr, y_tr, sample_weight=w_tr)
        p_hgb_reg = np.clip(m_full.predict(X.loc[va_m]), 0, 1)
        np.save(f_hgb, p_hgb_reg)
        log(f"  hgb_reg 최적iter={best_iter} 최종학습완료 ({time.time()-ts:.0f}s)  단독={sc(p_hgb_reg):.2f}")

    # --- CatBoost Regressor (RMSE), 시간순ES(내장 eval_set) ---
    f_cat = f"{CD}/{tag}_cat_reg.npy"
    if os.path.exists(f_cat):
        p_cat_reg = np.load(f_cat)
    else:
        ts = time.time()
        m = CatBoostRegressor(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, depth=6,
                              loss_function="RMSE", verbose=0, early_stopping_rounds=50, random_seed=42)
        m.fit(X_tr.iloc[tr_i], y_tr[tr_i], sample_weight=w_tr[tr_i],
             eval_set=(X_tr.iloc[es_i], y_tr[es_i]))
        p_cat_reg = np.clip(m.predict(X.loc[va_m]), 0, 1)
        np.save(f_cat, p_cat_reg)
        log(f"  cat_reg best_iter={m.best_iteration_} 완료 ({time.time()-ts:.0f}s)  단독={sc(p_cat_reg):.2f}")

    row = dict(v35local=sc(v35l), d6_cls=sc(d6_cls), hgb_reg=sc(p_hgb_reg), cat_reg=sc(p_cat_reg))
    # base3에서 d6 자리를 hgb_reg로 교체했을 때 전체블렌드 영향
    base3_swapped = np.mean([p_hgb_reg, np.load(f"phase90_cache/{tag}_base_d8.npy"),
                             np.load(f"phase90_cache/{tag}_base_sub.npy")], axis=0)
    v35l_swapped = 0.55 * base3_swapped + 0.45 * hur
    row["v35l_swapped_hgbreg"] = sc(v35l_swapped)
    log(f"  d6->hgb_reg 교체 스왑이득: {row['v35l_swapped_hgbreg']-row['v35local']:+.2f}")
    results[tag] = row

print()
print("=" * 100)
print(f"{'fold':<6}{'v35local':>10}{'d6_cls':>10}{'hgb_reg':>10}{'cat_reg':>10}{'스왑이득':>10}")
for tag, r in results.items():
    print(f"{tag:<6}{r['v35local']:10.2f}{r['d6_cls']:10.2f}{r['hgb_reg']:10.2f}{r['cat_reg']:10.2f}"
         f"{r['v35l_swapped_hgbreg']-r['v35local']:+10.2f}")
gains_clean = [results[t]["v35l_swapped_hgbreg"] - results[t]["v35local"] for t in ["A", "C"]]
gain_b = results["B"]["v35l_swapped_hgbreg"] - results["B"]["v35local"]
print(f"\n클린폴드(A,C) 최소이득={min(gains_clean):+.2f}  (참고 B={gain_b:+.2f})")
log(f"총 {time.time()-t0:.0f}s")
