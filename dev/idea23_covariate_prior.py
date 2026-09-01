"""아이디어L(파일럿) — multires head1/head2의 prior를 전체평균(g_tr) 대신
'그 투수의 직전시즌 rate'(inseason.py와 동일 패턴, 없으면 전체평균)로 교체.
단일시드 빠른 파일럿 -> 유망하면 시드반복.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea23_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_PS = 15.0
SEED = 42


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64)

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50, random_seed=SEED)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
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
    log(f"  v35local={sc(v35l):.2f}")

    sub_tr = meta.loc[tr_m, ["pitcher_id"]].copy()
    sub_tr["season"] = seasons[tr_m]
    sub_tr["sh"] = same_hand[tr_m]
    sub_tr["y"] = y[tr_m]
    g_tr = float(sub_tr["y"].mean())

    # 직전시즌 rate 테이블 (train 데이터로만 구축)
    season_tbl = sub_tr.groupby(["pitcher_id", "season"])["y"].agg(S="sum", N="count").reset_index()
    piv_s = season_tbl.pivot(index="pitcher_id", columns="season", values="S")
    piv_n = season_tbl.pivot(index="pitcher_id", columns="season", values="N")
    all_seasons = sorted(season_tbl["season"].unique())
    piv_s = piv_s.reindex(columns=all_seasons).fillna(0).cumsum(axis=1)
    piv_n = piv_n.reindex(columns=all_seasons).fillna(0).cumsum(axis=1)
    piv_rate = (piv_s / piv_n.replace(0, np.nan))

    idx_prev = pd.MultiIndex.from_arrays([sub_tr["pitcher_id"], sub_tr["season"] - 1])
    prior_vals = piv_rate.stack(future_stack=True).reindex(idx_prev).to_numpy()
    prior_cov = pd.Series(prior_vals).fillna(g_tr).to_numpy(np.float64)
    log(f"  공변량prior: mean={prior_cov.mean():.4f}  std={prior_cov.std():.4f}  (기존 flat prior={g_tr:.4f})")

    # ---- 기존(flat mean prior) 재현 ----
    ps = sub_tr.groupby(["pitcher_id", "season"])["y"].agg(s="sum", n="count")
    sub_tr = sub_tr.join(ps, on=["pitcher_id", "season"])
    h1_flat = ((sub_tr["s"] - sub_tr["y"]) + K_PS * g_tr) / ((sub_tr["n"] - 1) + K_PS)
    psh = sub_tr.groupby(["pitcher_id", "season", "sh"])["y"].agg(s2="sum", n2="count")
    sub_tr = sub_tr.join(psh, on=["pitcher_id", "season", "sh"])
    h2_flat = ((sub_tr["s2"] - sub_tr["y"]) + K_PS * h1_flat) / ((sub_tr["n2"] - 1) + K_PS)
    h1_flat = h1_flat.to_numpy(np.float64); h2_flat = h2_flat.to_numpy(np.float64)

    # ---- 신규(공변량 prior) ----
    h1_cov = ((sub_tr["s"].to_numpy() - sub_tr["y"].to_numpy()) + K_PS * prior_cov) / \
             ((sub_tr["n"].to_numpy() - 1) + K_PS)
    h2_cov = ((sub_tr["s2"].to_numpy() - sub_tr["y"].to_numpy()) + K_PS * h1_cov) / \
             ((sub_tr["n2"].to_numpy() - 1) + K_PS)

    for variant, h1_, h2_ in [("flat", h1_flat, h2_flat), ("cov", h1_cov, h2_cov)]:
        f_out = f"{CD}/{tag}_{variant}.npy"
        if os.path.exists(f_out):
            p = np.load(f_out)
        else:
            Ymat_tr = np.column_stack([y[tr_m], h1_, h2_])
            ts = time.time()
            n_es = int(tr_m.sum() * 0.92)
            m = CatBoostRegressor(**CAT_PARAMS)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
                 eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
            p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f_out, p)
            log(f"    [{variant}] 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        results.setdefault(tag, {})[f"{variant}_solo"] = sc(p)
        for wv in [0.1, 0.15, 0.2]:
            blend = (1 - wv) * v35l + wv * p
            results[tag][f"{variant}_w{wv}"] = sc(blend)

    for wv in [0.1, 0.15, 0.2]:
        diff = results[tag][f"cov_w{wv}"] - results[tag][f"flat_w{wv}"]
        log(f"  w={wv}: flat={results[tag][f'flat_w{wv}']:.2f}  cov={results[tag][f'cov_w{wv}']:.2f}  "
           f"(공변량prior 개선분 {diff:+.2f})")

print()
print("=" * 90)
for tag in results:
    print(f"\nfold {tag}: flat단독={results[tag]['flat_solo']:.2f}  cov단독={results[tag]['cov_solo']:.2f}")
    for wv in [0.1, 0.15, 0.2]:
        print(f"  w={wv}: flat={results[tag][f'flat_w{wv}']:.2f}  cov={results[tag][f'cov_w{wv}']:.2f}  "
             f"diff={results[tag][f'cov_w{wv}']-results[tag][f'flat_w{wv}']:+.2f}")
t = time.time() - t0
print(f"\n[{t:5.0f}s] 총 {t:.0f}s")
