"""아이디어E 시드반복 확인 — idea13 파일럿에서 m=0.1,o=0.2가 유망했으나(+4.25) 단일시드라
신뢰불가. seed=7을 추가해 2시드 평균으로 재판정한다. seed=42 캐시(idea13_cache)는 재사용.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea13_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_PS = 15.0
SEEDS = [42, 7]
BEST = (0.1, 0.2)  # (w_multires, w_ordinal) 파일럿 최고 조합


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64)

log("투구단위 라벨 복원...")
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])

def _diff_label(rate_col):
    c = np.round(meta[rate_col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(meta))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab

lab_reverse = _diff_label("asof_pitcher_reverse_rate")
lab_middle = _diff_label("asof_pitcher_middle_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.08, n_iter_no_change=20)

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
    log(f"  v35local={sc(v35l):.2f}")

    p_mr_seeds, p_or_seeds = [], []
    for seed in SEEDS:
        f_mr = f"{CD}/{tag}_multires_s{seed}.npy"
        if os.path.exists(f_mr):
            p_mr = np.load(f_mr)
        else:
            sub_tr = meta.loc[tr_m, ["pitcher_id"]].copy()
            sub_tr["season"] = seasons[tr_m]
            sub_tr["sh"] = same_hand[tr_m]
            sub_tr["y"] = y[tr_m]
            ps = sub_tr.groupby(["pitcher_id", "season"])["y"].agg(s="sum", n="count")
            sub_tr = sub_tr.join(ps, on=["pitcher_id", "season"])
            g_tr = float(sub_tr["y"].mean())
            h1_tr = ((sub_tr["s"] - sub_tr["y"]) + K_PS * g_tr) / ((sub_tr["n"] - 1) + K_PS)
            psh = sub_tr.groupby(["pitcher_id", "season", "sh"])["y"].agg(s2="sum", n2="count")
            sub_tr = sub_tr.join(psh, on=["pitcher_id", "season", "sh"])
            h2_tr = ((sub_tr["s2"] - sub_tr["y"]) + K_PS * h1_tr) / ((sub_tr["n2"] - 1) + K_PS)
            h1_tr = h1_tr.to_numpy(np.float64); h2_tr = h2_tr.to_numpy(np.float64)
            Ymat_tr = np.column_stack([y[tr_m], h1_tr, h2_tr])
            ts = time.time()
            n_es = int(tr_m.sum() * 0.92)
            m = CatBoostRegressor(**CAT_PARAMS, random_seed=seed)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
                 eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
            p_mr = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f_mr, p_mr)
            log(f"    multires s{seed} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)  단독={sc(p_mr):.2f}")
        p_mr_seeds.append(p_mr)

        f_or = f"{CD}/{tag}_ordinal_s{seed}.npy"
        if os.path.exists(f_or):
            p_or = np.load(f_or)
        else:
            v1 = tr_m & valid_lab
            ts = time.time()
            m1 = HistGradientBoostingClassifier(**HGB_CLS, random_state=seed)
            m1.fit(X.loc[v1], (1 - lab_reverse[v1]), sample_weight=w[v1])
            not_rev = v1 & (lab_reverse == 0)
            m2 = HistGradientBoostingClassifier(**HGB_CLS, random_state=seed)
            m2.fit(X.loc[not_rev], (1 - lab_middle[not_rev]), sample_weight=w[not_rev])
            not_rev_mid = not_rev & (lab_middle == 0)
            m3 = HistGradientBoostingClassifier(**HGB_CLS, random_state=seed)
            m3.fit(X.loc[not_rev_mid], y[not_rev_mid], sample_weight=w[not_rev_mid])
            po1 = m1.predict_proba(X.loc[va_m])[:, 1]
            po2 = m2.predict_proba(X.loc[va_m])[:, 1]
            po3 = m3.predict_proba(X.loc[va_m])[:, 1]
            p_or = po1 * po2 * po3
            np.save(f_or, p_or)
            log(f"    ordinal s{seed} 학습완료 ({time.time()-ts:.0f}s)  단독={sc(p_or):.2f}")
        p_or_seeds.append(p_or)

    p_mr_avg = np.mean(p_mr_seeds, axis=0)
    p_or_avg = np.mean(p_or_seeds, axis=0)
    mr_spread = max(sc(p) for p in p_mr_seeds) - min(sc(p) for p in p_mr_seeds)
    or_spread = max(sc(p) for p in p_or_seeds) - min(sc(p) for p in p_or_seeds)
    wm, wo = BEST
    wb = 1 - wm - wo
    blend = wb * v35l + wm * p_mr_avg + wo * p_or_avg
    row = {"v35local": sc(v35l), "blend": sc(blend), "mr_spread": mr_spread, "or_spread": or_spread}
    log(f"  multires 시드폭={mr_spread:.2f}  ordinal 시드폭={or_spread:.2f}")
    log(f"  최종블렌드(m={wm},o={wo}, 2시드평균) = {row['blend']:.2f}  (v35l대비 {row['blend']-row['v35local']:+.2f})")
    results[tag] = row

print()
print("=" * 90)
print(f"{'fold':<6}{'v35local':>10}{'blend':>10}{'gain':>10}{'mr시드폭':>10}{'or시드폭':>10}")
for tag, r in results.items():
    print(f"{tag:<6}{r['v35local']:10.2f}{r['blend']:10.2f}{r['blend']-r['v35local']:+10.2f}"
         f"{r['mr_spread']:10.2f}{r['or_spread']:10.2f}")
gains_clean = [results[t]["blend"] - results[t]["v35local"] for t in ["A", "C"]]
gain_b = results["B"]["blend"] - results["B"]["v35local"]
max_spread = max(max(results[t]["mr_spread"], results[t]["or_spread"]) for t in ["A", "C"])
trustworthy = min(gains_clean) > max_spread
print(f"\nm=0.1,o=0.2 최종판정: 클린폴드(A,C) 최소이득={min(gains_clean):+.2f}  시드폭최대={max_spread:.2f}  "
     f"{'신뢰가능 -> 채택' if trustworthy else '신뢰불가 -> 기각'}  (참고 B={gain_b:+.2f})")
pd.DataFrame(results).T.to_csv("idea13b_results.csv")
log(f"총 {time.time()-t0:.0f}s")
