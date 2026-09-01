"""아이디어K — v42 최적블렌드(base=.30,hur=.40,mr=.10,or=.20)에 4가지 보정 비교:
무보정 / Platt(logit 1피처) / Isotonic(idea17에서 폭주 확인됨, 재확인용) / Beta(log p, -log(1-p) 2피처).

idea17과 동일 구조: fold 내부를 row_num 기준 앞/뒤 반으로 나눠 앞에서 보정기 학습, 뒤에서 평가
(미래 데이터로 과거 보정 적용 -- 실전과 동일).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CD = "idea13_cache"
t0 = time.time()
EPS = 1e-15
W = (0.30, 0.40, 0.10, 0.20)  # base, hur, mr, or (v42 최적조합)


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def score(p, yv):
    p = np.clip(p, 0, 1)
    r = yv.mean(); BS = r * (1 - r)
    return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)


def brier(p, yv):
    return float(np.mean((np.clip(p, 0, 1) - yv) ** 2))


def logloss(p, yv):
    pc = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))


def ece(p, yv, n_bins=10):
    p = np.clip(p, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(p[m].mean() - yv[m].mean())
    return e


meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
row_num = meta["row_num"].to_numpy(np.float64)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    va_m = seasons == val
    yv = y[va_m]

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    p_mr = np.mean([np.load(f"{CD}/{tag}_multires_s{s}.npy") for s in [42, 7]], axis=0)
    p_or = np.mean([np.load(f"{CD}/{tag}_ordinal_s{s}.npy") for s in [42, 7]], axis=0)
    p_raw = np.clip(W[0] * base3 + W[1] * hur + W[2] * p_mr + W[3] * p_or, EPS, 1 - EPS)

    row_va = row_num[va_m]
    order = np.argsort(row_va)
    half = len(order) // 2
    fit_idx, eval_idx = order[:half], order[half:]
    p_fit, y_fit = p_raw[fit_idx], yv[fit_idx]
    p_eval, y_eval = p_raw[eval_idx], yv[eval_idx]

    row = {}

    row["raw"] = dict(score=score(p_eval, y_eval), brier=brier(p_eval, y_eval),
                      logloss=logloss(p_eval, y_eval), ece=ece(p_eval, y_eval),
                      mean_pred=p_eval.mean(), mean_true=y_eval.mean())

    logit_fit = np.log(p_fit / (1 - p_fit)).reshape(-1, 1)
    logit_eval = np.log(p_eval / (1 - p_eval)).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit_fit, y_fit)
    p_platt = platt.predict_proba(logit_eval)[:, 1]
    row["platt"] = dict(score=score(p_platt, y_eval), brier=brier(p_platt, y_eval),
                        logloss=logloss(p_platt, y_eval), ece=ece(p_platt, y_eval),
                        mean_pred=p_platt.mean(), mean_true=y_eval.mean())

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_fit, y_fit)
    p_iso = iso.predict(p_eval)
    row["isotonic"] = dict(score=score(p_iso, y_eval), brier=brier(p_iso, y_eval),
                           logloss=logloss(p_iso, y_eval), ece=ece(p_iso, y_eval),
                           mean_pred=p_iso.mean(), mean_true=y_eval.mean())

    Xb_fit = np.column_stack([np.log(p_fit), -np.log(1 - p_fit)])
    Xb_eval = np.column_stack([np.log(p_eval), -np.log(1 - p_eval)])
    beta = LogisticRegression(C=1e6, max_iter=1000).fit(Xb_fit, y_fit)
    p_beta = beta.predict_proba(Xb_eval)[:, 1]
    row["beta"] = dict(score=score(p_beta, y_eval), brier=brier(p_beta, y_eval),
                       logloss=logloss(p_beta, y_eval), ece=ece(p_beta, y_eval),
                       mean_pred=p_beta.mean(), mean_true=y_eval.mean())

    results[tag] = row
    log(f"fold {tag} (실제평균={y_eval.mean():.4f}):")
    for method in ["raw", "platt", "isotonic", "beta"]:
        r = row[method]
        log(f"  {method:10s} score={r['score']:9.2f}  brier={r['brier']:.5f}  logloss={r['logloss']:.5f}  "
           f"ece={r['ece']:.5f}  예측평균={r['mean_pred']:.4f}")

print()
print("=" * 100)
print(f"{'method':<10}{'A':>10}{'C':>10}{'B':>10}{'clean최소':>10}{'clean대비(raw)':>14}")
for method in ["raw", "platt", "isotonic", "beta"]:
    sa, sc_, sb = results["A"][method]["score"], results["C"][method]["score"], results["B"][method]["score"]
    min_ac = min(sa, sc_)
    diff = min_ac - min(results["A"]["raw"]["score"], results["C"]["raw"]["score"])
    print(f"{method:<10}{sa:10.2f}{sc_:10.2f}{sb:10.2f}{min_ac:10.2f}{diff:+14.2f}")
log(f"총 {time.time()-t0:.0f}s")
