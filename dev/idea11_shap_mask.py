"""아이디어 11 — SHAP 상위 K개 피처 마스킹 (phase94_shap_magnitude.csv 기반).

idea1(D_ablate, x_ability_here 제거) 결과: 제거하면 3폴드 다 손해(-2~-12).
'비중은 늘어도 온전히 대체는 못 한다'는 게 이미 확인됨. 하지만 idea8의 논리(단독은
나빠져도 다양성으로 앙상블엔 도움될 수 있음)를 SHAP 크기 기준으로도 검증한다.
idea8은 '의미 블록'(trackman/batter 등) 기준, 이건 'SHAP 크기' 기준이라 다른 축이다.

참조: phase94는 season 2023~2024 표본 30만행으로 CatBoost 3변종 평균 SHAP -> 근사.
train<=upto 로만 다시 뽑는 게 이상적이나 시간상 기존 랭킹 재사용(리스크: 약한 근사).

*** v38/v39 교훈: 시드 2개 반복, fold A/C 우선 판정, fold B는 참고만. ***
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea11_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
TOPKS = [10, 20, 40]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

shap_rank = pd.read_csv("phase94_shap_magnitude.csv", index_col=0)["magnitude"].sort_values(ascending=False)
shap_rank = shap_rank[shap_rank.index.isin(X.columns)]
log(f"  SHAP 랭킹 {len(shap_rank)}개 로드, 상위5: {list(shap_rank.index[:5])}")

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)

HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)


def fit_pred(tag, name, drop_cols, seed, tr_m, va_m, w):
    f = f"{CD}/{tag}_{name}_s{seed}.npy"
    if os.path.exists(f):
        return np.load(f)
    Xa = X.drop(columns=drop_cols) if drop_cols else X
    p = dict(HGB); p["random_state"] = seed
    ts = time.time()
    m = HistGradientBoostingClassifier(**p).fit(Xa.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
    pr = m.predict_proba(Xa.loc[va_m])[:, 1]
    np.save(f, pr)
    log(f"    {name}/s{seed} iters={m.n_iter_} feat={Xa.shape[1]} ({time.time()-ts:.0f}s)")
    return pr


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    base_seeds = [fit_pred(tag, "full", None, s, tr_m, va_m, w) for s in SEEDS]
    base_scores = [sc(p) for p in base_seeds]
    base_avg = np.mean(base_seeds, axis=0)
    seed_spread = max(base_scores) - min(base_scores)
    log(f"  [기준선] 시드별 {[round(x,2) for x in base_scores]}  시드폭={seed_spread:.2f}  "
        f"2시드평균={sc(base_avg):.2f}")

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    row = dict(seed_spread=seed_spread, base_avg=sc(base_avg), v35local=sc(v35l))

    for K in TOPKS:
        drop = list(shap_rank.index[:K])
        ps = [fit_pred(tag, f"top{K}", drop, s, tr_m, va_m, w) for s in SEEDS]
        avg = np.mean(ps, axis=0)
        corr = np.corrcoef(avg, base_avg)[0, 1]
        log(f"  [top{K}마스킹] 단독={sc(avg):.2f}  corr(기준선)={corr:.4f}")
        row[f"top{K}_solo"] = sc(avg)
        row[f"top{K}_corr"] = corr
        for wv in [0.15, 0.3]:
            ens = 0.5 * base_avg + 0.5 * avg   # 기준선 + 마스킹모델 앙상블
            blend = (1 - wv) * v35l + wv * ens
            row[f"top{K}_w{wv}"] = sc(blend)
            log(f"    v35local+(기준선+top{K}마스킹앙상블)(w={wv}) = {row[f'top{K}_w{wv}']:.2f}  "
               f"(v35l대비 {row[f'top{K}_w{wv}']-sc(v35l):+.2f})")
    results[tag] = row

print()
print("=" * 90)
for K in TOPKS:
    print(f"\n--- top{K} 마스킹 ---")
    print(f"{'fold':<6}{'시드폭':>9}{'단독':>9}{'corr':>8}{'v35local':>10}" +
         "".join(f"{'w='+str(w):>9}" for w in [0.15, 0.3]))
    for tag, r in results.items():
        print(f"{tag:<6}{r['seed_spread']:9.2f}{r[f'top{K}_solo']:9.2f}{r[f'top{K}_corr']:8.4f}"
             f"{r['v35local']:10.2f}" + "".join(f"{r[f'top{K}_w{w}']:9.2f}" for w in [0.15, 0.3]))
    for wv in [0.15, 0.3]:
        gains_clean = [results[t][f"top{K}_w{wv}"] - results[t]["v35local"] for t in ["A", "C"]]
        spreads = [results[t]["seed_spread"] for t in ["A", "C"]]
        ok = min(gains_clean) > max(spreads)
        print(f"  w={wv}: 클린폴드 최소이득={min(gains_clean):+.2f}  시드폭최대={max(spreads):.2f}  "
             f"{'유효' if ok else '신뢰불가'}")
pd.DataFrame(results).T.to_csv("idea11_results.csv")
log(f"총 {time.time()-t0:.0f}s")
