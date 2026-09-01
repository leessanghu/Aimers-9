"""idea47 — aux head 메커니즘 대조군. head0만 있는 평범한 CatBoostRegressor를
w=0.10으로 블렌드했을 때도 이득이 나는지 확인. 만약 그렇다면 지금까지의
"미활용도 규칙"/"aux head 편향" 이론은 그냥 앙상블 다양성(다른 loss/모델타입)의
사후 합리화일 수 있다. midaxis(head0=y, head1=1-lab_middle)와 완전히 동일한
설정(피처/파라미터/시드/가중치)에서 head1만 뺀 것이 유일한 차이.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea47_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

# midaxis와 동일 파라미터, loss만 단일head RMSE로
CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function="RMSE", early_stopping_rounds=50)

A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
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

    base = A([f"phase90_cache/{tag}_base_{n}.npy" for n in ["d6", "d8", "sub"]])
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) *
                   np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
    mr = A([f"idea13_cache/{tag}_multires_s{k}.npy" for k in [42, 7]])
    od = A([f"idea13_cache/{tag}_ordinal_s{k}.npy" for k in [42, 7]])
    v47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
    b47 = sc(v47)
    md = A([f"idea31_cache/{tag}_midaxis_s{k}.npy" for k in [42, 7]])
    log(f"  v47local={b47:.2f}  (참고: midaxis w0.10 델타 = {sc(0.90*v47+0.10*md)-b47:+.2f})")

    ps = []
    for seed in SEEDS:
        f = f"{CD}/{tag}_control_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        ts = time.time()
        n_es = int(tr_m.sum() * 0.92)
        m = CatBoostRegressor(**CAT, random_seed=seed)
        m.fit(X.loc[tr_m].iloc[:n_es], y[tr_m][:n_es], sample_weight=w[tr_m][:n_es],
              eval_set=(X.loc[tr_m].iloc[n_es:], y[tr_m][n_es:]))
        p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)
        np.save(f, p); ps.append(p)
        log(f"    s{seed} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    ctrl = np.mean(ps, axis=0)
    spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
    d = sc(0.90 * v47 + 0.10 * ctrl) - b47
    results[tag] = dict(solo=sc(ctrl), spread=spread, delta=d, mid_delta=sc(0.90*v47+0.10*md)-b47)
    log(f"  대조군(head0만) 단독={sc(ctrl):.2f} 시드폭={spread:.2f}  w0.10 델타={d:+.2f}")

print()
print("=" * 78)
print("결론: aux head(midaxis) vs 대조군(단일head, 동일설정) — w=0.10 로컬Δ")
print("=" * 78)
for tag, r in results.items():
    print(f"fold {tag}: 대조군Δ={r['delta']:+.2f}  midaxisΔ={r['mid_delta']:+.2f}  "
          f"차이={r['mid_delta']-r['delta']:+.2f}  (대조군 시드폭={r['spread']:.2f})")
print()
print("판정: 대조군Δ가 midaxisΔ와 비슷하면 -> aux head 메커니즘 무의미(다양성 효과일 뿐)")
print("      대조군Δ가 뚜렷이 작으면(또는 음수) -> aux head 메커니즘 실재 확인")
log(f"총 {time.time()-t0:.0f}s")
