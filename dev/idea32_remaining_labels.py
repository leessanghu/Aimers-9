"""middle축(v50, 실측 +7.72) 성공 레시피를 남은 복원 라벨에 확대 적용.
reverse: Hurdle에서 middle과 묶여(core_fail) 쓰였지 독립 aux head로는 미시도
ball/strike: 존(판정) 축 -- 커맨드 축과 다른 정보. 미시도

fold A만 먼저 스크리닝(2시드). aux head 편향 +6.15 감안해 fold A -5 이상이면 후보.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea32_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
LABELS = ["reverse", "ball", "strike"]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(rate_col):
    c = np.round(meta[rate_col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(meta))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab


labs = {}
for name in LABELS:
    labs[name] = recover(f"asof_pitcher_{name}_rate")
    v = ~np.isnan(labs[name])
    log(f"  {name}: 유효행 {v.sum():,} ({v.mean()*100:.2f}%), 발생률={np.nanmean(labs[name])*100:.2f}%")

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

upto, val, tag = 2023, 2024, "A"
tr_m = seasons <= upto
va_m = seasons == val
yv = y[va_m]
r = yv.mean(); BS = r * (1 - r)
w = 0.5 ** ((upto - seasons) / 2.0)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
               for n in ["d6", "d8"]], axis=0)
v35l = 0.55 * base3 + 0.45 * hur
log(f"fold A: v35local={sc(v35l):.2f}")

results = {}
for name in LABELS:
    log(f"===== {name} 축 =====")
    lab = labs[name]
    valid = ~np.isnan(lab)
    m_tr = tr_m & valid
    # head1 = 1 - label (성공방향으로 정렬)
    head = np.where(valid, 1.0 - lab, np.nan)
    Ymat_tr = np.column_stack([y[m_tr], head[m_tr]])

    ps = []
    for seed in SEEDS:
        f = f"{CD}/{tag}_{name}_s{seed}.npy"
        if os.path.exists(f):
            p = np.load(f)
        else:
            ts = time.time()
            n_es = int(m_tr.sum() * 0.92)
            mdl = CatBoostRegressor(**CAT_PARAMS, random_seed=seed)
            mdl.fit(X.loc[m_tr].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[m_tr][:n_es],
                   eval_set=(X.loc[m_tr].iloc[n_es:], Ymat_tr[n_es:]))
            p = np.clip(mdl.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f, p)
            log(f"    s{seed} 완료 best_iter={mdl.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
        ps.append(p)
    avg = np.mean(ps, axis=0)
    spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
    row = {"solo": sc(avg), "spread": spread}
    for wv in [0.1, 0.15, 0.2]:
        row[f"w{wv}"] = sc((1 - wv) * v35l + wv * avg) - sc(v35l)
    results[name] = row
    log(f"  {name}: 단독={row['solo']:.2f} 시드폭={spread:.2f}  "
       f"w0.1={row['w0.1']:+.2f} w0.15={row['w0.15']:+.2f} w0.2={row['w0.2']:+.2f}")

print()
print("=" * 90)
print(f"{'라벨':<10}{'단독':>10}{'시드폭':>8}" + "".join(f"{'w='+str(w):>9}" for w in [0.1, 0.15, 0.2]))
for name, r_ in results.items():
    print(f"{name:<10}{r_['solo']:10.2f}{r_['spread']:8.2f}" + "".join(f"{r_[f'w{w}']:+9.2f}" for w in [0.1, 0.15, 0.2]))
print()
print("참고: middle축(v50)은 fold A w0.1=+1.08 -> 실측 +7.72 (편향 +6.64)")
print("판정: aux head 편향 +6.15 감안, fold A 최고이득이 -5 이상이면 프로덕션 후보")
for name, r_ in results.items():
    best = max(r_[f"w{w}"] for w in [0.1, 0.15, 0.2])
    print(f"  {name:<10} 최고 {best:+.2f} -> 보정추정 {best+6.15:+.2f}  {'후보' if best > -5 else '기각'}")
log(f"총 {time.time()-t0:.0f}s")
