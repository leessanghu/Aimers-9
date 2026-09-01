"""제4범주("기타") + 중증(reverse&middle 동시) aux head 스크리닝.

확립된 규칙: aux head 이득 ∝ 해당 축의 기존모델 미활용도
  middle SHAP122위/splits2  -> 실측 +7.72
  ball   SHAP  7위/splits41 -> 실측 +1.83
포렌식 발견: success+reverse+middle 합이 0인 13.17%가 "기타" 범주인데
대응 피처가 아예 없음 -> 미활용도 최대. 3.41%는 reverse&middle 동시(중증).

fold A 2시드 스크리닝. 판정: aux head 편향 +6.15 감안.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea33_cache"
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


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid))
tot = y + lab_rev + lab_mid  # 유효행에서만 의미

lab_other = np.where(valid, (tot == 0).astype(float), np.nan)   # 제4범주
lab_severe = np.where(valid, (tot == 2).astype(float), np.nan)  # reverse&middle 동시
log(f"  유효행 {valid.sum():,} ({valid.mean()*100:.2f}%)")
log(f"  기타(합=0)  발생률 {np.nanmean(lab_other)*100:.2f}%")
log(f"  중증(합=2)  발생률 {np.nanmean(lab_severe)*100:.2f}%")

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
mr = np.mean([np.load(f"idea13_cache/{tag}_multires_s{k}.npy") for k in [42, 7]], axis=0)
od = np.mean([np.load(f"idea13_cache/{tag}_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
# v47 구성을 기준으로 (실제 프로덕션 기준선과 일치)
v47l = 0.30 * base3 + 0.40 * hur + 0.10 * mr + 0.20 * od
log(f"fold A: v47local={sc(v47l):.2f}  (참고 v50=midaxis추가시 932.29)")

CANDS = [("other", lab_other), ("severe", lab_severe)]
results = {}
for name, lab in CANDS:
    log(f"===== {name} 축 =====")
    m_tr = tr_m & valid
    head = np.where(valid, 1.0 - lab, np.nan)  # 성공방향 정렬(해당사건 아님=1)
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
        rem = 1.0 - wv
        p = rem * 0.30 * base3 + rem * 0.40 * hur + rem * 0.10 * mr + rem * 0.20 * od + wv * avg
        row[f"w{wv}"] = sc(p) - sc(v47l)
    results[name] = row
    log(f"  {name}: 단독={row['solo']:.2f} 시드폭={spread:.2f}  "
       f"w0.1={row['w0.1']:+.2f} w0.15={row['w0.15']:+.2f} w0.2={row['w0.2']:+.2f}")

print()
print("=" * 80)
print(f"{'축':<10}{'단독':>10}{'시드폭':>8}" + "".join(f"{'w='+str(w):>9}" for w in [0.1, 0.15, 0.2]))
for name, r_ in results.items():
    print(f"{name:<10}{r_['solo']:10.2f}{r_['spread']:8.2f}" + "".join(f"{r_[f'w{w}']:+9.2f}" for w in [0.1, 0.15, 0.2]))
print()
print("참고: middle축 fold A +1.08 -> 실측 +7.72 / ball축 +0.43 -> 실측 +1.83")
for name, r_ in results.items():
    best = max(r_[f"w{w}"] for w in [0.1, 0.15, 0.2])
    print(f"  {name:<8} 최고 {best:+.2f} -> 보정추정 {best+6.15:+.2f}")
log(f"총 {time.time()-t0:.0f}s")
