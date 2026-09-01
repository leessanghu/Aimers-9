"""idea46 — middle + other 를 3-head 공유트리 하나로 통합.

동기: 현재 v58은 midaxis / otheraxis 를 독립 CatBoost 2개로 운용한다.
      두 축 모두 개별 실측 성공(+7.72 / +3.25)이고,
      다중head 통합(unified5)이 실측 +6.99로 성공한 전례가 있다.
      한 트리 안에서 두 축의 상호작용을 포착할 여지가 있는지 검증.

head0 = y (control_success)          <- 추론시 이것만 사용
head1 = 1 - lab_middle               (middle 아님 = 제구 양호 방향)
head2 = 1 - lab_other                (기타범주 아님)
복원불가 행은 NaN -> MultiRMSEWithMissingValues가 해당 head만 제외.

측정규약(local_leaderboard.py): fold A **전체2024**, 기준=v47local,
블렌딩=(1-w)x기존 + w x신규 (비례축소). 캘리브: 실측Δ = 6.29 + 5.02 x 로컬Δ.
비교대상: 같은 규약으로 mid(.10)+other(.10) 을 함께 넣은 구성(= v58 등가).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea46_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
CAL_A, CAL_B = 6.29, 5.02


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


def recover(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    co = c[order]
    d = np.empty(len(meta))
    d[:-1] = co[1:] - co[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid))
tot = y + lab_rev + lab_mid
lab_other = np.where(valid, (tot == 0).astype(np.float64), np.nan)
h1 = np.where(valid, 1.0 - lab_mid, np.nan)
h2 = np.where(valid, 1.0 - lab_other, np.nan)
log(f"  유효 {valid.sum():,} ({valid.mean()*100:.2f}%)  middle율={np.nanmean(lab_mid)*100:.2f}% "
    f"기타율={np.nanmean(lab_other)*100:.2f}%")

CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

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
    md = A([f"idea31_cache/{tag}_midaxis_s{k}.npy" for k in [42, 7]])
    v47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
    b47 = sc(v47)
    ot = A([f"idea33_cache/{tag}_other_s{k}.npy" for k in [42, 7]]) if tag == "A" else None
    log(f"  v47local={b47:.2f}")

    Ymat = np.column_stack([y, h1, h2])
    ps = []
    for seed in SEEDS:
        f = f"{CD}/{tag}_midother_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        ts = time.time()
        n_es = int(tr_m.sum() * 0.92)
        m = CatBoostRegressor(**CAT, random_seed=seed)
        m.fit(X.loc[tr_m].iloc[:n_es], Ymat[tr_m][:n_es], sample_weight=w[tr_m][:n_es],
              eval_set=(X.loc[tr_m].iloc[n_es:], Ymat[tr_m][n_es:]))
        p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
        np.save(f, p); ps.append(p)
        log(f"    s{seed} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    uni = np.mean(ps, axis=0)
    spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
    log(f"  통합3head 단독={sc(uni):.2f} 시드폭={spread:.2f}")

    row = {"solo": sc(uni), "spread": spread}
    for wv in [0.10, 0.15, 0.20]:
        row[f"uni{wv}"] = sc((1 - wv) * v47 + wv * uni) - b47
    row["mid.10"] = sc(0.90 * v47 + 0.10 * md) - b47
    if ot is not None:
        row["other.10"] = sc(0.90 * v47 + 0.10 * ot) - b47
        row["mid.10+other.10"] = sc(0.80 * v47 + 0.10 * md + 0.10 * ot) - b47
        row["uni.10+mid.10"] = sc(0.80 * v47 + 0.10 * uni + 0.10 * md) - b47
    results[tag] = row

print()
print("=" * 84)
print("middle+other 3-head 통합 vs 개별 운용 (규약: v47 기준, 비례축소, 전체시즌)")
print("=" * 84)
for tag, r in results.items():
    print(f"\n[fold {tag}]  통합3head 단독={r['solo']:.2f} 시드폭={r['spread']:.2f}")
    for k in ["mid.10", "other.10", "mid.10+other.10", "uni0.1", "uni0.15", "uni0.2", "uni.10+mid.10"]:
        if k in r:
            pred = CAL_A + CAL_B * r[k] if tag == "A" else float("nan")
            ps_ = f"{pred:+.2f}" if tag == "A" else "-"
            print(f"   {k:<20} 로컬Δ={r[k]:+7.2f}   예상실측Δ={ps_:>8}")
print()
print("참고 실측: mid단독 +7.72 / other단독 +3.25 / v58(둘 다) = 1077.15 (v50 대비 +3.25)")
log(f"총 {time.time()-t0:.0f}s")
