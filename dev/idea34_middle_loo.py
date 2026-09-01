"""middle축 형태변형 — idea31(이진 per-pitch 타겟)의 대안으로 투수-시즌 LOO
평활 middle율을 head1로 사용. 목적: per-pitch 이진 타겟은 노이즈가 커서
(이벤트 희소) 공유트리가 우연한 분할을 학습할 수 있음 -> 투수-시즌 단위로
평활한 연속값을 쓰면 같은 정보를 덜 노이즈있게 전달할 수 있는지 검증.

head0=y, head1=1-LOO(middle율 | 투수-시즌, K=15 사전분포 평활, 자기자신 제외)
idea31과 동일 fold A/C 스크리닝 절차, 동일 blend 가중치로 비교.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea34_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
K_PS = 15.0


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()

log("투구단위 middle 라벨 복원...")
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


lab_middle = _diff_label("asof_pitcher_middle_rate")
valid_lab = ~np.isnan(lab_middle)
log(f"  라벨 유효행 {valid_lab.sum()}/{len(meta)} ({valid_lab.mean()*100:.2f}%)")

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & valid_lab
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

    # 투수-시즌 LOO 평활 (train마스크 내에서만 집계 -> 학습마스크 밖 정보 누출 없음)
    g_mid = lab_middle[tr_m].mean()
    sub = pd.DataFrame({"pid": pid[tr_m], "season": seasons[tr_m], "mid": lab_middle[tr_m]})
    agg = sub.groupby(["pid", "season"])["mid"].agg(s="sum", n="count")
    sub = sub.join(agg, on=["pid", "season"])
    loo = ((sub["s"] - sub["mid"]) + K_PS * g_mid) / ((sub["n"] - 1) + K_PS)
    h_loo_tr = loo.to_numpy(np.float64)
    head1_tr = 1.0 - h_loo_tr

    Ymat_tr = np.column_stack([y[tr_m], head1_tr])

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_midloo_s{seed}.npy"
        if os.path.exists(f_out):
            p = np.load(f_out)
        else:
            ts = time.time()
            n_es = int(tr_m.sum() * 0.92)
            m = CatBoostRegressor(**CAT_PARAMS, random_seed=seed)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
                 eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
            p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f_out, p)
            log(f"    s{seed} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        p_seeds.append(p)

    p_avg = np.mean(p_seeds, axis=0)
    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    row = {"solo": sc(p_avg), "spread": spread}
    for wv in [0.05, 0.1, 0.15, 0.2]:
        blend = (1 - wv) * v35l + wv * p_avg
        row[f"w{wv}"] = sc(blend) - sc(v35l)
    results[tag] = row
    log(f"  fold {tag}: 단독={row['solo']:.2f} 시드폭={spread:.2f}  "
       f"w0.05={row['w0.05']:+.2f} w0.1={row['w0.1']:+.2f} w0.15={row['w0.15']:+.2f} w0.2={row['w0.2']:+.2f}")

print()
print("=" * 80)
print("idea31(이진 per-pitch) 대비 idea34(투수-시즌 LOO 평활) 비교")
print(f"참고 idea31: fold A w0.1=+1.08(시드폭1.54) fold C w0.1=+7.68(시드폭10.92)")
for tag, r_ in results.items():
    print(f"fold {tag}: 단독={r_['solo']:.2f} 시드폭={r_['spread']:.2f}  "
         f"w0.05={r_['w0.05']:+.2f} w0.1={r_['w0.1']:+.2f} w0.15={r_['w0.15']:+.2f} w0.2={r_['w0.2']:+.2f}")
log(f"총 {time.time()-t0:.0f}s")
