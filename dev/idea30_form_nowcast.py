"""아이디어A — 폼 나우캐스팅 (규정 준수 버전).
학습시점(train만): 그 투수의 향후 50투구 실제 성공률을 auxiliary head로 삼음(미래정보,
train 내부에서만 사용 -- 공식 학습데이터라 합법).
추론시점: head0(y)만 사용. head_form은 test 행에서 계산 안 함(계산 자체가 불필요 --
공유트리 정규화 신호로만 학습에 관여, multires와 동일 메커니즘).
Rule.md §4 준수: 어떤 test 행도 다른 test 행을 참조하지 않음.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea30_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
WINDOW = 50


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
rn = meta["row_num"].to_numpy()
g_global = float(y.mean())

CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

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

    # --- head_form: train 내부에서만, 향후 WINDOW투구 실제 성공률(자기 포함 중심 아님, 자기 이후) ---
    tr_idx = np.where(tr_m)[0]  # 전체 meta 기준 원본 위치 (역스캐터용)
    sub_tr = pd.DataFrame({"pid": pid[tr_m], "rn": rn[tr_m], "y": y[tr_m]}, index=tr_idx).sort_values(["pid", "rn"])
    grp = sub_tr.groupby("pid")["y"]
    # 순수 미래(자기 이후) WINDOW개 합/카운트: reverse -> shift(1) -> rolling -> reverse
    fut_sum = grp.transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).sum().iloc[::-1])
    fut_cnt = grp.transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).count().iloc[::-1])
    g_tr = float(sub_tr["y"].mean())
    head_form = (fut_sum + 10.0 * g_tr) / (fut_cnt + 10.0)
    head_form = head_form.fillna(g_tr)
    # 원래 순서로 되돌림 (sub_tr.index는 meta 전체 기준 원본 인덱스)
    hf_series = pd.Series(head_form.to_numpy(), index=sub_tr.index)
    head_form_full = hf_series.reindex(range(len(meta))).to_numpy(np.float64)
    head_form_tr = head_form_full[tr_m]

    Ymat_tr = np.column_stack([y[tr_m], head_form_tr])

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_formcast_s{seed}.npy"
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
    for wv in [0.1, 0.15, 0.2]:
        blend = (1 - wv) * v35l + wv * p_avg
        results.setdefault(tag, {})[f"w{wv}"] = sc(blend) - sc(v35l)
    results[tag]["v35local"] = sc(v35l)
    results[tag]["solo"] = sc(p_avg)
    results[tag]["spread"] = spread
    log(f"  formcast 2시드평균 단독={sc(p_avg):.2f}  시드폭={spread:.2f}")
    for wv in [0.1, 0.15, 0.2]:
        log(f"    w={wv} 이득={results[tag][f'w{wv}']:+.2f}")

print()
print("=" * 90)
print(f"{'fold':<6}{'단독':>10}{'시드폭':>8}" + "".join(f"{'w='+str(w):>9}" for w in [0.1, 0.15, 0.2]))
for tag, r in results.items():
    print(f"{tag:<6}{r['solo']:10.2f}{r['spread']:8.2f}" + "".join(f"{r[f'w{w}']:+9.2f}" for w in [0.1, 0.15, 0.2]))
gain_a = max(results["A"][f"w{w}"] for w in [0.1, 0.15, 0.2])
print(f"\n[신기준] 주검증 fold A 최고이득={gain_a:+.2f}  {'양수->통과후보' if gain_a>0 else '음수->기각'}")
log(f"총 {time.time()-t0:.0f}s")
