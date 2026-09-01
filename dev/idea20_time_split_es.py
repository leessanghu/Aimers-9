"""아이디어H — HGB 조기종료(early_stopping) 검증셋을 랜덤(sklearn 기본)에서 시간순 뒤쪽
8%로 교체. sklearn 소스 확인: early_stopping=True일 때 train_test_split(랜덤,stratify)을
내부에서 쓴다. 시간 드리프트가 핵심인 이 데이터에서 랜덤 홀드아웃은 조기종료를
못 걸리게 만들 수 있다(phase16 로그: n_iter_=500=max_iter, 조기종료 미발동).

방법: early_stopping=False로 끄고, time_split_es()로 만든 시간순 뒤쪽 8%를 수동으로
n_iter_no_change 로직처럼 직접 구현하기는 sklearn API상 어려우므로, 대신
warm_start + staged 방식 대신 간단하게: max_iter를 시간순검증 기준으로 사전에 스윕해
최적 iter를 찾고, 그 값으로 고정 학습(no early stopping)한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea20_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
ITER_GRID = [100, 200, 300, 400, 500, 700, 900]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

BASE_PARAMS = dict(max_depth=6, max_leaf_nodes=31, learning_rate=0.03, l2_regularization=5.0)

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

    Xtr, ytr, wtr = X.loc[tr_m], y[tr_m], w[tr_m]
    n = len(Xtr)
    cut = int(n * 0.92)  # 시간순 앞 92% 학습, 뒤 8% 시간순 검증(time_split_es와 동일 비율)
    tr_i, es_i = np.arange(cut), np.arange(cut, n)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n_}.npy") for n_ in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n_}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n_}.npy")
                   for n_ in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    d6_random = np.load(f"phase90_cache/{tag}_base_d6.npy")  # 기존 랜덤ES d6 (참고 비교용)
    log(f"  v35local={sc(v35l):.2f}  기존d6(랜덤ES)={sc(d6_random):.2f}")

    for seed in SEEDS:
        f_iter = f"{CD}/{tag}_best_iter_s{seed}.npy"
        if os.path.exists(f_iter):
            best_iter = int(np.load(f_iter)[0])
        else:
            ts = time.time()
            best_score, best_iter = -1e18, ITER_GRID[0]
            for it in ITER_GRID:
                m = HistGradientBoostingClassifier(**BASE_PARAMS, max_iter=it, early_stopping=False,
                                                   random_state=seed)
                m.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=wtr[tr_i])
                p_es = m.predict_proba(Xtr.iloc[es_i])[:, 1]
                yv_es = ytr[es_i]
                r_es = yv_es.mean(); bs_es = r_es * (1 - r_es)
                s_es = 1e5 * (1 - np.mean((p_es - yv_es) ** 2) / bs_es)
                if s_es > best_score:
                    best_score, best_iter = s_es, it
            np.save(f_iter, np.array([best_iter]))
            log(f"    s{seed}: 시간순검증 최적 iter={best_iter} (score={best_score:.2f}, {time.time()-ts:.0f}s)")

        f_pred = f"{CD}/{tag}_timesplit_s{seed}.npy"
        if os.path.exists(f_pred):
            p = np.load(f_pred)
        else:
            ts = time.time()
            m_full = HistGradientBoostingClassifier(**BASE_PARAMS, max_iter=best_iter, early_stopping=False,
                                                     random_state=seed)
            m_full.fit(Xtr, ytr, sample_weight=wtr)
            p = m_full.predict_proba(X.loc[va_m])[:, 1]
            np.save(f_pred, p)
            log(f"    s{seed}: 전체데이터 재학습(iter={best_iter}) 완료 ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        results.setdefault(tag, {})[f"s{seed}_p"] = p
        results[tag][f"s{seed}_iter"] = best_iter

    p_avg = np.mean([results[tag][f"s{s}_p"] for s in SEEDS], axis=0)
    spread = max(sc(results[tag][f"s{s}_p"]) for s in SEEDS) - min(sc(results[tag][f"s{s}_p"]) for s in SEEDS)
    results[tag]["spread"] = spread
    results[tag]["v35local"] = sc(v35l)
    results[tag]["d6_random_score"] = sc(d6_random)
    results[tag]["timesplit_avg_score"] = sc(p_avg)
    log(f"  d6(랜덤ES) 단독={sc(d6_random):.2f}  d6(시간순ES) 2시드평균 단독={sc(p_avg):.2f}  "
       f"(차이 {sc(p_avg)-sc(d6_random):+.2f})  시드폭={spread:.2f}")

    # base3의 d6 슬롯만 시간순ES 버전으로 교체해서 전체 블렌드에 미치는 영향도 확인
    base3_swapped = np.mean([p_avg, np.load(f"phase90_cache/{tag}_base_d8.npy"),
                             np.load(f"phase90_cache/{tag}_base_sub.npy")], axis=0)
    v35l_swapped = 0.55 * base3_swapped + 0.45 * hur
    results[tag]["v35l_swapped"] = sc(v35l_swapped)
    log(f"  base3에서 d6만 시간순ES로 교체 -> v35local={sc(v35l_swapped):.2f}  (원래대비 {sc(v35l_swapped)-sc(v35l):+.2f})")

print()
print("=" * 100)
print(f"{'fold':<6}{'d6랜덤ES':>10}{'d6시간ES':>10}{'차이':>8}{'시드폭':>8}{'v35l원본':>10}{'v35l(d6교체)':>12}{'교체이득':>8}")
for tag, r in results.items():
    diff = r["timesplit_avg_score"] - r["d6_random_score"]
    swap_gain = r["v35l_swapped"] - r["v35local"]
    print(f"{tag:<6}{r['d6_random_score']:10.2f}{r['timesplit_avg_score']:10.2f}{diff:+8.2f}"
         f"{r['spread']:8.2f}{r['v35local']:10.2f}{r['v35l_swapped']:12.2f}{swap_gain:+8.2f}")
log(f"총 {time.time()-t0:.0f}s")
