"""아이디어 10 — 음의 상관 학습 (Negative Correlation Learning, 다양성 명시적 강제).

p1이 학습된 뒤, p2의 타겟을 y' = y + λ*(y - p1)로 바꿔 p1이 틀린 방향을 증폭한다.
p1이 이미 잘 맞히는 곳은 정보가 적고(y-p1≈0), 못 맞히는 곳에 p2가 강제로 집중하게 된다.

주의(간소화): 엄밀하게는 y' 생성에 쓰는 p1 예측이 OOF(교차적합)여야 새는 게 없는데,
시간 제약상 이번 파일럿은 p1을 tr_m에 학습해 tr_m 자체에 in-sample 예측한 값으로 y'을
만든다(약한 형태의 정보 누출 위험 -- GBDT가 노이즈 타겟에 완벽적합은 안 하므로 실용적
근사로 채택하되, 결과가 좋으면 정식 OOF 버전으로 재검증 필요).

*** v38/v39 교훈: 판정은 fold A/C 우선, fold B는 참고만. 단일 시드 파일럿 -> 유망하면
    이어서 시드반복 검증. ***
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

CD = "idea10_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
LAMBDAS = [0.3, 0.6, 1.0]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)

HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)
HGB_REG = dict(HGB_CLS, loss="squared_error")

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    f1 = f"{CD}/{tag}_p1.npy"
    f1tr = f"{CD}/{tag}_p1_insample.npy"
    if os.path.exists(f1):
        p1_va = np.load(f1); p1_tr = np.load(f1tr)
    else:
        ts = time.time()
        m1 = HistGradientBoostingClassifier(**HGB_CLS).fit(X.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
        p1_va = m1.predict_proba(X.loc[va_m])[:, 1]
        p1_tr = m1.predict_proba(X.loc[tr_m])[:, 1]
        np.save(f1, p1_va); np.save(f1tr, p1_tr)
        log(f"  p1 학습완료 iters={m1.n_iter_} ({time.time()-ts:.0f}s)")
    log(f"  p1 단독 score={sc(p1_va):.2f}")

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    row = dict(p1=sc(p1_va), v35local=sc(v35l))

    y_tr = y[tr_m]
    for lam in LAMBDAS:
        f2 = f"{CD}/{tag}_p2_lam{lam}.npy"
        if os.path.exists(f2):
            p2_va = np.load(f2)
        else:
            # 클립하지 않는다: 이진타겟에 클립을 걸면 y+lam*(y-p1)이 항상 0/1로
            # 되돌아가버려 lam이 무의미해지는 버그가 있었다(1차 시도에서 확인됨).
            # 회귀(squared_error)라 [0,1] 밖 타겟도 학습에 문제없다. 클립은 최종
            # 예측 출력(p2_va)에서만 건다.
            y_amp = y_tr + lam * (y_tr - p1_tr)
            ts = time.time()
            m2 = HistGradientBoostingRegressor(**HGB_REG).fit(X.loc[tr_m], y_amp, sample_weight=w[tr_m])
            p2_va = np.clip(m2.predict(X.loc[va_m]), 0, 1)
            np.save(f2, p2_va)
            log(f"    lam={lam} p2 학습완료 iters={m2.n_iter_} ({time.time()-ts:.0f}s)")
        corr = np.corrcoef(p1_va, p2_va)[0, 1]
        avg = 0.5 * p1_va + 0.5 * p2_va
        log(f"  lam={lam}: p2단독={sc(p2_va):.2f}  corr(p1,p2)={corr:.4f}  p1+p2평균={sc(avg):.2f}")
        row[f"lam{lam}_p2"] = sc(p2_va)
        row[f"lam{lam}_corr"] = corr
        for wv in [0.1, 0.2, 0.3]:
            blend = (1 - wv) * v35l + wv * avg
            row[f"lam{lam}_w{wv}"] = sc(blend)
            log(f"    v35local+avg(w={wv}) = {row[f'lam{lam}_w{wv}']:.2f}  "
               f"(v35l대비 {row[f'lam{lam}_w{wv}']-sc(v35l):+.2f})")
    results[tag] = row

print()
print("=" * 90)
for lam in LAMBDAS:
    print(f"\n--- lambda={lam} ---")
    print(f"{'fold':<6}{'p2단독':>9}{'corr(p1,p2)':>12}{'v35local':>10}" +
         "".join(f"{'w='+str(w):>9}" for w in [0.1, 0.2, 0.3]))
    for tag, r in results.items():
        print(f"{tag:<6}{r[f'lam{lam}_p2']:9.2f}{r[f'lam{lam}_corr']:12.4f}{r['v35local']:10.2f}" +
             "".join(f"{r[f'lam{lam}_w{w}']:9.2f}" for w in [0.1, 0.2, 0.3]))
    for wv in [0.1, 0.2, 0.3]:
        gains_clean = [results[t][f"lam{lam}_w{wv}"] - results[t]["v35local"] for t in ["A", "C"]]
        gain_b = results["B"][f"lam{lam}_w{wv}"] - results["B"]["v35local"]
        print(f"  w={wv}: 클린폴드(A,C) 최소이득={min(gains_clean):+.2f}  (참고 B={gain_b:+.2f})")
pd.DataFrame(results).T.to_csv("idea10_results.csv")
log(f"총 {time.time()-t0:.0f}s")
