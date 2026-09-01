"""아이디어E — 확률공간 블렌드(idea13) 대신 로짓공간 가중평균으로 v40/v41을 합친다.

pred_E = sigmoid((1-a-b)*logit(base) + a*logit(multires) + b*logit(ordinal))

idea13b가 이미 캐싱한 2시드평균 예측(multires, ordinal)과 v35local(base)을 그대로
재사용 -- 추가 학습 없이 그리드서치만 한다. a,b 그리드는 사용자 제안 그대로:
a,b in [0.15,0.25,0.35,0.45], a+b<=0.65.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

CD = "idea13_cache"
t0 = time.time()
EPS = 1e-6
A_GRID = [0.15, 0.25, 0.35, 0.45]
B_GRID = [0.15, 0.25, 0.35, 0.45]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = np.clip(0.55 * base3 + 0.45 * hur, EPS, 1 - EPS)

    p_mr = np.mean([np.load(f"{CD}/{tag}_multires_s{s}.npy") for s in [42, 7]], axis=0)
    p_or = np.mean([np.load(f"{CD}/{tag}_ordinal_s{s}.npy") for s in [42, 7]], axis=0)

    lg_base, lg_mr, lg_or = logit(v35l), logit(p_mr), logit(p_or)
    row = {"v35local": sc(v35l)}
    for a in A_GRID:
        for b in B_GRID:
            if a + b > 0.65:
                continue
            z = (1 - a - b) * lg_base + a * lg_mr + b * lg_or
            row[f"a{a}_b{b}"] = sc(sigmoid(z))
    results[tag] = row
    log(f"fold {tag}: v35local={row['v35local']:.2f}")

combos = [(a, b) for a in A_GRID for b in B_GRID if a + b <= 0.65]
print()
print("=" * 100)
print(f"{'combo':<14}{'A':>10}{'C':>10}{'B':>10}{'클린최소이득':>14}{'참고B':>10}")
best = None
for a, b in combos:
    key = f"a{a}_b{b}"
    ga = results["A"][key] - results["A"]["v35local"]
    gc = results["C"][key] - results["C"]["v35local"]
    gb = results["B"][key] - results["B"]["v35local"]
    min_clean = min(ga, gc)
    print(f"a={a} b={b}  {results['A'][key]:10.2f}{results['C'][key]:10.2f}{results['B'][key]:10.2f}"
         f"{min_clean:14.2f}{gb:10.2f}")
    if best is None or min_clean > best[0]:
        best = (min_clean, a, b, gb)

print(f"\n최고조합: a={best[1]} b={best[2]}  클린폴드 최소이득={best[0]:+.2f}  (참고 B={best[3]:+.2f})")
print("\n--- idea13(확률공간 블렌드) 대비 비교용: m=0.1,o=0.2 확률공간 결과 ---")
print("(idea13b 로그 참조: fold A gain=+5.75, fold C는 대기 중이면 idea13b 결과 직접 확인)")

pd.DataFrame(results).T.to_csv("idea14_results.csv")
log(f"총 {time.time()-t0:.0f}s")
