"""가중치 동시 재최적화 — base(HGB+Cat), Hurdle, MultiRes(v40), Ordinal(v41) 4개 멤버를
손으로 하나씩 그리드서치하는 대신 심플렉스 전체를 동시에 탐색한다.
실측 검증된 K_PS=15 원본 multires/ordinal(idea13_cache, 2시드평균) 사용.
목적함수: fold A/C 중 최소값을 최대화(robust) + fold B 참고.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

CD = "idea13_cache"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

data = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    p_mr = np.mean([np.load(f"{CD}/{tag}_multires_s{s}.npy") for s in [42, 7]], axis=0)
    p_or = np.mean([np.load(f"{CD}/{tag}_ordinal_s{s}.npy") for s in [42, 7]], axis=0)

    data[tag] = dict(yv=yv, BS=BS, base=base3, hur=hur, mr=p_mr, ordv=p_or)


def sc(tag, p):
    d = data[tag]
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - d["yv"]) ** 2) / d["BS"])


def blend(tag, w):
    d = data[tag]
    return w[0] * d["base"] + w[1] * d["hur"] + w[2] * d["mr"] + w[3] * d["ordv"]


v35l_score = {t: sc(t, data[t]["base"] * 0.55 + data[t]["hur"] * 0.45) for t in ["A", "C", "B"]}
log(f"기준(v35local): A={v35l_score['A']:.2f}  C={v35l_score['C']:.2f}  B={v35l_score['B']:.2f}")

log("심플렉스 그리드서치 (0.05 단위)...")
step = 0.05
grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
best = None
results = []
for w_base in grid:
    for w_hur in grid:
        if w_base + w_hur > 1.0 + 1e-9:
            continue
        for w_mr in grid:
            if w_base + w_hur + w_mr > 1.0 + 1e-9:
                continue
            w_or = round(1.0 - w_base - w_hur - w_mr, 2)
            if w_or < 0 or w_or > 1.0:
                continue
            w = (w_base, w_hur, w_mr, w_or)
            sa = sc("A", blend("A", w))
            sc_ = sc("C", blend("C", w))
            min_ac = min(sa, sc_)
            results.append((min_ac, sa, sc_, w))
            if best is None or min_ac > best[0]:
                best = (min_ac, sa, sc_, w)

results.sort(key=lambda r: -r[0])
log(f"그리드 조합 수: {len(results)}")
print()
print("=" * 100)
print(f"{'rank':<5}{'min(A,C)':>10}{'A':>10}{'C':>10}{'B':>10}{'w_base':>8}{'w_hur':>8}{'w_mr':>8}{'w_or':>8}")
for i, (min_ac, sa, sc_, w) in enumerate(results[:15]):
    sb = sc("B", blend("B", w))
    print(f"{i+1:<5}{min_ac:10.2f}{sa:10.2f}{sc_:10.2f}{sb:10.2f}"
         f"{w[0]:8.2f}{w[1]:8.2f}{w[2]:8.2f}{w[3]:8.2f}")

print(f"\n기준선(v35local): min(A,C)={min(v35l_score['A'], v35l_score['C']):.2f}")
wb = best[3]
sb_best = sc("B", blend("B", wb))
print(f"\n최고조합: base={wb[0]} hur={wb[1]} mr={wb[2]} or={wb[3]}")
print(f"  A={best[1]:.2f} (v35l대비 {best[1]-v35l_score['A']:+.2f})")
print(f"  C={best[2]:.2f} (v35l대비 {best[2]-v35l_score['C']:+.2f})")
print(f"  B={sb_best:.2f} (v35l대비 {sb_best-v35l_score['B']:+.2f}, 참고용)")
log(f"총 {time.time()-t0:.0f}s")
