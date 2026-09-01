"""아이디어M — Bates-Granger 닫힌형해 최적선형결합. idea16(그리드서치, step 0.05)과 비교.
w* = argmin_w E[(sum(w_i p_i) - y)^2]  s.t. sum(w)=1, w>=0 (QP로 풀이)
fold A/C 각각 독립적으로 최적화 -> 두 폴드에서 얼마나 다른 w*가 나오는지도 확인(불안정성 진단).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CD = "idea13_cache"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def sc(p, yv):
    r = yv.mean(); BS = r * (1 - r)
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)


meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

members = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    va_m = seasons == val
    yv = y[va_m]
    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    p_mr = np.mean([np.load(f"{CD}/{tag}_multires_s{s}.npy") for s in [42, 7]], axis=0)
    p_or = np.mean([np.load(f"{CD}/{tag}_ordinal_s{s}.npy") for s in [42, 7]], axis=0)
    members[tag] = dict(yv=yv, P=np.column_stack([base3, hur, p_mr, p_or]))

NAMES = ["base", "hurdle", "multires", "ordinal"]


def solve_qp(P, yv):
    n = P.shape[1]

    def obj(w):
        pred = P @ w
        return np.mean((pred - yv) ** 2)

    def grad(w):
        pred = P @ w
        return 2.0 / len(yv) * (P.T @ (pred - yv))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


log("closed-form(QP) 최적화 (fold별 독립)...")
w_by_fold = {}
for tag in ["A", "C", "B"]:
    P, yv = members[tag]["P"], members[tag]["yv"]
    w = solve_qp(P, yv)
    w_by_fold[tag] = w
    pred = np.clip(P @ w, 0, 1)
    log(f"  fold {tag}: w={dict(zip(NAMES, np.round(w, 4)))}  score={sc(pred, yv):.2f}")

# idea16 그리드 최적점(0.30,0.40,0.10,0.20)을 fold별로 재평가 -> 비교
grid_w = np.array([0.30, 0.40, 0.10, 0.20])
log(f"\nidea16 그리드 최적점 {dict(zip(NAMES, grid_w))} 재평가:")
for tag in ["A", "C", "B"]:
    P, yv = members[tag]["P"], members[tag]["yv"]
    pred = np.clip(P @ grid_w, 0, 1)
    log(f"  fold {tag}: score={sc(pred, yv):.2f}")

# A+C 데이터를 합쳐서 하나의 공통 w*로 풀기 (fold별 불안정성 대신 안정적 절충)
log("\nfold A+C 결합(공통 w*) closed-form...")
P_ac = np.vstack([members["A"]["P"], members["C"]["P"]])
y_ac = np.concatenate([members["A"]["yv"], members["C"]["yv"]])
w_ac = solve_qp(P_ac, y_ac)
log(f"  결합 w*={dict(zip(NAMES, np.round(w_ac, 4)))}")
for tag in ["A", "C", "B"]:
    P, yv = members[tag]["P"], members[tag]["yv"]
    pred = np.clip(P @ w_ac, 0, 1)
    log(f"  fold {tag}: score={sc(pred, yv):.2f}")

log(f"\n총 {time.time()-t0:.0f}s")
