"""기각된 아이디어들의 점수를 '신호' vs '보정'으로 분해한다.

항등식:
    Score/1e5 = (2Cov(p,y) - Var(p) - bias^2) / BSref

affine 재보정 p* = a + b*p 로 도달 가능한 최대 점수는 정확히:
    Score_potential = 1e5 * rho^2      (rho = corr(p, y))

여러 모델을 선형결합했을 때의 상한도 같은 논리로:
    Score_pair_max = 1e5 * R^2         (y ~ 1 + p1 + p2 회귀의 R^2)

따라서 delta가 음수였던 아이디어라도 rho^2 / R^2 가 높으면 '신호가 없어서'가 아니라
'보정이 망가져서' 진 것이고, 보정을 분리하는 구조로 살릴 수 있다.
"""
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

SEARCH_DIRS = [".", ".."]


def find(pattern):
    out = []
    for d in SEARCH_DIRS:
        out += sorted(glob.glob(os.path.join(d, pattern)))
    # 중복 basename 제거 (dev/ 우선)
    seen, uniq = set(), []
    for p in out:
        b = os.path.basename(p)
        if b not in seen:
            seen.add(b)
            uniq.append(p)
    return uniq


def decompose(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    r = y.mean()
    bsref = r * (1 - r)
    bs = np.mean((p - y) ** 2)
    score = max(0.0, 1e5 * (1 - bs / bsref))

    bias = p.mean() - r
    varp = p.var()
    cov = np.mean((p - p.mean()) * (y - r))
    rho = cov / (p.std() * y.std()) if p.std() > 0 else 0.0
    b_opt = cov / varp if varp > 0 else 0.0

    return {"score": score, "potential": 1e5 * rho ** 2, "rho": rho, "bias": bias,
            "bias_pen": 1e5 * bias ** 2 / bsref, "b_opt": b_opt, "pred_std": p.std()}


def pair_max(y, preds):
    """여러 예측의 최적 선형결합으로 도달 가능한 최대 점수 = 1e5 * R^2."""
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(p, dtype=np.float64) for p in preds])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1 - resid.var() / y.var()
    return 1e5 * r2, coef


def show(rows):
    hdr = (f"{'모델/실험':<32}{'score':>9}{'잠재력':>9}{'차이':>8}"
           f"{'rho':>8}{'bias':>9}{'bias손실':>9}{'b_opt':>7}{'pred_std':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, d in rows:
        print(f"{name:<32}{d['score']:9.1f}{d['potential']:9.1f}{d['potential']-d['score']:+8.1f}"
              f"{d['rho']:8.4f}{d['bias']:+9.4f}{d['bias_pen']:9.1f}{d['b_opt']:7.3f}{d['pred_std']:9.4f}")


def main():
    # ---- 2024 폴드 y 확보 ----
    y2024 = None
    for path in find("phase3_tabm_preds/fold_2024_pred_*.csv") + find("phase3_preds/fold_2024_pred_*.csv"):
        d = pd.read_csv(path)
        if "y_valid" in d.columns:
            y2024 = d["y_valid"].to_numpy()
            print(f"y_valid(2024) 로드: {path}  n={len(y2024):,}  mean={y2024.mean():.4f}")
            break
    if y2024 is None:
        raise SystemExit("2024 폴드 y_valid를 찾지 못함")

    named = {}   # name -> pred array (2024 폴드 한정)

    gp = "phase61_cache/gbdt_v25_valid_pred.npy"
    if os.path.exists(gp):
        p = np.load(gp)
        if len(p) == len(y2024):
            named["GBDT v25 (현 최고, LB 981.44)"] = p

    for path in find("phase3_preds/fold_2024_pred_*.csv") + find("phase3_tabm_preds/fold_2024_pred_*.csv"):
        d = pd.read_csv(path)
        col = [c for c in d.columns if c.startswith("pred")]
        if col and "y_valid" in d.columns and len(d) == len(y2024):
            nm = os.path.basename(path).replace("fold_2024_pred_", "").replace(".csv", "")
            named[f"{nm} (구버전)"] = d[col[0]].to_numpy()

    for path in find("phase61_cache/nn_*_valid_pred.npy"):
        p = np.load(path)
        if len(p) == len(y2024):
            nm = os.path.basename(path).replace("nn_", "").replace("_valid_pred.npy", "")
            named[f"phase61 {nm} (멀티태스크+recency)"] = p

    rp = "phase61_cache/residual_boost_valid_pred.npy"
    if os.path.exists(rp):
        p = np.load(rp)
        if len(p) == len(y2024):
            named["GBDT + residual-NN"] = p

    for path in find("phase60_cache/*_valid_pred.npy"):
        p = np.load(path)
        if len(p) == len(y2024):
            nm = os.path.basename(path).replace("_valid_pred.npy", "")
            named[f"phase60 {nm}"] = p

    print()
    print("=" * 96)
    print("2024 폴드 — 단일 모델 분해")
    print("=" * 96)
    rows = [(n, decompose(y2024, p)) for n, p in named.items()]
    rows.sort(key=lambda kv: -kv[1]["potential"])
    show(rows)

    # ---- 블렌드 상한 ----
    gk = next((k for k in named if k.startswith("GBDT v25")), None)
    if gk and len(named) > 1:
        print()
        print("=" * 96)
        print("GBDT v25와의 결합 상한 (최적 선형결합 = 도달 가능한 최대치)")
        print("=" * 96)
        base_sc = decompose(y2024, named[gk])["potential"]
        print(f"{'상대 모델':<40}{'단독잠재력':>11}{'결합상한':>10}{'추가이득':>10}{'오차상관':>10}")
        print("-" * 81)
        pg = named[gk]
        eg = y2024 - pg
        for n, p in named.items():
            if n == gk:
                continue
            solo = decompose(y2024, p)["potential"]
            mx, _ = pair_max(y2024, [pg, p])
            ec = np.corrcoef(eg, y2024 - p)[0, 1]
            print(f"{n:<40}{solo:11.1f}{mx:10.1f}{mx-base_sc:+10.1f}{ec:10.4f}")

        others = [p for n, p in named.items() if n != gk]
        if len(others) >= 2:
            mx, coef = pair_max(y2024, [pg] + others)
            print(f"\n전체 결합 상한: {mx:.1f}  (GBDT 단독 잠재력 {base_sc:.1f} 대비 {mx-base_sc:+.1f})")

    print()
    print("읽는 법:")
    print("  잠재력   = affine 재보정 후 최대 점수 (1e5*rho^2). 그 예측이 가진 '진짜 신호'")
    print("  차이     = 잠재력 - score. 보정만 고쳐서 얻을 수 있는 점수")
    print("  b_opt    = 최적 스케일. <1 이면 과신(예측 폭을 줄여야), >1 이면 과소")
    print("  결합상한 = 두 모델 최적 선형결합의 최대 점수. GBDT 단독 잠재력보다 크면 NN이 새 정보를 가진 것")
    print("  오차상관 = 두 모델 잔차의 상관. 낮을수록 다양성이 커서 블렌딩 이득이 큼")


if __name__ == "__main__":
    main()
