"""검증 지표 재설계 — 연도 난이도를 상쇄하는 지표를 실측 이력으로 골라낸다.

문제: fold A(2024)는 그 해 투수 실력분산이 0.001893으로 최저라 Resolution 상한 자체가
낮고, 8-10월(저Resolution 구간)을 30% 포함한다. fold B는 레벨드리프트로 Reliability가
fold A의 36배라 -804점. 즉 **BSS 원값은 연도 난이도에 심하게 오염**돼 있다.

착안: 보정 경로는 이미 닫혔다(v48 프로브로 최대 +2.4 확인). 남은 이득은 전부
Resolution이다. 그러면 Reliability를 제거하고 Resolution만 재면 연도 난이도가
상쇄되고, 여러 폴드를 합칠 수도 있다.

검증법: 실측 LB 델타를 아는 aux head 5종으로 각 후보지표의 Pearson 상관을 잰다.
현행 지표(fold A 전체2024 BSS델타)의 Pearson +0.818을 이겨야 채택.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()
KNOWN_LB = {"midaxis": 7.72, "other": 3.25, "ball": 1.83, "strike": 0.20, "unified5": 6.99}

meta = pd.read_parquet("featcache_meta.parquet")
mo = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                 usecols=["game_month"])["game_month"].to_numpy()
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)

FOLD = {"A": 2024, "B": 2023, "C": 2022}
CAND_SRC = {
    "midaxis": ("idea31_cache", "midaxis", [42, 7]),
    "other": ("idea33_cache", "other", [42, 7]),
    "ball": ("idea32_cache", "ball", [42, 7]),
    "strike": ("idea32_cache", "strike", [42, 7, 2024]),
    "unified5": ("idea12_cache", "head0", [42, 7]),
}


def v47(tag):
    b = A([f"phase90_cache/{tag}_base_{n}.npy" for n in ["d6", "d8", "sub"]])
    h = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) *
                 np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
    m = A([f"idea13_cache/{tag}_multires_s{k}.npy" for k in [42, 7]])
    o = A([f"idea13_cache/{tag}_ordinal_s{k}.npy" for k in [42, 7]])
    return 0.30 * b + 0.40 * h + 0.10 * m + 0.20 * o


def load_cand(tag, name):
    d, stem, seeds = CAND_SRC[name]
    import os
    ps = [f"{d}/{tag}_{stem}_s{k}.npy" for k in seeds]
    ps = [p for p in ps if os.path.exists(p)]
    return A(ps) if ps else None


def bss(p, yv):
    r = yv.mean(); U = r * (1 - r)
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / U)


def bss_decal(p, yv):
    """Reliability 제거: 예측 평균을 검증셋 평균에 맞춘 뒤 BSS. Resolution만 남음."""
    p = np.clip(p, 0, 1)
    p2 = np.clip(p - (p.mean() - yv.mean()), 0, 1)
    return bss(p2, yv)


def resolution(p, yv, nb=50):
    """Murphy Resolution 직접 계산 (분위 50개)."""
    r = yv.mean()
    q = pd.qcut(pd.Series(np.clip(p, 0, 1)), nb, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": yv, "q": q})
    g = d.groupby("q")["y"].agg(n="size", ym="mean")
    return float((g["n"] * (g["ym"] - r) ** 2).sum() / len(p) / (r * (1 - r)) * 1e5)


# --- 각 후보에 대해 여러 지표로 델타 계산 ---
METRICS = {}
for tag, season in FOLD.items():
    va = seasons == season
    yv = y[va]
    mv = mo[va]
    seg37 = (mv >= 3) & (mv <= 7)
    try:
        base = v47(tag)
    except Exception:
        continue
    for mname, fn, mask in [
        (f"{tag}_BSS_full", bss, None),
        (f"{tag}_BSS_37", bss, seg37),
        (f"{tag}_decal_full", bss_decal, None),
        (f"{tag}_decal_37", bss_decal, seg37),
        (f"{tag}_reso_full", resolution, None),
        (f"{tag}_reso_37", resolution, seg37),
    ]:
        vals = {}
        m = np.ones(len(yv), bool) if mask is None else mask
        b0 = fn(base[m], yv[m])
        for cname in CAND_SRC:
            c = load_cand(tag, cname)
            if c is None:
                continue
            vals[cname] = fn((0.90 * base + 0.10 * c)[m], yv[m]) - b0
        if len(vals) >= 4:
            METRICS[mname] = vals

# --- 조합 지표 (A+C, A+B+C 평균) ---
def combine(names, label):
    common = None
    for n in names:
        if n not in METRICS:
            return
        common = set(METRICS[n]) if common is None else (common & set(METRICS[n]))
    if not common or len(common) < 4:
        return
    METRICS[label] = {c: float(np.mean([METRICS[n][c] for n in names])) for c in common}


combine(["A_decal_full", "C_decal_full"], "AC_decal")
combine(["A_decal_full", "B_decal_full", "C_decal_full"], "ABC_decal")
combine(["A_reso_full", "C_reso_full"], "AC_reso")
combine(["A_decal_37", "C_decal_37"], "AC_decal_37")

print()
print("=" * 92)
print("검증지표 후보 vs 실측 LB 델타 (aux head 5종)")
print("=" * 92)
lb_all = KNOWN_LB
rows = []
for mname, vals in METRICS.items():
    common = [c for c in vals if c in lb_all]
    if len(common) < 4:
        continue
    x = np.array([vals[c] for c in common])
    z = np.array([lb_all[c] for c in common])
    pear = float(np.corrcoef(x, z)[0, 1])
    spear = float(pd.Series(x).corr(pd.Series(z), method="spearman"))
    # 회귀 잔차
    Amat = np.column_stack([np.ones(len(x)), x])
    res = z - Amat @ np.linalg.lstsq(Amat, z, rcond=None)[0]
    rows.append((mname, len(common), pear, spear, res.std()))
rows.sort(key=lambda r: -r[2])
print(f"{'지표':<18}{'n':>4}{'Pearson':>10}{'Spearman':>10}{'잔차SD':>9}  판정")
for mname, n, pe, sp, rs in rows:
    mark = " <-- 현행" if mname == "A_BSS_full" else ""
    v = "채택후보" if pe > 0.818 else ""
    print(f"{mname:<18}{n:4d}{pe:10.3f}{sp:10.3f}{rs:9.2f}  {v}{mark}")
print()
print("현행 지표(A_BSS_full) Pearson +0.818을 이겨야 의미 있음.")
print(f"[{time.time()-t0:5.0f}s] 완료")
