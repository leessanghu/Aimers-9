"""아이디어 2 — 타자축 x 신뢰도 분리 (+ 아이디어1 교훈 검증용 대조군).

아이디어1 결과: 합성피처(here+bat, here+count 등) 4개 조합 전부 3폴드에서 손해
                (최소이득 -18 ~ -34). 12개 측정 중 플러스는 1개뿐.
교훈: 축을 '합치면' 트리가 정보를 뭉갠 지름길 축을 쓰게 되어 손해다.
      실제로 Part1 스크리닝에서 xh_bat(-13.42), xh_count(-14.18)가 이미 경고했었다
      (원본 ref_bat_inseason은 +12.94인데 합성하면 신호가 사라짐).

아이디어2 방향: 합치지 말고 '분리'한다.
    상호작용(피처 x 저표본플래그)은 합성과 반대 연산이다. 저표본 구간에서만 값을 갖는
    별도 축을 만들 뿐, 두 축을 하나로 뭉개지 않는다.
    근거: 오늘 유일하게 성공한 피처계열이 trackman x low-n (+9.6, v27 실측 +18.1의 일부).
    그리고 타자 표본수 범위가 매우 넓다 (0~13927, 중앙값 2263, 10%분위 247).

실험 세트:
    A_binflag  타자 저표본 이진플래그 x 타자신호 4종        (trackman x lown 방식 그대로)
    B_trust    연속 신뢰도 가중 (n/(n+K)) x 타자신호        (수축 형태)
    C_both     A+B
    D_ablate   x_ability_here 제거 (아이디어1 교훈 직접 검증 대조군)

판정: 3폴드(A/C/B) 최소이득 > 2 이면 채택.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea2_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
g = float(meta["global_rate"].iloc[0])

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)
log(f"  X={X.shape}  step={step.sum():,}")

# ---------------- 신규 피처 ----------------
bat_n = meta["asof_batter_n"].fillna(0).to_numpy(np.float64)
bin_n = X["bat_inseason_n"].to_numpy(np.float64)
BAT_SIGNALS = ["bat_inseason_smooth", "bat_inseason_minus_career",
               "asof_batter_success_rate_smooth", "bat_ly_rate"]

# 저표본 임계값 = train 중앙값 (fit 시점 상수, 배치 무관)
lown_thr = float(np.median(bat_n))
lown = (bat_n <= lown_thr).astype(np.float64)
log(f"  타자 저표본 임계값(중앙값)={lown_thr:.0f}  해당비율={lown.mean():.3f}")

NEW_A, NEW_B = {}, {}
NEW_A["bat_lown_flag"] = lown
for c in BAT_SIGNALS:
    NEW_A[f"{c}_x_batlown"] = X[c].to_numpy(np.float64) * lown

# 연속 신뢰도: in-season 표본수 기반 (bat_inseason_n은 log1p 스케일일 수 있으므로 원복)
raw_bin = np.expm1(np.clip(bin_n, 0, 20)) if bin_n.max() < 20 else bin_n
for K in [200.0]:
    trust = raw_bin / (raw_bin + K)
    NEW_B[f"bat_trust_K{int(K)}"] = trust
    NEW_B[f"bat_sig_x_trust_K{int(K)}"] = (X["bat_inseason_smooth"].to_numpy(np.float64) - g) * trust
    NEW_B[f"bat_dev_x_trust_K{int(K)}"] = X["bat_inseason_minus_career"].to_numpy(np.float64) * trust
log(f"  A세트 {len(NEW_A)}개, B세트 {len(NEW_B)}개")
for k, v in list(NEW_A.items())[:2] + list(NEW_B.items())[:2]:
    print(f"    {k:<34} mean={v.mean():.4f} std={v.std():.4f}")

HGB_VARIANTS = [
    ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)

SETS = {
    "A_binflag": ("add", NEW_A),
    "B_trust": ("add", NEW_B),
    "C_both": ("add", {**NEW_A, **NEW_B}),
    "D_ablate": ("drop", ["x_ability_here"]),
}


def run_fold(train_upto, valid_season, tag):
    log(f"===== fold {tag}: train<={train_upto} -> valid={valid_season} =====")
    tr_m = (seasons <= train_upto) & step
    va_m = seasons == valid_season
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((train_upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    out = {}
    base = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n, _ in HGB_VARIANTS], axis=0)
    out["base"] = score(base)
    log(f"  base(캐시) = {out['base']:.2f}")

    for setname, (mode, payload) in SETS.items():
        preds = []
        for vn, extra in HGB_VARIANTS:
            f = f"{CD}/{tag}_{setname}_{vn}.npy"
            if os.path.exists(f):
                preds.append(np.load(f))
                continue
            if mode == "add":
                Xa = X.copy()
                for c, v in payload.items():
                    Xa[c] = v
            else:
                Xa = X.drop(columns=[c for c in payload if c in X.columns])
            p = dict(BASE_HGB); p.update(extra)
            ts = time.time()
            m = HistGradientBoostingClassifier(**p).fit(Xa.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
            pr = m.predict_proba(Xa.loc[va_m])[:, 1]
            np.save(f, pr)
            preds.append(pr)
            log(f"    {setname}/{vn} 완료 iters={m.n_iter_} feat={Xa.shape[1]} ({time.time()-ts:.0f}s)")
            del Xa
        out[setname] = score(np.mean(preds, axis=0))
        log(f"  {setname} = {out[setname]:.2f}  (base대비 {out[setname]-out['base']:+.2f})")
    return out


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    results[tag] = run_fold(upto, val, tag)

print()
print("=" * 78)
print(f"{'조합':<14}" + "".join(f"{t:>19}" for t in ["A(2024)", "C(2022)", "B(2023)"]) + f"{'최소이득':>10}")
print("-" * 78)
for setname in ["base"] + list(SETS):
    row = f"{setname:<14}"
    gains = []
    for t in ["A", "C", "B"]:
        v = results[t][setname]
        d = v - results[t]["base"]
        gains.append(d)
        row += f"{v:13.2f}({d:+5.2f})"
    row += f"{min(gains):10.2f}" if setname != "base" else f"{0.0:10.2f}"
    print(row)

adds = [s for s in SETS if SETS[s][0] == "add"]
best = max(adds, key=lambda s: min(results[t][s] - results[t]["base"] for t in ["A", "C", "B"]))
best_min = min(results[t][best] - results[t]["base"] for t in ["A", "C", "B"])
print()
print(f"신규피처 조합 중 3폴드 최소이득 최대: {best} ({best_min:+.2f})")
print("=> 채택 검토" if best_min > 2 else "=> 기각")
abl = [results[t]["D_ablate"] - results[t]["base"] for t in ["A", "C", "B"]]
print(f"\n[대조군] x_ability_here 제거 효과: A{abl[0]:+.2f} C{abl[1]:+.2f} B{abl[2]:+.2f}")
print("  -> 제거해도 손해 없으면 '합성축이 정보를 뭉갠다'는 아이디어1 교훈 지지")
pd.DataFrame(results).to_csv("idea2_results.csv")
log(f"총 {time.time()-t0:.0f}s")
