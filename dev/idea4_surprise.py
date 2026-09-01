"""아이디어 4 — In-season Change-point Evidence (Codex 제안).

기존 inseason_success/n/minus_career는 단순 편차다. 트리가 진짜 필요한 건
"이 표본 크기에서 이 편차가 나왔으면 통계적으로 유의한 변화인가"다.

    surprise_axis = (S_seas - N_seas*p0) / sqrt(BetaBinom_Var(N_seas, a0, b0))
    a0 = p0*K0, b0 = (1-p0)*K0   (커리어 성공률 p0을 K0 강도의 베타사전으로)
    BetaBinom_Var(N,a,b) = N*a*b*(a+b+N) / [(a+b)^2*(a+b+1)]

axis ∈ {success, middle, reverse, ball}. 부호 분리:
    {axis}_shift_up   = max(surprise, 0)
    {axis}_shift_down = max(-surprise, 0)

cmd_index와 다른 점: cmd_index는 두 확률의 선형결합(A-B)이라 트리가 몇 split으로
복원 가능해서 증분이 0이었다(phase93으로 확정, "성공사례" 아니라 재표현 함정 그 자체).
surprise는 나눗셈(비선형 스케일링)이라 트리가 근사하기 더 어렵지만, 그래도 단조변환이라
낙관은 금물 -> 판정은 반드시 3폴드 재학습으로만 (phase93 증분 0이어도 정상).

검증 범위(시간상 축소): direct HGB3 baseline + hurdle core_fail head, 각 3폴드.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea4_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K0 = 30.0   # 베타사전 강도 (K_BATTER, lastyear k와 같은 스케일)


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

# core_fail 라벨 (reverse or middle)
R_ = np.round(meta["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_)
M_ = np.round(meta["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_)
n_o = n_[ordr]
d_r = np.zeros(len(meta)); d_m = np.zeros(len(meta))
d_r[ordr[:-1]] = np.diff(R_[ordr]); d_m[ordr[:-1]] = np.diff(M_[ordr])
core_fail = np.where(step, ((d_r > 0) | (d_m > 0)).astype(np.float64), np.nan)
assert (y[step & (core_fail == 1)] == 0).all()
log(f"  core_fail 복원 {step.sum():,}행")

log("시즌종료 누적 테이블 (N/S/B/R/M_end) 구성...")
sub = meta.sort_values(["pitcher_id", "row_num"])
last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
n_bl = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
last_outcome = last["control_success"].to_numpy(np.float64)
last["N_end"] = n_bl + 1
last["S_end"] = np.round(last["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_bl) + last_outcome
last["B_end"] = np.round(last["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_bl)
last["R_end"] = np.round(last["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_bl)
last["M_end"] = np.round(last["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_bl)
end_tbl = last[["pitcher_id", "season", "N_end", "S_end", "B_end", "R_end", "M_end"]]

sr = sorted(meta["season"].unique().tolist())
idx_now = pd.MultiIndex.from_arrays([meta["pitcher_id"], meta["season"] - 1])


def piv(col):
    p = end_tbl.pivot(index="pitcher_id", columns="season", values=col)
    p = p.reindex(columns=sr).ffill(axis=1).stack(future_stack=True)
    return np.nan_to_num(p.reindex(idx_now).to_numpy().astype(np.float64), nan=0.0)


N_end_row = piv("N_end")
end_vals = {ax: piv(f"{c}_end") for ax, c in [("success", "S"), ("ball", "B"), ("reverse", "R"), ("middle", "M")]}

n_now = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
n_seas = np.clip(n_now - N_end_row, 0, None)
now_vals = {}
now_vals["success"] = meta["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now
now_vals["ball"] = meta["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now
now_vals["reverse"] = meta["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now
now_vals["middle"] = meta["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_now

career_p0 = {}
for ax, col in [("success", "asof_pitcher_success_rate"), ("ball", "asof_pitcher_ball_rate"),
                ("reverse", "asof_pitcher_reverse_rate"), ("middle", "asof_pitcher_middle_rate")]:
    career_p0[ax] = meta[col].fillna(meta[col].mean()).to_numpy(np.float64)

NEW = {}
for ax in ["success", "middle", "reverse", "ball"]:
    s_seas = np.clip(now_vals[ax] - end_vals[ax], 0, None)
    s_seas = np.minimum(s_seas, n_seas)  # 근사오차 방어
    p0 = np.clip(career_p0[ax], 1e-4, 1 - 1e-4)
    a0 = p0 * K0
    b0 = (1 - p0) * K0
    var = n_seas * a0 * b0 * (a0 + b0 + n_seas) / ((a0 + b0) ** 2 * (a0 + b0 + 1))
    surprise = np.divide(s_seas - n_seas * p0, np.sqrt(var), out=np.zeros(len(meta)), where=var > 1e-9)
    NEW[f"{ax}_shift_up"] = np.clip(surprise, 0, None)
    NEW[f"{ax}_shift_down"] = np.clip(-surprise, 0, None)

log(f"신규 피처 {len(NEW)}개 생성 (K0={K0})")
for k, v in NEW.items():
    print(f"    {k:<20} mean={v.mean():.4f} std={v.std():.4f} p90={np.percentile(v,90):.3f}")

HGB_VARIANTS = [
    ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)


def run_target(tag, train_upto, valid_season, target_name, target_arr, target_mask, base_cache_key):
    """target_arr: 학습에 쓸 라벨(예: y 또는 core_fail). target_mask: 그 라벨이 유효한 행."""
    tr_m = (seasons <= train_upto) & step & target_mask
    va_m = (seasons == valid_season) & target_mask
    yv = target_arr[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((train_upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    # 기준선
    bf = f"{CD}/{tag}_{base_cache_key}_base.npy"
    if os.path.exists(bf):
        base_preds = [np.load(bf)]
    else:
        base_preds = []
        for vn, extra in HGB_VARIANTS[:1]:  # 기준선도 d6 하나로 (surprise 세트와 공정 비교)
            p = dict(BASE_HGB); p.update(extra)
            m = HistGradientBoostingClassifier(**p).fit(X.loc[tr_m], target_arr[tr_m], sample_weight=w[tr_m])
            base_preds.append(m.predict_proba(X.loc[va_m])[:, 1])
        np.save(bf, base_preds[0])
    s_base = score(np.mean(base_preds, axis=0))

    nf = f"{CD}/{tag}_{base_cache_key}_new.npy"
    if os.path.exists(nf):
        p_new = np.load(nf)
    else:
        Xa = X.copy()
        for c, v in NEW.items():
            Xa[c] = v
        ts = time.time()
        m = HistGradientBoostingClassifier(**dict(BASE_HGB, **HGB_VARIANTS[0][1])).fit(
            Xa.loc[tr_m], target_arr[tr_m], sample_weight=w[tr_m])
        p_new = m.predict_proba(Xa.loc[va_m])[:, 1]
        np.save(nf, p_new)
        log(f"    [{target_name}/{tag}] 학습완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
    s_new = score(p_new)
    return s_base, s_new


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    all_true = np.ones(len(meta), dtype=bool)
    b1, n1 = run_target(tag, upto, val, "direct", y, all_true, "direct")
    log(f"  [direct]    base={b1:.2f}  new={n1:.2f}  delta={n1-b1:+.2f}")
    valid_cf = ~np.isnan(core_fail)
    b2, n2 = run_target(tag, upto, val, "core_fail", np.nan_to_num(core_fail), valid_cf, "corefail")
    log(f"  [core_fail] base={b2:.2f}  new={n2:.2f}  delta={n2-b2:+.2f}")
    results[tag] = dict(direct_base=b1, direct_new=n1, direct_delta=n1 - b1,
                        cf_base=b2, cf_new=n2, cf_delta=n2 - b2)

print()
print("=" * 78)
print(f"{'fold':<6}{'direct_base':>13}{'direct_new':>13}{'direct_Δ':>10}"
     f"{'cf_base':>10}{'cf_new':>10}{'cf_Δ':>8}")
for tag, r in results.items():
    print(f"{tag:<6}{r['direct_base']:13.2f}{r['direct_new']:13.2f}{r['direct_delta']:+10.2f}"
         f"{r['cf_base']:10.2f}{r['cf_new']:10.2f}{r['cf_delta']:+8.2f}")

min_direct = min(r["direct_delta"] for r in results.values())
min_cf = min(r["cf_delta"] for r in results.values())
print(f"\ndirect 3폴드 최소이득 = {min_direct:+.2f}   {'채택검토' if min_direct > 2 else '기각'}")
print(f"core_fail 3폴드 최소이득 = {min_cf:+.2f}   {'채택검토' if min_cf > 2 else '기각'}")
pd.DataFrame(results).T.to_csv("idea4_results.csv")
log(f"총 {time.time()-t0:.0f}s")
