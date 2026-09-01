"""아이디어 8 — 피처 블록 배깅 (구조적 다양성) + 노이즈 바닥 동시 측정.

진단: 멤버 간 상관이 0.86~0.94 밑으로 안 내려간다. 우리가 쓴 다양성 수단(seed, depth,
rsm, 행 드롭아웃)은 전부 '무작위로 조금씩 흔들기'라, 랜덤으로 컬럼 40%를 빼도
같은 정보 경로(inseason_success가 빠져도 inseason_reverse, x_ability_here가 남음)가
살아있어 모델이 같은 곳으로 수렴한다.

대안: 자연 블록을 통째로 제거해 그 정보 경로 자체를 차단한다.
    trackman을 못 보는 모델   -> 순수 기록 기반 판단
    batter를 못 보는 모델     -> 투수 능력만으로 판단
    inseason을 못 보는 모델   -> 커리어 통계로만 판단
각자 다른 '전문가 인격'이 되어, seed/rsm이 만들 수 없는 종류의 차이가 생긴다.
공통 51개(상황/카운트/시즌 등 기본정보)는 항상 유지한다.

*** 이번엔 v38 실패(단일학습 노이즈에 속음)를 반복하지 않도록 처음부터 시드 반복을 넣는다. ***
    - 각 블록드롭 모델을 시드 2개로 학습
    - 기준선(base3)도 동일 조건 시드 2개
    - 이득이 시드 간 변동보다 큰지 반드시 확인

평가: fold A/C 우선(깨끗한 폴드). fold B는 regime단절이라 큰 수가 나와도 신뢰하지 않는다
      (v38/v39가 정확히 fold B 큰 수에 속아서 실측 실패).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea8_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
cols = list(X.columns)

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)

BLOCKS = {
    "trackman": [c for c in cols if c.startswith("tm_")],
    "batter": [c for c in cols if c.startswith("bat_") or "batter" in c],
    "inseason": [c for c in cols if c.startswith("inseason_")],
    "form_role": [c for c in cols if c.startswith("form") or c.startswith("role")],
    "crosses": [c for c in cols if c.startswith("x_")],
    "lastyear_vol": [c for c in cols if c.startswith("ly_") or c.startswith("vol_")],
}
for k, v in BLOCKS.items():
    log(f"  블록 {k}: {len(v)}개 제거 -> 잔여 {len(cols)-len(v)}개")

HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)


def fit_pred(tag, name, drop_cols, seed, tr_m, va_m, w):
    f = f"{CD}/{tag}_{name}_s{seed}.npy"
    if os.path.exists(f):
        return np.load(f)
    Xa = X.drop(columns=drop_cols) if drop_cols else X
    p = dict(HGB); p["random_state"] = seed
    ts = time.time()
    m = HistGradientBoostingClassifier(**p).fit(Xa.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
    pr = m.predict_proba(Xa.loc[va_m])[:, 1]
    np.save(f, pr)
    log(f"    {name}/s{seed} iters={m.n_iter_} feat={Xa.shape[1]} ({time.time()-ts:.0f}s)")
    return pr


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    # 기준선: 블록 제거 없음, 시드 2개 (노이즈 바닥 측정용)
    base_seeds = [fit_pred(tag, "full", None, s, tr_m, va_m, w) for s in SEEDS]
    base_scores = [sc(p) for p in base_seeds]
    base_avg = np.mean(base_seeds, axis=0)
    log(f"  [기준선] 시드별 {[round(x,2) for x in base_scores]}  "
        f"시드폭={max(base_scores)-min(base_scores):.2f}  2시드평균={sc(base_avg):.2f}")

    # 블록 드롭 모델들
    block_preds = {}
    for name, drop in BLOCKS.items():
        ps = [fit_pred(tag, name, drop, s, tr_m, va_m, w) for s in SEEDS]
        block_preds[name] = np.mean(ps, axis=0)
        log(f"  [{name}드롭] 단독={sc(block_preds[name]):.2f}  "
           f"corr(기준선)={np.corrcoef(block_preds[name], base_avg)[0,1]:.4f}")

    # 블록 앙상블: 기준선 + 모든 블록드롭 모델 평균
    all_members = [base_avg] + list(block_preds.values())
    ens_all = np.mean(all_members, axis=0)
    log(f"  [전체앙상블 {len(all_members)}멤버] {sc(ens_all):.2f}  (2시드평균 기준선 대비 {sc(ens_all)-sc(base_avg):+.2f})")

    # v35local(base3+hurdle)에 블록앙상블을 추가했을 때
    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    row = dict(base_seed_spread=max(base_scores) - min(base_scores), base_avg=sc(base_avg),
               ens_all=sc(ens_all), v35local=sc(v35l))
    for wv in [0.1, 0.2, 0.3, 0.4]:
        row[f"w{wv}"] = sc((1 - wv) * v35l + wv * ens_all)
        log(f"  v35local + 블록앙상블(w={wv}) = {row[f'w{wv}']:.2f}  (v35l대비 {row[f'w{wv}']-sc(v35l):+.2f})")
    row["corrs"] = {k: float(np.corrcoef(v, base_avg)[0, 1]) for k, v in block_preds.items()}
    results[tag] = row

print()
print("=" * 84)
print("*** 판정 기준: 이득이 '기준선 시드폭'보다 확실히 커야 신뢰 가능 ***")
print(f"{'fold':<6}{'시드폭':>9}{'기준선':>10}{'블록앙상블':>11}{'v35local':>10}" +
      "".join(f"{'w='+str(w):>9}" for w in [0.1, 0.2, 0.3, 0.4]))
for tag, r in results.items():
    print(f"{tag:<6}{r['base_seed_spread']:9.2f}{r['base_avg']:10.2f}{r['ens_all']:11.2f}{r['v35local']:10.2f}" +
          "".join(f"{r[f'w{w}']:9.2f}" for w in [0.1, 0.2, 0.3, 0.4]))

print()
for wv in [0.1, 0.2, 0.3, 0.4]:
    gains = [results[t][f"w{wv}"] - results[t]["v35local"] for t in results]
    spreads = [results[t]["base_seed_spread"] for t in results]
    ok = min(gains) > max(spreads)
    print(f"w={wv}: 이득 {[round(g,2) for g in gains]}  최소={min(gains):+.2f}  "
         f"시드폭최대={max(spreads):.2f}  {'노이즈 초과(유효)' if ok else '노이즈 이내(신뢰불가)'}")
print()
print("블록별 상관(기준선 대비, 낮을수록 다양성 큼):")
for tag, r in results.items():
    print(f"  fold {tag}: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(r["corrs"].items(), key=lambda x: x[1])))
pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "corrs"} for k, v in results.items()}).to_csv("idea8_results.csv")
log(f"총 {time.time()-t0:.0f}s")
