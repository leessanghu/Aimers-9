"""idea50 — CatBoost 노이즈타겟 정규화 계열 전수 스크리닝 (축5 확장).

배경: v60a의 midother는 CatBoost 기본 설정(boosting_type=Plain, langevin=False,
random_strength=1.0, bootstrap 미설정)이다. CatBoost 1.2.10에서 사용 가능한
정규화 파라미터 21개 중 우리가 써본 건 rsm 하나뿐.
control_success는 p~0.49 동전던지기에 가까운 극단적 노이즈 타겟이므로
'노이즈 타겟 전용' 정규화가 이론적으로 잘 맞는다.

후보 (전부 노이즈/과적합 억제 계열):
  random_strength=3/10  : 분할 점수에 노이즈 주입. 노이즈 타겟에서 우연한
                          분할 선택을 억제. langevin의 분할레벨 버전.
  posterior_sampling    : CatBoost의 SGLD 기반 사후표본추출.
                          langevin+model_shrink_rate+diffusion_temperature 동시설정.
  model_shrink_rate     : 매 이터레이션마다 기존 모델을 축소(연속적 정규화).
  bootstrap Bayesian    : 베이지안 부트스트랩(bagging_temperature로 강도조절).
  bootstrap MVS         : Minimal Variance Sampling. 그래디언트 크기 기반 표본추출.
  grow_policy Lossguide : 비대칭 트리. 구조 다양성.
  score_function L2     : 분할 점수함수 변경(기본 Cosine).

기준선: v60a 로컬 933.78 (실측 1080.60). midother(3-head) 구조 그대로,
CatBoost 학습 옵션만 변경. 2시드.
주의: MultiRMSE 계열은 일부 옵션 미지원 가능 -> 예외처리하고 지원여부 기록.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea50_cache"
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
pid = meta["pitcher_id"].to_numpy()
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    co = c[order]
    d = np.empty(len(meta))
    d[:-1] = co[1:] - co[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid))
tot = y + lab_rev + lab_mid
lab_other = np.where(valid, (tot == 0).astype(np.float64), np.nan)
h1 = np.where(valid, 1.0 - lab_mid, np.nan)
h2 = np.where(valid, 1.0 - lab_other, np.nan)

tr_m = seasons <= 2023
va_m = seasons == 2024
yv = y[va_m]
r = yv.mean(); BS = r * (1 - r)
w = 0.5 ** ((2023 - seasons) / 2.0)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)

base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
REST = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
uni0 = A([f"idea46_cache/A_midother_s{k}.npy" for k in [42, 7]])
v60a = lambda u, wu=0.20: sc((1 - wu) * REST + wu * u)
BASE = v60a(uni0)
log(f"v60a 로컬 기준선 = {BASE:.2f} (실측 1080.60)")

Ymat = np.column_stack([y[tr_m], h1[tr_m], h2[tr_m]])
n_es = int(tr_m.sum() * 0.92)

VARIANTS = [
    ("rs3", {"random_strength": 3.0}),
    ("rs10", {"random_strength": 10.0}),
    ("posterior", {"posterior_sampling": True}),
    ("shrink", {"model_shrink_rate": 0.01}),
    ("bayes", {"bootstrap_type": "Bayesian", "bagging_temperature": 1.0}),
    ("mvs", {"bootstrap_type": "MVS"}),
    ("lossguide", {"grow_policy": "Lossguide", "max_leaves": 31}),
    ("scoreL2", {"score_function": "L2"}),
]

print()
print("=" * 76)
print(f"{'변종':<14}{'단독':>10}{'시드폭':>9}{'v60a로컬':>11}{'Δ':>9}  비고")
print("=" * 76)
print(f"{'(기준)midother':<14}{sc(uni0):10.2f}{'-':>9}{BASE:11.2f}{0.0:+9.2f}")
for name, extra in VARIANTS:
    ps, err = [], None
    for seed in SEEDS:
        f = f"{CD}/A_{name}_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        try:
            ts = time.time()
            params = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                         verbose=0, loss_function="MultiRMSEWithMissingValues",
                         early_stopping_rounds=50, random_seed=seed)
            params.update(extra)
            m = CatBoostRegressor(**params)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat[:n_es], sample_weight=w[tr_m][:n_es],
                  eval_set=(X.loc[tr_m].iloc[n_es:], Ymat[n_es:]))
            p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f, p); ps.append(p)
            log(f"    [{name}/s{seed}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
        except Exception as e:
            err = str(e).split("\n")[0][:60]
            break
    if err:
        print(f"{name:<14}{'미지원':>10}{'-':>9}{'-':>11}{'-':>9}  {err}")
        continue
    u = np.mean(ps, axis=0)
    spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
    v = v60a(u)
    print(f"{name:<14}{sc(u):10.2f}{spread:9.2f}{v:11.2f}{v-BASE:+9.2f}")
print()
print("판정: Δ가 시드폭보다 크고 양수여야 신호. (로컬Δ>0은 외삽구간이므로 실측 필수)")
log(f"총 {time.time()-t0:.0f}s")
