"""idea49 — v60a 실제 구성(midother 3-head)에 6개 최적화축을 전부 재검증.

과거 실험들은 대부분 옛 아키텍처(67~132피처, 단일head 분류기, 다른 블렌드)에서
했다. v60a는 구조가 많이 바뀌었으니(3-head MultiRMSE, 6피처증가, 비례축소 가중치)
동일 결론이 유지되는지 재확인한다.

기준선: v60a_local(fold A 전체2024) = 933.78 (실측 1080.60과 매칭 확인됨)
축1 가중치: midother 미세그리드 (0.15~0.25)
축2 훈련방법: 시드 앙상블 (2시드->5시드)
축5 gradient: boosting_type=Ordered, langevin=True
축6 피처우선순위: feature_weights(SHAP상위 축소), baseline(잔차학습)
축3(모델추가)/축4(loss함수)는 과거 결정적 기각(-18~-47, RMSE<Logloss 9점차)이라
재검증 비용 대비 기대값 낮음 -> 스킵, 이유만 기록.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea49_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


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

upto, val, tag = 2023, 2024, "A"
tr_m = seasons <= upto
va_m = seasons == val
yv = y[va_m]
r = yv.mean(); BS = r * (1 - r)
w = 0.5 ** ((upto - seasons) / 2.0)


def sc(p):
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)


A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
REST = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od  # 비율 유지, 정규화 전
uni_base = A([f"idea46_cache/A_midother_s{k}.npy" for k in [42, 7]])


def v60a_score(uni_pred, w_uni=0.20):
    pot = 1.0 - w_uni
    return sc(pot * REST + w_uni * uni_pred)


BASELINE = v60a_score(uni_base)
log(f"v60a 로컬 기준선 = {BASELINE:.2f} (실측 1080.60)")

Ymat_tr = np.column_stack([y[tr_m], h1[tr_m], h2[tr_m]])
n_es = int(tr_m.sum() * 0.92)

results = {}


def train_variant(name, extra_params, seeds):
    ps = []
    for seed in seeds:
        f = f"{CD}/A_{name}_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        ts = time.time()
        params = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                     loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50,
                     random_seed=seed)
        params.update(extra_params)
        m = CatBoostRegressor(**params)
        m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
              eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
        p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
        np.save(f, p); ps.append(p)
        log(f"    [{name}/s{seed}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    return np.mean(ps, axis=0)


print()
print("=" * 74)
print("축1: 가중치 미세그리드 (midother 비중, 나머지는 v42비율 유지)")
print("=" * 74)
for wu in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    v = v60a_score(uni_base, wu)
    print(f"  w_midother={wu:.2f}  로컬={v:8.2f}  Δ={v-BASELINE:+.2f}")

print()
print("=" * 74)
print("축2: 시드 앙상블 (2시드 -> 5시드)")
print("=" * 74)
p5 = train_variant("baseline5seed", {}, [42, 7, 2024, 1234, 99])
d5 = v60a_score(p5) - BASELINE
print(f"  5시드 평균  로컬={v60a_score(p5):.2f}  Δ={d5:+.2f}")

print()
print("=" * 74)
print("축5: gradient 최적화 (Ordered boosting / Langevin)")
print("=" * 74)
p_ord = train_variant("ordered", {"boosting_type": "Ordered"}, [42, 7])
d_ord = v60a_score(p_ord) - BASELINE
print(f"  Ordered boosting  로컬={v60a_score(p_ord):.2f}  Δ={d_ord:+.2f}")

p_lang = train_variant("langevin", {"langevin": True}, [42, 7])
d_lang = v60a_score(p_lang) - BASELINE
print(f"  Langevin(SGLD)    로컬={v60a_score(p_lang):.2f}  Δ={d_lang:+.2f}")

p_both = train_variant("ordered_langevin", {"boosting_type": "Ordered", "langevin": True}, [42, 7])
d_both = v60a_score(p_both) - BASELINE
print(f"  Ordered+Langevin  로컬={v60a_score(p_both):.2f}  Δ={d_both:+.2f}")

print()
print("=" * 74)
print("축6: 피처 우선순위 (feature_weights로 SHAP상위 축소 -> 미탐색축 강제)")
print("=" * 74)
# ability계열(이미 포화) 축소, middle계열(미활용) 증폭
shrink_cols = [c for c in X.columns if any(k in c for k in
              ["ability", "x_ability", "asof_pitcher_success_rate", "inseason_success"])]
boost_cols = [c for c in X.columns if "middle" in c]
fw = {c: 0.5 for c in shrink_cols}
fw.update({c: 2.0 for c in boost_cols})
log(f"  축소 {len(shrink_cols)}개(x0.5) / 증폭 {len(boost_cols)}개(x2.0)")
p_fw = train_variant("featweight", {"feature_weights": fw}, [42, 7])
d_fw = v60a_score(p_fw) - BASELINE
print(f"  feature_weights   로컬={v60a_score(p_fw):.2f}  Δ={d_fw:+.2f}")

print()
print("=" * 74)
print("스킵된 축 (과거 이 아키텍처 계열에서 이미 결정적으로 기각)")
print("=" * 74)
print("  축3 모델추가: LGB/XGB corr=0.943(base) -> 희석만. capacity 확장(phase68) -18~-47.")
print("  축4 loss함수: phase79 blend_logloss 880.7 > blend_rmse 871.4 (9점차, 결정적)")
log(f"총 {time.time()-t0:.0f}s")
