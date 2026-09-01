"""아이디어 6 — Masked Multi-Task Shared-Tree Boosting (Codex 제안).

배경: 현재 direct와 hurdle(core_fail, success|no_core)은 완전히 별도 모델이라 각자
noisy gradient로 독립적으로 split을 고른다. 하나의 CatBoost가 tree structure를
공유하면서 3개 타깃을 동시에 학습하면:
    - success에서만 우연히 좋은 noisy split은 세 head 전부에 안 통하면 채택되기 어렵다
      (공동 gain 규제 -> 노이즈 과적합 억제, 오늘 진단인 '분산 병목'을 정면으로 겨냥)
    - success/core_fail 양쪽에 재현되는 split이 우선된다
    - success|no_core 행은 target=NaN 마스킹으로 해당 loss에서 제외

CatBoostRegressor(loss_function="MultiRMSEWithMissingValues") 지원 확인됨(catboost 1.2.10).

head0 = success (전체 행)
head1 = core_fail (복원 가능 행만, 나머지 NaN)
head2 = success|no_core_fail (core_fail=0 행만, 나머지 NaN)

p_direct = clip(head0)
p_hurdle_shared = (1-clip(head1)) * clip(head2)
p_multi = 0.5*p_direct + 0.5*p_hurdle_shared  (공유트리 내부의 두 경로 평균)

검증: CatBoost 한 설정, A/C/B. head0/hurdle_shared/평균/v29local+10~30% 블렌드 각각 기록.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea6_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)

R_ = np.round(meta["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_)
M_ = np.round(meta["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_)
n_o = n_[ordr]
d_r = np.zeros(len(meta)); d_m = np.zeros(len(meta))
d_r[ordr[:-1]] = np.diff(R_[ordr]); d_m[ordr[:-1]] = np.diff(M_[ordr])
core_fail = np.where(step, ((d_r > 0) | (d_m > 0)).astype(np.float64), np.nan)
assert (y[step & (core_fail == 1)] == 0).all()
log(f"  core_fail 복원 {step.sum():,}행")

CAT_PARAMS = dict(iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  random_seed=42, loss_function="MultiRMSEWithMissingValues",
                  early_stopping_rounds=50)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_full = (seasons <= upto)
    tr_m = tr_full & step   # core_fail 라벨 필요하므로 복원가능행만 학습에 사용
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    Y0 = y.copy()
    Y1 = core_fail.copy()
    Y2 = np.where(core_fail == 0, y, np.nan)
    Ymat = np.column_stack([Y0, Y1, Y2])

    tr_i = np.where(tr_m)[0]
    n_tr = len(tr_i)
    es_cut = int(n_tr * 0.92)
    tr_sub, es_sub = tr_i[:es_cut], tr_i[es_cut:]  # phase2_common.time_split_es와 동일 아이디어(뒤 8%)

    f0 = f"{CD}/{tag}_head0.npy"; f1 = f"{CD}/{tag}_head1.npy"; f2 = f"{CD}/{tag}_head2.npy"
    if os.path.exists(f0):
        h0, h1, h2 = np.load(f0), np.load(f1), np.load(f2)
    else:
        ts = time.time()
        m = CatBoostRegressor(**CAT_PARAMS)
        m.fit(X.iloc[tr_sub], Ymat[tr_sub], sample_weight=w[tr_sub],
             eval_set=(X.iloc[es_sub], Ymat[es_sub]))
        pred = m.predict(X.loc[va_m])
        h0, h1, h2 = pred[:, 0], pred[:, 1], pred[:, 2]
        np.save(f0, h0); np.save(f1, h1); np.save(f2, h2)
        log(f"  학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)")

    p_direct = np.clip(h0, 0, 1)
    p_hurdle_shared = (1 - np.clip(h1, 0, 1)) * np.clip(h2, 0, 1)
    p_multi = 0.5 * p_direct + 0.5 * p_hurdle_shared

    s_direct, s_hurdle, s_multi = score(p_direct), score(p_hurdle_shared), score(p_multi)
    log(f"  head0(direct)={s_direct:.2f}  hurdle_shared={s_hurdle:.2f}  0.5/0.5평균={s_multi:.2f}")

    v29 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    s_v29 = score(v29)
    row = dict(direct=s_direct, hurdle=s_hurdle, multi=s_multi, v29local=s_v29,
              corr_multi_v29=np.corrcoef(p_multi, v29)[0, 1])
    log(f"  v29local={s_v29:.2f}  상관(multi,v29local)={row['corr_multi_v29']:.4f}")
    for wv in [0.1, 0.2, 0.3]:
        blend = (1 - wv) * v29 + wv * p_multi
        row[f"w{wv}"] = score(blend)
        log(f"  v29local+multi(w={wv}) = {row[f'w{wv}']:.2f}  (v29local대비 {row[f'w{wv}']-s_v29:+.2f})")
    results[tag] = row

print()
print("=" * 90)
hdr = f"{'fold':<6}{'direct':>9}{'hurdle':>9}{'multi':>9}{'v29local':>10}{'corr':>8}"
for w in [0.1, 0.2, 0.3]:
    hdr += f"{'w='+str(w):>9}"
print(hdr)
for tag, r in results.items():
    row = f"{tag:<6}{r['direct']:9.2f}{r['hurdle']:9.2f}{r['multi']:9.2f}{r['v29local']:10.2f}{r['corr_multi_v29']:8.4f}"
    for w in [0.1, 0.2, 0.3]:
        row += f"{r[f'w{w}']:9.2f}"
    print(row)

print()
for wv in [0.1, 0.2, 0.3]:
    gains = [results[t][f"w{wv}"] - results[t]["v29local"] for t in ["A", "C", "B"]]
    print(f"w={wv}: 폴드별 이득 {[round(g,2) for g in gains]}  최소={min(gains):+.2f}  "
         f"{'채택검토' if min(gains) > 2 else '기각'}")
pd.DataFrame(results).T.to_csv("idea6_results.csv")
log(f"총 {time.time()-t0:.0f}s")
