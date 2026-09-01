"""아이디어 5 — Cross-fitted Latent Target (Codex 제안).

배경: 설명가능분산이 1.05%뿐이라 대부분의 gradient가 투수능력이 아니라 베르누이 잡음.
직접 0/1에 gradient를 거는 대신, train fold 내부에서(검증시즌 절대 미사용) leave-one-out
empirical Bayes로 축소된 soft label을 만들어 그걸 회귀 타겟으로 쓴다.

    cell = (pitcher_id, season, pressure_bucket, same_hand)
    p_ps_loo   = LOO 축소된 pitcher-season 성공률 (global로 축소, K_PS)
    y_soft_i   = (cell_sum_loo_i + K_CELL * p_ps_loo_i) / (cell_count_loo_i + K_CELL)
    (자기 행은 cell/pitcher-season 양쪽 다 LOO로 제외)

phase79의 RMSE 실험과 다른 점: 그건 noisy 0/1에 그대로 RMSE를 썼다. 이번엔 조건부 성공확률
추정치(soft label) 자체를 타겟으로 회귀한다 -> HistGradientBoostingRegressor 사용.

최종: p = (1-w)*p_v35_local + w*p_denoised. 채택조건 = 단독성능이 아니라 블렌드가
3폴드 모두(A/C/B) 플러스인지. w in {0.1, 0.2, 0.3}.

검증: HGB d6 하나만, pressure_bucket x same_hand x pitcher-season 해상도 하나만.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CD = "idea5_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_PS = 15.0     # pitcher-season -> global 축소 강도
K_CELL = 30.0   # cell -> pitcher-season 축소 강도


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
g = float(meta["global_rate"].iloc[0])
pid = meta["pitcher_id"].to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64)  # 이미 0/1 인코딩됨
balls = meta["balls_before"].to_numpy(np.float64)
strikes = meta["strikes_before"].to_numpy(np.float64)
pressure_bucket = np.sign(balls - strikes).astype(np.int64)  # -1(유리)/0(균형)/+1(불리)

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)


def build_y_soft(tr_m):
    """tr_m(학습행 마스크) 내부에서만 LOO 축소. 검증시즌 정보는 절대 안 들어감."""
    idx = np.where(tr_m)[0]
    sub = pd.DataFrame({"pid": pid[idx], "season": seasons[idx], "y": y[idx],
                        "pb": pressure_bucket[idx], "sh": same_hand[idx]}, index=idx)

    # pitcher-season 합계/개수 -> LOO
    ps_grp = sub.groupby(["pid", "season"])["y"].agg(ps_sum="sum", ps_n="count")
    sub = sub.join(ps_grp, on=["pid", "season"])
    ps_sum_loo = sub["ps_sum"] - sub["y"]
    ps_n_loo = sub["ps_n"] - 1
    p_ps_loo = (ps_sum_loo + K_PS * g) / (ps_n_loo + K_PS)

    # cell(pid,season,pb,sh) 합계/개수 -> LOO
    cell_grp = sub.groupby(["pid", "season", "pb", "sh"])["y"].agg(c_sum="sum", c_n="count")
    sub = sub.join(cell_grp, on=["pid", "season", "pb", "sh"])
    c_sum_loo = sub["c_sum"] - sub["y"]
    c_n_loo = sub["c_n"] - 1

    y_soft_sub = (c_sum_loo + K_CELL * p_ps_loo) / (c_n_loo + K_CELL)
    out = np.full(len(meta), np.nan)
    out[idx] = y_soft_sub.to_numpy(np.float64)
    return out


HGB_REG = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42,
              loss="squared_error")

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w_rec = 0.5 ** ((upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    f = f"{CD}/{tag}_denoised.npy"
    if os.path.exists(f):
        p_den = np.load(f)
    else:
        y_soft = build_y_soft(tr_m)
        log(f"  y_soft 생성 완료  mean={np.nanmean(y_soft[tr_m]):.4f} std={np.nanstd(y_soft[tr_m]):.4f}"
           f"  (원 y std={y[tr_m].std():.4f})")
        ts = time.time()
        m = HistGradientBoostingRegressor(**HGB_REG).fit(X.loc[tr_m], y_soft[tr_m], sample_weight=w_rec[tr_m])
        p_den = m.predict(X.loc[va_m])
        np.save(f, p_den)
        log(f"  학습완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")

    base = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    s_base = score(base)
    s_den = score(p_den)
    log(f"  base={s_base:.2f}  denoised단독={s_den:.2f}  상관(base,denoised)={np.corrcoef(base,p_den)[0,1]:.4f}")

    row = dict(base=s_base, denoised=s_den, corr=np.corrcoef(base, p_den)[0, 1])
    for wv in [0.1, 0.2, 0.3]:
        blend = (1 - wv) * base + wv * np.clip(p_den, 0, 1)
        row[f"w{wv}"] = score(blend)
        log(f"  w={wv}: score={row[f'w{wv}']:.2f}  (base대비 {row[f'w{wv}']-s_base:+.2f})")
    results[tag] = row

print()
print("=" * 78)
print(f"{'fold':<6}{'base':>10}{'denoised':>10}{'corr':>8}" + "".join(f"{'w='+str(w):>10}" for w in [0.1, 0.2, 0.3]))
for tag, r in results.items():
    print(f"{tag:<6}{r['base']:10.2f}{r['denoised']:10.2f}{r['corr']:8.4f}" +
         "".join(f"{r[f'w{w}']:10.2f}" for w in [0.1, 0.2, 0.3]))

print()
for wv in [0.1, 0.2, 0.3]:
    gains = [results[t][f"w{wv}"] - results[t]["base"] for t in ["A", "C", "B"]]
    print(f"w={wv}: 폴드별 이득 {[round(g,2) for g in gains]}  최소={min(gains):+.2f}  "
         f"{'채택검토' if min(gains) > 2 else '기각'}")
pd.DataFrame(results).T.to_csv("idea5_results.csv")
log(f"총 {time.time()-t0:.0f}s")
