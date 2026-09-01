"""idea54 — 새 정보축 3종을 독립적으로 fold A/C 검증. local_leaderboard 규약 준수
(전체2024, (1-w)*v47 + w*신규, w=0.10).

A) cond_ball  — not-dangerous(not middle/reverse) 조건에서만 1-ball head.
   근거: not-dangerous 안에서 P(succ|ball=1)=0.591 vs P(succ|ball=0)=0.941,
   격차 0.35로 지금까지 조사된 조건부 분할 중 최대. 전체행 ballaxis(v52,+1.83)보다
   조건부가 신호를 더 선명하게 줄 수 있는지 검증.
B) count_resid — y - E[y|balls,strikes] 잔차를 aux head로.
   근거: count_state는 이미 162피처에 있으나 '타겟을 잔차화'하는 건 미시도.
C) future50_multi — y + 향후50투구 성공률 + 향후50투구 middle율.
   근거: v49 formcast(y+향후50성공률 단일)가 실측 +4. middle까지 추가하면
   multires/midother 패턴(멀티헤드 확장)과 결합 가능한지 검증.

전부 독립 학습, 서로 비교하지 않고 v47 대비만 잰다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea54_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
CAL_A, CAL_B = 6.29, 5.02


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
bb = meta["balls_before"].to_numpy(np.float64)
st = meta["strikes_before"].to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    co = c[order]
    d = np.empty(len(meta)); d[:-1] = co[1:] - co[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta)); lab[order] = d
    return lab


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
lab_ball = recover("asof_pitcher_ball_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid) | np.isnan(lab_ball))
dang = valid & ((lab_mid > 0) | (lab_rev > 0))
notdang = valid & ~dang

# A) 조건부 ball head: not-dangerous 행에서만 1-ball, 나머지 NaN
h_cond_ball = np.where(notdang, 1.0 - lab_ball, np.nan)
log(f"  cond_ball 유효 {np.isfinite(h_cond_ball).mean()*100:.1f}% (not-dangerous 부분집합)")

# B) count residual: y - E[y|balls,strikes] (train 전용, season-1 누출 없음 -> 여기선
#    전역 count prior를 train<=upto 내부에서 fold별로 재계산해야 함. 아래 fold루프에서 처리)
count_state = bb * 4 + st

# C) future50: 향후 50투구 성공률/middle율 (formcast와 동일 슬라이딩 방식)
WINDOW = 50
sub = pd.DataFrame({"pid": pid, "row_num": meta["row_num"].to_numpy(), "y": y, "mid": lab_mid})
sub = sub.sort_values(["pid", "row_num"])
grp = sub.groupby("pid")
K_F = 10.0
g_glob = float(y.mean())
g_mid = float(np.nanmean(lab_mid))
fut_y_sum = grp["y"].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).sum().iloc[::-1])
fut_y_cnt = grp["y"].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).count().iloc[::-1])
fut_m_sum = grp["mid"].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).sum().iloc[::-1])
fut_m_cnt = grp["mid"].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).count().iloc[::-1])
h_fut_succ_s = (fut_y_sum + K_F * g_glob) / (fut_y_cnt + K_F)
h_fut_mid_s = (fut_m_sum + K_F * g_mid) / (fut_m_cnt + K_F)
h_fut_succ = pd.Series(h_fut_succ_s.to_numpy(), index=sub.index).reindex(range(len(meta))).to_numpy(np.float64)
h_fut_notmid = 1.0 - pd.Series(h_fut_mid_s.to_numpy(), index=sub.index).reindex(range(len(meta))).to_numpy(np.float64)
log(f"  future50 커버리지 {np.isfinite(h_fut_succ).mean()*100:.1f}%")

CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)
    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base = A([f"phase90_cache/{tag}_base_{n}.npy" for n in ["d6", "d8", "sub"]])
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) *
                   np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
    mr = A([f"idea13_cache/{tag}_multires_s{k}.npy" for k in [42, 7]])
    od = A([f"idea13_cache/{tag}_ordinal_s{k}.npy" for k in [42, 7]])
    V47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
    B47 = sc(V47)
    log(f"  v47local={B47:.2f}")

    # count prior: train<=upto 내부에서만 계산 (leakage 안전)
    ctab = pd.DataFrame({"cs": count_state[tr_m], "y": y[tr_m]}).groupby("cs")["y"].agg(["sum", "count"])
    K_C = 500.0
    ctab["prior"] = (ctab["sum"] + K_C * y[tr_m].mean()) / (ctab["count"] + K_C)
    cprior_all = pd.Series(count_state).map(ctab["prior"]).fillna(y[tr_m].mean()).to_numpy(np.float64)
    h_count_resid = y - cprior_all

    n_es = int(tr_m.sum() * 0.92)
    VARIANTS = {
        "cond_ball": [y, h_cond_ball],
        "count_resid": [y, h_count_resid],
        "future50_multi": [y, h_fut_succ, h_fut_notmid],
    }
    for name, heads in VARIANTS.items():
        Ymat = np.column_stack([np.asarray(h, dtype=np.float64) for h in heads])
        ps = []
        for seed in SEEDS:
            f = f"{CD}/{tag}_{name}_s{seed}.npy"
            if os.path.exists(f):
                ps.append(np.load(f)); continue
            ts = time.time()
            m = CatBoostRegressor(**CAT, random_seed=seed)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat[tr_m][:n_es], sample_weight=w[tr_m][:n_es],
                  eval_set=(X.loc[tr_m].iloc[n_es:], Ymat[tr_m][n_es:]))
            p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f, p); ps.append(p)
            log(f"    [{name}/s{seed}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
        u = np.mean(ps, axis=0)
        spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
        d = sc(0.90 * V47 + 0.10 * u) - B47
        results.setdefault(name, {})[tag] = dict(solo=sc(u), spread=spread, delta=d)
        log(f"  {name:<16} 단독={sc(u):.2f} 시드폭={spread:.2f} Δ={d:+.2f}")

print()
print("=" * 78)
print("새 정보축 3종 fold A/C 결과 (v47 기준, w=0.10)")
print("=" * 78)
for name, r in results.items():
    a = r.get("A", {}); c = r.get("C", {})
    pred = CAL_A + CAL_B * a.get("delta", 0)
    print(f"{name:<16} foldA_Δ={a.get('delta',0):+7.2f}(폭{a.get('spread',0):.2f})  "
          f"foldC_Δ={c.get('delta',0):+7.2f}(폭{c.get('spread',0):.2f})  예상실측Δ={pred:+.2f}")
print()
print("참고: v60a로컬=933.78(실측1080.60) 별도 비교. 손익분기 로컬Δ=-1.25")
log(f"총 {time.time()-t0:.0f}s")
