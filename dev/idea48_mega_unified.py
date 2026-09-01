"""idea48 — '메가 통합' 공유트리. 지금까지 확인된 양수 신호를 전부 한 트리에.

근거(전부 이 세션에서 실측/로컬로 확인된 사실):
  multires(y+투수시즌LOO+투수x손LOO)      실측 +10.17
  ordinal(순차분해 3단)                   실측  +5.08
  unified5(y+not_rev+not_mid|not_rev+LOO2) 실측  +6.99
  midother(y+1-middle+1-other)            로컬 첫 양수(uni0.20 +1.25, v58등가 -1.50 대비 +2.75)
공통점: '공유트리가 여러 head를 동시에 만족하는 split만 채택 -> 노이즈 split 억제'.
지금까지는 이 head들을 서로 다른 모델(멤버)로 쪼개 각자 0.10~0.20의 가중치를
따로 먹었다. 만약 정규화 효과가 head 수에 비례해서 강해진다면, 전부 한 트리에
합치는 게 개별 운용보다 나을 수 있다 -> 검증되지 않은 지점.

head 구성 (전부 '성공/양호 방향'으로 정렬, 실측된 것만 사용 -> strike/severe/reverse
는 로컬조차 약해서 제외):
  head0 = y
  head1 = 1 - lab_middle       (실측 성공 +7.72)
  head2 = 1 - lab_other        (실측 성공 +3.25)
  head3 = 1 - lab_ball         (실측 성공 +1.83)
  head4 = 투수-시즌 LOO(K=15)   (multires 성분, 실측 성공에 기여)
  head5 = 투수x손 LOO(K=15)     (multires 성분)
추론시엔 head0만 사용. Rule §4 준수(train 누적통계 차분/LOO만, test 행간참조 없음).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea48_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
K_PS = 15.0
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
lab_ball = recover("asof_pitcher_ball_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid) | np.isnan(lab_ball))
tot = y + lab_rev + lab_mid
lab_other = np.where(valid, (tot == 0).astype(np.float64), np.nan)
h_mid = np.where(valid, 1.0 - lab_mid, np.nan)
h_other = np.where(valid, 1.0 - lab_other, np.nan)
h_ball = np.where(valid, 1.0 - lab_ball, np.nan)
log(f"  유효 {valid.sum():,} ({valid.mean()*100:.2f}%)")

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

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base = A([f"phase90_cache/{tag}_base_{n}.npy" for n in ["d6", "d8", "sub"]])
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) *
                   np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
    mr = A([f"idea13_cache/{tag}_multires_s{k}.npy" for k in [42, 7]])
    od = A([f"idea13_cache/{tag}_ordinal_s{k}.npy" for k in [42, 7]])
    v47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
    b47 = sc(v47)
    log(f"  v47local={b47:.2f}")

    # 투수-시즌/투수x손 LOO (multires와 동일 구성)
    same_hand = X["same_hand"].to_numpy(np.float64) if "same_hand" in X.columns else np.zeros(len(X))
    ytr = y[tr_m]
    sub = pd.DataFrame({"pid": pid[tr_m], "season": seasons[tr_m], "sh": same_hand[tr_m], "y": ytr})
    g_glob = ytr.mean()
    ps_ = sub.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
    sub = sub.join(ps_, on=["pid", "season"])
    hl1 = ((sub["s"] - sub["y"]) + K_PS * g_glob) / ((sub["n"] - 1) + K_PS)
    psh = sub.groupby(["pid", "season", "sh"])["y"].agg(s2="sum", n2="count")
    sub = sub.join(psh, on=["pid", "season", "sh"])
    hl2 = ((sub["s2"] - sub["y"]) + K_PS * hl1) / ((sub["n2"] - 1) + K_PS)

    Ymat_tr = np.column_stack([ytr, h_mid[tr_m], h_other[tr_m], h_ball[tr_m],
                               hl1.to_numpy(np.float64), hl2.to_numpy(np.float64)])
    ps = []
    for seed in SEEDS:
        f = f"{CD}/{tag}_mega_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        ts = time.time()
        n_es = int(tr_m.sum() * 0.92)
        m = CatBoostRegressor(**CAT, random_seed=seed)
        m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
              eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
        p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
        np.save(f, p); ps.append(p)
        log(f"    s{seed} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    mega = np.mean(ps, axis=0)
    spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
    row = {"solo": sc(mega), "spread": spread}
    for wv in [0.15, 0.20, 0.25, 0.30]:
        row[f"w{wv}"] = sc((1 - wv) * v47 + wv * mega) - b47
    results[tag] = row
    log(f"  메가통합 단독={sc(mega):.2f} 시드폭={spread:.2f}  " +
        "  ".join(f"w{wv}={row[f'w{wv}']:+.2f}" for wv in [0.15, 0.20, 0.25, 0.30]))

print()
print("=" * 84)
print("메가 통합(6-head) vs 개별운용 실측이력 비교")
print("=" * 84)
for tag, r in results.items():
    print(f"\n[fold {tag}] 단독={r['solo']:.2f} 시드폭={r['spread']:.2f}")
    for wv in [0.15, 0.20, 0.25, 0.30]:
        d = r[f"w{wv}"]
        pred = f"{CAL_A + CAL_B*d:+.2f}" if tag == "A" else "-"
        print(f"   w={wv:.2f}  로컬Δ={d:+7.2f}   예상실측Δ={pred:>8}")
print()
print("참고: midother(2head) uni0.20 로컬Δ=+1.25 / mid단독 -0.18(실측+7.72) / other단독 -0.19(실측+3.25)")
print("판정: 6head가 midother(2head)의 +1.25보다 크면 '헤드수 비례 정규화' 가설 지지")
log(f"총 {time.time()-t0:.0f}s")
