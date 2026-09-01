"""감사 영역 5 — 현재 최고모델의 Brier 손실을 구간별로 분해.
225점(Brier 0.00056)이 현실적으로 어느 구간에서 나올 수 있는지 수치로 특정.

각 구간 g에 대해:
  share      = 표본 비중
  Brier_g    = 그 구간의 Brier
  base_g     = r_g(1-r_g)  (구간 자체 baseline)
  손실기여   = share*Brier_g / BSref_global  (전체 점수에서 차지하는 손실)
  bias_g     = mean(p) - r_g
  상수보정이득 = 1e5 * share * bias_g^2 / BSref_global   <- 구간별 상수보정만으로 얻을 값
  완전해결상한 = 1e5 * share * (Brier_g - base_g_resid) / BSref  <- 구간을 r_g로 대체시
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()
VAL = 2024


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


meta = pd.read_parquet("featcache_meta.parquet")
X = pd.read_parquet("featcache_X.parquet")
y = meta["control_success"].to_numpy(np.float64)
s = meta["season"].to_numpy(np.float64)
va = s == VAL
yv = y[va]
R = yv.mean(); BSREF = R * (1 - R)

# 현재 최고 구성(v47)의 로컬 근사: base3 + hurdle (multires/ordinal은 폴드캐시 있음)
base3 = np.mean([np.load(f"phase90_cache/A_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) * np.load(f"phase90_cache/A_snc_{n}.npy")
               for n in ["d6", "d8"]], axis=0)
mr = np.mean([np.load(f"idea13_cache/A_multires_s{k}.npy") for k in [42, 7]], axis=0)
od = np.mean([np.load(f"idea13_cache/A_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
p = np.clip(0.30 * base3 + 0.40 * hur + 0.10 * mr + 0.20 * od, 1e-9, 1 - 1e-9)
score = 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)
log(f"fold A 검증 n={va.sum():,}  r={R:.4f}  BSref={BSREF:.6f}")
log(f"v47구성 로컬 점수 = {score:.2f}  (전체 Brier={np.mean((p-yv)**2):.6f})")
log(f"목표: Brier를 0.00056 줄이면 +225점")
print()

# 세그먼트 정의
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["row_id", "pitcher_id", "batter_id", "inning", "balls_before",
                           "strikes_before", "pitcher_hand", "batter_hand", "game_type",
                           "asof_pitcher_n", "asof_batter_n"])
raw["row_num"] = raw["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
raw = raw.set_index("row_num").loc[meta["row_num"].to_numpy()].reset_index()
rv = raw[va].reset_index(drop=True)

hist_p = set(raw.loc[s <= 2023, "pitcher_id"])
hist_b = set(raw.loc[s <= 2023, "batter_id"])
tm_matched = X.loc[va, "tm_matched"].to_numpy() if "tm_matched" in X.columns else np.ones(va.sum())

SEGS = {
    "투수 seen/unseen": np.where(rv["pitcher_id"].isin(hist_p), "seen", "UNSEEN"),
    "타자 seen/unseen": np.where(rv["batter_id"].isin(hist_b), "seen", "UNSEEN"),
    "투수 누적표본": pd.cut(rv["asof_pitcher_n"].fillna(0), [-1, 100, 500, 2000, 1e9],
                       labels=["~100", "100-500", "500-2k", "2k+"]).astype(str),
    "타자 누적표본": pd.cut(rv["asof_batter_n"].fillna(0), [-1, 100, 500, 2000, 1e9],
                       labels=["~100", "100-500", "500-2k", "2k+"]).astype(str),
    "count_state": (rv["balls_before"] * 4 + rv["strikes_before"]).astype(str),
    "inning": pd.cut(rv["inning"], [0, 3, 6, 9], labels=["1-3", "4-6", "7-9"]).astype(str),
    "hand matchup": (rv["pitcher_hand"].astype(str) + "v" + rv["batter_hand"].astype(str)),
    "game_type": rv["game_type"].astype(str),
    "Trackman 매칭": np.where(tm_matched > 0, "matched", "UNMATCHED"),
    "예측확률 구간": pd.cut(p, [0, .40, .45, .50, .55, .60, 1.0]).astype(str),
    "모델 불일치도": pd.qcut(np.abs(base3 - hur), 4, labels=["낮음", "중", "높음", "최고"]).astype(str),
}

rows = []
for segname, lab in SEGS.items():
    lab = pd.Series(np.asarray(lab), name="g")
    for gname, idx in lab.groupby(lab).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 200:
            continue
        yy, pp = yv[idx], p[idx]
        share = len(idx) / len(yv)
        br = np.mean((pp - yy) ** 2)
        rg = yy.mean(); baseg = rg * (1 - rg)
        bias = pp.mean() - rg
        # 이 구간에 상수보정만 했을 때 전체점수 이득
        const_gain = 1e5 * share * bias ** 2 / BSREF
        # 이 구간을 r_g(구간평균)로 완전 대체했을 때 (= 구간내 판별 포기, 보정만 완벽)
        br_perfect_cal = np.mean((rg - yy) ** 2)
        # 이 구간의 Brier를 0으로 만들면(완전해결) 얻는 전체점수
        full_gain = 1e5 * share * br / BSREF
        rows.append(dict(seg=segname, g=str(gname), share=share, n=len(idx), brier=br,
                         base_g=baseg, rate=rg, pred=pp.mean(), bias=bias,
                         loss_contrib=1e5 * share * br / BSREF,
                         const_gain=const_gain, full_gain=full_gain))

res = pd.DataFrame(rows)
res.to_csv("audit_error_decomp.csv", index=False)

for segname in SEGS:
    d = res[res.seg == segname].sort_values("share", ascending=False)
    print("=" * 100)
    print(f"[{segname}]")
    print(f"{'구간':<14}{'비중':>7}{'실제율':>8}{'예측평균':>9}{'bias':>8}{'Brier':>9}"
          f"{'손실기여':>9}{'상수보정이득':>12}")
    for _, r_ in d.iterrows():
        print(f"{r_.g:<14}{r_.share*100:6.1f}%{r_.rate:8.4f}{r_.pred:9.4f}{r_.bias:+8.4f}"
              f"{r_.brier:9.5f}{r_.loss_contrib:9.1f}{r_.const_gain:12.2f}")
    print(f"{'  -> 이 분할의 상수보정 총이득':<40}{d.const_gain.sum():8.2f}점")
print()
print("=" * 100)
print("각 분할별 '구간별 상수보정만으로 얻을 수 있는 총이득' 랭킹 (225점과 비교)")
tot = res.groupby("seg")["const_gain"].sum().sort_values(ascending=False)
for k, v in tot.items():
    print(f"  {k:<20} {v:8.2f}점   ({v/225*100:5.1f}% of 225)")
log(f"총 {time.time()-t0:.0f}s")
