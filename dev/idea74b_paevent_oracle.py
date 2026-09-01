"""PA 진행/종료 차분 라벨의 v66 잔차 오라클 상한과 시즌 안정성.

다음 행은 train 라벨 생성에만 사용한다. 결과는 새 aux target의 가치 판단용이며
test에서 이 라벨을 직접 복원하는 코드는 만들지 않는다.
"""

import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def build_event(df):
    d = df.sort_values("row_num").reset_index()
    same_pa = np.r_[
        (d.pitcher_id.to_numpy()[1:] == d.pitcher_id.to_numpy()[:-1])
        & (d.batter_id.to_numpy()[1:] == d.batter_id.to_numpy()[:-1])
        & (d.inning.to_numpy()[1:] == d.inning.to_numpy()[:-1])
        & (d.top_bottom.to_numpy()[1:] == d.top_bottom.to_numpy()[:-1]), False]
    b = d.balls_before.to_numpy(); s = d.strikes_before.to_numpy()
    bn = np.r_[b[1:], -99]; sn = np.r_[s[1:], -99]
    # 0=볼로 PA계속, 1=스트라이크 증가, 2=2스트라이크 파울, 3=PA종료/기타
    e = np.full(len(d), 3, dtype=np.int8)
    e[same_pa & (bn == b + 1) & (sn == s)] = 0
    e[same_pa & (sn == s + 1) & (bn == b)] = 1
    e[same_pa & (bn == b) & (sn == s) & (s == 2)] = 2
    out = np.empty(len(d), dtype=np.int8); out[d["index"].to_numpy()] = e
    return out


def v66_valid():
    avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
    base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
    hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
                   np.load(f"phase90_cache/A_snc_{n}.npy") for n in ("d6", "d8")], axis=0)
    mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
    od = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
    mo = avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42, 7)])
    cb = avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42, 7)])
    cr = avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42, 7)])
    f5 = avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42, 7)])
    return (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
            + .08 * cb + .08 * cr + .08 * f5)


def score(y, p):
    return 1e5 * (1 - np.mean((p-y)**2) / (y.mean() * (1-y.mean())))


def crossfit_group_correction(y, p, code, order, k):
    """2024 앞/뒤 절반에서 학습한 group residual을 반대 절반에 적용."""
    pred = p.copy()
    halves = order < np.median(order)
    for fit, val in ((halves, ~halves), (~halves, halves)):
        z = pd.DataFrame({"c": code[fit], "e": y[fit]-p[fit]}).groupby("c").e.agg(["sum", "count"])
        corr = (z["sum"] / (z["count"] + k)).to_dict()
        pred[val] += pd.Series(code[val]).map(corr).fillna(0).to_numpy()
    return pred


use = ["row_id", "season", "inning", "top_bottom", "balls_before", "strikes_before",
       "pitcher_id", "batter_id", "control_success"]
df = pd.read_csv("../data/train.csv", usecols=use, encoding="utf-8-sig")
df["row_num"] = df.row_id.str.replace("TRAIN_", "", regex=False).astype(int)
event = build_event(df)
cls5 = np.load("cls5_labels.npy").astype(int)
ptype = np.load("pitchtype_labels.npy").astype(int)
count = df.balls_before.to_numpy() * 3 + df.strikes_before.to_numpy()
yall = df.control_success.to_numpy(float)

names = ["cont_ball", "cont_strike", "foul2", "pa_end"]
print("시즌별 event 비중/성공률")
for yr in sorted(df.season.unique()):
    print(f"\n{yr}")
    sy = df.season.to_numpy() == yr
    for c, name in enumerate(names):
        m = sy & (event == c)
        print(f"  {name:12s} share={m[sy].mean():.4f} success={yall[m].mean():.5f}")

va = df.season.to_numpy() == 2024
y = yall[va]; p = v66_valid(); assert len(y) == len(p)
e = event[va]; c5 = cls5[va]; pt = ptype[va]; cnt = count[va]
order = df.loc[va, "row_num"].to_numpy()
base = score(y, p)
print(f"\nv66 local={base:.3f}")
codes = {
    "event4": e,
    "event4_x_count12": e*12+cnt,
    "event4_x_cls5": e*5+c5,
    "event4_x_cls15": e*15+c5*3+pt,
    "event4_x_count_x_cls15": (e*12+cnt)*15+c5*3+pt,
}
for name, code in codes.items():
    best = (-1e9, None)
    print(f"\n{name} classes={len(np.unique(code))}")
    for k in (0, 50, 200, 1000, 5000):
        q = crossfit_group_correction(y, p, code, order, k)
        delta = score(y, q)-base
        print(f"  K={k:4d} delta={delta:+.3f}")
        if delta > best[0]: best = (delta, k)
    print(f"  best={best[0]:+.3f} @K={best[1]}")

# 가장 중요한 nd&strike 안의 PA 종료 격차가 시즌마다 유지되는지 확인.
print("\nnd&strike: event별 성공률 시즌 안정성")
for yr in sorted(df.season.unique()):
    sy = (df.season.to_numpy() == yr) & (cls5 == 3)
    vals = []
    for c in (1, 2, 3):
        m = sy & (event == c)
        vals.append(f"{names[c]} n={m.sum():6d} p={yall[m].mean():.4f}")
    print(yr, " | ".join(vals))
