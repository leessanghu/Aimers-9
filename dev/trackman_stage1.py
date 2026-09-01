"""Trackman D(상황별 모집단 분포) + B(구종간 기하학) 1단계 스크리닝.
D: target-free, count/inning/hand별 rel_speed/spin_rate/movement 분포 (전체표본, 저위험)
B: (pitcher_id,season) x pitch_type_group 중심점 간 거리 (fastball-breaking-offspeed)
   -- 지금까지 구종 "내부" SD만 썼지 구종 "간" 거리는 처음.
partial_gain(phase64b 방식)으로 1차 스캔. baseline은 phase90_cache의 fold A d6.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def partial_gain(y, p, z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    cz = np.linalg.lstsq(A, z, rcond=None)[0]
    ry, rz = y - A @ cy, z - A @ cz
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


log("train.csv 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
df = df.sort_values("row_num").reset_index(drop=True)
y_all = df["control_success"].to_numpy(np.float64)
g = float(y_all.mean())
va_m = (df["season"] == 2024).to_numpy()

p_base = np.load("phase90_cache/A_base_d6.npy")
assert p_base.shape[0] == va_m.sum()

# ============ D: 상황별 Trackman 모집단 분포 (target-free) ============
log("트릭맨 로드 (D용, 상황 컨텍스트만)...")
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
USE_D = ["season", "balls_before", "strikes_before", "inning", "pitcher_hand", "batter_hand"] + PHYS
tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig", usecols=USE_D)
tm["count_state"] = tm["balls_before"] * 4 + tm["strikes_before"]
HAND_MAP = {"Right": 2, "Left": 1}
tm["pitcher_hand"] = tm["pitcher_hand"].map(HAND_MAP)
tm["batter_hand"] = tm["batter_hand"].map(HAND_MAP)
log(f"  트릭맨 {len(tm):,}행")

results = []
D_FAMILIES = [
    ("D_count_speed", ["season", "count_state"], "rel_speed"),
    ("D_count_spin", ["season", "count_state"], "spin_rate"),
    ("D_count_ivb", ["season", "count_state"], "induced_vert_break"),
    ("D_count_hb", ["season", "count_state"], "horz_break"),
    ("D_inning_speed", ["season", "inning"], "rel_speed"),
    ("D_hand_speed", ["season", "pitcher_hand", "batter_hand"], "rel_speed"),
    ("D_hand_break", ["season", "pitcher_hand", "batter_hand"], "induced_vert_break"),
]
# train.csv 쪽 count_state를 붙여서 join (season 기준, 리그 전체 통계라 개인 leakage 없음
# -- 각 행은 자기 자신을 제외한 전체분포를 봐야 하나, 표본이 워낙 커서(수십만) 자기포함 편향 무시 가능)
df["count_state"] = df["balls_before"] * 4 + df["strikes_before"]

for fam_id, keys, src in D_FAMILIES:
    agg = tm.groupby(keys)[src].agg(["mean", "std"]).reset_index()
    agg.columns = list(keys) + [f"{fam_id}_mean", f"{fam_id}_std"]
    merged = df[["row_num"] + keys].merge(agg, on=keys, how="left")
    assert len(merged) == len(df)
    for stat in ["mean", "std"]:
        col = f"{fam_id}_{stat}"
        z = merged[col].to_numpy(np.float64)[va_m]
        gain, pc = partial_gain(y_all[va_m], p_base, z)
        results.append(dict(family=col, keys="+".join(keys) + f"|{src}", gain=gain, partial_corr=pc))
        log(f"  {col}: gain={gain:+.2f}  pc={pc:+.4f}")

# ============ B: 구종간 기하학 (pitcher-season centroid distance) ============
log("트릭맨 로드 (B용, 투수매핑 필요)...")
m = pd.read_csv("pitcher_map.csv").sort_values("sim", ascending=False).drop_duplicates("tm_id")
t2p = m.set_index("tm_id")["pitcher_id"]
USE_B = ["season", "pitcher_trackman_id", "pitch_type_group"] + PHYS + ["rel_height", "rel_side"]
tmb = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig", usecols=USE_B)
tmb = tmb.rename(columns={"pitcher_trackman_id": "tm_id"})
tmb["pitcher_id"] = tmb["tm_id"].map(t2p)
tmb = tmb.dropna(subset=["pitcher_id"])
tmb["pitcher_id"] = tmb["pitcher_id"].astype(np.int64)
tmb = tmb[tmb["pitch_type_group"].isin(["fastball", "breaking", "offspeed"])]
log(f"  매핑후 {len(tmb):,}행")

PB = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]
cent = tmb.groupby(["pitcher_id", "season", "pitch_type_group"])[PB + ["rel_height", "rel_side"]].mean()
cnt = tmb.groupby(["pitcher_id", "season", "pitch_type_group"]).size().rename("n")
cent = cent.join(cnt).reset_index()

def pair_dist(cent, cols, g1, g2):
    a = cent[cent.pitch_type_group == g1].set_index(["pitcher_id", "season"])
    b = cent[cent.pitch_type_group == g2].set_index(["pitcher_id", "season"])
    idx = a.index.intersection(b.index)
    d = np.sqrt(((a.loc[idx, cols].to_numpy() - b.loc[idx, cols].to_numpy()) ** 2).sum(axis=1))
    return pd.Series(d, index=idx)

PAIRS = [("fastball", "breaking"), ("fastball", "offspeed"), ("breaking", "offspeed")]
B_results = {}
for g1, g2 in PAIRS:
    B_results[f"B_movedist_{g1}_{g2}"] = pair_dist(cent, ["induced_vert_break", "horz_break"], g1, g2)
    B_results[f"B_reldist_{g1}_{g2}"] = pair_dist(cent, ["rel_height", "rel_side"], g1, g2)
    B_results[f"B_speeddist_{g1}_{g2}"] = pair_dist(cent, ["rel_speed"], g1, g2)

log("B: 직전시즌(season-1) 값을 각 행에 lookup...")
for name, ser in B_results.items():
    ser = ser.reset_index()
    ser.columns = ["pitcher_id", "season", "val"]
    ser["season"] = ser["season"] + 1  # 직전시즌 값을 다음 시즌 행에 붙임 (leakage 안전)
    merged = df[["row_num", "pitcher_id", "season"]].merge(ser, on=["pitcher_id", "season"], how="left")
    assert len(merged) == len(df)
    z = merged["val"].to_numpy(np.float64)[va_m]
    gain, pc = partial_gain(y_all[va_m], p_base, z)
    results.append(dict(family=name, keys="pitcher_id+season-1", gain=gain, partial_corr=pc))
    log(f"  {name}: gain={gain:+.2f}  pc={pc:+.4f}  결측률={np.isnan(z).mean()*100:.1f}%")

res = pd.DataFrame(results).sort_values("gain", ascending=False)
print()
print("=" * 90)
print(res.to_string(index=False))
res.to_csv("trackman_stage1_results.csv", index=False)
survivors = res[res.gain > 4.0]
print(f"\n1단계 관대한 기준(1시그마=4.0) 통과: {len(survivors)}/{len(res)}개")
if len(survivors):
    print(survivors.to_string(index=False))
log(f"총 {time.time()-t0:.0f}s")
