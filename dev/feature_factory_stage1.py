"""Feature Factory 1단계 — OOF/expanding target-encoding 후보 12개 family를
값싼 partial_gain(편상관, phase64b 방식)으로 빠르게 스캔. 모델학습 없음.

leakage-safe: 각 행은 row_num 기준 '이전 행들'만 사용하는 전역 expanding 통계
(season 안 나누고 전체 히스토리 누적 -- OOF spec의 "previous rows only" 패턴).
baseline p는 phase90_cache의 fold A(train<=2023) 검증예측(v35local, base+hurdle)을 재사용.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()
N_REF = 253507  # phase93/64b 기준 표본크기, sigma=1/sqrt(n)


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


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
df = df.sort_values("row_num").reset_index(drop=True)
y_all = df["control_success"].to_numpy(np.float64)
g = float(y_all.mean())

df["count_state"] = df["balls_before"] * 4 + df["strikes_before"]
df["li_bin"] = pd.qcut(df["li"], 8, labels=False, duplicates="drop")
df["score_diff_bin"] = pd.cut(df["score_diff_pitcher_team"], bins=[-99, -3, -1, 0, 1, 3, 99],
                              labels=False)

va_m = (df["season"] == 2024).to_numpy()
p_base = np.load("phase90_cache/A_base_d6.npy")  # fold A(train<=2023) 검증(2024) 예측, d6 하나만이라도 baseline으론 충분
# phase90_cache 예측 길이가 va_m 개수와 일치하는지 확인
assert p_base.shape[0] == va_m.sum(), f"길이 불일치: {p_base.shape[0]} vs {va_m.sum()}"

FAMILIES = [
    ("OOF001", ["season", "game_type", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF011", ["season", "game_type", "inning"], "expanding_global"),
    ("OOF021", ["game_type", "base_state", "outs_before"], "expanding_global"),
    ("OOF031", ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF041", ["pitcher_hand", "batter_hand", "base_state", "outs_before"], "expanding_global"),
    ("OOF091", ["season", "game_month", "game_type"], "expanding_global"),
    ("OOF101", ["season", "game_month", "pitcher_hand", "batter_hand"], "expanding_global"),
    ("OOF111", ["base_state", "num_runners_on", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF121", ["game_type", "li_bin", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF131", ["game_type", "score_diff_bin", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF051", ["pitcher_team_id", "game_type", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF061", ["batter_team_id", "game_type", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF071", ["pitcher_team_id", "batter_hand", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF081", ["batter_team_id", "pitcher_hand", "balls_before", "strikes_before"], "expanding_global"),
    ("OOF141", ["pitcher_id", "season"], "expanding_global"),
    ("OOF151", ["pitcher_id", "season", "batter_hand"], "expanding_global"),
    ("OOF161", ["batter_id", "season", "pitcher_hand"], "expanding_global"),
]

TF_FAMILIES = [
    ("TF001", ["season", "game_type", "balls_before", "strikes_before"], "li"),
    ("TF011", ["season", "game_type", "inning"], "li"),
    ("TF021", ["game_type", "base_state", "outs_before"], "home_win_expectancy"),
    ("TF031", ["game_type", "base_state", "balls_before", "strikes_before"], "score_diff_pitcher_team"),
    ("TF041", ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"], "li"),
    ("TF051", ["pitcher_team_id", "season", "game_month"], "asof_pitcher_success_rate"),
    ("TF061", ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"], "score_diff_pitcher_team"),
    ("TF071", ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"], "asof_pitcher_ball_rate"),
    ("TF081", ["season", "game_month", "pitcher_hand"], "li"),
    ("TF091", ["season", "game_month", "pitcher_hand"], "asof_pitcher_breaking_rate"),
    ("TF101", ["season", "game_month", "pitcher_hand"], "asof_pitcher_offspeed_rate"),
    ("TF111", ["pitcher_team_id", "batter_hand", "balls_before", "strikes_before"], "asof_pitcher_fastball_rate"),
    ("TF121", ["pitcher_team_id", "batter_hand", "balls_before", "strikes_before"], "asof_pitcher_breaking_rate"),
    ("TF131", ["pitcher_team_id", "batter_hand", "balls_before", "strikes_before"], "asof_pitcher_offspeed_rate"),
    ("TF141", ["game_type", "base_state", "num_runners_on"], "li"),
    ("TF151", ["game_type", "base_state", "num_runners_on"], "asof_pitcher_middle_rate"),
    ("TF161", ["pitcher_hand", "batter_hand", "base_state", "outs_before"], "asof_batter_middle_rate"),
    ("TF171", ["season", "game_month", "game_type"], "li"),
    ("TF181", ["season", "game_month", "game_type"], "asof_pitcher_prev5_game_success_rate"),
]

K_VALUES = [20.0, 50.0]

log(f"{len(FAMILIES)}개 family x {len(K_VALUES)}개 K = {len(FAMILIES)*len(K_VALUES)}개 후보 스캔...")
results = []
for fam_id, keys, window in FAMILIES:
    grp = df.groupby(keys)["control_success"]
    cum_n = grp.cumcount()
    cum_s = grp.cumsum() - df["control_success"]  # 이전 행까지의 누적합(자기 자신 제외)
    for k in K_VALUES:
        rate = (cum_s + k * g) / (cum_n + k)
        z_va = rate.to_numpy()[va_m]
        gain, pc = partial_gain(y_all[va_m], p_base, z_va)
        results.append(dict(family=fam_id, keys="+".join(keys), k=k, gain=gain, partial_corr=pc,
                            mean_n=float(cum_n[va_m].mean())))
        log(f"  {fam_id} (k={k}): gain={gain:+.2f}  pc={pc:+.4f}  평균지지표본={cum_n[va_m].mean():.0f}")

log(f"\ntarget-free {len(TF_FAMILIES)}개 family 스캔...")
for fam_id, keys, src in TF_FAMILIES:
    grp = df.groupby(keys)[src]
    cum_n = grp.cumcount()
    cum_mean_shifted = grp.apply(lambda s: s.shift(1).expanding().mean())
    cum_std_shifted = grp.apply(lambda s: s.shift(1).expanding().std())
    for agg_name, z_full in [("mean", cum_mean_shifted), ("std", cum_std_shifted)]:
        z_va = z_full.to_numpy()[va_m] if hasattr(z_full, "to_numpy") else np.asarray(z_full)[va_m]
        gain, pc = partial_gain(y_all[va_m], p_base, z_va)
        results.append(dict(family=f"{fam_id}_{agg_name}", keys="+".join(keys) + f"|{src}", k=np.nan,
                            gain=gain, partial_corr=pc, mean_n=float(cum_n[va_m].mean())))
        log(f"  {fam_id}_{agg_name} ({src}): gain={gain:+.2f}  pc={pc:+.4f}")

res = pd.DataFrame(results).sort_values("gain", ascending=False)
print()
print("=" * 90)
print(res.to_string(index=False))
res.to_csv("feature_factory_stage1_results.csv", index=False)

threshold_loose = 4.0  # 1시그마 (관대한 1단계 기준 -- 통과하면 2단계로)
survivors = res[res["gain"] > threshold_loose]
print(f"\n1단계 관대한 기준(1시그마={threshold_loose}) 통과: {len(survivors)}/{len(res)}개")
if len(survivors):
    print(survivors[["family", "keys", "k", "gain"]].to_string(index=False))
log(f"총 {time.time()-t0:.0f}s")
