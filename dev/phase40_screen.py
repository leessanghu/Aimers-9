"""남은 후보 3종을 잔차 지표로 사전 선별 (학습 없이). baseline = v14 구성(970.5 실증).

검증된 판정 규칙 (phase39): 예상이득 <2 기각 / >5 검증가치
  구종 +5.6 -> 실제 +6.7 (일치) / workload,form +0.4 -> 실제 -4.2 (가치없음 정확히 판정)
  단 이 지표는 '선형 추출 가능한 이득의 상한'이라 음수는 예측 못 한다.

후보: (A) 작년 한 시즌 피처 7개  (B) disjoint prev-game 블록  (C) Current Ability Teacher
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchcount_recover import recover_denominator
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
print(f"v14 피처 준비 ({time.time()-t0:.0f}s)", flush=True)

# ---------- (A) 작년 피처 ----------
dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)

# ---------- (B) disjoint prev-game 블록 ----------
n1 = recover_denominator(df, ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"], 200)
n3 = recover_denominator(df, ["asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev3_game_middle_rate"], 450)
n5 = recover_denominator(df, ["asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev5_game_middle_rate"], 700)
r1 = df["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
r3 = df["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
r5 = df["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
c1, c3, c5 = np.round(r1 * n1), np.round(r3 * n3), np.round(r5 * n5)
ok = ((~np.isnan(n1)) & (~np.isnan(n3)) & (~np.isnan(n5))
      & (n1 <= n3) & (n3 <= n5) & (c1 <= c3) & (c3 <= c5))
d23 = np.where(ok & ((n3 - n1) > 0), (c3 - c1) / np.maximum(n3 - n1, 1), np.nan)
d45 = np.where(ok & ((n5 - n3) > 0), (c5 - c3) / np.maximum(n5 - n3, 1), np.nan)
dblk = pd.DataFrame({
    "blk_r23": np.nan_to_num(d23, nan=g),
    "blk_r45": np.nan_to_num(d45, nan=g),
    "blk_trend_s": np.nan_to_num(r1 - d23, nan=0.0),
    "blk_trend_l": np.nan_to_num(d23 - d45, nan=0.0),
    "blk_ok": ok.astype(np.float64)}, index=df.index)
print(f"  disjoint 블록 일관 행 {ok.mean()*100:.1f}%", flush=True)

fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
ytr, yva = fold["y_train"], fold["y_valid"]
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
ti, ei = time_split_es(len(tr))


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X)], axis=1)


Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])

# ---------- (C) Teacher: 시즌 잔여 성공률 타깃, 2024를 안 본 모델 ----------
print(f"\nTeacher 학습 ({time.time()-t0:.0f}s)", flush=True)
d = df.sort_values(["pitcher_id", "season", "row_num"])
grp = d.groupby(["pitcher_id", "season"])["control_success"]
tot_s, tot_n = grp.transform("sum"), grp.transform("size")
cum_s, cum_n = grp.cumsum(), grp.cumcount() + 1
fut_s, fut_n = tot_s - cum_s, tot_n - cum_n
K_T = 50.0
tgt = ((fut_s + K_T * g) / (fut_n + K_T)).reindex(df.index)
fut_n = fut_n.reindex(df.index)

TF = ["asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
      "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
      "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
      "asof_pitcher_prev5_game_success_rate", "season", "game_month"]
Xt_all = pd.concat([df[TF].fillna(g), dins[INS]], axis=1).astype(np.float64)
m_tr = (df["season"] <= 2023).to_numpy() & (fut_n.to_numpy() >= 100)
samp = np.where(m_tr)[0][::4]
teach = HistGradientBoostingRegressor(max_depth=6, max_leaf_nodes=31, max_iter=400,
                                      learning_rate=0.05, l2_regularization=5.0,
                                      random_state=SEED).fit(Xt_all.iloc[samp], tgt.to_numpy()[samp])
theta = teach.predict(Xt_all)
dteach = pd.DataFrame({"theta": theta,
                       "theta_minus_prior": theta - pp,
                       "theta_minus_inseason": theta - dins["inseason_success_smooth"].to_numpy()},
                      index=df.index)
print(f"  학습표본 {len(samp):,}  theta SD={theta.std():.5f} (실력 진짜SD 0.0555 대비)", flush=True)
print(f"  theta_minus_inseason SD={dteach['theta_minus_inseason'].std():.5f}", flush=True)

print(f"\nv14 모델 학습 -> 잔차 ({time.time()-t0:.0f}s)", flush=True)
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                        verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
print(f"  v14 구성 score={max(0, evaluate(yva, p)['bss']*1e5):.1f}", flush=True)

e = yva - p
r = yva.mean()
bv = r * (1 - r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p ** 2])
XtX = Xd.T @ Xd + 1e-3 * np.eye(Xd.shape[1])


def value(zdf, tag):
    tot, lines = 0.0, []
    for col in zdf.columns:
        z = zdf[col].to_numpy(np.float64)
        if z.std() < 1e-12:
            continue
        zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
        v = zp.var()
        if v < 1e-14:
            lines.append(f"    {col:24s} (전부 설명됨)")
            continue
        gain = (np.cov(e, zp)[0, 1] ** 2 / v) / bv * 1e5
        tot += gain
        lines.append(f"    {col:24s} corr={np.corrcoef(e, zp)[0, 1]:+.5f}  예상이득={gain:+7.2f}")
    print(f"\n  [{tag}]  합계 = {tot:+.1f}", flush=True)
    for l in lines:
        print(l, flush=True)


print("\n" + "=" * 74 + "\n후보 선별 (규칙: <2 기각 / >5 검증가치)\n" + "=" * 74, flush=True)
value(dly.loc[va], "(A) 작년 피처 7개")
value(dblk.loc[va], "(B) disjoint prev-game 블록")
value(dteach.loc[va], "(C) Teacher")
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
