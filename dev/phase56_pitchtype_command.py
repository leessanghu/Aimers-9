"""구종별 커맨드 프로파일 확장(Domain.md Direction A) 검증 — baseline = v15 구성.

pt_dev(성공률 전용)는 실측 검증된 유일한 Trackman 성공 사례(+6.7). 이 스크립트는 같은
'구종 항등식 주변화' 메커니즘을 reverse/middle/ball/strike 4개로 확장한 버전을 검증한다.

주의(v17 전례): 같은 라벨 원천(pitchlabels.py 복원)을 쓴 label-conditioned 버전이 실측
-10.1/-3으로 실패했다. 이번엔 조건화 방식이 다르지만(구종 주변화 vs 일반 컨텍스트),
잔차가치 + 직접폴드 재확인 없이는 채택하지 않는다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchlabels import LABELS
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from pitchtype_command import PT_CMD_COLS, build_command_tables, build_matched_with_labels, transform_command
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
gr = build_global_rates(df)
dly = transform_lastyear(df, build_lastyear_table(df), gr, sr, k=30.0)

print("구종별 커맨드 프로파일(reverse/middle/ball/strike) 테이블 구축...", flush=True)
matched_cmd = build_matched_with_labels(df)
matched_cmd_valid = matched_cmd.dropna(subset=[f"lab_{n}" for n in LABELS])
cmd_tables = build_command_tables(matched_cmd_valid, sr)
glob_label = {n: float(matched_cmd[f"lab_{n}"].mean(skipna=True)) for n in LABELS}
dcmd = transform_command(df, cmd_tables, glob_label, sr)
print(f"  매칭 {len(matched_cmd_valid):,}행  ({time.time()-t0:.0f}s)", flush=True)
for c in PT_CMD_COLS:
    print(f"  {c:18s} SD={dcmd[c].std():.5f}  mean={dcmd[c].mean():.5f}", flush=True)

fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
ytr, yva = fold["y_train"], fold["y_valid"]
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
ti, ei = time_split_es(len(tr))


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                        verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
base = max(0, evaluate(yva, p)["bss"] * 1e5)
print(f"\nv15 단일시드 구성 score={base:.1f}  ({time.time()-t0:.0f}s)", flush=True)

e = yva - p
r = yva.mean()
bv = r * (1 - r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p ** 2])
XtX = Xd.T @ Xd + 1e-3 * np.eye(Xd.shape[1])

print("\n=== 구종 커맨드프로파일 개별 잔차가치 ===", flush=True)
tot = 0.0
for c in PT_CMD_COLS:
    z = dcmd.loc[va, c].to_numpy(np.float64)
    if z.std() < 1e-12:
        continue
    zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
    v = zp.var()
    if v < 1e-14:
        print(f"  {c:18s} (전부 설명됨)", flush=True)
        continue
    gain = (np.cov(e, zp)[0, 1] ** 2 / v) / bv * 1e5
    tot += gain
    print(f"  {c:18s} corr(e,z_perp)={np.corrcoef(e, zp)[0,1]:+.5f}  개별={gain:+6.2f}", flush=True)
print(f"  개별 합산(참고) = {tot:+.1f}", flush=True)

print("\n=== 직접 폴드 재확인 (v15구성 + 커맨드프로파일 8개, HGB+cat단일) ===", flush=True)
Xtr2 = pd.concat([Xtr, dcmd.loc[tr].reset_index(drop=True)], axis=1)
Xva2 = pd.concat([Xva, dcmd.loc[va].reset_index(drop=True)], axis=1)
h2 = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                    l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                    n_iter_no_change=20, random_state=SEED).fit(Xtr2, ytr)
cb2 = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                         verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb2.fit(Xtr2.iloc[ti], ytr[ti], eval_set=(Xtr2.iloc[ei], ytr[ei]))
p2 = 0.5 * h2.predict_proba(Xva2)[:, 1] + 0.5 * cb2.predict_proba(Xva2)[:, 1]
s2 = max(0, evaluate(yva, p2)["bss"] * 1e5)
print(f"  baseline(v15)={base:.1f}  +커맨드프로파일8={s2:.1f}  delta={s2-base:+.1f}", flush=True)

m19 = df["season"] == 2019
print(f"\n  2019행 |pt_reverse_dev| 최대={dcmd.loc[m19,'pt_reverse_dev'].abs().max():.2e} (0이어야 정상)", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
