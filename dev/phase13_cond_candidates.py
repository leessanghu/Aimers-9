"""신규 조건부 분할 후보 4개 + 이닝 K 재조정 — 2024 폴드 단일 검증(가장 빠르고 실전과 가장 가까움).

baseline = v7c 실전 구성 (58 base + 5 in-season + 2 platoon + 2 inning[K=570]) = 실제 948.970점

phase12b에서 cov 기반으로 유도한 실질상한/K:
  투수x볼카운트  79.6점  K=1256
  타자x투수손    78.4점  K=1275
  투수x월        76.7점  K=1304
  투수x상대팀    58.1점  K=1721
  (대조군) 투수x이닝 24.5점 K=4076 <- 현재 570을 쓰고 있어 과소축소. 재조정 arm 추가.

로컬 2024 폴드 델타 x 0.47 ~= 실제 점수 (이닝: 로컬+19.9 -> 실제+9.29로 검량됨)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from cond_split import (build_cond_offset, build_cond_table, build_entity_season_end,
                        lookup_prior_rate, transform_cond)
from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import build_fold
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]
TRAIN_MAX, VALID_SEASON = 2023, 2024


def run_hgb(Xtr, ytr, Xva, yva, tag):
    t = time.time()
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xtr, ytr)
    bss = evaluate(yva, hgb.predict_proba(Xva)[:, 1])["bss"]
    print(f"  [{tag:22s}] {Xtr.shape[1]}피처  BSS={bss:.6f}  score={max(0,bss*100000):7.1f}  ({time.time()-t:.0f}s)", flush=True)
    return bss


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df["control_success"].mean())
    sr = sorted(df["season"].unique().tolist())

    # ----- v7c baseline 재현 -----
    season_end = build_season_end_table(df)
    df_ins = transform_inseason(df, season_end, g, sr)
    piv = _pivots_from_table(season_end, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior_p = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    df_plt = transform_platoon(df, build_platoon_table(df), prior_p, sr, k=K_PLATOON)
    inn_tbl, inn_off = build_inning_table(df), build_inning_offset(df)
    df_inn570 = transform_inning(df, inn_tbl, inn_off, prior_p, sr, k=570.0)
    df_inn4076 = transform_inning(df, inn_tbl, inn_off, prior_p, sr, k=4076.0)

    # 타자쪽 prior (asof_batter_* 로 복원)
    b_end = build_entity_season_end(df, "batter_id", "asof_batter_n", "asof_batter_success_rate")
    prior_b = lookup_prior_rate(df, b_end, "batter_id", g, sr)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    # ----- 신규 후보 정의 -----
    count_ctx = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    cands = [
        ("pcount", "pitcher_id", count_ctx, prior_p, 1256.0),
        ("bplat", "batter_id", df["pitcher_hand"].to_numpy(), prior_b, 1275.0),
        ("pmonth", "pitcher_id", df["game_month"].to_numpy(), prior_p, 1304.0),
        ("popp", "pitcher_id", df["batter_team_id"].to_numpy(), prior_p, 1721.0),
    ]
    cand_feats = {}
    for name, ent, ctx, prior, k in cands:
        tbl = build_cond_table(df, ent, ctx)
        off = build_cond_offset(df, ctx)
        f = transform_cond(df, tbl, off, ent, ctx, prior, sr, k, name)
        cand_feats[name] = f
        print(f"  {name}: 셀={len(tbl):,}  diff_SD={f[f'{name}_diff'].std():.5f}", flush=True)

    # ----- 폴드 구성 -----
    fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= TRAIN_MAX].index, df[df.season == VALID_SEASON].index

    def stack(base_frame, idx_, extra=()):
        parts = [base_frame.reset_index(drop=True),
                 df_ins.loc[idx_, INSEASON_COLS].reset_index(drop=True),
                 df_plt.loc[idx_].reset_index(drop=True)]
        parts += [e.loc[idx_].reset_index(drop=True) for e in extra]
        return pd.concat(parts, axis=1)

    Xtr_b = stack(fold["X_train"], tr, [df_inn570])
    Xva_b = stack(fold["X_valid"], va, [df_inn570])

    print(f"\n{'='*66}\n2024 폴드 (baseline = v7c 구성, 실제 948.970점)\n{'='*66}", flush=True)
    res = {"baseline(v7c)": run_hgb(Xtr_b, ytr, Xva_b, yva, "baseline(v7c)")}

    # 이닝 K 재조정
    res["inning_K4076"] = run_hgb(stack(fold["X_train"], tr, [df_inn4076]), ytr,
                                  stack(fold["X_valid"], va, [df_inn4076]), yva, "inning K570->4076")

    # 신규 후보 개별 추가
    for name, *_ in cands:
        f = cand_feats[name]
        res[f"+{name}"] = run_hgb(pd.concat([Xtr_b, f.loc[tr].reset_index(drop=True)], axis=1), ytr,
                                  pd.concat([Xva_b, f.loc[va].reset_index(drop=True)], axis=1), yva, f"+{name}")

    b = res["baseline(v7c)"]
    print(f"\n{'='*66}\nbaseline 대비 (로컬 델타 x0.47 ~= 실제 예상)\n{'='*66}", flush=True)
    for k_, v in res.items():
        if k_ == "baseline(v7c)":
            continue
        d = 100000 * (v - b)
        print(f"  {k_:20s} delta={d:+7.1f}   실제예상={d*0.47:+6.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
