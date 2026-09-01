"""플래툰 스플릿 피처 3폴드 검증.

근거(노이즈 제거 분산 회계):
  투수 실력 개인차 진짜SD=0.0555 / 플래툰 스플릿 개인차 진짜SD=0.0438 (79%)
  -> 상한 약 192점. 같이 측정한 투수x카운트 상호작용은 진짜SD=0.0117로 거의 노이즈라 폐기.

baseline = v4 (58 base + 5 in-season, 실제 925.908점 구성)
arm      = baseline + platoon_diff/platoon_n (K 민감도 2종)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import FOLDS, build_fold
from platoon import build_platoon_table, transform_platoon

SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)

INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]


def get_prior_rate(df, season_end_table, global_success_rate, seasons_range):
    """각 행의 '직전 시즌 끝 시점' 투수 marginal 성공률 (플래툰 축소의 기준점)."""
    pivots = _pivots_from_table(season_end_table, seasons_range)
    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = pivots["rate"].reindex(lookup_idx).to_numpy()
    return pd.Series(vals).fillna(global_success_rate).to_numpy(np.float64)


def run_eval(X_train, y_train, X_valid, y_valid, tag):
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    p_rf = rf.predict_proba(X_valid)[:, 1]
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    rows = {
        "rf": evaluate(y_valid, p_rf)["bss"],
        "hgb": evaluate(y_valid, p_hgb)["bss"],
        "rf015_hgb085": evaluate(y_valid, 0.15 * p_rf + 0.85 * p_hgb)["bss"],
    }
    print(f"  [{tag}]", flush=True)
    for k, v in rows.items():
        print(f"    {k:15s} BSS={v:.6f}  score={max(0, v*100000):.1f}", flush=True)
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    global_success_rate = float(df["control_success"].mean())
    seasons_range = sorted(df["season"].unique().tolist())

    print("in-season + 플래툰 테이블 구성...", flush=True)
    season_end = build_season_end_table(df)
    df_inseason = transform_inseason(df, season_end, global_success_rate, seasons_range)
    prior_rate = get_prior_rate(df, season_end, global_success_rate, seasons_range)
    platoon_table = build_platoon_table(df)
    print(f"  플래툰 셀 {len(platoon_table):,}개  ({time.time()-t0:.0f}s)", flush=True)

    platoon_by_k = {}
    for k in (520.0, 150.0):
        pf = transform_platoon(df, platoon_table, prior_rate, seasons_range, k=k)
        platoon_by_k[k] = pf
        d = pf["platoon_diff"]
        print(f"  K={k:.0f}: platoon_diff SD={d.std():.5f}  |비영값|비율={(d.abs()>1e-9).mean():.3f}", flush=True)

    all_results = {}
    for train_max, valid_season in FOLDS:
        print(f"\n{'='*60}\nFOLD train<={train_max} valid={valid_season}\n{'='*60}", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True)
        y_train, y_valid = fold["y_train"], fold["y_valid"]
        tr_idx = df[df["season"] <= train_max].index
        va_idx = df[df["season"] == valid_season].index

        Xtr = pd.concat([fold["X_train"].reset_index(drop=True),
                         df_inseason.loc[tr_idx, INSEASON_COLS].reset_index(drop=True)], axis=1)
        Xva = pd.concat([fold["X_valid"].reset_index(drop=True),
                         df_inseason.loc[va_idx, INSEASON_COLS].reset_index(drop=True)], axis=1)

        print(f"\n--- baseline (v4, {Xtr.shape[1]}피처) ---", flush=True)
        fold_res = {"baseline": run_eval(Xtr, y_train, Xva, y_valid, "baseline")}

        for k, pf in platoon_by_k.items():
            name = f"+platoon_K{k:.0f}"
            xt = pd.concat([Xtr, pf.loc[tr_idx].reset_index(drop=True)], axis=1)
            xv = pd.concat([Xva, pf.loc[va_idx].reset_index(drop=True)], axis=1)
            print(f"\n--- {name} ({xt.shape[1]}피처) ---", flush=True)
            fold_res[name] = run_eval(xt, y_train, xv, y_valid, name)

        print(f"\n--- {valid_season} baseline 대비 delta score ---", flush=True)
        b = fold_res["baseline"]
        for name in fold_res:
            if name == "baseline":
                continue
            print(f"    {name:18s} " + "  ".join(
                f"{k}={100000*(fold_res[name][k]-b[k]):+7.1f}" for k in b), flush=True)
        all_results[valid_season] = fold_res

    print(f"\n{'='*60}\n전체 요약 — baseline 대비 delta score\n{'='*60}", flush=True)
    for model in ["rf", "hgb", "rf015_hgb085"]:
        print(f"\n [{model}]", flush=True)
        for season, fr in all_results.items():
            b = fr["baseline"][model]
            line = "  ".join(f"{n}={100000*(fr[n][model]-b):+7.1f}"
                             for n in fr if n != "baseline")
            print(f"   {season}  baseline={max(0,b*100000):7.1f} | {line}", flush=True)
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
