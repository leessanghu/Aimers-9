"""v5(규칙위반, test.csv 행 순서 참조) 폐기 후 대체안 검증.

1순위: workload/역할 피처 — prev1/prev5_game_success_rate에 숨은 분모(투구 수)를
       연분수형 근사로 복원. recent1_pitch_n, recent5_pitch_n, recent1_vs_avg5_ratio.
2순위: 폼(form) 피처 — prev1/3/5_game_success_rate - inseason_success_smooth (자기 시즌
       기준선 대비 최근 편차), prev1-prev5 트렌드.

둘 다 그 행 자신의 컬럼만 사용(asof_pitcher_prev*, inseason_success_smooth는 '직전 시즌
끝 시점'만 참조) — test.csv 다른 행 참조 절대 없음. build_fold의 fit은 train_fold에서만.

2022/2023/2024 3-fold 전부 검증.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from inseason import build_season_end_table, transform_inseason
from metrics import evaluate
from phase2_common import FOLDS, build_fold, time_split_es
from pitchcount_recover import build_workload_features

SEED = 42

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)

INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]


def build_form_features(df, inseason_success_smooth):
    p1 = df["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
    p3 = df["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
    p5 = df["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    base = inseason_success_smooth.to_numpy(np.float64)

    out = pd.DataFrame(index=df.index)
    out["form_prev1"] = np.nan_to_num(p1 - base, nan=0.0)
    out["form_prev3"] = np.nan_to_num(p3 - base, nan=0.0)
    out["form_prev5"] = np.nan_to_num(p5 - base, nan=0.0)
    out["form_trend"] = np.nan_to_num(p1 - p5, nan=0.0)
    return out


def run_eval(X_train, y_train, X_valid, y_valid, tag):
    rows = {}
    tr_idx, es_idx = time_split_es(len(X_train))

    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    p_rf = rf.predict_proba(X_valid)[:, 1]
    rows["rf"] = evaluate(y_valid, p_rf)["bss"]

    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    rows["hgb"] = evaluate(y_valid, p_hgb)["bss"]

    rows["rf015_hgb085"] = evaluate(y_valid, 0.15 * p_rf + 0.85 * p_hgb)["bss"]

    lgb = LGBMRegressor(**LGBM_PARAMS)
    lgb.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
           eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
           callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    p_lgb = np.clip(lgb.predict(X_valid), 0.0, 1.0)
    rows["lgbm_a"] = evaluate(y_valid, p_lgb)["bss"]

    print(f"  [{tag}]")
    for k, v in rows.items():
        print(f"    {k:15s} BSS={v:.6f}  score={max(0, v*100000):.1f}")
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    global_success_rate = float(df["control_success"].mean())

    print("season_end_table + workload 복원 (전체 df, row-wise, 안전) ...", flush=True)
    season_end = build_season_end_table(df)
    seasons_range = sorted(df["season"].unique().tolist())
    df_inseason = transform_inseason(df, season_end, global_success_rate, seasons_range)
    df_workload = build_workload_features(df)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    # 복원 정합성 체크: recent1_pitch_n(log1p) 역변환값이 그럴듯한 범위인지
    q1_check = np.expm1(df_workload["recent1_pitch_n"])
    print(f"  recent1 투구수 복원 분포: median={np.median(q1_check[q1_check>0]):.1f} "
          f"p90={np.percentile(q1_check[q1_check>0], 90):.1f} max={q1_check.max():.1f}", flush=True)

    all_results = {}
    for train_max, valid_season in FOLDS:
        print(f"\n{'='*60}\nFOLD train<={train_max} valid={valid_season}\n{'='*60}", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True)
        X_train_base, X_valid_base = fold["X_train"], fold["X_valid"]
        y_train, y_valid = fold["y_train"], fold["y_valid"]

        train_idx = df[df["season"] <= train_max].index
        valid_idx = df[df["season"] == valid_season].index

        ins_train = df_inseason.loc[train_idx, INSEASON_COLS].reset_index(drop=True)
        ins_valid = df_inseason.loc[valid_idx, INSEASON_COLS].reset_index(drop=True)

        wl_train = df_workload.loc[train_idx].reset_index(drop=True)
        wl_valid = df_workload.loc[valid_idx].reset_index(drop=True)

        form_train = build_form_features(df.loc[train_idx], df_inseason.loc[train_idx, "inseason_success_smooth"]).reset_index(drop=True)
        form_valid = build_form_features(df.loc[valid_idx], df_inseason.loc[valid_idx, "inseason_success_smooth"]).reset_index(drop=True)

        X_train_v4 = pd.concat([X_train_base.reset_index(drop=True), ins_train], axis=1)
        X_valid_v4 = pd.concat([X_valid_base.reset_index(drop=True), ins_valid], axis=1)

        X_train_ext = pd.concat([X_train_v4, wl_train, form_train], axis=1)
        X_valid_ext = pd.concat([X_valid_v4, wl_valid, form_valid], axis=1)

        print(f"\n--- baseline (v4, {X_train_v4.shape[1]}피처) ---", flush=True)
        base_res = run_eval(X_train_v4, y_train, X_valid_v4, y_valid, "v4-baseline")

        print(f"\n--- +workload+form ({X_train_ext.shape[1]}피처) ---", flush=True)
        ext_res = run_eval(X_train_ext, y_train, X_valid_ext, y_valid, "v4+workload+form")

        print(f"\n--- {valid_season} 비교 (score 기준) ---", flush=True)
        for k in base_res:
            d = (ext_res[k] - base_res[k]) * 100000
            print(f"    {k:15s} baseline_score={max(0,base_res[k]*100000):7.1f}  "
                  f"ext_score={max(0,ext_res[k]*100000):7.1f}  delta={d:+7.1f}")

        all_results[valid_season] = {"base": base_res, "ext": ext_res}

    print(f"\n{'='*60}\n전체 요약 (delta score, +가 개선)\n{'='*60}", flush=True)
    for season, r in all_results.items():
        print(f" {season}:", flush=True)
        for k in r["base"]:
            d = (r["ext"][k] - r["base"][k]) * 100000
            print(f"   {k:15s} {d:+7.1f}", flush=True)

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
