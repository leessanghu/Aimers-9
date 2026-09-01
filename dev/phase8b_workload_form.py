"""phase8 재검증 — 투구수 복원을 success+middle 공동분모 방식으로 고친 뒤,
workload / form 기여도를 분리(ablation)해서 3폴드 측정.

1차 결과(구 복원, median 25구 = 약 40% 과소추정):
  2022 +23.6 / 2023 +80.0 / 2024 -13.8 (rf015_hgb085 기준)
  -> 2024에서 손해. workload가 노이즈였는지 form이 진짜인지 분리 필요.

arm:
  baseline      v4 (63피처)
  +form         v4 + form 4개  (prev1/3/5 - inseason 기준선, prev1-prev5 트렌드)
  +workload     v4 + workload 4개 (복원된 투구수/역할/이상부하)
  +both         v4 + 8개

leakage: 두 피처군 모두 그 행 자신의 컬럼만 사용. test.csv 행 간 참조 없음.
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

    # v4 배포조합 + lgbm 3자 블렌딩도 참고로 측정
    rows["rf010_hgb060_lgb030"] = evaluate(y_valid, 0.10 * p_rf + 0.60 * p_hgb + 0.30 * p_lgb)["bss"]

    print(f"  [{tag}]", flush=True)
    for k, v in rows.items():
        print(f"    {k:22s} BSS={v:.6f}  score={max(0, v*100000):.1f}", flush=True)
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    global_success_rate = float(df["control_success"].mean())

    print("season_end_table + workload(success+middle 공동분모) 복원 ...", flush=True)
    season_end = build_season_end_table(df)
    seasons_range = sorted(df["season"].unique().tolist())
    df_inseason = transform_inseason(df, season_end, global_success_rate, seasons_range)
    df_workload = build_workload_features(df)
    q1 = np.expm1(df_workload["recent1_pitch_n"])
    q1 = q1[q1 > 0]
    print(f"  복원 prev1 투구수: median={np.median(q1):.1f} p25={np.percentile(q1,25):.1f} "
          f"p75={np.percentile(q1,75):.1f} ({time.time()-t0:.0f}s)", flush=True)

    all_results = {}
    for train_max, valid_season in FOLDS:
        print(f"\n{'='*64}\nFOLD train<={train_max} valid={valid_season}\n{'='*64}", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True)
        y_train, y_valid = fold["y_train"], fold["y_valid"]

        train_idx = df[df["season"] <= train_max].index
        valid_idx = df[df["season"] == valid_season].index

        def slice_(frame, idx, cols=None):
            s = frame.loc[idx] if cols is None else frame.loc[idx, cols]
            return s.reset_index(drop=True)

        Xtr = pd.concat([fold["X_train"].reset_index(drop=True),
                         slice_(df_inseason, train_idx, INSEASON_COLS)], axis=1)
        Xva = pd.concat([fold["X_valid"].reset_index(drop=True),
                         slice_(df_inseason, valid_idx, INSEASON_COLS)], axis=1)

        wl_tr, wl_va = slice_(df_workload, train_idx), slice_(df_workload, valid_idx)
        fm_tr = build_form_features(df.loc[train_idx],
                                    df_inseason.loc[train_idx, "inseason_success_smooth"]).reset_index(drop=True)
        fm_va = build_form_features(df.loc[valid_idx],
                                    df_inseason.loc[valid_idx, "inseason_success_smooth"]).reset_index(drop=True)

        arms = {
            "baseline":  (Xtr, Xva),
            "+form":     (pd.concat([Xtr, fm_tr], axis=1), pd.concat([Xva, fm_va], axis=1)),
            "+workload": (pd.concat([Xtr, wl_tr], axis=1), pd.concat([Xva, wl_va], axis=1)),
            "+both":     (pd.concat([Xtr, wl_tr, fm_tr], axis=1), pd.concat([Xva, wl_va, fm_va], axis=1)),
        }

        fold_res = {}
        for name, (xt, xv) in arms.items():
            print(f"\n--- {name} ({xt.shape[1]}피처) ---", flush=True)
            fold_res[name] = run_eval(xt, y_train, xv, y_valid, name)

        print(f"\n--- {valid_season} baseline 대비 delta score ---", flush=True)
        b = fold_res["baseline"]
        for name in ["+form", "+workload", "+both"]:
            deltas = "  ".join(f"{k}={100000*(fold_res[name][k]-b[k]):+7.1f}" for k in b)
            print(f"    {name:10s} {deltas}", flush=True)

        all_results[valid_season] = fold_res

    print(f"\n{'='*64}\n전체 요약 — baseline 대비 delta score\n{'='*64}", flush=True)
    models = list(next(iter(all_results.values()))["baseline"].keys())
    for model in models:
        print(f"\n [{model}]", flush=True)
        for season, fr in all_results.items():
            b = fr["baseline"][model]
            line = "  ".join(f"{n}={100000*(fr[n][model]-b):+7.1f}" for n in ["+form", "+workload", "+both"])
            print(f"   {season}  baseline={max(0,b*100000):7.1f} | {line}", flush=True)

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
