"""in-season(시즌 한정) 피처 검증 — 2024 폴드.

핵심 아이디어: asof_pitcher_success_rate * asof_pitcher_n = 누적 성공 횟수를 정확히 복원 가능
(이미 검증됨, 오차 <0.008). 그래서 "이번 시즌 들어와서만"의 성공/볼/역방향 횟수를
(현재 누적) - (직전 시즌 끝 시점 누적) 으로 구할 수 있다.

leakage 안전성: 각 행은 자기 투수의 "직전 시즌 끝 시점" 값만 참조한다. 그 값은 해당 투수의
그 이전 시즌 데이터에서만 나오므로 — 같은 시즌의 다른 행(=test 행 간 참조)은 전혀 안 쓴다.
실제 2025 test에도 그대로 적용 가능(2024까지의 train.csv만 있으면 계산 가능).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from metrics import evaluate
from phase2_common import build_fold, time_split_es

SEED = 42
K_SMOOTH = 15.0  # in-season rate를 prior(직전 시즌 끝 누적률)로 당기는 스무딩 강도

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)


def build_season_end_table(df):
    """(pitcher_id, season) -> 그 시즌이 '끝난 후' 시점의 진짜 누적 n/성공/볼/역방향 횟수.

    asof_pitcher_n은 '이 투구 직전까지' 값이라 시즌 마지막 행 기준으로 보면 그 마지막
    투구 자체가 누락된다(off-by-one). N_end/S_end는 그 마지막 투구의 실제 결과
    (control_success, train.csv에 라벨로 있음)를 더해 정확히 보정한다.
    B_end/R_end는 그 마지막 투구가 볼성/역방향 중 무엇이었는지 행 단위로 알 수 없어
    보정 없이 둔다(최대 투구 1개분의 근사 오차, k=15 스무딩에서 영향 미미)."""
    sub = df.sort_values(["pitcher_id", "row_num"])
    last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
    n_before_last = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    last_outcome = last["control_success"].to_numpy(np.float64)

    last["N_end"] = n_before_last + 1  # 마지막 투구 자신도 포함
    last["S_end"] = np.round(last["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64)
                             * n_before_last) + last_outcome  # 마지막 투구의 실제 성공여부 반영
    last["B_end"] = np.round(last["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_before_last)
    last["R_end"] = np.round(last["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_before_last)
    last["prior_success_rate"] = last["S_end"] / last["N_end"].replace(0, np.nan)
    return last[["pitcher_id", "season", "N_end", "S_end", "B_end", "R_end", "prior_success_rate"]]


def add_inseason_features(df, season_end_table, global_success_rate):
    """각 행에 '직전 시즌(시즌 갭 있으면 그 이전 가장 최근 시즌)' 끝 시점 값을 붙인다.
    pivot(pitcher x season) 후 시즌 축으로 ffill -> (pitcher, season-1) 조회.
    같은 시즌의 다른 행은 절대 참조하지 않는다 (row 독립성 규칙 준수)."""
    out = df.copy()
    seasons_range = sorted(df["season"].unique())

    pivots = {}
    for col, val_col in [("N_end", "N_end"), ("S_end", "S_end"), ("B_end", "B_end"),
                         ("R_end", "R_end"), ("rate", "prior_success_rate")]:
        p = season_end_table.pivot(index="pitcher_id", columns="season", values=val_col)
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        pivots[col] = p.stack(future_stack=True)  # MultiIndex (pitcher_id, season) -> 그 시즌까지 ffill된 값

    lookup_idx = pd.MultiIndex.from_arrays([out["pitcher_id"], out["season"] - 1])
    for col in ["N_end", "S_end", "B_end", "R_end"]:
        out[col] = pivots[col].reindex(lookup_idx).to_numpy()
        out[col] = np.nan_to_num(out[col].astype(np.float64), nan=0.0)
    out["prior_success_rate"] = pivots["rate"].reindex(lookup_idx).to_numpy()
    out["prior_success_rate"] = pd.Series(out["prior_success_rate"]).fillna(global_success_rate).to_numpy()

    n_now = out["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(out["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
    b_now = np.round(out["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now)
    r_now = np.round(out["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - out["N_end"].to_numpy(np.float64), 0, None)
    s_season = np.clip(s_now - out["S_end"].to_numpy(np.float64), 0, None)
    b_season = np.clip(b_now - out["B_end"].to_numpy(np.float64), 0, None)
    r_season = np.clip(r_now - out["R_end"].to_numpy(np.float64), 0, None)

    prior_rate = out["prior_success_rate"].fillna(global_success_rate).to_numpy(np.float64)
    rate_season_raw = np.divide(s_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    ball_season_raw = np.divide(b_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    rev_season_raw = np.divide(r_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)

    out["inseason_success_smooth"] = (n_season * np.nan_to_num(rate_season_raw) + K_SMOOTH * prior_rate) / (n_season + K_SMOOTH)
    out["inseason_ball_smooth"] = (n_season * np.nan_to_num(ball_season_raw) + K_SMOOTH * 0.4) / (n_season + K_SMOOTH)
    out["inseason_reverse_smooth"] = (n_season * np.nan_to_num(rev_season_raw) + K_SMOOTH * 0.25) / (n_season + K_SMOOTH)
    out["inseason_n"] = np.log1p(n_season)
    out["inseason_is_first_appearance"] = (n_season == 0).astype(np.float64)

    return out.sort_index()


def run_eval(X_train, y_train, X_valid, y_valid, tag):
    rows = {}
    tr_idx, es_idx = time_split_es(len(X_train))

    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    p = rf.predict_proba(X_valid)[:, 1]
    rows["rf"] = evaluate(y_valid, p)["bss"]

    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p = hgb.predict_proba(X_valid)[:, 1]
    rows["hgb"] = evaluate(y_valid, p)["bss"]

    p_rf = rf.predict_proba(X_valid)[:, 1]
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    rows["rf015_hgb085"] = evaluate(y_valid, 0.15 * p_rf + 0.85 * p_hgb)["bss"]

    lgb = LGBMRegressor(**LGBM_PARAMS)
    lgb.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
           eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
           callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    p = np.clip(lgb.predict(X_valid), 0.0, 1.0)
    rows["lgbm_a"] = evaluate(y_valid, p)["bss"]

    print(f"[{tag}]")
    for k, v in rows.items():
        print(f"  {k:15s} BSS={v:.6f}  score={max(0,v*100000):.1f}")
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    global_success_rate = df["control_success"].mean()

    print("season_end_table 구성...", flush=True)
    season_end = build_season_end_table(df)
    df_ext = add_inseason_features(df, season_end, global_success_rate)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    # 2024 폴드: train<=2023, valid=2024 (기존 established 프로토콜)
    fold = build_fold(df_ext, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
    X_train_base, X_valid_base = fold["X_train"], fold["X_valid"]
    y_train, y_valid = fold["y_train"], fold["y_valid"]

    inseason_cols = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                     "inseason_n", "inseason_is_first_appearance"]
    train_fold_idx = df_ext[df_ext["season"] <= 2023].index
    valid_fold_idx = df_ext[df_ext["season"] == 2024].index
    inseason_train = df_ext.loc[train_fold_idx, inseason_cols].reset_index(drop=True)
    inseason_valid = df_ext.loc[valid_fold_idx, inseason_cols].reset_index(drop=True)

    print(f"\n표본 체크: 2024 valid 중 in-season 첫 등장(n_season=0) 비율 = "
          f"{inseason_valid['inseason_is_first_appearance'].mean():.3f}", flush=True)
    print(f"inseason_success_smooth 분포:\n{inseason_valid['inseason_success_smooth'].describe()}", flush=True)

    print("\n===== baseline (58피처, in-season 없음) =====", flush=True)
    base_result = run_eval(X_train_base, y_train, X_valid_base, y_valid, "baseline")

    X_train_ext = pd.concat([X_train_base.reset_index(drop=True), inseason_train], axis=1)
    X_valid_ext = pd.concat([X_valid_base.reset_index(drop=True), inseason_valid], axis=1)

    print("\n===== +in-season 피처(5개 추가, 63피처) =====", flush=True)
    ext_result = run_eval(X_train_ext, y_train, X_valid_ext, y_valid, "+inseason")

    print("\n===== 비교 =====", flush=True)
    for k in base_result:
        delta = ext_result[k] - base_result[k]
        print(f"  {k:15s}  baseline={base_result[k]:.6f}  +inseason={ext_result[k]:.6f}  delta={delta:+.6f}")

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
