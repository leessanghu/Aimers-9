# script.py
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리를 fit된 통계만으로 재현 (dev/features.py FeatureBuilder와 동일 로직)
# =======================

def build_features(df, stats):
    out = {}

    cat_cols = stats["cat_cols"]
    cats = stats["cat_encoder"].transform(df[cat_cols].astype(str))
    for i, c in enumerate(cat_cols):
        out[f"cat_{c}"] = cats[:, i]

    num_median = stats["num_median"]
    for c in stats["raw_num_cols"]:
        out[c] = df[c].fillna(num_median.get(c, 0.0)).to_numpy(dtype=np.float64)

    out["pitcher_hand"] = df["pitcher_hand"].to_numpy(dtype=np.float64)
    out["batter_hand"] = df["batter_hand"].to_numpy(dtype=np.float64)
    out["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float64).to_numpy()
    out["count_state"] = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy(dtype=np.float64)
    out["hand_matchup"] = (df["pitcher_hand"] * 10 + df["batter_hand"]).to_numpy(dtype=np.float64)

    rate_global_mean = stats["rate_global_mean"]
    smooth_k_rate = stats["smooth_k_rate"]
    smoothed = {}
    for n_col, rate_cols in stats["rate_groups"].items():
        n = df[n_col].fillna(0).to_numpy(dtype=np.float64)
        out[f"flag_{n_col}_zero"] = (n == 0).astype(np.float64)
        out[n_col] = np.log1p(n)
        for c in rate_cols:
            raw = df[c].fillna(0).to_numpy(dtype=np.float64)
            gm = rate_global_mean[c]
            sm = (n * raw + smooth_k_rate * gm) / (n + smooth_k_rate)
            out[f"{c}_smooth"] = sm
            smoothed[c] = sm

    no_n_rate_cols = stats["no_n_rate_cols"]
    miss_flag = df[no_n_rate_cols[0]].isna().to_numpy()
    out["flag_prev_game_missing"] = miss_flag.astype(np.float64)
    for c in no_n_rate_cols:
        out[c] = df[c].fillna(rate_global_mean[c]).to_numpy(dtype=np.float64)

    out["diff_success_rate"] = smoothed["asof_pitcher_success_rate"] - smoothed["asof_batter_success_rate"]
    out["diff_middle_rate"] = smoothed["asof_pitcher_middle_rate"] - smoothed["asof_batter_middle_rate"]

    id_count = stats["id_count"]
    for col in stats["id_count_cols"]:
        cnt = df[col].map(id_count[col]).fillna(0).to_numpy(dtype=np.float64)
        out[f"{col}_count"] = np.log1p(cnt)

    team_count = stats["team_count"]
    for col in stats["team_cols"]:
        cnt = df[col].map(team_count[col]).fillna(0).to_numpy(dtype=np.float64)
        out[f"{col}_count"] = np.log1p(cnt)

    team_te = stats["team_te"]
    global_y_mean = stats["global_y_mean"]
    for col in stats["team_cols"]:
        out[f"{col}_te"] = df[col].map(team_te[col]).fillna(global_y_mean).to_numpy(dtype=np.float64)

    return pd.DataFrame(out, index=df.index)


# =======================
# in-season(시즌 한정) 피처 — dev/inseason.py와 동일 로직 (학습 리포지토리 의존 없이 재구현)
# leakage 안전성: 각 행은 자기 투수의 '직전 시즌 끝 시점'만 참조 (같은 시즌 다른 행 안 씀)
# =======================

def build_inseason_features(df, inseason_stats):
    season_end_table = inseason_stats["season_end_table"]
    global_success_rate = inseason_stats["global_success_rate"]
    seasons_range = inseason_stats["seasons_range"]
    K_SMOOTH = 15.0
    BALL_PRIOR, REVERSE_PRIOR = 0.4, 0.25

    pivots = {}
    for col, val_col in [("N_end", "N_end"), ("S_end", "S_end"), ("B_end", "B_end"),
                         ("R_end", "R_end"), ("rate", "prior_success_rate")]:
        p = season_end_table.pivot(index="pitcher_id", columns="season", values=val_col)
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        pivots[col] = p.stack(future_stack=True)

    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    ends = {}
    for col in ["N_end", "S_end", "B_end", "R_end"]:
        vals = pivots[col].reindex(lookup_idx).to_numpy()
        ends[col] = np.nan_to_num(vals.astype(np.float64), nan=0.0)
    prior_rate_vals = pivots["rate"].reindex(lookup_idx).to_numpy()
    prior_rate = pd.Series(prior_rate_vals).fillna(global_success_rate).to_numpy(np.float64)

    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(df["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
    b_now = np.round(df["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now)
    r_now = np.round(df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - ends["N_end"], 0, None)
    s_season = np.clip(s_now - ends["S_end"], 0, None)
    b_season = np.clip(b_now - ends["B_end"], 0, None)
    r_season = np.clip(r_now - ends["R_end"], 0, None)

    rate_raw = np.divide(s_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    ball_raw = np.divide(b_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    rev_raw = np.divide(r_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)

    out = pd.DataFrame(index=df.index)
    out["inseason_success_smooth"] = (n_season * np.nan_to_num(rate_raw) + K_SMOOTH * prior_rate) / (n_season + K_SMOOTH)
    out["inseason_ball_smooth"] = (n_season * np.nan_to_num(ball_raw) + K_SMOOTH * BALL_PRIOR) / (n_season + K_SMOOTH)
    out["inseason_reverse_smooth"] = (n_season * np.nan_to_num(rev_raw) + K_SMOOTH * REVERSE_PRIOR) / (n_season + K_SMOOTH)
    out["inseason_n"] = np.log1p(n_season)
    out["inseason_is_first_appearance"] = (n_season == 0).astype(np.float64)
    return out


def get_prior_pitcher_rate(df, inseason_stats):
    """각 행의 '직전 시즌 끝 시점' 투수 marginal 성공률 (플래툰 축소 기준점)."""
    season_end_table = inseason_stats["season_end_table"]
    global_success_rate = inseason_stats["global_success_rate"]
    seasons_range = inseason_stats["seasons_range"]

    p = season_end_table.pivot(index="pitcher_id", columns="season", values="prior_success_rate")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    pivot_rate = p.stack(future_stack=True)

    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = pivot_rate.reindex(lookup_idx).to_numpy()
    return pd.Series(vals).fillna(global_success_rate).to_numpy(np.float64)


# =======================
# platoon 피처 — (투수, 타자손) 조건부 성공률의 그 투수 자신 대비 편차
#
# 규칙 준수: 각 행은 자기 투수의 '직전 시즌 끝 시점까지' 누적된 (pitcher_id, batter_hand)
# 조회만 한다(pivot+ffill+stack, in-season과 동일 구조). 같은 시즌의 다른 행이나
# test.csv의 다른 행은 전혀 참조하지 않는다.
#
# 근거: 주최측 asof_* 컬럼은 전부 marginal(투수 전체 성공률)이라 "이 투수는 좌타에
# 유독 약하다" 같은 조건부 개인차를 모델이 볼 수 없다. 노이즈 제거 후 진짜 개인차
# SD=0.0438 (투수 실력 개인차 SD=0.0555의 79%) — 그 행 컬럼만으론 계산 불가능한 정보.
# =======================

def build_platoon_features(df, platoon_stats, prior_rate):
    platoon_table = platoon_stats["platoon_table"]
    seasons_range = platoon_stats["seasons_range"]
    k = platoon_stats["k"]

    def lookup(value_col):
        p = platoon_table.pivot_table(index=["pitcher_id", "batter_hand"], columns="season",
                                      values=value_col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["batter_hand"], df["season"] - 1])
        return p.stack(future_stack=True).reindex(idx).to_numpy()

    s_cell = np.nan_to_num(lookup("s").astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(lookup("n").astype(np.float64), nan=0.0)

    prior = np.asarray(prior_rate, dtype=np.float64)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["platoon_diff"] = rate_smooth - prior
    out["platoon_n"] = np.log1p(n_cell)
    return out


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    ARTIFACT_PATH = os.path.join(MODEL_DIR, "model_artifacts_v7b.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model...")
    artifacts = joblib.load(ARTIFACT_PATH)
    rf, hgb, stats = artifacts["rf"], artifacts["hgb"], artifacts["stats"]
    inseason_stats = artifacts["inseason_stats"]
    platoon_stats = artifacts["platoon_stats"]
    w_rf, w_hgb = artifacts["w_rf"], artifacts["w_hgb"]
    print(f" OK. w_rf={w_rf}  w_hgb={w_hgb}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features (58 base + 5 in-season + 2 platoon)...")
    ids = test[ID_COL].tolist()
    X_base = build_features(test, stats).reset_index(drop=True)
    X_inseason = build_inseason_features(test, inseason_stats).reset_index(drop=True)
    prior_rate = get_prior_pitcher_rate(test, inseason_stats)
    X_platoon = build_platoon_features(test, platoon_stats, prior_rate).reset_index(drop=True)
    X = pd.concat([X_base, X_inseason, X_platoon], axis=1)
    X.index = test.index
    X = X[list(rf.feature_names_in_)]
    print(f" features={X.shape[1]}")

    print("Inference model...")
    if len(X):
        p_rf = rf.predict_proba(X)[:, 1]
        p_hgb = hgb.predict_proba(X)[:, 1]
        preds = w_rf * p_rf + w_hgb * p_hgb
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
