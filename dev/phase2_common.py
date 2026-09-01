"""Phase 2 공통 유틸: rolling fold 구성, seen/unseen 분리 평가, 가중치 민감도."""

import numpy as np
import pandas as pd

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate

DATA_PATH = "../data/train.csv"
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]  # (train<=X, valid==X+1)
FOLD_WEIGHTS = {
    "default_0.2_0.3_0.5": {2022: 0.2, 2023: 0.3, 2024: 0.5},
    "equal": {2022: 1 / 3, 2023: 1 / 3, 2024: 1 / 3},
    "strong_0.1_0.2_0.7": {2022: 0.1, 2023: 0.2, 2024: 0.7},
}
CAT_FEATURES = ["cat_top_bottom", "cat_game_type", "cat_base_state", "count_state", "hand_matchup"]

_DF_CACHE = None


def load_full():
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return _DF_CACHE


def build_fold(df, train_max_season, valid_season, extra_features=None, seed=42, include_team_te=True,
               team_te_mode="shuffled_kfold"):
    train_fold = df[df["season"] <= train_max_season].reset_index(drop=True)
    valid_fold = df[df["season"] == valid_season].reset_index(drop=True)

    fb = FeatureBuilder(seed=seed, include_raw_rates=False, extra_features=extra_features,
                        include_team_te=include_team_te, team_te_mode=team_te_mode).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)
    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype(pd.CategoricalDtype(categories=X_train[c].cat.categories))

    known_pitchers = set(train_fold["pitcher_id"].unique())
    known_batters = set(train_fold["batter_id"].unique())
    seen_p = valid_fold["pitcher_id"].isin(known_pitchers).to_numpy()
    seen_b = valid_fold["batter_id"].isin(known_batters).to_numpy()

    return {
        "train_fold": train_fold, "valid_fold": valid_fold,
        "X_train": X_train, "X_valid": X_valid,
        "y_train": train_fold[TARGET_COL].to_numpy(), "y_valid": valid_fold[TARGET_COL].to_numpy(),
        "seen_pitcher_mask": seen_p, "seen_batter_mask": seen_b,
        "row_id": valid_fold["row_id"].to_numpy(),
    }


def time_split_es(n, frac=0.08):
    cut = int(n * (1 - frac))
    return np.arange(cut), np.arange(cut, n)


def rich_eval(y_valid, pred, seen_pitcher_mask, seen_batter_mask):
    m = evaluate(y_valid, pred)
    out = {
        "bss": m["bss"], "score": m["leaderboard_score"], "brier": m["brier_score"],
        "target_mean": float(y_valid.mean()), "pred_mean": float(pred.mean()),
        "calib_diff": float(pred.mean() - y_valid.mean()),
    }
    for name, mask in [("pitcher", seen_pitcher_mask), ("batter", seen_batter_mask)]:
        for tag, mm in [("seen", mask), ("unseen", ~mask)]:
            if mm.sum() > 0:
                mm_m = evaluate(y_valid[mm], pred[mm])
                out[f"{name}_{tag}_bss"] = mm_m["bss"]
                out[f"{name}_{tag}_n"] = int(mm.sum())
            else:
                out[f"{name}_{tag}_bss"] = None
                out[f"{name}_{tag}_n"] = 0
    return out


def weighted_bss(fold_bss_by_season, weights):
    return sum(fold_bss_by_season[s] * w for s, w in weights.items())


def all_weight_sensitivity(fold_bss_by_season):
    return {name: weighted_bss(fold_bss_by_season, w) for name, w in FOLD_WEIGHTS.items()}
