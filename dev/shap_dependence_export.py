"""shap_cache/*.joblib(shap_dependence.py 결과)를 읽어 feature dependence plot용 binned JSON을 만든다.
season별/game_type별로 feature 값을 20분위 bin으로 나눠 평균 SHAP을 계산 -> 대시보드에서 바로 그림."""
import json
import os

import joblib
import numpy as np
import pandas as pd

CACHE_DIR = "shap_cache"
OUT_PATH = "shap_dependence_data.json"
N_BINS = 20

FEATURES_BY_VERSION = {
    "v18": ["prev1_hidden_total_n", "prev3_hidden_total_n", "prev5_hidden_total_n",
            "prev3_hidden_avg_n", "prev5_hidden_avg_n", "prev1_vs_prev3_workload", "prev3_vs_prev5_workload"],
    "v20": ["vol_std", "vol_min", "vol_max", "vol_range", "vol_n_seasons"],
    "v22": ["arsenal_entropy", "arsenal_top_share"],
}
COMMON_FEATS = [
    "inseason_success_smooth", "inseason_reverse_smooth", "inseason_n",
    "platoon_diff", "platoon_n", "inning_diff", "inning_n",
    "pt_pred", "pt_dev", "pt_n",
    "ly_success", "ly_reverse", "ly_ball", "ly_middle", "ly_n", "ly_minus_career",
    "ability_composite", "pitcher_team_id_te", "batter_team_id_te",
    "cat_game_type", "season", "asof_pitcher_success_rate_smooth",
    "x_ability_here", "x_kal_minus_career", "x_prev5_minus_career",
    "x_rev_over_succ", "x_p_over_b", "x_platoon_x_samehand",
]


def bin_dependence(v, s, group_vals, n_bins=N_BINS):
    """v: feature value array, s: shap array, group_vals: labels (season or game_type) per row."""
    out = {}
    for g in sorted(pd.unique(group_vals), key=lambda x: str(x)):
        mask = group_vals == g
        vv, ss = v[mask], s[mask]
        if len(vv) < n_bins * 5 or np.std(vv) == 0:
            continue
        try:
            qs = np.quantile(vv, np.linspace(0, 1, n_bins + 1))
            qs = np.unique(qs)
        except Exception:
            continue
        if len(qs) < 3:
            continue
        bin_idx = np.clip(np.digitize(vv, qs[1:-1]), 0, len(qs) - 2)
        pts = []
        for b in range(len(qs) - 1):
            m = bin_idx == b
            if m.sum() == 0:
                continue
            pts.append({"x": float(vv[m].mean()), "y": float(ss[m].mean()), "n": int(m.sum())})
        out[str(g)] = pts
    return out


def main():
    result = {}
    for tag in ("v18", "v20", "v22"):
        cache_path = f"{CACHE_DIR}/{tag}.joblib"
        if not os.path.exists(cache_path):
            print(f"skip {tag}: no cache")
            continue
        cached = joblib.load(cache_path)
        X, shap_vals, order = cached["X"], cached["shap_vals"], cached["order"]
        idx_map = {c: i for i, c in enumerate(order)}

        season = X["season"].to_numpy()
        game_type = X["cat_game_type"].to_numpy() if "cat_game_type" in X.columns else None

        targets = FEATURES_BY_VERSION.get(tag, []) + COMMON_FEATS
        for name in targets:
            if name not in idx_map or name not in X.columns:
                continue
            key = f"{tag}::{name}"
            if key in result:
                continue
            v = X[name].to_numpy()
            s = shap_vals[:, idx_map[name]]
            entry = {
                "version": tag, "feature": name,
                "overall_mean_abs_shap": float(np.abs(s).mean()),
                "overall_corr": float(np.corrcoef(v, s)[0, 1]) if np.std(v) > 0 else None,
                "by_season": bin_dependence(v, s, season),
            }
            if game_type is not None:
                entry["by_game_type"] = bin_dependence(v, s, game_type)
            result[key] = entry
            print(f"  {key}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"\nwrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.2f}MB), {len(result)} feature entries")


if __name__ == "__main__":
    main()
