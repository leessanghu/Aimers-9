"""Phase 4-5: 검증 폴드를 2020~2024로 확장해서 RF/HGB(811점 냈던 원래 구성)를 재검증.

폴드: train<=2019->2020, train<=2020->2021, train<=2021->2022, train<=2022->2023, train<=2023->2024
(2019은 그 이전 데이터가 없어 valid 대상으로 쓸 수 없음 — train<=2019->2020이 사실상 "가장 이른" 폴드)

RF/HGB는 원래 811점 제출본과 동일한 피처 구성(전체 58 + team_te, extra_features 없음)으로 검증.
결과는 dev/phase4_preds/fold_{season}_rfhgb.csv 로 저장 (다른 모델과 나중에 블렌딩).
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from phase2_common import build_fold, load_full, rich_eval, all_weight_sensitivity

SEED = 42
OUT_DIR = "phase4_preds"

FOLDS_EXT = [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)

# 811점 제출본과 동일 가중치 (참고용 비교)
W_RF_ORIG, W_HGB_ORIG = 0.15, 0.85


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_full()

    fold_bss_rf, fold_bss_hgb, fold_bss_blend = {}, {}, {}
    all_rows = []

    for train_max, valid_season in FOLDS_EXT:
        print(f"\n===== fold: train<=season{train_max} -> valid=season{valid_season} =====", flush=True)
        tf = time.time()
        # 811점 제출본과 동일: extra_features 없음, team_te 포함 (원래 FeatureBuilder 기본값)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED,
                          include_team_te=True)
        X_train, X_valid = fold["X_train"], fold["X_valid"]
        y_train, y_valid = fold["y_train"], fold["y_valid"]
        print(f"  train={len(y_train):,}  valid={len(y_valid):,}  features={X_train.shape[1]}  "
              f"({time.time()-tf:.0f}s)", flush=True)

        preds_df = pd.DataFrame({"row_id": fold["row_id"], "y_valid": y_valid})

        tm = time.time()
        rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
        p_rf = rf.predict_proba(X_valid)[:, 1]
        preds_df["pred_rf"] = p_rf
        m_rf = rich_eval(y_valid, p_rf, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
        fold_bss_rf[valid_season] = m_rf["bss"]
        print(f"  RF   BSS={m_rf['bss']:.6f}  score={m_rf['score']:.1f}  ({time.time()-tm:.0f}s)", flush=True)

        tm = time.time()
        hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
        p_hgb = hgb.predict_proba(X_valid)[:, 1]
        preds_df["pred_hgb"] = p_hgb
        m_hgb = rich_eval(y_valid, p_hgb, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
        fold_bss_hgb[valid_season] = m_hgb["bss"]
        print(f"  HGB  BSS={m_hgb['bss']:.6f}  score={m_hgb['score']:.1f}  ({time.time()-tm:.0f}s)", flush=True)

        p_blend = W_RF_ORIG * p_rf + W_HGB_ORIG * p_hgb
        preds_df["pred_rf015_hgb085"] = p_blend
        m_blend = rich_eval(y_valid, p_blend, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
        fold_bss_blend[valid_season] = m_blend["bss"]
        print(f"  RF0.15+HGB0.85(원래 811점 구성)  BSS={m_blend['bss']:.6f}  score={m_blend['score']:.1f}",
              flush=True)

        for name, m in [("rf", m_rf), ("hgb", m_hgb), ("rf015_hgb085", m_blend)]:
            row = dict(m)
            row.update({"model": name, "valid_season": valid_season})
            all_rows.append(row)

        preds_df.to_csv(os.path.join(OUT_DIR, f"fold_{valid_season}_rfhgb.csv"), index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase4_rfhgb_extended_summary.csv"), index=False, encoding="utf-8")

    print("\n===== 폴드별 BSS 한눈에 =====", flush=True)
    print(f"{'model':15s}  " + "  ".join(f"{s}" for _, s in FOLDS_EXT), flush=True)
    for name, d in [("rf", fold_bss_rf), ("hgb", fold_bss_hgb), ("rf015_hgb085", fold_bss_blend)]:
        print(f"{name:15s}  " + "  ".join(f"{d[s]:.5f}" for _, s in FOLDS_EXT), flush=True)

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
