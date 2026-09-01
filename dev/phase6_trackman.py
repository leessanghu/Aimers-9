"""Trackman 물리 피처 검증 — 2024 폴드.

앞선 발견:
  - team 매핑: 일정 지문 상관 0.93~0.94 (헝가리안 1:1)
  - pitcher 매핑: 손잡이 일치율 100%, 시즌간 일관성 99.25%, 투구수비 0.955
  - 물리지표가 '과거 제구성공률'을 제거한 뒤에도 부분상관 0.12~0.18 잔존

여기서는 실제 모델(RF/HGB/LGBM)에 넣어서 2024 폴드 BSS가 오르는지 최종 판정한다.
leakage 방지: 2024 폴드 검증이므로 Trackman은 season<=2023 만 사용.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason_v2 import build_season_end_table, build_global_rates, transform_inseason_v2
from metrics import evaluate
from phase2_common import time_split_es

SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)

TM_NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "zone_speed"]


def build_trackman_profile(max_season):
    """pitcher_id -> Trackman 물리 프로필 (max_season 이하 데이터만 사용)."""
    pmap = pd.read_csv("pitcher_map.csv").set_index("pitcher_id")["tm_id"]
    tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group"] + TM_NUM)
    tm = tm[tm.season <= max_season]

    agg = {f"tm_{c}_mean": (c, "mean") for c in TM_NUM}
    agg.update({f"tm_{c}_std": (c, "std") for c in
                ["rel_speed", "spin_rate", "extension", "rel_height", "rel_side"]})
    agg["tm_n"] = ("rel_speed", "size")
    prof = tm.groupby("pitcher_trackman_id").agg(**agg)

    # 구종군 비율
    mix = (tm.pivot_table(index="pitcher_trackman_id", columns="pitch_type_group",
                          aggfunc="size", fill_value=0))
    mix = (mix.T / mix.sum(1)).T.add_prefix("tm_mix_")
    prof = prof.join(mix)

    # 최근 시즌 구속 - 직전 시즌 구속 (쇠퇴 신호)
    sp = tm.groupby(["pitcher_trackman_id", "season"])["rel_speed"].mean().unstack()
    if max_season in sp.columns and (max_season - 1) in sp.columns:
        prof["tm_speed_trend"] = sp[max_season] - sp[max_season - 1]
    else:
        prof["tm_speed_trend"] = np.nan

    inv = pmap.reset_index().set_index("tm_id")["pitcher_id"]
    prof = prof.join(inv, how="inner").set_index("pitcher_id")
    return prof


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
    rows["lgbm_a"] = evaluate(y_valid, np.clip(lgb.predict(X_valid), 0.0, 1.0))["bss"]
    print(f"[{tag}]")
    for k, v in rows.items():
        print(f"  {k:15s} BSS={v:.6f}  score={max(0,v*100000):.1f}")
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)

    season_end = build_season_end_table(df)
    global_rates = build_global_rates(df)
    seasons_range = sorted(df["season"].unique().tolist())
    v2 = transform_inseason_v2(df, season_end, global_rates, seasons_range, k_smooth_list=(15,))
    inseason_cols = ["inseason_success_smooth_k15", "inseason_n", "inseason_is_first_appearance"]
    print(f"in-season 준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    prof = build_trackman_profile(max_season=2023)
    print(f"Trackman 프로필: {len(prof)}명, {prof.shape[1]}개 지표 ({time.time()-t0:.0f}s)", flush=True)

    tm_feat = df[["pitcher_id"]].join(prof, on="pitcher_id")
    tm_cols = [c for c in prof.columns if c.startswith("tm_")]
    tm_feat = tm_feat[tm_cols]
    tm_feat["tm_has_profile"] = tm_feat["tm_n"].notna().astype(float)
    cover_2024 = tm_feat.loc[df.season == 2024, "tm_has_profile"].mean()
    print(f"  2024 행 커버리지: {cover_2024:.3f}", flush=True)

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df[df["season"] <= 2023])
    X_tr_base = fb.transform_train_oof(df[df["season"] <= 2023]).reset_index(drop=True)
    X_va_base = fb.transform(df[df["season"] == 2024]).reset_index(drop=True)
    tri = df.index[df["season"] <= 2023]
    vai = df.index[df["season"] == 2024]
    y_tr = df.loc[tri, TARGET_COL].to_numpy()
    y_va = df.loc[vai, TARGET_COL].to_numpy()

    A_tr = pd.concat([X_tr_base, v2.loc[tri, inseason_cols].reset_index(drop=True)], axis=1)
    A_va = pd.concat([X_va_base, v2.loc[vai, inseason_cols].reset_index(drop=True)], axis=1)

    print("\n===== baseline (58 + in-season success k15) =====", flush=True)
    base = run_eval(A_tr, y_tr, A_va, y_va, "baseline")

    B_tr = pd.concat([A_tr, tm_feat.loc[tri].reset_index(drop=True)], axis=1)
    B_va = pd.concat([A_va, tm_feat.loc[vai].reset_index(drop=True)], axis=1)
    print(f"\n===== + Trackman ({len(tm_cols)+1}개 추가, 총 {B_tr.shape[1]}피처) =====", flush=True)
    tmr = run_eval(B_tr, y_tr, B_va, y_va, "+trackman")

    print("\n===== 비교 =====", flush=True)
    for k in base:
        d = tmr[k] - base[k]
        print(f"  {k:15s}  base={base[k]:.6f}  +tm={tmr[k]:.6f}  delta={d:+.6f} ({d*100000:+.1f}점)")
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
