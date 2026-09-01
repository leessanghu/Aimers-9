"""Trackman 최소 피처 재시도 + 조건부 가치 검증.

앞선 실패(20개 전부 추가 -> 전 모델 -15~-28점)의 원인 가설:
  (1) 피처 과다로 희석  (2) 21% 결측이 '매핑 성공 여부'라는 가짜 분기를 만듦
따라서 이번엔:
  - 부분상관 상위 1~2개만 사용
  - 결측은 리그 평균으로 채워 가짜 분기 제거 (has_profile 플래그도 제거)
  - 추가로, 경험 적은 투수(asof_pitcher_n 하위) 구간에서만 이득이 있는지 별도 측정
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FeatureBuilder, TARGET_COL
from inseason_v2 import build_season_end_table, build_global_rates, transform_inseason_v2
from metrics import evaluate
from phase2_common import time_split_es

SEED = 42
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)


def build_tm_features(max_season, cols_wanted):
    pmap = pd.read_csv("pitcher_map.csv").set_index("pitcher_id")["tm_id"]
    tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "induced_vert_break", "horz_break",
                              "extension", "rel_speed"])
    tm = tm[tm.season <= max_season]
    prof = tm.groupby("pitcher_trackman_id").agg(
        tm_ivb=("induced_vert_break", "mean"),
        tm_hb=("horz_break", "mean"),
        tm_ext_std=("extension", "std"),
        tm_speed_std=("rel_speed", "std"))
    inv = pmap.reset_index().set_index("tm_id")["pitcher_id"]
    prof = prof.join(inv, how="inner").set_index("pitcher_id")
    return prof[cols_wanted]


def run(X_tr, y_tr, X_va, y_va, mask_low, tag):
    tr_idx, es_idx = time_split_es(len(X_tr))
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_tr, y_tr)
    p_hgb = hgb.predict_proba(X_va)[:, 1]
    lgb = LGBMRegressor(**LGBM_PARAMS)
    lgb.fit(X_tr.iloc[tr_idx], y_tr[tr_idx].astype(np.float64),
           eval_set=[(X_tr.iloc[es_idx], y_tr[es_idx].astype(np.float64))], eval_metric="l2",
           callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    p_lgb = np.clip(lgb.predict(X_va), 0.0, 1.0)
    out = {"hgb": evaluate(y_va, p_hgb)["bss"], "lgbm": evaluate(y_va, p_lgb)["bss"],
           "hgb_lowN": evaluate(y_va[mask_low], p_hgb[mask_low])["bss"],
           "lgbm_lowN": evaluate(y_va[mask_low], p_lgb[mask_low])["bss"]}
    print(f"[{tag}]  hgb={out['hgb']:.6f}  lgbm={out['lgbm']:.6f}   "
          f"| 경험적은투수 hgb={out['hgb_lowN']:.6f} lgbm={out['lgbm_lowN']:.6f}", flush=True)
    return out


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    season_end = build_season_end_table(df)
    v2 = transform_inseason_v2(df, season_end, build_global_rates(df),
                               sorted(df["season"].unique().tolist()), k_smooth_list=(15,))
    ins = ["inseason_success_smooth_k15", "inseason_n", "inseason_is_first_appearance"]

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df[df["season"] <= 2023])
    X_tr = pd.concat([fb.transform_train_oof(df[df["season"] <= 2023]).reset_index(drop=True),
                      v2.loc[df.index[df.season <= 2023], ins].reset_index(drop=True)], axis=1)
    X_va = pd.concat([fb.transform(df[df["season"] == 2024]).reset_index(drop=True),
                      v2.loc[df.index[df.season == 2024], ins].reset_index(drop=True)], axis=1)
    y_tr = df.loc[df.season <= 2023, TARGET_COL].to_numpy()
    y_va = df.loc[df.season == 2024, TARGET_COL].to_numpy()

    n_va = df.loc[df.season == 2024, "asof_pitcher_n"].fillna(0).to_numpy()
    mask_low = n_va < 500
    print(f"경험 적은 투수(asof_pitcher_n<500) 비중: {mask_low.mean():.3f} (n={mask_low.sum():,})", flush=True)

    prof = build_tm_features(2023, ["tm_ivb", "tm_hb", "tm_ext_std", "tm_speed_std"])
    print(f"Trackman 프로필 {len(prof)}명 ({time.time()-t0:.0f}s)\n", flush=True)

    res = {}
    res["baseline"] = run(X_tr, y_tr, X_va, y_va, mask_low, "baseline")

    for name, cols in [("+ivb", ["tm_ivb"]), ("+ivb+hb", ["tm_ivb", "tm_hb"]),
                       ("+ivb+hb+ext_std+speed_std", ["tm_ivb", "tm_hb", "tm_ext_std", "tm_speed_std"])]:
        f_tr = df.loc[df.season <= 2023, ["pitcher_id"]].join(prof[cols], on="pitcher_id")[cols]
        f_va = df.loc[df.season == 2024, ["pitcher_id"]].join(prof[cols], on="pitcher_id")[cols]
        # 결측은 리그 평균으로 -> '매핑 성공 여부' 가짜 분기 제거
        fill = f_tr.mean()
        f_tr, f_va = f_tr.fillna(fill).reset_index(drop=True), f_va.fillna(fill).reset_index(drop=True)
        res[name] = run(pd.concat([X_tr, f_tr], axis=1), y_tr,
                        pd.concat([X_va, f_va], axis=1), y_va, mask_low, name)

    print("\n===== baseline 대비 (점수) =====", flush=True)
    b = res["baseline"]
    for k, v in res.items():
        if k == "baseline":
            continue
        print(f"  {k:28s} hgb={100000*(v['hgb']-b['hgb']):+7.1f}  lgbm={100000*(v['lgbm']-b['lgbm']):+7.1f}"
              f"   | lowN hgb={100000*(v['hgb_lowN']-b['hgb_lowN']):+7.1f}"
              f"  lgbm={100000*(v['lgbm_lowN']-b['lgbm_lowN']):+7.1f}")
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
