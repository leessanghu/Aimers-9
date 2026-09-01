"""v17a/v17b = v15(976.099 실증) + 투구단위 라벨 복원 조건부 (+선택적으로 나머지 14개).

핵심 발견: 같은 투수의 연속 행에서 asof_pitcher_n 증가량이 정확히 +1인 비율 = 100.00%.
  누적 카운트 차분으로 reverse/middle/ball/strike '투구 단위 라벨'을 100.000% 정확도로
  복원(success로 대조검증). 주최측이 라벨을 사실상 5개 준 셈이었다.

phase43 잔차 스크리닝 (baseline=v15, 학습 없이):
  (A) 라벨 x 볼카운트     +3.1 (애매)   (B) 라벨 x 이닝         +2.5 (애매)
  (C) lastyear strike     +0.2 (기각)   (D)(E) pitchmix+JS      +0.4 (기각)
  (F) Trackman pitch_of_pa +0.2 (기각)  전체 합동               +6.2

로컬 신뢰도가 낮아 "다 넣고 실측으로 검증" 요청에 따라 2개 버전을 한 번에 만든다:
  v17a = v15 + (A)(B) 라벨조건부 9개                = 100피처  (기여자만, 귀속 깨끗)
  v17b = v15 + (A)(B)(C)(D)(E)(F) 전체 19개          = 110피처  (v17b-v17a로 나머지 10개 가치 확정)
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from crosses import add_crosses
from extra_v17 import (build_popa_table, build_pitchmix_table, build_strike_table,
                       export_stats_cde, export_stats_f, global_mix_rates, global_strike_rate,
                       transform_pitchmix_arsenal, transform_popa, transform_strike)
from features import FeatureBuilder, TARGET_COL
from inning_split import (K_INNING, build_inning_offset, build_inning_table, transform_inning,
                          export_stats as inning_export_stats)
from inseason import (build_season_end_table, transform_inseason, _pivots_from_table,
                      export_stats as inseason_export_stats)
from lastyear import (build_global_rates, build_lastyear_table, transform_lastyear,
                      export_stats as lastyear_export_stats)
from phase2_common import time_split_es
from pitchlabels import (build_cond_table, build_global_offsets, recover_pitch_labels,
                         transform_cond_labels, export_stats as labels_export_stats)
from pitchtype import (K_CONTROL, K_MIX, build_matched, build_pitchtype_tables, transform_pitchtype,
                       export_stats as pitchtype_export_stats)
from platoon import build_platoon_table, transform_platoon, export_stats as platoon_export_stats, K_PLATOON

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def fit_models(X, y, tag, t0):
    print(f"\n[{tag}] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{tag}] CatBoost 학습...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=SEED, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
    print(f"  완료 best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)
    strip_rng(hgb)
    return hgb, cb


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    y = df[TARGET_COL].to_numpy()
    g = float(df[TARGET_COL].mean())
    sr = sorted(df["season"].unique().tolist())
    print(f"train={len(df):,}  season {sr[0]}~{sr[-1]}", flush=True)

    print("\n[1] base 피처...", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df)
    X_base = fb.transform_train_oof(df).reset_index(drop=True)

    print("[2] in-season / platoon / inning / 구종 / 작년 (v15와 동일)...", flush=True)
    se = build_season_end_table(df)
    X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
    matched = build_matched(df)
    pt_tables = build_pitchtype_tables(matched, sr)
    X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)
    gr = build_global_rates(df)
    ly_tbl = build_lastyear_table(df)
    X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[3] 투구단위 라벨 복원 + 조건부 (A)(B)...", flush=True)
    labels = recover_pitch_labels(df)
    valid_pct = labels["lab_reverse"].notna().mean() * 100
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    inn = np.clip(df["inning"].to_numpy(np.int64), 1, 9)
    tbl_c = build_cond_table(df, labels, cs)
    gl_c, bc_c = build_global_offsets(df, labels, cs)
    X_lc = transform_cond_labels(df, tbl_c, gl_c, bc_c, cs, "lc", sr, k=400.0).reset_index(drop=True)
    tbl_i = build_cond_table(df, labels, inn)
    gl_i, bc_i = build_global_offsets(df, labels, inn)
    X_li = transform_cond_labels(df, tbl_i, gl_i, bc_i, inn, "li", sr, k=400.0).reset_index(drop=True)
    print(f"  라벨 유효율={valid_pct:.2f}%  lc_reverse_dev SD={X_lc['lc_reverse_dev'].std():.5f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    X_common = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X_common)
    X_ab = pd.concat([X_common, C, X_ly, X_lc, X_li], axis=1)
    print(f"\nv17a 최종 피처 수={X_ab.shape[1]} (v15 91 + 라벨조건부 9)", flush=True)

    hgb_a, cb_a = fit_models(X_ab, y, "v17a", t0)

    print("\n[4] 나머지 (C)(D)(E)(F) — v17b 전용...", flush=True)
    kt = build_strike_table(df)
    gk = global_strike_rate(df)
    X_c = transform_strike(df, kt, gk, sr, k=30.0).reset_index(drop=True)
    mtd = build_pitchmix_table(df)
    gmix = global_mix_rates(df)
    X_de = transform_pitchmix_arsenal(df, mtd, gmix, sr, k=30.0).reset_index(drop=True)
    popa_prof = build_popa_table(df)
    X_f = transform_popa(df, popa_prof, sr).reset_index(drop=True)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    X_b = pd.concat([X_ab, X_c, X_de, X_f], axis=1)
    print(f"v17b 최종 피처 수={X_b.shape[1]} (v17a {X_ab.shape[1]} + 나머지 10)", flush=True)

    hgb_b, cb_b = fit_models(X_b, y, "v17b", t0)

    os.makedirs(OUT_DIR, exist_ok=True)
    common_stats = {
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(se, g, sr),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "pitchtype_stats": pitchtype_export_stats(pt_tables, g, sr, k_control=K_CONTROL, k_mix=K_MIX),
        "lastyear_stats": lastyear_export_stats(ly_tbl, gr, sr, k=30.0),
        "labels_c_stats": labels_export_stats(tbl_c, gl_c, bc_c, sr, k=400.0),
        "labels_i_stats": labels_export_stats(tbl_i, gl_i, bc_i, sr, k=400.0),
        "w_hgb": 0.5, "w_cat": 0.5,
    }

    art_a = dict(common_stats, hgb=hgb_a, cat=cb_a, feature_order=list(X_ab.columns))
    out_a = os.path.join(OUT_DIR, "model_artifacts_v17a.pkl")
    joblib.dump(art_a, out_a)
    print(f"\n저장: {out_a} ({os.path.getsize(out_a)/1e6:.1f}MB)", flush=True)

    art_b = dict(common_stats, hgb=hgb_b, cat=cb_b, feature_order=list(X_b.columns),
                extra_cde_stats=export_stats_cde(kt, gk, mtd, gmix, sr, k_strike=30.0, k_mix=30.0),
                extra_f_stats=export_stats_f(popa_prof, sr))
    out_b = os.path.join(OUT_DIR, "model_artifacts_v17b.pkl")
    joblib.dump(art_b, out_b)
    print(f"저장: {out_b} ({os.path.getsize(out_b)/1e6:.1f}MB)", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
