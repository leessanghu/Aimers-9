"""v20 = v15(976.099 실증, 최고점) + 커리어 시즌간 변동성 5개 (단일 변수 변경) = 96피처.

v18(cat 3시드 평균 단독, 973.660 실증)과 v19(v18+hidden_denom, 972.759 실증) 둘 다
v15보다 나빴다 -> v15 이후 시도한 모든 변경(모델/피처 불문)이 실측에서 전부 손해였음.
그래서 v20은 v18/v19 위가 아니라 v15(HGB+CatBoost 단일시드 50:50) 그대로 위에 얹는다
(hidden_denom 계열은 반복 실패 + 65~69% 분모매칭 모호성 확인되어 영구 폐기).

커리어 변동성(AMEX/Home Credit류 "과거 관측치를 std/min/max로 압축" 패턴 적용, phase53):
  vol_std/min/max/range: 그 투수의 '직전 시즌까지' 시즌별 고립 성공률의 표준편차/최저/최고/범위
  vol_n_seasons: 관측된 과거 시즌 수
  가설: 지금까지 '평균 제구력'만 썼지 '시즌마다 얼마나 들쭉날쭉한가'는 피처가 없었다.

phase53 (2024폴드, baseline=v15 정확 재현 812.9):
  개별 잔차가치: vol_std +0.03, vol_min +0.49, vol_max +0.00, vol_range +0.08, vol_n_seasons +1.00 (합산 +1.6, 약함)
  직접 폴드 재확인: 812.9 -> 817.8 (+5.0, 채택 기준선 통과)
  leakage 체크 정상 (2019행 vol_n_seasons=0)

주의: season_end_table의 깨끗한 누적카운트 차분이라 hidden_denom류의 분모매칭 모호성
문제는 없다(그쪽은 rate 두 개만으로 n을 역산해야 해서 원천적으로 모호했음).

--- 이하 v15 설명 ---
v15 = v14(970.5 실증) + 작년 한 시즌 피처 7개 = 91피처.
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

from career_volatility import K_VOL, build_volatility_table, export_stats as vol_export_stats, transform_volatility
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from inning_split import (K_INNING, build_inning_offset, build_inning_table, transform_inning,
                          export_stats as inning_export_stats)
from inseason import (build_season_end_table, transform_inseason, _pivots_from_table,
                      export_stats as inseason_export_stats)
from lastyear import (build_global_rates, build_lastyear_table, transform_lastyear,
                      export_stats as lastyear_export_stats)
from phase2_common import time_split_es
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
    print(f"  {X_base.shape[1]}피처 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[2] in-season (v12과 동일 K=15)...", flush=True)
    se = build_season_end_table(df)
    X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    print("\n[3] platoon(K=520) + inning(K=570) - v12과 동일...", flush=True)
    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)

    print("\n[4] 구종 피처 (신규 3개, Trackman 투구단위 매칭)...", flush=True)
    matched = build_matched(df)
    pt_tables = build_pitchtype_tables(matched, sr)
    X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)
    print(f"  매칭 {len(matched):,}행  pt_dev SD={X_pt.pt_dev.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4b] 작년 한 시즌 피처 (신규 7개)...", flush=True)
    gr = build_global_rates(df)
    ly_tbl = build_lastyear_table(df)
    X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)
    print(f"  ly_reverse SD={X_ly['ly_reverse'].std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4c] 커리어 시즌간 변동성 (신규 5개)...", flush=True)
    vol_tbl = build_volatility_table(se)
    X_vol = transform_volatility(df, vol_tbl, sr, k=K_VOL).reset_index(drop=True)
    print(f"  vol_std SD={X_vol['vol_std'].std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_vol], axis=1)
    print(f"\n교차항 {C.shape[1]}개 + 작년 {X_ly.shape[1]}개 + 변동성 {X_vol.shape[1]}개 -> 최종 피처 수={X.shape[1]} (v15 91 -> 96)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6] CatBoost 학습 (단일 시드, v15와 동일)...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=SEED, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
    print(f"  완료 best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    common = {
        "hgb": hgb, "cat": cb,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(se, g, sr),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "pitchtype_stats": pitchtype_export_stats(pt_tables, g, sr, k_control=K_CONTROL, k_mix=K_MIX),
        "lastyear_stats": lastyear_export_stats(ly_tbl, gr, sr, k=30.0),
        "volatility_stats": vol_export_stats(vol_tbl, sr, k=K_VOL),
        "feature_order": list(X.columns),
    }
    for tag, w_hgb, w_cat in [("v20", 0.5, 0.5)]:
        artifacts = dict(common, w_hgb=w_hgb, w_cat=w_cat)
        out = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out)
        print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb={w_hgb} w_cat={w_cat}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
