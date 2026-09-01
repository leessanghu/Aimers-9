"""v27 = v26(1007.56 실증, 현 최고점) + 3-seed 평균 + 타자 in-season + trackman 저표본 상호작용.

v26 이후 확인된 것 (전부 진짜 미지시즌 폴드 train<=2023->valid=2024, GBDT 132피처 score=848.1 기준):

  [모델은 병목이 아님 — 세 실험이 독립적으로 같은 결론]
    phase68 capacity sweep : depth 6->8->10 전부 손해 (-14.6, -46.8). 과적합만 늘어남.
    phase67 MLP 블렌드     : MLP 단독 552.9, 결합상한 897.5 (GBDT 잠재력 893.4 대비 +4.1).
                             최적 가중치가 MLP=0%. 섞을수록 나빠짐.
    phase69 RandomForest   : bagging 계열을 섞어도 -2.7.
    -> 유일하게 유효했던 건 CatBoost 3seed 평균 (+2.9). 이건 정보를 더 뽑는 게 아니라
       순수 분산감소라, 오히려 'феature가 병목'이라는 결론을 강화한다.

  [phase70 신규 정보원 탐색]
    타자 in-season       +17.3  <- 투수 쪽만 만들고 타자 쪽은 공식컬럼 4개뿐이었음
    trackman x 저표본     +9.6  <- 6.1 -> 15.7 (상호작용 명시)
    TTO(타순 순회)        +1.4  <- 기각. 전부 0.8시그마. phase65에서 inning 오라클이
                                  19점뿐이었던 것과 정합적.
    보정                 +45.3  <- 실재하나 2025 편향을 forward로 추정해야 해서 리스크 있음.
                                  이번 버전에는 넣지 않고 단독 변수로 따로 검증한다.

v27 변경 3가지 (전부 위에서 측정된 것만):
  1) CatBoost 단일시드 -> 3시드(42/7/2024) 평균           +2.9
  2) 타자 in-season 블록 5개 (batterform.py)              +17.3
  3) trackman x 저표본 상호작용 17개                       +9.6

피처 132 -> 154.
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

from batterform import K_BATTER, build_batter_table, transform_batter, export_stats as batter_export_stats
from career_volatility import K_VOL, build_volatility_table, transform_volatility, export_stats as vol_export_stats
from count_split import K_COUNT, build_count_table, transform_count, export_stats as count_export_stats
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role, export_stats as role_export_stats
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
from trackman_profile import (build_trackman_profile, transform_trackman, add_lown_interactions,
                              export_stats as trackman_export_stats, K_PROFILE)

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
TM_CACHE = "phase64_trackman_profile.parquet"
SEED = 42

HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


def recency_weight(seasons, half_life=2.0):
    age = seasons.max() - seasons
    return 0.5 ** (age / half_life)


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
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False, team_te_mode="expanding").fit(df)
    X_base = fb.transform_train_oof(df).reset_index(drop=True)
    print(f"  {X_base.shape[1]}피처 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[2] in-season (K=15)...", flush=True)
    se = build_season_end_table(df)
    X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    print("\n[3] platoon(K=520) + inning(K=570)...", flush=True)
    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)

    print("\n[3b] count_state 조건부 (신규 2개, K=880)...", flush=True)
    ctb = build_count_table(df)
    X_cnt = transform_count(df, ctb, prior, sr, k=K_COUNT).reset_index(drop=True)
    print(f"  count_diff SD={X_cnt.count_diff.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4] 구종 피처 (3개, Trackman 투구단위 매칭)...", flush=True)
    matched = build_matched(df)
    pt_tables = build_pitchtype_tables(matched, sr)
    X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)
    print(f"  매칭 {len(matched):,}행  pt_dev SD={X_pt.pt_dev.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4b] 작년 한 시즌 피처 (7개)...", flush=True)
    gr = build_global_rates(df)
    ly_tbl = build_lastyear_table(df)
    X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)
    print(f"  ly_reverse SD={X_ly['ly_reverse'].std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4c] 커리어 시즌간 변동성 (5개, v20에서 기각 -> phase64b에서 재확인)...", flush=True)
    vol_tbl = build_volatility_table(se)
    X_vol = transform_volatility(df, vol_tbl, sr, k=K_VOL).reset_index(drop=True)
    print(f"  vol_std SD={X_vol['vol_std'].std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4d] 역할(선발/불펜) 프로파일 (신규 7개)...", flush=True)
    role_tbl = build_role_table(df)
    X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)
    print(f"  role_ppa 중앙값={X_role.role_ppa.median():.1f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4e] 폼 피처 (신규 11개, 자기 베이스라인 대비 logit차)...", flush=True)
    base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
    X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                            base_middle).reset_index(drop=True)
    print(f"  form5_success SD={X_form.form5_success.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4f] Trackman 물리 프로파일 (신규 17개, 최초 실행시 오래 걸림)...", flush=True)
    if os.path.exists(TM_CACHE):
        prof = pd.read_parquet(TM_CACHE)
        print(f"  캐시 로드 ({time.time()-t0:.0f}s)", flush=True)
    else:
        prof = build_trackman_profile()
        prof.to_parquet(TM_CACHE)
        print(f"  신규 계산+캐시 ({time.time()-t0:.0f}s)", flush=True)
    X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)
    print(f"  매칭율={100*X_tm.tm_matched.mean():.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4g] trackman x 저표본 상호작용 (신규 17개)...", flush=True)
    # 임계값은 fit(=train) 시점 상수. 배치에 따라 달라지면 행 독립성 위반이므로 반드시 고정한다.
    lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
    X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)
    print(f"  저표본 임계값(train 중앙값)={lown_thr:.0f}  해당비율={100*X_tmx.tm_lown_flag.mean():.1f}%"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4h] 타자 in-season 블록 (신규 5개)...", flush=True)
    bat_tbl = build_batter_table(df)
    X_bat = transform_batter(df, bat_tbl, sr, g, k=K_BATTER).reset_index(drop=True)
    print(f"  bat_inseason_smooth SD={X_bat.bat_inseason_smooth.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat], axis=1)
    print(f"\n최종 피처 수={X.shape[1]} (v26 132 -> v27 {X.shape[1]})", flush=True)

    print("\n[5] 최근시즌 가중치 (half-life=2)...", flush=True)
    w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)
    print(f"  weight range=[{w.min():.4f}, {w.max():.4f}]  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6] HGB 학습 (sample_weight 적용)...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y, sample_weight=w)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[7] CatBoost 3시드 학습 (phase69에서 +2.9 확인, 분산감소라 손해 불가능)...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cats = []
    for s in (42, 7, 2024):
        cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                                random_seed=s, verbose=0, early_stopping_rounds=50,
                                min_data_in_leaf=200, loss_function="Logloss")
        cb.fit(X.iloc[tr_i], y[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
        cats.append(cb)
        print(f"  seed={s} best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    common = {
        "hgb": hgb, "cats": cats,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(se, g, sr),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "count_stats": count_export_stats(ctb, sr, k=K_COUNT),
        "pitchtype_stats": pitchtype_export_stats(pt_tables, g, sr, k_control=K_CONTROL, k_mix=K_MIX),
        "lastyear_stats": lastyear_export_stats(ly_tbl, gr, sr, k=30.0),
        "volatility_stats": vol_export_stats(vol_tbl, sr, k=K_VOL),
        "role_stats": role_export_stats(role_tbl, sr),
        "trackman_stats": trackman_export_stats(prof, sr, k=K_PROFILE, lown_threshold=lown_thr),
        "batter_stats": batter_export_stats(bat_tbl, sr, g, k=K_BATTER),
        "form_base_middle_global": float(base_middle[0]),
        "feature_order": list(X.columns),
    }
    for tag, w_hgb, w_cat in [("v27", 0.5, 0.5)]:
        artifacts = dict(common, w_hgb=w_hgb, w_cat=w_cat)
        out = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out)
        print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb={w_hgb} w_cat={w_cat}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
