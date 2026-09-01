"""v28 = v27(1025.66 실증, 현 최고점) + in-season 라벨차원 보충 + 타자 in-season middle + 타자 플래툰.

핵심 교훈 (phase73/74에서 확인) — magnitude와 증분은 다른 축이다:
  inseason_cmd_index(성공-의도반대-위험코스)는 단독설명력 564로 기존 최강 피처(569)와 동급이지만
  증분은 +0.47뿐이었다. 모델이 이미 success/reverse를 갖고 있어 '새 정보'가 아니라
  '아는 것의 재조합'이기 때문. 검증용으로 넣은 inseason_success_smooth도 단독 569 / 증분 0.20.
  -> 우리가 원하는 건 'magnitude 높으면서 모델이 아직 모르는' 피처다.

무엇이 이겼나 (v26 이후 전부 같은 패턴):
  이긴 것 = '직접 결과 x 당해시즌 x 큰 표본' 클래스
    bat_inseason_smooth (v27)   +17.1  6.6시그마
    bat_inseason_middle (v28)   +15.1  6.2시그마
  진 것 = 간접 대리지표 / 조건부 스플릿
    batter x count               +1.0  1.6시그마   -> 채택 안 함
    trackman 물리(원본)           +3.2            -> 저표본 상호작용으로만 살아남
    TTO                          +1.4            -> 기각
    arsenal/volatility/hidden    ~0              -> 기각

v28 추가 3블록 (증분 합동 21.5, 개별합 19.0보다 커서 상호보완적):
  1) inseason_full 4개  : inseason.py가 success/ball/reverse만 저장하고 middle/strike는
                          빠져 있던 비대칭 보충. lastyear는 ly_middle을 이미 쓰고 있었다.  +1.6
  2) 타자 middle 2개    : batterform.py의 success 차분 트릭을 asof_batter_middle_rate에 적용. +15.2
  3) 타자 플래툰 2개    : (타자, 투수손) 조건부. K=2486 실측.                              +2.2

  실현율 0.6(v25->v26, v26->v27 두 번 모두 60%) 적용 예상 실측 = +12.9점

피처 154 -> 162.
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

import batter_split as bsplit
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
from inseason_full import (build_global_priors, build_season_end_table_full, transform_inseason_full,
                           export_stats as inseason_full_export_stats)
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

    print("\n[4h] 타자 in-season 블록 (5개)...", flush=True)
    bat_tbl = build_batter_table(df)
    X_bat = transform_batter(df, bat_tbl, sr, g, k=K_BATTER).reset_index(drop=True)
    print(f"  bat_inseason_smooth SD={X_bat.bat_inseason_smooth.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4i] in-season 라벨차원 보충 middle/strike (신규 4개)...", flush=True)
    # inseason.py는 N/S/B/R만 저장해 middle/strike가 빠져 있었다. 분모를 success 쪽과
    # 정확히 맞추기 위해 inseason의 N_end를 그대로 넘긴다.
    n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    se_full = build_season_end_table_full(df)
    ins_priors = build_global_priors(df)
    X_isf = transform_inseason_full(df, se_full, ins_priors, sr, n_end_row,
                                    X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                    X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
    print(f"  inseason_middle_smooth SD={X_isf.inseason_middle_smooth.std():.5f}"
          f"  cmd_index SD={X_isf.inseason_cmd_index.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4j] 타자 middle in-season + 타자 플래툰 (신규 4개)...", flush=True)
    g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
    bmid_tbl = bsplit.build_batter_middle_table(df)
    X_bmid = bsplit.transform_batter_middle(df, bmid_tbl, sr, g_bmid).reset_index(drop=True)
    bmarg = bsplit.build_batter_marginal(df)
    b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
    bplat_tbl = bsplit.build_bplatoon_table(df)
    X_bplat = bsplit.transform_bplatoon(df, bplat_tbl, b_prior, sr, k=bsplit.K_BPLATOON).reset_index(drop=True)
    print(f"  bat_inseason_middle SD={X_bmid.bat_inseason_middle.std():.5f}"
          f"  bplatoon_diff SD={X_bplat.bplatoon_diff.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
                   X_isf, X_bmid, X_bplat], axis=1)
    print(f"\n최종 피처 수={X.shape[1]} (v27 154 -> v28 {X.shape[1]})", flush=True)

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
        "inseason_full_stats": inseason_full_export_stats(se_full, ins_priors, sr),
        "batter_split_stats": bsplit.export_stats(None, bplat_tbl, bmid_tbl, bmarg, sr,
                                                  g, g_bmid, k_bplatoon=bsplit.K_BPLATOON),
        "form_base_middle_global": float(base_middle[0]),
        "feature_order": list(X.columns),
    }
    for tag, w_hgb, w_cat in [("v28", 0.5, 0.5)]:
        artifacts = dict(common, w_hgb=w_hgb, w_cat=w_cat)
        out = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out)
        print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb={w_hgb} w_cat={w_cat}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
