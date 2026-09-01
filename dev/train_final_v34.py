"""v29 = v28(피처 162개 동일, 실측 1025.12) + 모델 다양화 앙상블.

배경 (phase80/81):
    현재 프로덕션 = HGB 1개 + CatBoost 3시드(같은 depth=6) 블렌드.
    HGB<->CatBoost 예측상관 0.909인데, CatBoost끼리는 depth/rsm을 바꿔도 상관이
    0.965 밑으로 안 내려간다 (symmetric 트리 구조 자체가 같아서). 즉 3-seed 평균은
    노이즈만 줄이지 진짜 다양성을 못 만든다.

    depth/subsample을 계열 내에서 다양화한 6개(HGB 3 + CatBoost 3)를 2024 폴드에서
    테스트한 결과:
        현재 2모델 블렌드(hgb_d6+cat_d6, 0.5/0.5)     880.70
        6개 계열균형 단순평균(0.5*HGB3평균+0.5*Cat3평균) 891.35   (+10.65)
        6개 greedy 가중(9:9:8:8:5:1)                   893.68   (+12.98)
    greedy와 단순평균 차이가 +2.33뿐이라 이득의 대부분은 '다양화 자체'에서 오고
    폴드별 가중치 미세조정 기여는 작다 -> 과적합 위험이 적은 단순 계열균형 평균을 쓴다.

    LightGBM/XGBoost는 phase80에서 튜닝 부족(88~90s 학습, 나머지는 200~500s)으로
    최약체(785.9, 777.7)였고 greedy가 0번 선택했다. phase81에서 optuna로 공정
    튜닝 검증 진행 중 -> 이번 버전엔 포함하지 않는다(신규 의존성 리스크도 있음).

v29 변경 1가지:
    HGB 1개 + CatBoost 3시드(전부 depth=6) -> HGB 3변종 + CatBoost 3변종, 0.5/0.5 계열균형.
    피처는 v28과 완전히 동일(162개), recency weight(half_life=2.0)도 동일.
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

    print("\n[3b] count_state 조건부...", flush=True)
    ctb = build_count_table(df)
    X_cnt = transform_count(df, ctb, prior, sr, k=K_COUNT).reset_index(drop=True)

    print("\n[4] 구종 피처...", flush=True)
    matched = build_matched(df)
    pt_tables = build_pitchtype_tables(matched, sr)
    X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)

    print("\n[4b] 작년 한 시즌 피처...", flush=True)
    gr = build_global_rates(df)
    ly_tbl = build_lastyear_table(df)
    X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)

    print("\n[4c] 커리어 시즌간 변동성...", flush=True)
    vol_tbl = build_volatility_table(se)
    X_vol = transform_volatility(df, vol_tbl, sr, k=K_VOL).reset_index(drop=True)

    print("\n[4d] 역할(선발/불펜) 프로파일...", flush=True)
    role_tbl = build_role_table(df)
    X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)

    print("\n[4e] 폼 피처...", flush=True)
    base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
    X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                            base_middle).reset_index(drop=True)

    print("\n[4f] Trackman 물리 프로파일...", flush=True)
    if os.path.exists(TM_CACHE):
        prof = pd.read_parquet(TM_CACHE)
    else:
        prof = build_trackman_profile()
        prof.to_parquet(TM_CACHE)
    X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)

    print("\n[4g] trackman x 저표본 상호작용...", flush=True)
    lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
    X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)

    print("\n[4h] 타자 in-season 블록...", flush=True)
    bat_tbl = build_batter_table(df)
    X_bat = transform_batter(df, bat_tbl, sr, g, k=K_BATTER).reset_index(drop=True)

    print("\n[4i] in-season 라벨차원 보충...", flush=True)
    se_full = build_season_end_table_full(df)
    priors_full = build_global_priors(df)
    n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    X_isf = transform_inseason_full(df, se_full, priors_full, sr, n_end_row,
                                    X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                    X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)

    print("\n[4j] 타자 middle in-season + 타자 플래툰...", flush=True)
    g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
    bmid_tbl = bsplit.build_batter_middle_table(df)
    X_bmid = bsplit.transform_batter_middle(df, bmid_tbl, sr, g_bmid).reset_index(drop=True)
    bmarg = bsplit.build_batter_marginal(df)
    b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
    bplat_tbl = bsplit.build_bplatoon_table(df)
    X_bplat = bsplit.transform_bplatoon(df, bplat_tbl, b_prior, sr, k=bsplit.K_BPLATOON).reset_index(drop=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
                   X_isf, X_bmid, X_bplat], axis=1)
    print(f"\n최종 피처 수={X.shape[1]} (v28과 동일)", flush=True)

    print("\n[5] 최근시즌 가중치 (half-life=2)...", flush=True)
    w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)
    print(f"  weight range=[{w.min():.4f}, {w.max():.4f}]  ({time.time()-t0:.0f}s)", flush=True)

    tr_i, es_i = time_split_es(len(X))

    print("\n[6] HGB 3변종 학습 (phase80에서 다양성 확인: d6/d8/sub)...", flush=True)
    hgb_configs = [
        ("hgb_d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
        ("hgb_d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
        ("hgb_sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
    ]
    hgbs = []
    for name, extra in hgb_configs:
        params = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                     validation_fraction=0.1, n_iter_no_change=20)
        params.update(extra)
        m = HistGradientBoostingClassifier(**params).fit(X, y, sample_weight=w)
        hgbs.append(m)
        print(f"  {name} iters={m.n_iter_} ({time.time()-t0:.0f}s)", flush=True)

    print("\n[7] CatBoost 3변종 학습 (d6/d8/rsm)...", flush=True)
    cat_configs = [
        ("cat_d6", dict(depth=6, random_seed=42)),
        ("cat_d8", dict(depth=8, l2_leaf_reg=10.0, random_seed=7)),
        ("cat_rsm", dict(depth=6, rsm=0.6, random_seed=2024)),
    ]
    cats = []
    for name, extra in cat_configs:
        params = dict(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                     early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
        params.update(extra)
        cb = CatBoostClassifier(**params)
        cb.fit(X.iloc[tr_i], y[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
        cats.append(cb)
        print(f"  {name} best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    print("\n[8] Hurdle 3변종 (phase89 검증: v29로컬 893.68 -> 924.28 @ w=0.53, "
         "안전마진으로 w=0.45 사용)...", flush=True)
    n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)

    def cnt(col):
        return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)

    R_, M_ = [cnt(c) for c in ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"]]
    ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    pid_o = df["pitcher_id"].to_numpy()[ordr]
    n_o = n_[ordr]
    hstep = np.zeros(len(df), dtype=bool)
    hstep[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
    r_diff = np.zeros(len(df)); m_diff = np.zeros(len(df))
    r_diff[ordr[:-1]] = np.diff(R_[ordr]); m_diff[ordr[:-1]] = np.diff(M_[ordr])
    core_fail = np.where(hstep, ((r_diff > 0) | (m_diff > 0)).astype(np.float64), np.nan)
    assert (y[hstep & (core_fail == 1)] == 0).all(), "core_fail=1인데 success=1 -> 라벨 복원 버그"
    print(f"  복원 {hstep.sum():,}행 ({100*hstep.mean():.2f}%)  core_fail 비율={np.nanmean(core_fail):.4f}"
         f"  ({time.time()-t0:.0f}s)", flush=True)

    HURDLE_VARIANTS = [
        ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
        ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
        ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
    ]
    core_models = []
    succ_nc_models = []
    nc_m = hstep & (core_fail == 0)
    for name, extra in HURDLE_VARIANTS:
        params = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                     validation_fraction=0.1, n_iter_no_change=20)
        params.update(extra)
        cm = HistGradientBoostingClassifier(**params).fit(X.loc[hstep], core_fail[hstep], sample_weight=w[hstep])
        core_models.append(cm)
        print(f"  core_{name} iters={cm.n_iter_} ({time.time()-t0:.0f}s)", flush=True)
        sm = HistGradientBoostingClassifier(**params).fit(X.loc[nc_m], y[nc_m], sample_weight=w[nc_m])
        succ_nc_models.append(sm)
        print(f"  succ_nc_{name} iters={sm.n_iter_}  (학습행 {nc_m.sum():,}) ({time.time()-t0:.0f}s)", flush=True)

    for m in core_models + succ_nc_models:
        strip_rng(m)
    for m in hgbs:
        strip_rng(m)

    os.makedirs(OUT_DIR, exist_ok=True)
    common = {
        "hgbs": hgbs, "cats": cats,
        "core_fail_models": core_models, "succ_nc_models": succ_nc_models,
        "hurdle_weight": 0.45,
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
        "inseason_full_stats": inseason_full_export_stats(se_full, priors_full, sr),
        "batter_split_stats": bsplit.export_stats(bcount_table=None, bplatoon_table=bplat_tbl,
                                                  bmid_table=bmid_tbl, marginal=bmarg, seasons_range=sr,
                                                  global_rate=g, global_middle=g_bmid,
                                                  k_bplatoon=bsplit.K_BPLATOON, k_bmid=30.0),
        "form_base_middle_global": float(base_middle[0]),
        "feature_order": list(X.columns),
        "w_hgb": 0.5, "w_cat": 0.5,
    }
    out = os.path.join(OUT_DIR, "model_artifacts_v34.pkl")
    joblib.dump(common, out)
    print(f"\n저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)  HGB {len(hgbs)}변종  CatBoost {len(cats)}변종",
         flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
