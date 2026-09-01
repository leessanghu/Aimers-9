"""v31 = v30(CatBoost refit, 미제출) + Hurdle 인수분해 앙상블 멤버 추가.

배경(phase83): core_fail(reverse OR middle) 예측 x success|no_core 예측의 2단계
인수분해가 직행 모델과 상관 0.87~0.91로 낮아 다양성 멤버로 유효함을 확인.
v30 로컬(915.60) 기준 hurdle을 w=0.30으로 섞으면 934.39 (+18.79), v29 원점 대비 +40.71.
최종 예측 = 0.7 * v30(HGB3+Cat3refit) + 0.3 * p_hurdle.

v30 배경은 아래 원문 유지.
"""
"""v30 = v29(실측 1035.87) + CatBoost ES->refit (2024 후반 46.6% 미학습 버그 수정).

배경 (phase84):
    time_split_es(frac=0.08)는 전체 데이터 마지막 8%(118,008행)를 검증용으로 뗀다.
    이 118,008행은 전부 2024년이고 2024 전체(253,507행)의 46.55%다. 즉 v29의
    CatBoost 3개는 2024년 후반 절반 가까이를 한 번도 못 보고 학습됐다.

    수정: ES는 iteration 확정 용도로만 쓰고(v29 학습로그: d6=298, d8=207, rsm=263),
    그 iteration을 고정해서 전체 데이터로 재학습(refit)한다.

    로컬 검증(train<=2023->valid=2024 폴드, ES는 <=2023 내부 8%로 별도 확정):
        Cat_ES(기존방식)    d6=844.69  d8=828.51  rsm=835.98
        Cat_refit(신규)     d6=906.23  d8=892.48  rsm=895.56   (전부 +58~64)
        Cat_ES+refit 0.5/0.5 블렌드는 refit 단독보다 항상 나쁨(상관 0.96~0.97,
        다양성 기여 거의 없음) -> 섞지 않고 완전 교체.
        3개 config가 독립적으로 같은 방향/크기로 나와 노이즈 가능성은 낮음.

    최종 앙상블 (로컬):
        HGB3 + Cat3(ES, v29 현재)      891.35
        HGB3 + Cat3(refit만)           915.60   (+24.25)
    오늘 실측 실현율(모델다양화 +10.65 로컬 -> +10.21 실측, 96%)을 감안하면
    기대되는 실측 이득은 상당하나, CatBoost 단독 노이즈 바닥은 미측정이라 보수적으로 본다.

v30 변경 1가지: CatBoost 3변종을 ES 홀드아웃 없이 전체 2019~2024로, v29가 확정한
    iteration(298/207/263) 고정으로 재학습. HGB 3변종은 v29와 완전히 동일(불변).
    피처는 v28/v29와 동일(162개).
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

    print("\n[7] CatBoost 3변종 refit (ES 홀드아웃 없이 전체데이터, iteration 고정)...", flush=True)
    print("  v29 학습로그에서 확정된 iteration: d6=298, d8=207, rsm=263 (phase84 검증 완료)",
         flush=True)
    cat_configs = [
        ("cat_d6", dict(depth=6, random_seed=42, iterations=298)),
        ("cat_d8", dict(depth=8, l2_leaf_reg=10.0, random_seed=7, iterations=207)),
        ("cat_rsm", dict(depth=6, rsm=0.6, random_seed=2024, iterations=263)),
    ]
    cats = []
    for name, extra in cat_configs:
        params = dict(learning_rate=0.03, l2_leaf_reg=5.0, verbose=0, min_data_in_leaf=200,
                     loss_function="Logloss")
        params.update(extra)
        cb = CatBoostClassifier(**params)
        cb.fit(X, y, sample_weight=w)
        cats.append(cb)
        print(f"  {name} iterations={params['iterations']} (전체데이터 {len(X):,}행, "
             f"{time.time()-t0:.0f}s)", flush=True)

    print("\n[8] Hurdle 인수분해 (core_fail + success|no_core, phase83 검증: "
         "v30로컬 대비 +18.79 @ w=0.30)...", flush=True)
    n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)

    def cnt(col):
        return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)

    S_, R_, M_ = [cnt(c) for c in ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                                    "asof_pitcher_middle_rate"]]
    ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    pid_o = df["pitcher_id"].to_numpy()[ordr]
    n_o = n_[ordr]
    step = np.zeros(len(df), dtype=bool)
    step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
    r_diff = np.zeros(len(df)); m_diff = np.zeros(len(df))
    r_diff[ordr[:-1]] = np.diff(R_[ordr])
    m_diff[ordr[:-1]] = np.diff(M_[ordr])
    core_fail = np.where(step, ((r_diff > 0) | (m_diff > 0)).astype(np.float64), np.nan)
    chk = step & (core_fail == 1)
    assert (y[chk] == 0).all(), "core_fail=1인데 success=1 -> 라벨 복원 버그"
    print(f"  복원 {step.sum():,}행 ({100*step.mean():.2f}%)  core_fail 비율={np.nanmean(core_fail):.4f}"
         f"  ({time.time()-t0:.0f}s)", flush=True)

    HGB_H = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=42)
    core_model = HistGradientBoostingClassifier(**HGB_H).fit(
        X.loc[step], core_fail[step], sample_weight=w[step])
    print(f"  core_fail 모델 학습완료 iters={core_model.n_iter_} ({time.time()-t0:.0f}s)", flush=True)

    nc_m = step & (core_fail == 0)
    succ_nc_model = HistGradientBoostingClassifier(**HGB_H).fit(
        X.loc[nc_m], y[nc_m], sample_weight=w[nc_m])
    print(f"  success|no_core 모델 학습완료 iters={succ_nc_model.n_iter_}  (학습행 {nc_m.sum():,})"
         f"  ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(core_model)
    strip_rng(succ_nc_model)
    for m in hgbs:
        strip_rng(m)

    os.makedirs(OUT_DIR, exist_ok=True)
    common = {
        "hgbs": hgbs, "cats": cats,
        "core_fail_model": core_model, "succ_nc_model": succ_nc_model,
        "hurdle_weight": 0.30,
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
    out = os.path.join(OUT_DIR, "model_artifacts_v31.pkl")
    joblib.dump(common, out)
    print(f"\n저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)  HGB {len(hgbs)}변종  CatBoost {len(cats)}변종",
         flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
