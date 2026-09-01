"""v26 = v25(981.44 실증, 현 최고점) + 오라클 천장 실측(phase65)으로 찾은 신규 블록 4개.

배경: phase63(Brier 분해)로 모델/보정 개선의 상한이 +71점뿐임을 확인했고, phase64/64b
(재학습 delta 대신 GBDT 위 증분 잠재력을 부분상관으로 측정, split-half 자유도 편향 보정,
검증용 기존피처로 0점 재현해 방법론 검증됨), phase65(그룹핑별 오라클 천장, 잡음보정)로
신호의 위치를 직접 측정했다.

phase65 핵심 발견: pitcher_id 단독 천장=840, pitcher x count_state 천장=1223으로
platoon(1134)/inning(1029)보다 높은데 count_state 조건부 테이블이 없었다.

phase64b 개별 피처 재검증 (부분상관, 1시그마=4.0점):
  count_diff(신규)         +9.3  (4.9시그마) <- 최강
  form5_middle             +6.0  (3.9시그마)
  form5_success            +4.0  (3.2시그마)
  vol_max/n_seasons/range   +3.3/+2.0/+2.0  (기각됐던 career_volatility, 재확인시 유의)
  tm_ivb_sd                +1.9  (2.2시그마)
  [검증용] 기존 피처         0.0  <- 스크리너가 이미 아는 정보에 0을 준다는 것으로 방법론 검증

추가 블록 (전부 신규):
  count_split (2)   : pitcher x count_state 조건부, K=880 (전체 train 실측 노이즈보정 SD=0.0168)
  formfeat (11)     : 자기 베이스라인 대비 폼(logit차, 표본신뢰도 축소) — 리그평균이 아니라
                      투수 개인 inseason 추정치를 기준점으로 삼은 게 핵심 수정
  role (7)          : 선발/불펜 역할 + role x inning 피로 상호작용
  trackman (17)     : 릴리스포인트 반복성/무브먼트/피로/압박반응 물리 프로파일.
                      (투수 x 구종) 내부 SD로만 계산해 레퍼토리 다양성과 반복성을 분리.
  volatility (5)    : v20에서 기각됐다가 phase64b에서 부분상관으로 재확인된 것 재포함.

측정된 증분 상한(2024 폴드, GBDT 잠재력 895.9 위): 신규 3블록(역할+폼+trackman) 합동 +43.8.
count_split은 별도 검증 +9.3. 실측(전체 train, 압축 없는 신호)으로 확인한다.

feature 91 -> 91+2+11+7+17+5 = 133.
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
from trackman_profile import build_trackman_profile, transform_trackman, export_stats as trackman_export_stats, K_PROFILE

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

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm], axis=1)
    print(f"\n최종 피처 수={X.shape[1]} (v25 91 -> v26 {X.shape[1]})", flush=True)

    print("\n[5] 최근시즌 가중치 (half-life=2)...", flush=True)
    w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)
    print(f"  weight range=[{w.min():.4f}, {w.max():.4f}]  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6] HGB 학습 (sample_weight 적용)...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y, sample_weight=w)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[7] CatBoost 학습 (단일 시드, sample_weight 적용)...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=SEED, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(X.iloc[tr_i], y[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
    print(f"  완료 best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    common = {
        "hgb": hgb, "cat": cb,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(se, g, sr),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "count_stats": count_export_stats(ctb, sr, k=K_COUNT),
        "pitchtype_stats": pitchtype_export_stats(pt_tables, g, sr, k_control=K_CONTROL, k_mix=K_MIX),
        "lastyear_stats": lastyear_export_stats(ly_tbl, gr, sr, k=30.0),
        "volatility_stats": vol_export_stats(vol_tbl, sr, k=K_VOL),
        "role_stats": role_export_stats(role_tbl, sr),
        "trackman_stats": trackman_export_stats(prof, sr, k=K_PROFILE),
        "form_base_middle_global": float(base_middle[0]),
        "feature_order": list(X.columns),
    }
    for tag, w_hgb, w_cat in [("v26", 0.5, 0.5)]:
        artifacts = dict(common, w_hgb=w_hgb, w_cat=w_cat)
        out = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out)
        print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb={w_hgb} w_cat={w_cat}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
