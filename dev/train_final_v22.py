"""v22 = v15(976.099 실증, 최고점) + 구종 레퍼토리 엔트로피 2개 (단일 변수 변경) = 93피처.

공식 asof_pitcher_{fastball,breaking,offspeed}_rate + pitchmix_n을 그대로 쓰는 비선형 결합이라
leakage 걱정이 전혀 없다(crosses.py와 같은 철학, 새 정보원이 아니라 트리가 못 만드는 결합을 제공).

가설: 구종 믹스가 다양/예측불가능한(엔트로피 높은) 투수가 디셉션 효과로 제구에 유리할 수 있다.

phase54 (2024폴드, baseline=v15 정확 재현 812.9):
  개별 잔차가치: arsenal_entropy +0.10, arsenal_top_share +0.41 (합산 +0.5, 약함)
  직접 폴드 재확인: 812.9 -> 817.2 (+4.3, 채택 기준선 +5 살짝 못 미침)
  단, 오늘 로컬 최고치(변동성 +5.0)도 실측 -3.83이었던 전례가 있어 로컬 신호 신뢰도 자체가 낮음.
  seed=42 특이성 가능성(v21 seed10=977.74 실측)까지 감안해 일단 실측으로 직접 확인한다.

--- 이하 v15 설명 ---
v15 = v14(970.5 실증) + 작년 한 시즌 피처 7개 = 91피처.

단일 변수 변경: v14의 모든 설정(in-season K=15, platoon K=520, inning K=570, 구종,
교차항, HGB+CatBoost 50:50)을 그대로 두고 작년 피처만 추가한다.

채택 근거 — 잔차 기반 사전 선별(phase40/41, 학습 없이 계산):
  작년 피처 7개 합동 예상이득 = +17.4  (개별 합산 +21.5는 상관 때문에 과대계상)
  이 지표는 아는 답으로 검증됨: 구종 +5.6 -> 실제 +6.7 / workload,form +0.4 -> 실제 가치없음
  같은 기각 규칙으로 Teacher(+0.0)와 disjoint 블록(+0.8)은 학습 전에 걸러냄

핵심 발견: ly_success는 ly_reverse 위에 아무것도 더하지 않음(둘 다 +6.1).
  즉 진짜 새 정보는 '작년 reverse율'이었다. (reverse는 다음시즌 제구와 상관 -0.494로
  success +0.531에 맞먹는데 그동안 실력 추정에 전혀 안 쓰고 있었음)

v13이 -11이었던 이유도 이걸로 설명됨: 작년피처(+17)를 넣었지만 동시에 platoon K를
520->2500으로 올려 신호 진폭을 0.0119->0.0059(실패 구간)로 죽였다.

--- 이하 v14 설명 ---
v14 = v12(963.796 실증) + 구종 피처 3개(pt_pred, pt_dev, pt_n) = 84피처.

단일 변수 변경: K값(inseason=15, platoon=520, inning=570) 전부 v12 그대로, CatBoost도
v12와 동일하게 단일 시드. 구종 피처만 추가해서 v14-v12 = 구종 정보의 순수 효과가 되게 한다.

phase37 2024폴드 (baseline=v12 정확 재현): hgb +17.5 / cat3시드 -1.9 / 50:50블렌드 +7.8
시드 산포가 15~24점(노이즈 SD~11 확인됨)이라 확신은 약하지만 방향은 양수.

가중치 2개를 동시 저장(재학습 없이 A/B):
  v14a: w_hgb=0.5 w_cat=0.5 (v12와 동일 비중 -> 순수 피처 효과 측정용)
  v14b: w_hgb=0.2 w_cat=0.8 (기존 전체 이력에서 CatBoost가 압도적으로 강했던 것 반영)

근거 요약 (dev/pitchtype.py 참조):
  매칭정밀도 99.5~99.7% / 커버리지 61.5% / 재현상관 0.475(platoon 0.328보다 높음)
  신호진폭 0.0122 (성공구간 0.009~0.012에 위치, platoon 0.0119와 거의 동일)
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

from arsenal_entropy import ARSENAL_COLS, K_ARSENAL, export_stats as arsenal_export_stats, transform_arsenal
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

    print("\n[4c] 구종 레퍼토리 엔트로피 (신규 2개, row-local)...", flush=True)
    arsenal_global_mix = {c: float(df[c].mean(skipna=True)) for c in
                          ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]}
    X_ars = transform_arsenal(df, global_mix=arsenal_global_mix, k=K_ARSENAL).reset_index(drop=True)
    print(f"  arsenal_entropy SD={X_ars['arsenal_entropy'].std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_ars], axis=1)
    print(f"\n교차항 {C.shape[1]}개 + 작년 {X_ly.shape[1]}개 + 구종엔트로피 {X_ars.shape[1]}개 -> 최종 피처 수={X.shape[1]} (v15 91 -> 93)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6] CatBoost 학습 (단일 시드, v12과 동일)...", flush=True)
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
        "arsenal_stats": arsenal_export_stats(arsenal_global_mix),
        "feature_order": list(X.columns),
    }
    for tag, w_hgb, w_cat in [("v22", 0.5, 0.5)]:
        artifacts = dict(common, w_hgb=w_hgb, w_cat=w_cat)
        out = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out)
        print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb={w_hgb} w_cat={w_cat}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
