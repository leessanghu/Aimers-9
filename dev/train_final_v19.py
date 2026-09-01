"""v19 = v18(cat 3시드 평균, 미제출) + hidden denominator workload 7개 (단일 변수 추가).

v18: v15(976.099 실증, 최고점) + CatBoost 3시드 평균. phase46(2024폴드):
  hgb.5+cat단일.5(v15와 동일 구성)=812.9 -> hgb.5+cat3시드.5=815.2 (+2.3)
  시드평균은 편향 불변/분산만 감소라 이론적으로 손해 불가능한 유일한 카드. 그 자체로는 제출 안 함.

hidden denominator (phase49): asof_pitcher_prev{1,3,5}_game_success_rate/middle_rate 쌍에서
  최소 정수 분모를 역산해 그 게임의 실제 투구수를 복원(오차 eps=5.1e-7 이내 탐색).
  개별 잔차가치: prev1_vs_prev3_workload +2.21(대부분), 나머지 거의 0, 합산 참고값 +4.2.
  직접 폴드 재확인(v15구성+7개, HGB+cat단일): 812.9 -> 818.4 (+5.5).
  (외부 보고값 +9.23은 독립 재현 시 약 60%만 재현됨 — 과대계상 확인, 그래도 부호는 일치)
  주의: 같은 원천 데이터(prev1/3/5 rate쌍)로 만든 이전 pitchcount_recover.py 워크로드 피처가
  실제 제출에서 -4.2였던 전례 있음. 이번 버전은 신호가 1개 피처(workload 변화량)에 집중돼
  있고 절대 투구수(투수 역할의 대리변수, 이전 실패 원인 추정)가 아니라 성격이 다르다고 보고 시도.

가중치(0.5/0.5)는 그대로 둔다 — v14a/v14b 실측에서 가중치 이동 효과가 +0.05로 사실상 없었음.

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


def infer_min_denominator(success_rate, middle_rate, max_q, chunk=4000):
    q = np.arange(1, max_q + 1, dtype=np.float64)
    s_all = pd.Series(success_rate).to_numpy(dtype=np.float64)
    m_all = pd.Series(middle_rate).to_numpy(dtype=np.float64)
    inferred = np.ones(len(s_all), dtype=np.float64)
    for start in range(0, len(s_all), chunk):
        s = s_all[start:start + chunk, None]
        m = m_all[start:start + chunk, None]
        missing = np.isnan(s[:, 0]) | np.isnan(m[:, 0])
        s = np.nan_to_num(s, nan=0.0)
        m = np.nan_to_num(m, nan=0.0)
        err = np.maximum(np.abs(s * q - np.rint(s * q)), np.abs(m * q - np.rint(m * q))) / q
        valid = err <= 5.1e-7
        vals = np.where(valid.any(axis=1), valid.argmax(axis=1) + 1, err.argmin(axis=1) + 1)
        vals[missing] = 1.0
        inferred[start:start + len(vals)] = vals
    return inferred


def hidden_denominator_features(df):
    out = pd.DataFrame(index=df.index)
    for k, max_q in ((1, 160), (3, 480), (5, 800)):
        out[f"prev{k}_hidden_total_n"] = infer_min_denominator(
            df[f"asof_pitcher_prev{k}_game_success_rate"],
            df[f"asof_pitcher_prev{k}_game_middle_rate"],
            max_q,
        )
    out["prev3_hidden_avg_n"] = out["prev3_hidden_total_n"] / 3.0
    out["prev5_hidden_avg_n"] = out["prev5_hidden_total_n"] / 5.0
    out["prev1_vs_prev3_workload"] = out["prev1_hidden_total_n"] - out["prev3_hidden_avg_n"]
    out["prev3_vs_prev5_workload"] = out["prev3_hidden_avg_n"] - out["prev5_hidden_avg_n"]
    return out.astype(np.float64)


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

    print("\n[4c] hidden denominator workload 7 features (row-independent)...", flush=True)
    X_den = hidden_denominator_features(df).reset_index(drop=True)
    print(f"  prev1_den mean={X_den['prev1_hidden_total_n'].mean():.2f}  ({time.time()-t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_den], axis=1)
    print(f"\n교차항 {C.shape[1]}개 + 작년 {X_ly.shape[1]}개 + hidden_den {X_den.shape[1]}개 -> 최종 피처 수={X.shape[1]} (v15 91 -> 98)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6] CatBoost 3시드 학습 (v15는 단일시드 -> 시드평균으로 교체, 그 외 전부 동일)...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cats = []
    for s in (42, 7, 2024):
        cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                                random_seed=s, verbose=0, early_stopping_rounds=50,
                                min_data_in_leaf=200, loss_function="Logloss")
        cb.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
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
        "pitchtype_stats": pitchtype_export_stats(pt_tables, g, sr, k_control=K_CONTROL, k_mix=K_MIX),
        "lastyear_stats": lastyear_export_stats(ly_tbl, gr, sr, k=30.0),
        "feature_order": list(X.columns),
    }
    for tag, w_hgb, w_cat in [("v19", 0.5, 0.5)]:
        artifacts = dict(common, w_hgb=w_hgb, w_cat=w_cat)
        out = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out)
        print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb={w_hgb} w_cat={w_cat}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
