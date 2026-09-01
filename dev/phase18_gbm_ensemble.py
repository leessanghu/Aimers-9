"""GBM 계열 앙상블 재검증 — 현재 67피처 위에서.

왜 다시 하는가 (이전 '앙상블 실패' 결론의 구멍):
  기존 근거 5개 중 4개는 810점대 시절(= in-season/platoon/inning 이전) 실험이라 피처셋이 달랐다.
  최근 근거는 v7a/v7b 하나뿐인데 그건 RF였다. RF는 max_depth=10/min_samples_leaf=200으로
  편향이 큰 모델이라 섞으면 HGB의 날카로운 신호를 뭉갠다(플래툰에서 RF 2023 델타 -51.6).
  LGBM/XGB/CatBoost는 HGB와 같은 계열/비슷한 편향 -> 평균은 '분산 감소'에 가까운 다른 연산.

그리고 우리 진단이 이걸 지지한다:
  phase15/16에서 용량↑ -143, 용량↓ -43 -> 모델은 과적합 제약(분산 지배) 영역.
  그게 같은 계열 모델 평균이 가장 잘 듣는 조건이다.

주의: 로컬이 모델 배합을 과대평가한 전례가 있으므로(v7a), 개별 성능과 상관계수를 함께 본다.
      상관이 낮을수록 앙상블 이득이 진짜다.

baseline = v7c 구성 HGB 단독 = 실제 948.970점.
"""
import sys
import time
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
TRAIN_MAX, VALID_SEASON = 2023, 2024


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df["control_success"].mean())
    sr = sorted(df["season"].unique().tolist())
    se = build_season_end_table(df)
    dins = transform_inseason(df, se, g, sr)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
    dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
    dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=570.0)

    fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= TRAIN_MAX].index, df[df.season == VALID_SEASON].index

    def stack(bf, i):
        return pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                          dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True)], axis=1)

    Xtr, Xva = stack(fold["X_train"], tr), stack(fold["X_valid"], va)
    # 트리 모델들이 category dtype을 싫어하므로 전부 수치로 통일
    Xtr = Xtr.astype(np.float64)
    Xva = Xva.astype(np.float64)
    print(f"준비 완료 {Xtr.shape[1]}피처  ({time.time()-t0:.0f}s)\n" + "=" * 72, flush=True)

    preds = {}

    # ---- HGB (현 챔피언) ----
    t = time.time()
    hgb = HistGradientBoostingClassifier(
        max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
    preds["hgb"] = hgb.predict_proba(Xva)[:, 1]
    print(f"  hgb      BSS={evaluate(yva,preds['hgb'])['bss']:.6f}  ({time.time()-t:.0f}s)", flush=True)

    # ---- LGBM (HGB와 최대한 비슷한 정규화 강도로 맞춤) ----
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    tr_i, es_i = time_split_es(len(Xtr))
    t = time.time()
    lgb = LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, max_depth=6,
                         min_child_samples=200, reg_lambda=5.0, colsample_bytree=0.8,
                         subsample=0.9, subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1)
    lgb.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=[(Xtr.iloc[es_i], ytr[es_i])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    preds["lgbm"] = lgb.predict_proba(Xva)[:, 1]
    print(f"  lgbm     BSS={evaluate(yva,preds['lgbm'])['bss']:.6f}  iter={lgb.best_iteration_}  ({time.time()-t:.0f}s)", flush=True)

    # ---- XGBoost ----
    from xgboost import XGBClassifier
    t = time.time()
    xgb = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=200,
                        reg_lambda=5.0, subsample=0.9, colsample_bytree=0.8, random_state=SEED,
                        n_jobs=-1, eval_metric="logloss", early_stopping_rounds=50, tree_method="hist")
    xgb.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=[(Xtr.iloc[es_i], ytr[es_i])], verbose=False)
    preds["xgb"] = xgb.predict_proba(Xva)[:, 1]
    print(f"  xgb      BSS={evaluate(yva,preds['xgb'])['bss']:.6f}  iter={xgb.best_iteration}  ({time.time()-t:.0f}s)", flush=True)

    # ---- CatBoost ----
    from catboost import CatBoostClassifier
    t = time.time()
    cat = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                             random_seed=SEED, verbose=0, early_stopping_rounds=50,
                             min_data_in_leaf=200, loss_function="Logloss")
    cat.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
    preds["cat"] = cat.predict_proba(Xva)[:, 1]
    print(f"  cat      BSS={evaluate(yva,preds['cat'])['bss']:.6f}  iter={cat.best_iteration_}  ({time.time()-t:.0f}s)", flush=True)

    base = evaluate(yva, preds["hgb"])["bss"]

    print("\n" + "=" * 72 + "\n예측 상관 (낮을수록 앙상블 이득이 진짜)\n" + "=" * 72, flush=True)
    names = list(preds)
    for a, b in combinations(names, 2):
        print(f"  {a:5s}-{b:5s}  r={np.corrcoef(preds[a],preds[b])[0,1]:.4f}", flush=True)

    print("\n" + "=" * 72 + "\n앙상블 조합 (baseline=hgb 단독, 실제 948.970)\n" + "=" * 72, flush=True)
    combos = []
    for r_ in range(2, len(names) + 1):
        for c in combinations(names, r_):
            combos.append(c)
    rows = []
    for c in combos:
        p = np.mean([preds[k] for k in c], axis=0)
        rows.append(("+".join(c), evaluate(yva, p)["bss"]))
    # HGB 가중을 높인 변형도 확인
    for w in (0.5, 0.6, 0.7):
        rest = [k for k in names if k != "hgb"]
        p = w * preds["hgb"] + (1 - w) * np.mean([preds[k] for k in rest], axis=0)
        rows.append((f"hgb{w:.1f}+rest{1-w:.1f}", evaluate(yva, p)["bss"]))

    for nm, v in sorted(rows, key=lambda x: -x[1]):
        d = 100000 * (v - base)
        print(f"  {nm:24s} BSS={v:.6f}  score={max(0,v*100000):7.1f}  delta={d:+7.1f}  실제예상={d*0.47:+6.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
