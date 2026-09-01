"""전 축 통합 검증 — 2024 폴드.

지금까지 개별로 확인된 이득 (전부 baseline=v7c HGB 단독 대비):
  타자제거   -batter_asof(4개)            +33.0
  칼만       inseason 5개 -> 칼만 4개 교체  +10.7
  CatBoost   단독으로 HGB보다             +63
  GBM앙상블  hgb+lgbm+cat                 +66.1
  NN블렌드   noemb MLP, w=0.30            +50.6

이것들이 실제로 합쳐지는지(중복인지 가산인지)가 관건.
피처셋 2종 x (HGB/LGBM/Cat) -> 최고 조합에 NN 블렌딩까지.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table, K_SMOOTH
from kalman_ability import build_kalman_table, estimate_process_noise, transform_kalman
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from phase21_nn_variants import train_variant
from platoon import build_platoon_table, transform_platoon, K_PLATOON

torch.set_num_threads(4)
SEED = 42
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
BATTER_ASOF = ["flag_asof_batter_n_zero", "asof_batter_n",
               "asof_batter_success_rate_smooth", "asof_batter_middle_rate_smooth"]


def fit_gbms(Xtr, ytr, Xva, yva, tag, t0):
    out = {}
    t = time.time()
    m = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
    out["hgb"] = m.predict_proba(Xva)[:, 1]
    print(f"    hgb  BSS={evaluate(yva,out['hgb'])['bss']:.6f} ({time.time()-t:.0f}s)", flush=True)

    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    tr_i, es_i = time_split_es(len(Xtr))
    t = time.time()
    lg = LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, max_depth=6,
                        min_child_samples=200, reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9,
                        subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1)
    lg.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=[(Xtr.iloc[es_i], ytr[es_i])],
           callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    out["lgbm"] = lg.predict_proba(Xva)[:, 1]
    print(f"    lgbm BSS={evaluate(yva,out['lgbm'])['bss']:.6f} ({time.time()-t:.0f}s)", flush=True)

    from catboost import CatBoostClassifier
    t = time.time()
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=SEED, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
    out["cat"] = cb.predict_proba(Xva)[:, 1]
    print(f"    cat  BSS={evaluate(yva,out['cat'])['bss']:.6f} ({time.time()-t:.0f}s)", flush=True)
    return out


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

    q = estimate_process_noise(df)
    th, P = build_kalman_table(df, sr, q, g)
    n_season = np.expm1(dins["inseason_n"].to_numpy(np.float64))
    sm = dins["inseason_success_smooth"].to_numpy(np.float64)
    raw = np.clip(np.where(n_season > 0, (sm * (n_season + K_SMOOTH) - K_SMOOTH * pp) / np.maximum(n_season, 1e-9), np.nan), 0, 1)
    dkal = transform_kalman(df, th, P, g, inseason_n=n_season, inseason_rate=raw)

    fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= 2023].index, df[df.season == 2024].index

    def build(i, bf, use_kalman):
        parts = [bf.reset_index(drop=True)]
        parts.append(dkal.loc[i].reset_index(drop=True) if use_kalman
                     else dins.loc[i, INS].reset_index(drop=True))
        parts += [dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True)]
        X = pd.concat(parts, axis=1).astype(np.float64)
        return X.drop(columns=[c for c in BATTER_ASOF if c in X.columns])

    sets = {
        "F1: -batter_asof (inseason)": (build(tr, fold["X_train"], False), build(va, fold["X_valid"], False)),
        "F2: -batter_asof + 칼만교체": (build(tr, fold["X_train"], True), build(va, fold["X_valid"], True)),
    }

    # 기준점: v7c 원본 HGB (phase19b에서 저장)
    p_ref = np.load("phase19b_hgb_pred_2024.npy")
    base = evaluate(yva, p_ref)["bss"]
    print(f"기준 v7c HGB BSS={base:.6f} ({base*1e5:.1f}) = 실제 948.970점\n" + "=" * 76, flush=True)

    all_preds = {}
    for nm, (xt, xv) in sets.items():
        print(f"\n--- {nm}  ({xt.shape[1]}피처) ---", flush=True)
        all_preds[nm] = fit_gbms(xt, ytr, xv, yva, nm, t0)

    print(f"\n{'='*76}\nGBM 앙상블 (기준 v7c HGB 대비)\n{'='*76}", flush=True)
    best = (None, -9, None)
    for nm, pr in all_preds.items():
        for combo in [("hgb",), ("cat",), ("hgb", "lgbm"), ("hgb", "cat"), ("lgbm", "cat"),
                      ("hgb", "lgbm", "cat")]:
            p = np.mean([pr[k] for k in combo], axis=0)
            b = evaluate(yva, p)["bss"]
            d = 1e5 * (b - base)
            print(f"  {nm:28s} {'+'.join(combo):16s} score={max(0,b*1e5):7.1f}  delta={d:+7.1f}  실제예상={d*0.47:+6.1f}", flush=True)
            if b > best[1]:
                best = (f"{nm} | {'+'.join(combo)}", b, p)
    print(f"\n  >> 최고: {best[0]}  score={best[1]*1e5:.1f}", flush=True)

    # ---- 최고 피처셋에 NN(noemb) 블렌딩 ----
    best_set = best[0].split(" | ")[0]
    xt, xv = sets[best_set]
    print(f"\n{'='*76}\nNN(noemb) 학습 후 블렌딩 — 피처셋 [{best_set}]\n{'='*76}", flush=True)
    mu, sd = xt.to_numpy().mean(0), xt.to_numpy().std(0) + 1e-8
    Ztr = ((xt.to_numpy() - mu) / sd).astype(np.float32)
    Zva = ((xv.to_numpy() - mu) / sd).astype(np.float32)
    zero_tr = np.zeros(len(Ztr), np.int64)
    zero_va = np.zeros(len(Zva), np.int64)
    p_nn = train_variant("noemb", False, 8, 0.0, Ztr, zero_tr, zero_tr, ytr,
                         Zva, zero_va, zero_va, yva, 1, 1, t0=t0)
    np.save("phase23_nn_pred.npy", p_nn)
    bn = evaluate(yva, p_nn)["bss"]
    print(f"  NN 단독 BSS={bn:.6f} ({max(0,bn*1e5):.1f})  상관r={np.corrcoef(best[2],p_nn)[0,1]:.4f}", flush=True)

    print(f"\n{'='*76}\n최종: GBM앙상블 + NN\n{'='*76}", flush=True)
    for w in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        p = (1 - w) * best[2] + w * p_nn
        b = evaluate(yva, p)["bss"]
        d = 1e5 * (b - base)
        print(f"  w_nn={w:.2f}  score={max(0,b*1e5):7.1f}  delta={d:+7.1f}  실제예상={d*0.47:+6.1f}"
              f"  -> 예상 최종점수 {948.97 + d*0.47:7.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
