"""모델 구조 재검토 — 두 가지 가설을 한 번의 학습으로 측정.

배경: 모델 다양화(CatBoost/LGBM 앙상블)는 4번 시도해서 4번 다 실제 점수가 떨어졌다.
      이유는 트리 계열끼리는 편향이 같아 다양성이 안 생기기 때문으로 본다.
      반면 검증 로그의 진짜 이상신호는 따로 있다: 2023 폴드 baseline BSS가 '음수'다.
      = 용량 부족이 아니라 시즌 간 분포이동(drift)에 깨지는 것.

가설1 (수축): p' = r + lambda*(p - r). 드리프트가 큰 해엔 lambda<1이 손실을 줄인다.
              test 2025는 검증 폴드보다 한 시즌 더 멀어 드리프트 위험이 구조적으로 크다.
가설2 (GLM): 정규화 로지스틱은 트리와 편향이 근본적으로 다르다(계단 vs 매끄러운 외삽).
             '깨지는 방식'이 달라야 앙상블 이득이 난다 -> 트리끼리 실패한 것과 대조.

각 폴드의 valid 예측을 저장하므로, 이후 lambda/블렌딩 비중 탐색은 재학습 없이 가능.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from inseason import build_season_end_table, transform_inseason
from metrics import evaluate
from phase2_common import FOLDS, build_fold

SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df["control_success"].mean())
    sr = sorted(df["season"].unique().tolist())
    se = build_season_end_table(df)
    df_ins = transform_inseason(df, se, g, sr)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    store = {}
    for train_max, valid_season in FOLDS:
        print(f"\n{'='*60}\nFOLD train<={train_max} valid={valid_season}\n{'='*60}", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True)
        y_tr, y_va = fold["y_train"], fold["y_valid"]
        tr_idx = df[df["season"] <= train_max].index
        va_idx = df[df["season"] == valid_season].index

        Xtr = pd.concat([fold["X_train"].reset_index(drop=True),
                         df_ins.loc[tr_idx, INSEASON_COLS].reset_index(drop=True)], axis=1)
        Xva = pd.concat([fold["X_valid"].reset_index(drop=True),
                         df_ins.loc[va_idx, INSEASON_COLS].reset_index(drop=True)], axis=1)

        print("  RF...", flush=True)
        rf = RandomForestClassifier(**RF_PARAMS).fit(Xtr, y_tr)
        p_rf = rf.predict_proba(Xva)[:, 1]
        print(f"  HGB... ({time.time()-t0:.0f}s)", flush=True)
        hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xtr, y_tr)
        p_hgb = hgb.predict_proba(Xva)[:, 1]

        print(f"  GLM(로지스틱)... ({time.time()-t0:.0f}s)", flush=True)
        sc = StandardScaler()
        Ztr = sc.fit_transform(Xtr.astype(np.float64))
        Zva = sc.transform(Xva.astype(np.float64))
        lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", n_jobs=-1)
        lr.fit(Ztr, y_tr)
        p_lr = lr.predict_proba(Zva)[:, 1]

        r_tr = float(y_tr.mean())   # 학습 기준율(테스트에선 train 기준율만 알 수 있음)
        store[valid_season] = dict(y=y_va, p_rf=p_rf, p_hgb=p_hgb, p_lr=p_lr, r_tr=r_tr)

        base = 0.15 * p_rf + 0.85 * p_hgb
        print(f"  [{valid_season}] rf={evaluate(y_va,p_rf)['bss']:.6f}  hgb={evaluate(y_va,p_hgb)['bss']:.6f}  "
              f"glm={evaluate(y_va,p_lr)['bss']:.6f}  blend={evaluate(y_va,base)['bss']:.6f}", flush=True)
        print(f"       기준율 train={r_tr:.5f} valid={y_va.mean():.5f} 드리프트={y_va.mean()-r_tr:+.5f}", flush=True)

    np.savez("phase10_preds.npz", **{f"{s}_{k}": v for s, d in store.items() for k, v in d.items()})
    print(f"\n예측 저장 완료 phase10_preds.npz  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 가설1: 수축 계수 lambda ----------
    print(f"\n{'='*60}\n가설1: 수축 p' = r + lambda*(p-r)  (r=train 기준율)\n{'='*60}", flush=True)
    lams = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
    print(f"  {'lambda':>7s} " + "".join(f"{s:>10d}" for s in store) + f"{'평균':>10s}", flush=True)
    for lam in lams:
        row, vals = "", []
        for s, d in store.items():
            p = 0.15 * d["p_rf"] + 0.85 * d["p_hgb"]
            pp = d["r_tr"] + lam * (p - d["r_tr"])
            b = evaluate(d["y"], pp)["bss"]
            vals.append(b)
            row += f"{b*100000:10.1f}"
        print(f"  {lam:7.2f} {row}{np.mean(vals)*100000:10.1f}", flush=True)

    # ---------- 가설2: GLM 블렌딩 ----------
    print(f"\n{'='*60}\n가설2: 트리 blend에 GLM 섞기  p = (1-w)*blend + w*glm\n{'='*60}", flush=True)
    ws = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    print(f"  {'w_glm':>7s} " + "".join(f"{s:>10d}" for s in store) + f"{'평균':>10s}", flush=True)
    for w in ws:
        row, vals = "", []
        for s, d in store.items():
            p = (1 - w) * (0.15 * d["p_rf"] + 0.85 * d["p_hgb"]) + w * d["p_lr"]
            b = evaluate(d["y"], p)["bss"]
            vals.append(b)
            row += f"{b*100000:10.1f}"
        print(f"  {w:7.2f} {row}{np.mean(vals)*100000:10.1f}", flush=True)

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
