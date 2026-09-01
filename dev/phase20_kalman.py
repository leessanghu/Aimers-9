"""칼만(상태공간) 실력 추정 검증 — 2024 폴드.

핵심 차이: 지금까지 실패한 13개는 전부 '추가'였다(모델이 새 피처를 못 삼킴).
칼만은 asof(커리어누적) + inseason(시즌한정)이 근사하던 걸 하나의 최적 필터로 '교체'하는 것이라
피처 개수 비용 없이 갈 수 있다. 그리고 우리 최대 실패요인인 드리프트를 정면으로 겨냥한다.

arm:
  baseline        v7c 구성 (67피처) = 실제 948.970점
  +kalman         칼만 4개 추가 (69->71피처)  <- '추가' 경로, 실패 가능성 있음
  kalman_replace  inseason_success_smooth를 kal_post로 교체 (개수 동일)  <- 본命
  kalman_only     inseason 5개를 칼만 4개로 통째 교체 (66피처)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table, K_SMOOTH
from kalman_ability import build_kalman_table, estimate_process_noise, transform_kalman
from metrics import evaluate
from phase2_common import build_fold
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
TRAIN_MAX, VALID_SEASON = 2023, 2024


def run(Xtr, ytr, Xva, yva, tag):
    t = time.time()
    m = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr)
    b = evaluate(yva, m.predict_proba(Xva)[:, 1])["bss"]
    print(f"  [{tag:20s}] {Xtr.shape[1]:3d}피처  BSS={b:.6f}  score={max(0,b*100000):7.1f}  ({time.time()-t:.0f}s)", flush=True)
    return b


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

    # ---- 칼만 ----
    q = estimate_process_noise(df)
    print(f"추정된 시즌간 드리프트 분산 q={q:.6f}  (SD={np.sqrt(q):.4f})", flush=True)
    th, P = build_kalman_table(df, sr, q, g)

    # 시즌 내 raw 관측 복원: inseason_success_smooth 를 역산
    n_season = np.expm1(dins["inseason_n"].to_numpy(np.float64))
    sm = dins["inseason_success_smooth"].to_numpy(np.float64)
    raw = np.where(n_season > 0, (sm * (n_season + K_SMOOTH) - K_SMOOTH * pp) / np.maximum(n_season, 1e-9), np.nan)
    raw = np.clip(raw, 0.0, 1.0)

    dkal = transform_kalman(df, th, P, g, inseason_n=n_season, inseason_rate=raw)
    print(f"  kal_pred SD={dkal.kal_pred.std():.5f}  kal_post SD={dkal.kal_post.std():.5f}  "
          f"(비교) inseason_smooth SD={sm.std():.5f}", flush=True)
    print(f"  corr(kal_post, inseason_smooth)={np.corrcoef(dkal.kal_post, sm)[0,1]:.4f}", flush=True)

    fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= TRAIN_MAX].index, df[df.season == VALID_SEASON].index

    def stack(bf, i, ins_cols=INS, extra=()):
        parts = [bf.reset_index(drop=True)]
        if ins_cols:
            parts.append(dins.loc[i, ins_cols].reset_index(drop=True))
        parts += [dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True)]
        parts += [e.loc[i].reset_index(drop=True) for e in extra]
        return pd.concat(parts, axis=1)

    print(f"\n{'='*70}\n2024 폴드 (baseline = v7c, 실제 948.970점)\n{'='*70}", flush=True)
    base = run(stack(fold["X_train"], tr), ytr, stack(fold["X_valid"], va), yva, "baseline(v7c)")

    res = {}
    res["+kalman(추가)"] = run(stack(fold["X_train"], tr, INS, [dkal]), ytr,
                             stack(fold["X_valid"], va, INS, [dkal]), yva, "+kalman(추가)")

    ins_no_succ = [c for c in INS if c != "inseason_success_smooth"]
    kal_one = dkal[["kal_post"]]
    res["교체:succ->kal_post"] = run(stack(fold["X_train"], tr, ins_no_succ, [kal_one]), ytr,
                                   stack(fold["X_valid"], va, ins_no_succ, [kal_one]), yva, "교체 succ->kal_post")

    res["교체:inseason전체"] = run(stack(fold["X_train"], tr, None, [dkal]), ytr,
                                stack(fold["X_valid"], va, None, [dkal]), yva, "교체 inseason전체")

    print(f"\n{'='*70}\nbaseline 대비 (실제예상 = 델타 x0.47)\n{'='*70}", flush=True)
    for k, v in sorted(res.items(), key=lambda x: -x[1]):
        d = 100000 * (v - base)
        print(f"  {k:22s} {d:+7.1f}   실제예상={d*0.47:+6.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
