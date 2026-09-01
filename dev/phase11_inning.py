"""투수 x 이닝 조건부 분할 3폴드 검증.

baseline = v7b 실전 구성 (58 base + 5 in-season + 2 platoon, HGB 단독) = 실제 939.681점
arm      = + inning_diff/inning_n (K 민감도 2종)

RF를 뺀 이유: v7b(HGB단독 939.681) > v7a(RF15%+HGB85% 935.094). 검증 폴드는 실제 제출보다
적은 시즌으로 학습해 분산감소 수단(RF/블렌딩/수축)을 구조적으로 과대평가한다는 걸 확인함.
배포 구성과 검증 구성을 일치시킨다.

근거: 투수x이닝 상호작용 진짜SD=0.0209 (노이즈 제거) -> 상한 ~174점.
      비교) 투수x타자손 0.0438 -> 상한 192점, 실제 +14점 (실현율 ~7%)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import FOLDS, build_fold
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]


def get_prior_rate(df, season_end_table, global_success_rate, seasons_range):
    pivots = _pivots_from_table(season_end_table, seasons_range)
    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = pivots["rate"].reindex(lookup_idx).to_numpy()
    return pd.Series(vals).fillna(global_success_rate).to_numpy(np.float64)


def run_hgb(X_train, y_train, X_valid, y_valid, tag):
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p = hgb.predict_proba(X_valid)[:, 1]
    bss = evaluate(y_valid, p)["bss"]
    print(f"  [{tag:16s}] BSS={bss:.6f}  score={max(0, bss*100000):.1f}", flush=True)
    return bss


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df["control_success"].mean())
    sr = sorted(df["season"].unique().tolist())

    print("in-season / platoon / inning 테이블 구성...", flush=True)
    season_end = build_season_end_table(df)
    df_ins = transform_inseason(df, season_end, g, sr)
    prior_rate = get_prior_rate(df, season_end, g, sr)
    platoon_table = build_platoon_table(df)
    df_plt = transform_platoon(df, platoon_table, prior_rate, sr, k=K_PLATOON)
    inning_table = build_inning_table(df)
    print(f"  이닝 셀={len(inning_table):,}  ({time.time()-t0:.0f}s)", flush=True)

    all_results = {}
    for train_max, valid_season in FOLDS:
        print(f"\n{'='*60}\nFOLD train<={train_max} valid={valid_season}\n{'='*60}", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True)
        y_tr, y_va = fold["y_train"], fold["y_valid"]
        tr_idx = df[df["season"] <= train_max].index
        va_idx = df[df["season"] == valid_season].index

        # 전역 이닝 효과는 train fold에서만 계산 (미래 시즌 미사용)
        inn_off = build_inning_offset(df.loc[tr_idx])

        Xtr = pd.concat([fold["X_train"].reset_index(drop=True),
                         df_ins.loc[tr_idx, INSEASON_COLS].reset_index(drop=True),
                         df_plt.loc[tr_idx].reset_index(drop=True)], axis=1)
        Xva = pd.concat([fold["X_valid"].reset_index(drop=True),
                         df_ins.loc[va_idx, INSEASON_COLS].reset_index(drop=True),
                         df_plt.loc[va_idx].reset_index(drop=True)], axis=1)

        print(f"\n--- baseline (v7b 구성, {Xtr.shape[1]}피처) ---", flush=True)
        fold_res = {"baseline": run_hgb(Xtr, y_tr, Xva, y_va, "baseline")}

        for k in (570.0, 200.0):
            f_tr = transform_inning(df.loc[tr_idx], inning_table, inn_off, prior_rate[tr_idx], sr, k=k).reset_index(drop=True)
            f_va = transform_inning(df.loc[va_idx], inning_table, inn_off, prior_rate[va_idx], sr, k=k).reset_index(drop=True)
            xt = pd.concat([Xtr, f_tr], axis=1)
            xv = pd.concat([Xva, f_va], axis=1)
            name = f"+inning_K{k:.0f}"
            print(f"\n--- {name} ({xt.shape[1]}피처)  diff_SD={f_va['inning_diff'].std():.5f} ---", flush=True)
            fold_res[name] = run_hgb(xt, y_tr, xv, y_va, name)

        b = fold_res["baseline"]
        print(f"\n--- {valid_season} baseline 대비 ---", flush=True)
        for n_, v in fold_res.items():
            if n_ != "baseline":
                print(f"    {n_:16s} delta={100000*(v-b):+7.1f}", flush=True)
        all_results[valid_season] = fold_res

    print(f"\n{'='*60}\n전체 요약 (HGB 단독, baseline=v7b 구성)\n{'='*60}", flush=True)
    for season, fr in all_results.items():
        b = fr["baseline"]
        line = "  ".join(f"{n_}={100000*(v-b):+7.1f}" for n_, v in fr.items() if n_ != "baseline")
        print(f"  {season}  baseline={max(0,b*100000):7.1f} | {line}", flush=True)
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
