"""피처 추가의 '고정 비용'이 구조적인지 진단 + 용량 조정으로 제거되는지 검증.

관찰: 최근 8연패의 델타가 내용과 무관하게 -3~-7에 몰림
      (조건부분할 -3.5/-4.3/-4.8/-12.4, in-season확장 -6.4/-6.5/-7.0)
가설: 피처 추가 자체에 고정 비용이 있다. 현재 HGB 파라미터(max_leaf_nodes=31, max_iter=500)는
      58피처 시절 값인데 지금 67피처. 용량이 모자라 새 피처가 예산만 잡아먹는 상태일 수 있다.
      -> 사실이면 플래툰(+38)/이닝(+19.9)은 '관문을 넘을 만큼 컸을 뿐'이고,
         방금 버린 신호들은 용량을 늘리면 되살아난다.

진단: 순수 난수 피처 4개를 넣는다. 난수도 -5가 나오면 비용은 구조적(내용 무관)이 맞다.
      난수가 0에 가까우면 가설 기각 -> 그 피처들이 실제로 해로웠던 것.

baseline = v7c 실전 구성 = 실제 948.970점. 로컬 델타 x0.47 ~= 실제 예상.
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
from phase2_common import build_fold
from phase14_inseason_ext import generic_inseason
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
BASE_HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]
TRAIN_MAX, VALID_SEASON = 2023, 2024


def run(Xtr, ytr, Xva, yva, tag, **over):
    p = dict(BASE_HGB, **over)
    t = time.time()
    m = HistGradientBoostingClassifier(**p).fit(Xtr, ytr)
    bss = evaluate(yva, m.predict_proba(Xva)[:, 1])["bss"]
    print(f"  [{tag:30s}] {Xtr.shape[1]:3d}피처 iter={m.n_iter_:4d}  BSS={bss:.6f}  "
          f"score={max(0,bss*100000):7.1f}  ({time.time()-t:.0f}s)", flush=True)
    return bss


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df["control_success"].mean())
    sr = sorted(df["season"].unique().tolist())

    se = build_season_end_table(df)
    df_ins = transform_inseason(df, se, g, sr)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior_p = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
    df_plt = transform_platoon(df, build_platoon_table(df), prior_p, sr, k=K_PLATOON)
    df_inn = transform_inning(df, build_inning_table(df), build_inning_offset(df), prior_p, sr, k=570.0)
    bat = generic_inseason(df, "batter_id", "asof_batter_n",
                           ["asof_batter_success_rate", "asof_batter_middle_rate"], "binseason", sr)

    rng = np.random.default_rng(0)
    rnd = pd.DataFrame({f"rnd{i}": rng.standard_normal(len(df)) for i in range(4)}, index=df.index)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= TRAIN_MAX].index, df[df.season == VALID_SEASON].index

    def stack(bf, i, extra=()):
        parts = [bf.reset_index(drop=True), df_ins.loc[i, INSEASON_COLS].reset_index(drop=True),
                 df_plt.loc[i].reset_index(drop=True), df_inn.loc[i].reset_index(drop=True)]
        parts += [e.loc[i].reset_index(drop=True) for e in extra]
        return pd.concat(parts, axis=1)

    Xtr, Xva = stack(fold["X_train"], tr), stack(fold["X_valid"], va)
    Xtr_r, Xva_r = stack(fold["X_train"], tr, [rnd]), stack(fold["X_valid"], va, [rnd])
    Xtr_b, Xva_b = stack(fold["X_train"], tr, [bat]), stack(fold["X_valid"], va, [bat])

    print(f"\n{'='*74}\n[1] 진단: 고정 비용이 구조적인가 (현재 파라미터)\n{'='*74}", flush=True)
    base = run(Xtr, ytr, Xva, yva, "baseline(v7c)")
    rnd_b = run(Xtr_r, ytr, Xva_r, yva, "+난수4개")
    print(f"\n  >> 난수 델타 = {100000*(rnd_b-base):+.1f}   "
          f"(-3~-7이면 구조적 비용 확정 / 0근처면 가설 기각)", flush=True)

    print(f"\n{'='*74}\n[2] 용량 조정 (baseline 67피처)\n{'='*74}", flush=True)
    caps = {
        "leaves63_iter1000": dict(max_leaf_nodes=63, max_iter=1000),
        "leaves127_it1500_lr02": dict(max_leaf_nodes=127, max_iter=1500, learning_rate=0.02),
        "depth8_leaves63": dict(max_depth=8, max_leaf_nodes=63, max_iter=1000),
        "l2_1.0": dict(l2_regularization=1.0),
    }
    cap_res = {}
    for nm, ov in caps.items():
        cap_res[nm] = run(Xtr, ytr, Xva, yva, f"baseline {nm}", **ov)

    best_nm = max(cap_res, key=cap_res.get)
    print(f"\n  >> 최고 용량 설정: {best_nm}  delta={100000*(cap_res[best_nm]-base):+.1f}", flush=True)

    print(f"\n{'='*74}\n[3] 늘린 용량에서 버렸던 피처가 되살아나는가\n{'='*74}", flush=True)
    ov = caps[best_nm]
    b2 = cap_res[best_nm]
    r2 = run(Xtr_r, ytr, Xva_r, yva, f"+난수4개 [{best_nm}]", **ov)
    f2 = run(Xtr_b, ytr, Xva_b, yva, f"+batter_inseason [{best_nm}]", **ov)
    print(f"\n  난수 델타(용량↑) = {100000*(r2-b2):+.1f}  (기존 {100000*(rnd_b-base):+.1f})", flush=True)
    print(f"  batter 델타(용량↑) = {100000*(f2-b2):+.1f}  (기존 -6.4)", flush=True)

    print(f"\n{'='*74}\n최종 요약 — baseline(현재파라미터) 대비\n{'='*74}", flush=True)
    for nm, v in [("난수4개", rnd_b), *cap_res.items(),
                  (f"난수[{best_nm}]", r2), (f"batter[{best_nm}]", f2)]:
        d = 100000 * (v - base)
        print(f"  {nm:26s} {d:+7.1f}   실제예상={d*0.47:+6.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
