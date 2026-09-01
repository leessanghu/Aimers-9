"""in-season 메커니즘을 아직 안 쓴 통계군에 적용 — 2024 폴드 검증.

조건부 분할(platoon/inning)은 2승 6패로 고갈 판정. 방향 전환:
가장 크게 이긴 건 in-season(+114점)인데, 아직 '투수 자신의 success/ball/reverse'에만 썼다.
주최측이 준 나머지 누적 통계도 전부 커리어 누적이라 같은 드리프트 문제를 갖는다:

  A) 타자 in-season   : asof_batter_n / success_rate / middle_rate
                        (타자 정체성 진짜SD=0.0481로 투수 0.0555에 필적)
  B) 피치믹스 in-season: asof_pitcher_pitchmix_n / fastball / breaking / offspeed_rate
                        (구종 구성이 올 시즌 바뀐 투수 = 제구도 달라졌을 개연성)

복원 트릭은 동일: round(rate * n) = 누적 횟수, 직전 시즌 끝 누적을 빼면 시즌 한정.
leakage: 각 행은 자기 entity의 '직전 시즌 끝'만 참조. 같은 시즌/test 다른 행 참조 없음.

baseline = v7c 실전 구성 = 실제 948.970점.  로컬 델타 x0.47 ~= 실제 예상.
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
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
K_SMOOTH = 15.0
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                  "inseason_n", "inseason_is_first_appearance"]
TRAIN_MAX, VALID_SEASON = 2023, 2024
TARGET = "control_success"


def generic_inseason(df, entity, n_col, rate_cols, prefix, seasons_range, use_label_fix=True):
    """entity의 '이번 시즌 한정' 비율들을 복원. inseason.py 로직의 일반화.

    n_col/rate_cols는 그 entity의 '커리어 누적' asof 컬럼. 시즌 마지막 행 기준으로
    off-by-one(그 투구 자신이 빠짐)이 있어 n은 +1 보정하고, 대표 rate 하나는 라벨로 보정한다."""
    sub = df.sort_values([entity, "row_num"])
    last = sub.groupby([entity, "season"], as_index=False).last()
    n_before = last[n_col].fillna(0).to_numpy(np.float64)

    ends = {"N_end": n_before + 1}
    for i, c in enumerate(rate_cols):
        cnt = np.round(last[c].fillna(0).to_numpy(np.float64) * n_before)
        if i == 0 and use_label_fix:
            cnt = cnt + last[TARGET].to_numpy(np.float64)
        ends[f"S{i}_end"] = cnt
    end_tbl = pd.DataFrame({entity: last[entity], "season": last["season"], **ends})

    lookup = pd.MultiIndex.from_arrays([df[entity].to_numpy(), df["season"].to_numpy() - 1])
    got = {}
    for col in ends:
        p = end_tbl.pivot(index=entity, columns="season", values=col)
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        got[col] = np.nan_to_num(p.stack(future_stack=True).reindex(lookup).to_numpy().astype(np.float64), nan=0.0)

    n_now = df[n_col].fillna(0).to_numpy(np.float64)
    n_season = np.clip(n_now - got["N_end"], 0, None)

    out = pd.DataFrame(index=df.index)
    for i, c in enumerate(rate_cols):
        s_now = np.round(df[c].fillna(0).to_numpy(np.float64) * n_now)
        s_season = np.clip(s_now - got[f"S{i}_end"], 0, None)
        raw = np.divide(s_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
        gm = float(df[c].mean(skipna=True))
        out[f"{prefix}_{c.split('_')[-2]}_smooth"] = (
            (n_season * np.nan_to_num(raw) + K_SMOOTH * gm) / (n_season + K_SMOOTH))
    out[f"{prefix}_n"] = np.log1p(n_season)
    out[f"{prefix}_first"] = (n_season == 0).astype(np.float64)
    return out


def run_hgb(Xtr, ytr, Xva, yva, tag):
    t = time.time()
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xtr, ytr)
    bss = evaluate(yva, hgb.predict_proba(Xva)[:, 1])["bss"]
    print(f"  [{tag:20s}] {Xtr.shape[1]}피처  BSS={bss:.6f}  score={max(0,bss*100000):7.1f}  ({time.time()-t:.0f}s)", flush=True)
    return bss


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df[TARGET].mean())
    sr = sorted(df["season"].unique().tolist())

    season_end = build_season_end_table(df)
    df_ins = transform_inseason(df, season_end, g, sr)
    piv = _pivots_from_table(season_end, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior_p = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
    df_plt = transform_platoon(df, build_platoon_table(df), prior_p, sr, k=K_PLATOON)
    df_inn = transform_inning(df, build_inning_table(df), build_inning_offset(df), prior_p, sr, k=570.0)

    print(f"baseline 준비 ({time.time()-t0:.0f}s)", flush=True)

    bat = generic_inseason(df, "batter_id", "asof_batter_n",
                           ["asof_batter_success_rate", "asof_batter_middle_rate"], "binseason", sr)
    mix = generic_inseason(df, "pitcher_id", "asof_pitcher_pitchmix_n",
                           ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                            "asof_pitcher_offspeed_rate"], "mixinseason", sr, use_label_fix=False)
    for nm, f in [("타자 in-season", bat), ("피치믹스 in-season", mix)]:
        print(f"  {nm}: {list(f.columns)}", flush=True)
        print(f"    시즌첫등장비율={f[[c for c in f.columns if c.endswith('_first')][0]].mean():.4f}", flush=True)

    fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= TRAIN_MAX].index, df[df.season == VALID_SEASON].index

    def stack(bf, i, extra=()):
        parts = [bf.reset_index(drop=True), df_ins.loc[i, INSEASON_COLS].reset_index(drop=True),
                 df_plt.loc[i].reset_index(drop=True), df_inn.loc[i].reset_index(drop=True)]
        parts += [e.loc[i].reset_index(drop=True) for e in extra]
        return pd.concat(parts, axis=1)

    Xtr_b, Xva_b = stack(fold["X_train"], tr), stack(fold["X_valid"], va)
    print(f"\n{'='*62}\n2024 폴드 (baseline = v7c, 실제 948.970점)\n{'='*62}", flush=True)
    res = {"baseline": run_hgb(Xtr_b, ytr, Xva_b, yva, "baseline(v7c)")}
    res["+batter_inseason"] = run_hgb(stack(fold["X_train"], tr, [bat]), ytr,
                                      stack(fold["X_valid"], va, [bat]), yva, "+batter_inseason")
    res["+pitchmix_inseason"] = run_hgb(stack(fold["X_train"], tr, [mix]), ytr,
                                        stack(fold["X_valid"], va, [mix]), yva, "+pitchmix_inseason")
    res["+both"] = run_hgb(stack(fold["X_train"], tr, [bat, mix]), ytr,
                           stack(fold["X_valid"], va, [bat, mix]), yva, "+both")

    b = res["baseline"]
    print(f"\n{'='*62}\nbaseline 대비\n{'='*62}", flush=True)
    for k_, v in res.items():
        if k_ != "baseline":
            d = 100000 * (v - b)
            print(f"  {k_:22s} delta={d:+7.1f}   실제예상={d*0.47:+6.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
