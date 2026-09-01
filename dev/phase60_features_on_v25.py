"""v25(recency-weighted, 981.44 실증, 현재 최고점)를 새 기준선으로 놓고 hidden_denominator/
career_volatility/arsenal_entropy를 다시 스크리닝한다.

기존에 이 3개는 전부 v15(가중치 없는 균등학습) 대비로 검증돼서 기각/보류됐었다
(v18 973.66, v19 972.76, v20/v22는 로컬만 확인). 근데 v25가 학습 분포 자체를
바꿨으니(최근 시즌 가중), 그 기준에서 다시 봐야 공정하다. phase59와 동일하게
2023->2024 폴드, HGB+CatBoost 단일시드 50:50, recency_weight(half_life=2)로 통일."""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier

from arsenal_entropy import K_ARSENAL, transform_arsenal
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]

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
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
gr = build_global_rates(df)
dly = transform_lastyear(df, build_lastyear_table(df), gr, sr, k=30.0)

vol_tbl = build_volatility_table(se)
dvol = transform_volatility(df, vol_tbl, sr, k=K_VOL)

arsenal_global_mix = {c: float(df[c].mean(skipna=True)) for c in
                      ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]}
dars = transform_arsenal(df, global_mix=arsenal_global_mix, k=K_ARSENAL)


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


def hidden_denominator_features(d):
    out = pd.DataFrame(index=d.index)
    for k, max_q in ((1, 160), (3, 480), (5, 800)):
        out[f"prev{k}_hidden_total_n"] = infer_min_denominator(
            d[f"asof_pitcher_prev{k}_game_success_rate"], d[f"asof_pitcher_prev{k}_game_middle_rate"], max_q)
    out["prev3_hidden_avg_n"] = out["prev3_hidden_total_n"] / 3.0
    out["prev5_hidden_avg_n"] = out["prev5_hidden_total_n"] / 5.0
    out["prev1_vs_prev3_workload"] = out["prev1_hidden_total_n"] - out["prev3_hidden_avg_n"]
    out["prev3_vs_prev5_workload"] = out["prev3_hidden_avg_n"] - out["prev5_hidden_avg_n"]
    return out.astype(np.float64)


dhid = hidden_denominator_features(df)
print(f"베이스 + 신규 피처블록 준비 완료 ({time.time()-t0:.0f}s)", flush=True)


def recency_weight(seasons, half_life=2.0):
    age = seasons.max() - seasons
    return 0.5 ** (age / half_life)


def stack(i, extra=None):
    base = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                      dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                      dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    parts = [base, add_crosses(base), dly.loc[i].reset_index(drop=True)]
    if extra is not None:
        parts.append(extra.loc[i].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def train_eval(train_max, valid_season, extra_block=None, tag=""):
    global bf
    tr = df[df.season <= train_max].index
    va = df[df.season == valid_season].index
    fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True,
                      team_te_mode="expanding")
    ytr, yva = fold["y_train"], fold["y_valid"]
    bf = fold["X_train"]
    Xtr = stack(tr, extra_block)
    bf = fold["X_valid"]
    Xva = stack(va, extra_block)

    w = recency_weight(df.loc[tr, "season"].to_numpy(np.float64), half_life=2.0)
    ti, ei = time_split_es(len(Xtr))

    h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=SEED)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                            verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    h.fit(Xtr, ytr, sample_weight=w)
    cb.fit(Xtr.iloc[ti], ytr[ti], sample_weight=w[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))

    p_va = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
    score = max(0, evaluate(yva, p_va)["bss"] * 1e5)

    pool = Pool(Xva, yva)
    sv = cb.get_feature_importance(pool, type="ShapValues")[:, :-1]
    mags = np.abs(sv).mean(axis=0)
    col_mag = dict(zip(Xva.columns, mags))
    top_overall = sorted(col_mag.items(), key=lambda kv: -kv[1])[:3]
    mag_report = (f"\n    상위3(전체) magnitude: " +
                 ", ".join(f"{c}={v:.5f}" for c, v in top_overall))
    if extra_block is not None:
        extra_mag = {c: col_mag[c] for c in extra_block.columns}
        mag_report += (f"\n    신규 feature magnitude: " +
                       ", ".join(f"{c}={v:.5f}" for c, v in extra_mag.items()))

    print(f"[{tag}] valid={valid_season}  score={score:.1f}  n_extra={0 if extra_block is None else extra_block.shape[1]}  "
          f"({time.time()-t0:.0f}s){mag_report}", flush=True)
    return score


print("\n=== v25 재기준선 (2023->2024, recency_weight, team_te=expanding) ===", flush=True)
s_base = train_eval(2023, 2024, extra_block=None, tag="v25_baseline")

print("\n=== + hidden_denominator (v18/19 계열, 7개) ===", flush=True)
s_hid = train_eval(2023, 2024, extra_block=dhid, tag="v25+hidden_denom")

print("\n=== + career_volatility (v20 계열, 5개) ===", flush=True)
s_vol = train_eval(2023, 2024, extra_block=dvol, tag="v25+volatility")

print("\n=== + arsenal_entropy (v22 계열, 2개) ===", flush=True)
s_ars = train_eval(2023, 2024, extra_block=dars, tag="v25+arsenal")

print("\n=== 요약 ===", flush=True)
print(f"  v25_baseline      {s_base:.1f}")
print(f"  +hidden_denom      {s_hid:.1f}  (delta {s_hid-s_base:+.1f})")
print(f"  +volatility        {s_vol:.1f}  (delta {s_vol-s_base:+.1f})")
print(f"  +arsenal           {s_ars:.1f}  (delta {s_ars-s_base:+.1f})")

print(f"\n총 {time.time()-t0:.0f}s", flush=True)
