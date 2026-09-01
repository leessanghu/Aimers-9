"""예측 편향(계통 과대예측) 보정 후보 4개 검증.

발견: calib_gap(pred평균-실제평균) = 2024검증에서 +0.0092~+0.0105 (3seed 일관, 253507행
표준오차 ~0.001이라 10시그마급 진짜 편향). isotonic(비모수, 파라미터 수백개)은 held-out
과적합으로 참패(-90~-122)했지만, 그건 "유연성이 너무 커서"였지 "보정 방향 자체가 틀려서"가
아니다. 여기선 훨씬 저유연도(파라미터 1~2개) 보정을 시도한다.

원인 가설: season(2019~2024)이 원시 피처로 들어가는데 시즌별 성공률이 계속 하락 중
(0.565->0.486). 트리는 외삽을 못 해서 2025(미본값)를 2024처럼 취급 -> 실제 하락분만큼 과대예측.

후보:
  A. 상수 오프셋 (offset = mean(pred)-mean(actual) on held-out, 그 값을 빼기)
  B. 로짓 시프트 (로짓 공간에서 상수 이동 후 재변환)
  C. season 피처를 원시값 대신 '최근 시즌으로부터 거리'로 교체 (외삽 문제 원천 차단)
  D. 최근 시즌 가중 학습 (sample_weight로 최근 시즌 비중 상향)

3폴드(2022/2023/2024)에서 baseline 편향이 일관되는지도 같이 확인한다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

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
print(f"베이스 피처 준비 완료 ({time.time()-t0:.0f}s)", flush=True)


def stack(i, tr_idx, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_eval(train_max, valid_season, season_mode="raw", sample_weight=None, tag=""):
    tr = df[df.season <= train_max].index
    va = df[df.season == valid_season].index
    fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True,
                      team_te_mode="expanding")
    ytr, yva = fold["y_train"], fold["y_valid"]
    Xtr, Xva = stack(tr, tr, fold["X_train"]), stack(va, tr, fold["X_valid"])

    if season_mode == "distance":
        Xtr["season"] = Xtr["season"].to_numpy() - train_max
        Xva["season"] = Xva["season"].to_numpy() - train_max
    elif season_mode == "drop":
        Xtr = Xtr.drop(columns=["season"])
        Xva = Xva.drop(columns=["season"])

    ti, ei = time_split_es(len(Xtr))
    w = None
    if sample_weight is not None:
        w = sample_weight(df.loc[tr, "season"].to_numpy())

    h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=SEED)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                            verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    if w is None:
        h.fit(Xtr, ytr)
        cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
    else:
        h.fit(Xtr, ytr, sample_weight=w)
        cb.fit(Xtr.iloc[ti], ytr[ti], sample_weight=w[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))

    p_va = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
    p_ho = 0.5 * h.predict_proba(Xtr.iloc[ei])[:, 1] + 0.5 * cb.predict_proba(Xtr.iloc[ei])[:, 1]
    y_ho = ytr[ei]

    base_score = max(0, evaluate(yva, p_va)["bss"] * 1e5)
    gap = float(p_va.mean() - yva.mean())
    gap_ho = float(p_ho.mean() - y_ho.mean())

    offset_A = p_va - gap_ho
    s_A = max(0, evaluate(yva, np.clip(offset_A, 0, 1))["bss"] * 1e5)

    shift = float(logit(np.array([y_ho.mean()]))[0] - logit(np.array([p_ho.mean()]))[0])
    p_B = sigmoid(logit(p_va) + shift)
    s_B = max(0, evaluate(yva, p_B)["bss"] * 1e5)

    print(f"[{tag}] train<=<{train_max} valid={valid_season}  base={base_score:.1f}  gap={gap:+.4f} (ho={gap_ho:+.4f})  "
          f"offsetA={s_A:.1f}  logitB={s_B:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(tag=tag, train_max=train_max, valid_season=valid_season, base=base_score, gap=gap,
                gap_ho=gap_ho, s_A=s_A, s_B=s_B)


print("\n=== 1) 3폴드 편향 일관성 확인 (raw season, baseline) ===", flush=True)
fold_results = []
for tmax, vseason in [(2021, 2022), (2022, 2023), (2023, 2024)]:
    fold_results.append(train_eval(tmax, vseason, season_mode="raw", tag="baseline"))

print("\n=== 2) season 피처 처리 변경 (2023->2024 폴드) ===", flush=True)
r_distance = train_eval(2023, 2024, season_mode="distance", tag="season=distance")
r_drop = train_eval(2023, 2024, season_mode="drop", tag="season=drop")

print("\n=== 3) 최근 시즌 가중 학습 (2023->2024 폴드) ===", flush=True)


def recency_weight(seasons, half_life=2.0):
    age = seasons.max() - seasons
    return 0.5 ** (age / half_life)


r_weighted = train_eval(2023, 2024, season_mode="raw",
                        sample_weight=lambda s: recency_weight(s, half_life=2.0), tag="recency_weighted")

print("\n=== 요약 ===", flush=True)
for r in fold_results:
    print(f"  {r['tag']:20s} valid={r['valid_season']}  base={r['base']:.1f}  gap={r['gap']:+.4f}  "
          f"offsetA={r['s_A']:.1f}(delta{r['s_A']-r['base']:+.1f})  logitB={r['s_B']:.1f}(delta{r['s_B']-r['base']:+.1f})")
print(f"  {r_distance['tag']:20s} base={r_distance['base']:.1f}  gap={r_distance['gap']:+.4f}")
print(f"  {r_drop['tag']:20s} base={r_drop['base']:.1f}  gap={r_drop['gap']:+.4f}")
print(f"  {r_weighted['tag']:20s} base={r_weighted['base']:.1f}  gap={r_weighted['gap']:+.4f}  "
      f"offsetA={r_weighted['s_A']:.1f}  logitB={r_weighted['s_B']:.1f}")

print(f"\n총 {time.time()-t0:.0f}s", flush=True)
