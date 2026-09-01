"""아이디어 3 — Dynamic In-season Pitch Router (Codex 제안).

배경: phase14에서 '올해 구종비율'을 성공모델에 직접 피처로 넣었다가 -6.5로 실패했다.
원인: P(success|x) = Σ_t P(t|x)·P(success|t,x) 는 3중 상호작용 marginalization인데
트리가 split 몇개로 근사하기엔 너무 무겁다. 반면 pitchtype.py는 이 marginalization을
트리 밖에서 명시적으로 계산해 pt_pred/pt_dev로 넣고 있고, 이건 실제로 살아있는 피처다
(phase94 SHAP 상위 40위 안).

이번 아이디어: 그 파이프라인의 입력(q_hist, 구종의 과거 카운트조건부 확률)을
공식 asof_pitchmix 컬럼으로 복원한 '올해 실제 구종비율'로 최신화한다.

    q_hist(t|pitcher,count)   = 기존 pitchtype.py의 카운트조건부 구종확률 (과거시즌 누적)
    historical_mix(t)         = 그 투수의 과거시즌 누적 전체 구종비율 (카운트 무관)
    m_now(t)                  = asof_pitcher_{fastball,breaking,offspeed}_rate로 복원한 금년 구종비율
    m_post(t)                 = (current_n*m_now(t) + K*historical_mix(t)) / (current_n + K)
    q_dynamic(t|count)       ∝ q_hist(t|count) * m_post(t)/historical_mix(t),  재정규화

라킹(raking)/IPF 스타일 보정 — 조건부분포(q_hist)를 새 주변분포(m_post) 정보로 재조정.
current_n이 작으면 m_post≈historical_mix라 비율이 1로 수렴해 안전하게 q_hist로 폴백된다.

2단계 검증 (Codex 설계):
    Stage1: 2022/2023/2024 Trackman 매칭 행에서 q_hist 대비 q_dynamic의 실제 구종
            multiclass logloss가 3폴드 모두 개선되는지. target(control_success)과 무관해서
            노이즈가 훨씬 적다 -> 여기서 못 이기면 바로 기각.
    Stage2: 통과 시에만 pt_pred_dynamic을 만들어 v35 direct/hurdle 각각 3폴드 재학습.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import batter_split as bsplit
from batterform import K_BATTER, build_batter_table, transform_batter
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from count_split import K_COUNT, build_count_table, transform_count
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from inseason_full import build_global_priors, build_season_end_table_full, transform_inseason_full
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from pitchtype import (TYPES, build_matched, build_pitchtype_tables, transform_pitchtype,
                       K_CONTROL, K_MIX, _pv)
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
CD = "idea3_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_HM = 50.0          # historical_mix 자체의 수축(전역 구종비율로)
K_POST_LIST = [50.0, 150.0, 400.0]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())
seasons = df["season"].to_numpy(np.float64)

log("Trackman 매칭...")
matched = build_matched(df)
tables = build_pitchtype_tables(matched, sr)
log(f"  매칭 {len(matched):,}행")

pid = df["pitcher_id"].to_numpy()
cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
n_rows = len(df)

# 공식 asof_pitchmix -> m_now (4클래스: fastball/breaking/offspeed/other)
fb_r = df["asof_pitcher_fastball_rate"].fillna(0).to_numpy(np.float64)
br_r = df["asof_pitcher_breaking_rate"].fillna(0).to_numpy(np.float64)
of_r = df["asof_pitcher_offspeed_rate"].fillna(0).to_numpy(np.float64)
ot_r = np.clip(1.0 - fb_r - br_r - of_r, 0.0, 1.0)
m_now = {"fastball": fb_r, "breaking": br_r, "offspeed": of_r, "other": ot_r}
current_n = df["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)


def build_q(train_upto, valid_season, K_POST):
    """q_hist, q_dynamic(들), historical_mix -> 매칭행(valid_season)에서 평가용 배열 반환."""
    prev = np.full(n_rows, train_upto)  # season-1 대신 폴드의 train_upto를 조회기준으로 고정
    gt_n_all = {t: np.nan_to_num(_pv(tables["gtype"], "ptype", "n", sr)
                                  .reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    gt_tot = sum(gt_n_all.values())
    global_share = {t: np.divide(gt_n_all[t], gt_tot, out=np.full(n_rows, 1.0 / len(TYPES)), where=gt_tot > 0)
                    for t in TYPES}

    ct_n_all = {t: np.nan_to_num(_pv(tables["ctrl"], ["pitcher_id", "ptype"], "n", sr)
                                  .reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    hist_tot = sum(ct_n_all.values())
    hist_mix = {t: (ct_n_all[t] + K_HM * global_share[t]) / (hist_tot + K_HM) for t in TYPES}

    mx_n_all = {t: np.nan_to_num(_pv(tables["mix"], ["pitcher_id", "count_state", "ptype"], "n", sr)
                                  .reindex(pd.MultiIndex.from_arrays([pid, cs, [t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    gm_n_all = {t: np.nan_to_num(_pv(tables["gmix"], ["count_state", "ptype"], "n", sr)
                                  .reindex(pd.MultiIndex.from_arrays([cs, [t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    q_num = {t: mx_n_all[t] + K_MIX * gm_n_all[t] for t in TYPES}
    q_den = sum(q_num.values())
    q_hist = {t: np.divide(q_num[t], q_den, out=global_share[t].copy(), where=q_den > 0) for t in TYPES}

    m_post = {t: (current_n * m_now[t] + K_POST * hist_mix[t]) / (current_n + K_POST) for t in TYPES}
    ratio = {t: m_post[t] / np.maximum(hist_mix[t], 1e-4) for t in TYPES}
    q_dyn_raw = {t: q_hist[t] * ratio[t] for t in TYPES}
    q_dyn_tot = sum(q_dyn_raw.values())
    q_dyn = {t: np.divide(q_dyn_raw[t], q_dyn_tot, out=q_hist[t].copy(), where=q_dyn_tot > 0) for t in TYPES}
    return q_hist, q_dyn, hist_mix, ct_n_all


def mc_logloss(qdict, true_type, mask, eps=1e-6):
    p_true = np.zeros(mask.sum())
    tt = true_type[mask]
    for t in TYPES:
        p_true += np.where(tt == t, qdict[t][mask], 0.0)
    return -np.mean(np.log(np.clip(p_true, eps, 1.0)))


log("[Stage1] q_hist vs q_dynamic multiclass logloss (3폴드, Trackman 매칭행)...")
# build_matched는 원본 df의 인덱스 부분집합을 유지한 채 반환 (join 체인이 인덱스 보존)
true_type_full = pd.Series(index=df.index, dtype=object)
true_type_full.loc[matched.index] = matched["ptype"].to_numpy()
true_type_arr = true_type_full.to_numpy(dtype=object)
has_match = true_type_full.notna().to_numpy()
log(f"  매칭 표시 완료. 매칭행={has_match.sum():,}")

stage1_rows = []
for upto, val, tag in [(2021, 2022, "C"), (2022, 2023, "B"), (2023, 2024, "A")]:
    mask = has_match & (seasons == val)
    qh, _, _, _ = build_q(upto, val, K_POST_LIST[0])  # q_hist는 K_POST 무관, 1회만 계산해도 되지만 재사용 위해 루프안에서 재계산 생략
    ll_hist = mc_logloss(qh, true_type_arr, mask)
    row = dict(fold=tag, valid=val, n=int(mask.sum()), q_hist=ll_hist)
    for K_POST in K_POST_LIST:
        _, qd, _, _ = build_q(upto, val, K_POST)
        row[f"q_dyn_K{int(K_POST)}"] = mc_logloss(qd, true_type_arr, mask)
    stage1_rows.append(row)
    log(f"  fold {tag}(valid={val}, n={mask.sum():,}): " +
        "  ".join(f"{k}={v:.5f}" for k, v in row.items() if k not in ("fold", "valid", "n")))

s1 = pd.DataFrame(stage1_rows)
print()
print("=" * 78)
print("Stage1 결과 (낮을수록 좋음, q_hist 대비 개선 여부)")
print("=" * 78)
print(s1.to_string(index=False))
best_K = None
best_margin = -9e9
for K_POST in K_POST_LIST:
    col = f"q_dyn_K{int(K_POST)}"
    improve = s1["q_hist"] - s1[col]   # 양수면 개선
    print(f"  K={K_POST:5.0f}: 폴드별 개선 {list(improve.round(5))}  최소개선={improve.min():+.5f}")
    if improve.min() > best_margin:
        best_margin = improve.min()
        best_K = K_POST

print(f"\n최고 K={best_K}, 3폴드 최소개선={best_margin:+.5f}")
s1.to_csv("idea3_stage1_results.csv", index=False)

if best_margin <= 0:
    log("Stage1 기각: 3폴드 모두 개선하는 K 없음. Stage2 진행 안 함.")
    log(f"총 {time.time()-t0:.0f}s")
    sys.exit(0)

log(f"Stage1 통과 (K={best_K}). Stage2로 진행 — pt_pred_dynamic 피처 생성 + 3폴드 재학습...")

# ---------------- Stage2: 전체 피처 재구성 + pt_pred_dynamic 추가 ----------------
log("피처 재구성 (v28 162개, 캐시 재사용)...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)


def build_ptdyn_feature(train_upto, valid_season, K_POST):
    """전체 행(매칭 여부 무관)에 대해 pt_pred_dynamic 계산. ctrl_t는 기존 로직 재사용."""
    prev = np.full(n_rows, train_upto)
    prior = X["x_ability_here"].to_numpy(np.float64)  # 근사 prior (기존 transform_pitchtype는 in-season prior 사용)
    qh, qd, hist_mix, ct_n_all = build_q(train_upto, valid_season, K_POST)
    gt_s_all = {t: np.nan_to_num(_pv(tables["gtype"], "ptype", "s", sr)
                                  .reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    gt_n_all = {t: np.nan_to_num(_pv(tables["gtype"], "ptype", "n", sr)
                                  .reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    ct_s_all = {t: np.nan_to_num(_pv(tables["ctrl"], ["pitcher_id", "ptype"], "s", sr)
                                  .reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    pred_dyn = np.zeros(n_rows)
    pred_hist = np.zeros(n_rows)
    for t in TYPES:
        type_rate = np.divide(gt_s_all[t], gt_n_all[t], out=np.full(n_rows, g), where=gt_n_all[t] > 0)
        anchor = np.clip(prior + (type_rate - g), 1e-6, 1 - 1e-6)
        ctrl_t = (ct_s_all[t] + K_CONTROL * anchor) / (ct_n_all[t] + K_CONTROL)
        pred_dyn += qd[t] * ctrl_t
        pred_hist += qh[t] * ctrl_t
    return pred_dyn, pred_hist


HGB_VARIANTS = [
    ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)
SETS = {"E_ptdyn": None}   # payload는 fold별로 다시 계산(train_upto 의존)


def run_fold_stage2(train_upto, valid_season, tag):
    log(f"===== Stage2 fold {tag}: train<={train_upto} -> valid={valid_season} =====")
    pred_dyn, pred_hist = build_ptdyn_feature(train_upto, valid_season, best_K)
    Xa = X.copy()
    Xa["pt_pred_dynamic"] = pred_dyn
    Xa["pt_dev_dynamic"] = pred_dyn - X["x_ability_here"].to_numpy(np.float64)

    tr_m = (seasons <= train_upto) & step
    va_m = seasons == valid_season
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((train_upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    base = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n, _ in HGB_VARIANTS], axis=0)
    preds = []
    for vn, extra in HGB_VARIANTS:
        f = f"{CD}/{tag}_E_ptdyn_{vn}.npy"
        if os.path.exists(f):
            preds.append(np.load(f))
            continue
        p = dict(BASE_HGB); p.update(extra)
        ts = time.time()
        m = HistGradientBoostingClassifier(**p).fit(Xa.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
        pr = m.predict_proba(Xa.loc[va_m])[:, 1]
        np.save(f, pr)
        preds.append(pr)
        log(f"    E_ptdyn/{vn} 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
    new_score = score(np.mean(preds, axis=0))
    log(f"  base={score(base):.2f}  E_ptdyn={new_score:.2f}  (대비 {new_score-score(base):+.2f})")
    return score(base), new_score


res2 = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    b, n_ = run_fold_stage2(upto, val, tag)
    res2[tag] = dict(base=b, ptdyn=n_, gain=n_ - b)

print()
print("=" * 60)
print(f"{'fold':<6}{'base':>10}{'E_ptdyn':>10}{'이득':>10}")
for tag, r in res2.items():
    print(f"{tag:<6}{r['base']:10.2f}{r['ptdyn']:10.2f}{r['gain']:+10.2f}")
min_gain = min(r["gain"] for r in res2.values())
print(f"\n3폴드 최소이득 = {min_gain:+.2f}")
print("=> 채택 검토" if min_gain > 2 else "=> 기각")
pd.DataFrame(res2).to_csv("idea3_stage2_results.csv")
log(f"총 {time.time()-t0:.0f}s")
