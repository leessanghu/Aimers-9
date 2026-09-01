"""아이디어3 후속 — HGB 1개짜리 싼 target(BSS) Stage2로 Stage1 지문(C+/B-/A-) 재현 확인.

Stage1(구종예측 logloss, K=400 최고): C +0.02367 / B -0.00505 / A -0.01555
질문: 이 시계열 지문이 실제 target(control_success)에서도 같은 부호로 나오는가?
    같은 부호면    -> 2025 구조 진단 probe로 실측 제출할 가치 있음 (약한 alpha로)
    부호 불일치면  -> 구종예측 개선이 애초에 이 타겟의 좋은 프록시가 아니었다는 뜻, 접음

HGB 1개(d6)만 써서 최대한 싸게: 신규 3개 학습만 하면 됨(base는 phase90_cache 재사용).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pitchtype import TYPES, build_matched, build_pitchtype_tables, K_CONTROL, K_MIX, _pv

CD = "idea3_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_HM = 50.0
BEST_K_POST = 400.0   # Stage1에서 지문이 제일 뚜렷했던 K


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("데이터 로드 + Trackman 매칭...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y_full = df["control_success"].to_numpy(np.float64)
g = float(y_full.mean())
sr = sorted(df["season"].unique().tolist())
seasons = df["season"].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
n_rows = len(df)

matched = build_matched(df)
tables = build_pitchtype_tables(matched, sr)
log(f"  매칭 {len(matched):,}행")

fb_r = df["asof_pitcher_fastball_rate"].fillna(0).to_numpy(np.float64)
br_r = df["asof_pitcher_breaking_rate"].fillna(0).to_numpy(np.float64)
of_r = df["asof_pitcher_offspeed_rate"].fillna(0).to_numpy(np.float64)
ot_r = np.clip(1.0 - fb_r - br_r - of_r, 0.0, 1.0)
m_now = {"fastball": fb_r, "breaking": br_r, "offspeed": of_r, "other": ot_r}
current_n = df["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)


def build_ptdyn_feature(train_upto, prior):
    prev = np.full(n_rows, train_upto)
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

    m_post = {t: (current_n * m_now[t] + BEST_K_POST * hist_mix[t]) / (current_n + BEST_K_POST) for t in TYPES}
    ratio = {t: m_post[t] / np.maximum(hist_mix[t], 1e-4) for t in TYPES}
    q_dyn_raw = {t: q_hist[t] * ratio[t] for t in TYPES}
    q_dyn_tot = sum(q_dyn_raw.values())
    q_dyn = {t: np.divide(q_dyn_raw[t], q_dyn_tot, out=q_hist[t].copy(), where=q_dyn_tot > 0) for t in TYPES}

    gt_s_all = {t: np.nan_to_num(_pv(tables["gtype"], "ptype", "s", sr)
                                  .reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    ct_s_all = {t: np.nan_to_num(_pv(tables["ctrl"], ["pitcher_id", "ptype"], "s", sr)
                                  .reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev]))
                                  .to_numpy().astype(np.float64), nan=0.0) for t in TYPES}
    pred_dyn = np.zeros(n_rows)
    for t in TYPES:
        type_rate = np.divide(gt_s_all[t], gt_n_all[t], out=np.full(n_rows, g), where=gt_n_all[t] > 0)
        anchor = np.clip(prior + (type_rate - g), 1e-6, 1 - 1e-6)
        ctrl_t = (ct_s_all[t] + K_CONTROL * anchor) / (ct_n_all[t] + K_CONTROL)
        pred_dyn += q_dyn[t] * ctrl_t
    return pred_dyn


log("피처 캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)

HGB1 = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)

results = {}
for upto, val, tag in [(2021, 2022, "C"), (2022, 2023, "B"), (2023, 2024, "A")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    prior = X["x_ability_here"].to_numpy(np.float64)
    pred_dyn = build_ptdyn_feature(upto, prior)
    Xa = X.copy()
    Xa["pt_pred_dynamic"] = pred_dyn
    Xa["pt_dev_dynamic"] = pred_dyn - prior

    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y_full[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    base = np.load(f"phase90_cache/{tag}_base_d6.npy")
    f = f"{CD}/{tag}_targetprobe_d6.npy"
    if os.path.exists(f):
        p_new = np.load(f)
    else:
        ts = time.time()
        m = HistGradientBoostingClassifier(**HGB1).fit(Xa.loc[tr_m], y_full[tr_m], sample_weight=w[tr_m])
        p_new = m.predict_proba(Xa.loc[va_m])[:, 1]
        np.save(f, p_new)
        log(f"  학습 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
    s_base, s_new = score(base), score(p_new)
    results[tag] = dict(base=s_base, new=s_new, delta=s_new - s_base)
    log(f"  base={s_base:.2f}  new={s_new:.2f}  delta={s_new-s_base:+.2f}")

print()
print("=" * 60)
print("Stage1 지문(구종logloss, K=400): C +0.02367  B -0.00505  A -0.01555")
print("-" * 60)
print(f"{'fold':<6}{'base':>10}{'new':>10}{'delta':>10}")
for tag in ["C", "B", "A"]:
    r = results[tag]
    print(f"{tag:<6}{r['base']:10.2f}{r['new']:10.2f}{r['delta']:+10.2f}")

signs_stage1 = {"C": 1, "B": -1, "A": -1}
match = all(np.sign(results[t]["delta"]) == signs_stage1[t] or abs(results[t]["delta"]) < 3 for t in ["C", "B", "A"])
exact_match = all(np.sign(results[t]["delta"]) == signs_stage1[t] for t in ["C", "B", "A"])
print()
print(f"부호 완전일치(C+/B-/A-): {exact_match}")
print(f"C:{np.sign(results['C']['delta']):+.0f} B:{np.sign(results['B']['delta']):+.0f} A:{np.sign(results['A']['delta']):+.0f}")
if exact_match:
    print("=> 지문 재현됨. Codex 제안대로 alpha-probe 실측 제출 가치 있음. 프로덕션 p1 학습 진행.")
else:
    print("=> 지문 불일치. 구종예측 개선이 이 타겟의 좋은 프록시가 아니었음. 여기서 접는다.")
pd.DataFrame(results).T.to_csv("idea3_target_probe_results.csv")
log(f"총 {time.time()-t0:.0f}s")
