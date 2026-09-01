"""Phase95 block2/3/4 — multires / midaxis / ordinal 각각의 refit-closure 스크리닝.
cats-refit(idea35)은 fold A+6.75/fold C+24.50으로 강하게 양수였으나 실측(v53)은 -1.75로
반대 방향이었다. "ES방식 변경류=편향 거의 없음" 규칙이 이번 실험유형(공유트리/캐스케이드
refit)엔 안 맞을 수 있다는 뜻이므로, 이 스크리닝 결과도 예전만큼 신뢰하지 말고
반드시 개별 실측으로 재확인해야 한다. 세 블록을 한꺼번에 묶지 않고 별도 패키징한다.
fold A/C, featcache 재사용.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

CD90 = "phase90_cache"
CD13 = "idea13_cache"
CD31 = "idea31_cache"
CD = "idea37_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_PS = 15.0


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64) if "same_hand" in X.columns else np.zeros(len(X))

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(meta))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab


lab_reverse = diff_label("asof_pitcher_reverse_rate")
lab_middle = diff_label("asof_pitcher_middle_rate")
valid_rev = ~np.isnan(lab_reverse)
valid_mid = ~np.isnan(lab_middle)

CAT_PARAMS_MR = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                     loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, learning_rate=0.03, l2_regularization=5.0)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base3 = np.mean([np.load(f"{CD90}/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"{CD90}/{tag}_core_{n}.npy")) * np.load(f"{CD90}/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    mr_ES = np.mean([np.load(f"{CD13}/{tag}_multires_s{k}.npy") for k in [42, 7]], axis=0)
    od_ES = np.mean([np.load(f"{CD13}/{tag}_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
    md_ES = np.mean([np.load(f"{CD31}/{tag}_midaxis_s{k}.npy") for k in [42, 7]], axis=0)
    v47l = 0.30 * base3 + 0.40 * hur + 0.10 * mr_ES + 0.20 * od_ES
    v50l = 0.30 * base3 + 0.40 * hur + 0.10 * mr_ES + 0.20 * od_ES + 0.10 * md_ES  # 참고(가중치미보정)
    log(f"  v47local={sc(v47l):.2f}")

    Xtr = X.loc[tr_m].reset_index(drop=True)
    Xva = X.loc[va_m]
    ytr = y[tr_m]
    wtr = w[tr_m]
    n_tr = len(Xtr)
    n_es = int(n_tr * 0.92)

    # ---- multires refit ----
    f_rf = f"{CD}/{tag}_multires_refit.npy"
    if os.path.exists(f_rf):
        mr_rf = np.load(f_rf)
    else:
        sub = pd.DataFrame({"pid": pid[tr_m], "season": seasons[tr_m], "sh": same_hand[tr_m], "y": ytr})
        g_glob = ytr.mean()
        ps = sub.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
        sub = sub.join(ps, on=["pid", "season"])
        h1 = ((sub["s"] - sub["y"]) + K_PS * g_glob) / ((sub["n"] - 1) + K_PS)
        psh = sub.groupby(["pid", "season", "sh"])["y"].agg(s2="sum", n2="count")
        sub = sub.join(psh, on=["pid", "season", "sh"])
        h2 = ((sub["s2"] - sub["y"]) + K_PS * h1) / ((sub["n2"] - 1) + K_PS)
        Ymat = np.column_stack([ytr, h1.to_numpy(np.float64), h2.to_numpy(np.float64)])
        ts = time.time()
        m_es = CatBoostRegressor(**CAT_PARAMS_MR, random_seed=42)
        m_es.fit(Xtr.iloc[:n_es], Ymat[:n_es], sample_weight=wtr[:n_es],
                eval_set=(Xtr.iloc[n_es:], Ymat[n_es:]))
        best_iter = max(m_es.best_iteration_, 1)
        params_fixed = dict(CAT_PARAMS_MR); params_fixed.pop("early_stopping_rounds")
        params_fixed["iterations"] = best_iter
        m_rf = CatBoostRegressor(**params_fixed, random_seed=42)
        m_rf.fit(Xtr, Ymat, sample_weight=wtr)
        mr_rf = np.clip(m_rf.predict(Xva), 0.0, 1.0)[:, 0]
        np.save(f_rf, mr_rf)
        log(f"  [multires/{tag}] best_iter={best_iter} refit완료 ({time.time()-ts:.0f}s)")
    v_mr = 0.30 * base3 + 0.40 * hur + 0.10 * mr_rf + 0.20 * od_ES
    log(f"  multires refit: v47(mr=ES)={sc(v47l):.2f} v47(mr=refit)={sc(v_mr):.2f} delta={sc(v_mr)-sc(v47l):+.2f}")

    # ---- midaxis refit ----
    f_rf2 = f"{CD}/{tag}_midaxis_refit.npy"
    if os.path.exists(f_rf2):
        md_rf = np.load(f_rf2)
    else:
        head_mid = np.where(valid_mid[tr_m], 1.0 - lab_middle[tr_m], np.nan)
        Ymat = np.column_stack([ytr, head_mid])
        ts = time.time()
        m_es = CatBoostRegressor(**CAT_PARAMS_MR, random_seed=42)
        m_es.fit(Xtr.iloc[:n_es], Ymat[:n_es], sample_weight=wtr[:n_es],
                eval_set=(Xtr.iloc[n_es:], Ymat[n_es:]))
        best_iter = max(m_es.best_iteration_, 1)
        params_fixed = dict(CAT_PARAMS_MR); params_fixed.pop("early_stopping_rounds")
        params_fixed["iterations"] = best_iter
        m_rf = CatBoostRegressor(**params_fixed, random_seed=42)
        m_rf.fit(Xtr, Ymat, sample_weight=wtr)
        md_rf = np.clip(m_rf.predict(Xva), 0.0, 1.0)[:, 0]
        np.save(f_rf2, md_rf)
        log(f"  [midaxis/{tag}] best_iter={best_iter} refit완료 ({time.time()-ts:.0f}s)")
    v_md_ES = v50l
    v_md_rf = 0.30 * base3 + 0.40 * hur + 0.10 * mr_ES + 0.20 * od_ES + 0.10 * md_rf
    log(f"  midaxis refit: v50(md=ES)={sc(v_md_ES):.2f} v50(md=refit)={sc(v_md_rf):.2f} delta={sc(v_md_rf)-sc(v_md_ES):+.2f}")

    # ---- ordinal refit (3-stage HGB) ----
    f_rf3 = f"{CD}/{tag}_ordinal_refit.npy"
    if os.path.exists(f_rf3):
        od_rf = np.load(f_rf3)
    else:
        ts = time.time()
        valid_tr = valid_rev[tr_m] & valid_mid[tr_m]
        lr_tr = lab_reverse[tr_m]; lm_tr = lab_middle[tr_m]
        stages_iter = []
        # stage1
        m1es = HistGradientBoostingClassifier(**HGB_CLS, max_iter=500, early_stopping=True,
                                              validation_fraction=0.08, n_iter_no_change=20, random_state=42)
        m1es.fit(Xtr.loc[valid_tr], (1 - lr_tr[valid_tr]), sample_weight=wtr[valid_tr])
        it1 = m1es.n_iter_
        m1 = HistGradientBoostingClassifier(**HGB_CLS, max_iter=it1, early_stopping=False, random_state=42)
        m1.fit(Xtr.loc[valid_tr], (1 - lr_tr[valid_tr]), sample_weight=wtr[valid_tr])
        not_rev = valid_tr & (lr_tr == 0)
        m2es = HistGradientBoostingClassifier(**HGB_CLS, max_iter=500, early_stopping=True,
                                              validation_fraction=0.08, n_iter_no_change=20, random_state=42)
        m2es.fit(Xtr.loc[not_rev], (1 - lm_tr[not_rev]), sample_weight=wtr[not_rev])
        it2 = m2es.n_iter_
        m2 = HistGradientBoostingClassifier(**HGB_CLS, max_iter=it2, early_stopping=False, random_state=42)
        m2.fit(Xtr.loc[not_rev], (1 - lm_tr[not_rev]), sample_weight=wtr[not_rev])
        not_rev_mid = not_rev & (lm_tr == 0)
        m3es = HistGradientBoostingClassifier(**HGB_CLS, max_iter=500, early_stopping=True,
                                              validation_fraction=0.08, n_iter_no_change=20, random_state=42)
        m3es.fit(Xtr.loc[not_rev_mid], ytr[not_rev_mid], sample_weight=wtr[not_rev_mid])
        it3 = m3es.n_iter_
        m3 = HistGradientBoostingClassifier(**HGB_CLS, max_iter=it3, early_stopping=False, random_state=42)
        m3.fit(Xtr.loc[not_rev_mid], ytr[not_rev_mid], sample_weight=wtr[not_rev_mid])
        p1 = m1.predict_proba(Xva)[:, 1]; p2 = m2.predict_proba(Xva)[:, 1]; p3 = m3.predict_proba(Xva)[:, 1]
        od_rf = p1 * p2 * p3
        np.save(f_rf3, od_rf)
        log(f"  [ordinal/{tag}] iters={it1}/{it2}/{it3} refit완료 ({time.time()-ts:.0f}s)")
    v_od = 0.30 * base3 + 0.40 * hur + 0.10 * mr_ES + 0.20 * od_rf
    log(f"  ordinal refit: v47(od=ES)={sc(v47l):.2f} v47(od=refit)={sc(v_od):.2f} delta={sc(v_od)-sc(v47l):+.2f}")

    results[tag] = dict(mr=sc(v_mr) - sc(v47l), md=sc(v_md_rf) - sc(v_md_ES), od=sc(v_od) - sc(v47l))

print()
print("=" * 60)
for tag, r in results.items():
    print(f"fold {tag}: multires={r['mr']:+.2f}  midaxis={r['md']:+.2f}  ordinal={r['od']:+.2f}")
log(f"총 {time.time()-t0:.0f}s")
