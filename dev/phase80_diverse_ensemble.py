"""phase80 — 하이퍼파라미터 다양화 앙상블 (오늘 측정에서 도출된 유일한 큰 방향).

관측:
    HGB 단독      832.30
    CatBoost 단독 844.69
    0.5/0.5       880.70   <- 더 나은 단일 모델보다 +36.0, 개별평균(838.50) 대비 +42.2
    예측 상관 0.909

등상관 앙상블에서 노이즈 분산은 v(1+(m-1)rho)/m 이므로
    gain(1->2) = v(1-rho)/2
    gain(2->inf) = v(1+rho)/2 - v*rho = v(1-rho)/2      <- 정확히 같다
즉 1->2에서 얻은 +42.2 만큼이 2->다수에도 남아 있다. 폴드 천장 ~923.

과거 '모델은 병목 아님' 결론은 잘못된 질문이었다:
    phase68 capacity : depth를 '교체'      -> 앙상블 이득과 무관
    phase67 MLP      : 552.9짜리를 '추가'  -> 등강도 가정 위반 (GBDT보다 292점 약함)
    phase69 RF       : 약한 모델 '추가'     -> 동일
'비슷하게 강하면서 상관이 낮은 GBDT 변종 추가'는 한 번도 테스트한 적이 없다.

특히 피처 서브샘플링(CatBoost rsm, HGB max_features)은 멤버간 상관을 낮추는 고전
노브인데 미사용이다. phase76에서 피처를 '영구히' 빼는 건 실패했지만(z=0.2~0.9),
멤버마다 '다른 부분집합'을 주는 것은 전혀 다른 연산이다.

의존성 주의: lightgbm/xgboost는 submit/requirements.txt에 없다. 안전세트(sklearn+catboost)와
분리해서 측정하고, 이득이 리스크를 정당화할 때만 requirements 변경을 검토한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from phase2_common import time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
VALID_SEASON = 2024
CACHE_DIR = "phase80_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("데이터 로드 + 피처 재구성 (v28 162개)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

fb = FeatureBuilder(seed=42, include_raw_rates=False, team_te_mode="expanding").fit(df)
X_base = fb.transform_train_oof(df).reset_index(drop=True)
se = build_season_end_table(df)
X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
X_plt = transform_platoon(df, build_platoon_table(df), prior, sr, k=K_PLATOON).reset_index(drop=True)
it, io = build_inning_table(df), build_inning_offset(df)
X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
X_cnt = transform_count(df, build_count_table(df), prior, sr, k=K_COUNT).reset_index(drop=True)
X_pt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), prior, g, sr).reset_index(drop=True)
X_ly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0).reset_index(drop=True)
X_vol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL).reset_index(drop=True)
role_tbl = build_role_table(df)
X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                        base_middle).reset_index(drop=True)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)
lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)
X_bat = transform_batter(df, build_batter_table(df), sr, g, k=K_BATTER).reset_index(drop=True)
n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
X_isf = transform_inseason_full(df, build_season_end_table_full(df), build_global_priors(df), sr,
                                n_end_row, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
X_bmid = bsplit.transform_batter_middle(df, bsplit.build_batter_middle_table(df), sr, g_bmid).reset_index(drop=True)
bmarg = bsplit.build_batter_marginal(df)
b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
X_bplat = bsplit.transform_bplatoon(df, bsplit.build_bplatoon_table(df), b_prior, sr,
                                    k=bsplit.K_BPLATOON).reset_index(drop=True)

X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
C = add_crosses(X)
X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
               X_isf, X_bmid, X_bplat], axis=1).astype(np.float64)
log(f"피처 {X.shape[1]}개")

seasons = df["season"].to_numpy(np.float64)
tr_m = seasons <= VALID_SEASON - 1
va_m = seasons == VALID_SEASON
yv = y[va_m].astype(np.float64)
w_tr = recency_weight(seasons[tr_m], half_life=2.0)
r = yv.mean()
BSREF = r * (1 - r)
Xtr = X.loc[tr_m].reset_index(drop=True)
ytr = y[tr_m]
Xva = X.loc[va_m]
tr_i, es_i = time_split_es(int(tr_m.sum()))
log(f"train={tr_m.sum():,}  valid={va_m.sum():,}")


def score(p):
    return 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)


def run_hgb(name, **kw):
    f = f"{CACHE_DIR}/{name}.npy"
    if os.path.exists(f):
        p = np.load(f)
        log(f"  {name:<22} 캐시  score={score(p):.2f}")
        return p
    ts = time.time()
    params = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=20)
    params.update(kw)
    m = HistGradientBoostingClassifier(**params).fit(Xtr, ytr, sample_weight=w_tr)
    p = m.predict_proba(Xva)[:, 1]
    np.save(f, p)
    log(f"  {name:<22} score={score(p):7.2f}  iters={m.n_iter_}  ({time.time()-ts:.0f}s)")
    return p


def run_cat(name, **kw):
    f = f"{CACHE_DIR}/{name}.npy"
    if os.path.exists(f):
        p = np.load(f)
        log(f"  {name:<22} 캐시  score={score(p):.2f}")
        return p
    ts = time.time()
    params = dict(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    params.update(kw)
    m = CatBoostClassifier(**params)
    m.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
    p = m.predict_proba(Xva)[:, 1]
    np.save(f, p)
    log(f"  {name:<22} score={score(p):7.2f}  iters={m.best_iteration_}  ({time.time()-ts:.0f}s)")
    return p


preds = {}
log("[안전세트] HGB 변종...")
preds["hgb_d6"] = run_hgb("hgb_d6", max_depth=6, max_leaf_nodes=31, random_state=42)
preds["hgb_d4"] = run_hgb("hgb_d4", max_depth=4, max_leaf_nodes=63, learning_rate=0.05, random_state=7)
preds["hgb_d8"] = run_hgb("hgb_d8", max_depth=8, max_leaf_nodes=15, random_state=2024)
preds["hgb_sub"] = run_hgb("hgb_sub", max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)

log("[안전세트] CatBoost 변종...")
preds["cat_d6"] = run_cat("cat_d6", depth=6, random_seed=42)
preds["cat_d8"] = run_cat("cat_d8", depth=8, l2_leaf_reg=10.0, random_seed=7)
preds["cat_rsm"] = run_cat("cat_rsm", depth=6, rsm=0.6, random_seed=2024)
preds["cat_d5"] = run_cat("cat_d5", depth=5, learning_rate=0.05, random_seed=123)

SAFE = list(preds.keys())

log("[의존성리스크] LightGBM / XGBoost...")
try:
    import lightgbm as lgb
    f = f"{CACHE_DIR}/lgb.npy"
    if os.path.exists(f):
        preds["lgb"] = np.load(f)
    else:
        ts = time.time()
        m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=31, max_depth=6,
                               reg_lambda=5.0, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                               random_state=42, verbose=-1)
        m.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i],
              eval_set=[(Xtr.iloc[es_i], ytr[es_i])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds["lgb"] = m.predict_proba(Xva)[:, 1]
        np.save(f, preds["lgb"])
        log(f"  lgb score={score(preds['lgb']):.2f} ({time.time()-ts:.0f}s)")
    log(f"  lgb                    score={score(preds['lgb']):7.2f}")
except Exception as e:
    log(f"  lightgbm 실패: {e}")

try:
    import xgboost as xgb
    f = f"{CACHE_DIR}/xgb.npy"
    if os.path.exists(f):
        preds["xgb"] = np.load(f)
    else:
        ts = time.time()
        m = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=6, reg_lambda=5.0,
                              colsample_bytree=0.7, subsample=0.8, random_state=42,
                              early_stopping_rounds=50, eval_metric="logloss", tree_method="hist")
        m.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i],
              eval_set=[(Xtr.iloc[es_i], ytr[es_i])], verbose=False)
        preds["xgb"] = m.predict_proba(Xva)[:, 1]
        np.save(f, preds["xgb"])
        log(f"  xgb score={score(preds['xgb']):.2f} ({time.time()-ts:.0f}s)")
    log(f"  xgb                    score={score(preds['xgb']):7.2f}")
except Exception as e:
    log(f"  xgboost 실패: {e}")

names = list(preds.keys())
P = np.column_stack([preds[n] for n in names])

print()
print("=" * 62)
print("개별 점수")
print("-" * 62)
for n in names:
    print(f"  {n:<22}{score(preds[n]):10.2f}")

print()
print("예측 상관행렬")
print(pd.DataFrame(np.corrcoef(P.T), index=names, columns=names).round(3).to_string())


def greedy(cols, iters=40):
    """Caruana 방식 forward selection with replacement (과적합에 강함)."""
    chosen = []
    cur = np.zeros(len(yv))
    best_hist = []
    for _ in range(iters):
        bs, bi = -9e9, None
        for j in cols:
            cand = (cur * len(chosen) + preds[j]) / (len(chosen) + 1)
            s = score(cand)
            if s > bs:
                bs, bi = s, j
        chosen.append(bi)
        cur = (cur * (len(chosen) - 1) + preds[bi]) / len(chosen)
        best_hist.append(bs)
    return chosen, cur, best_hist


print()
print("=" * 62)
base2 = score(0.5 * preds["hgb_d6"] + 0.5 * preds["cat_d6"])
print(f"현행 2모델 블렌드 (hgb_d6 + cat_d6, 0.5/0.5) : {base2:10.2f}")
print(f"안전세트 {len(SAFE)}개 단순평균                      : {score(P[:, [names.index(n) for n in SAFE]].mean(axis=1)):10.2f}")
ch, cur, hist = greedy(SAFE)
print(f"안전세트 greedy 앙상블                        : {score(cur):10.2f}  ({score(cur)-base2:+.2f})")
print(f"   구성: {pd.Series(ch).value_counts().to_dict()}")
if len(names) > len(SAFE):
    print(f"전체({len(names)}개) 단순평균                       : {score(P.mean(axis=1)):10.2f}")
    ch2, cur2, _ = greedy(names)
    print(f"전체 greedy 앙상블                            : {score(cur2):10.2f}  ({score(cur2)-base2:+.2f})")
    print(f"   구성: {pd.Series(ch2).value_counts().to_dict()}")

pd.DataFrame({n: preds[n] for n in names}).to_parquet(f"{CACHE_DIR}/all_preds.parquet")
log(f"총 {time.time()-t0:.0f}s")
