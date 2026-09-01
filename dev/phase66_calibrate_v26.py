"""v26 아티팩트에 holdout 기반 affine 확률보정을 붙인다 (재학습 없음).

phase63 항등식: Score/1e5 = (2Cov(p,y) - Var(p) - bias^2)/BSref, 최적 affine 재보정은
b=Cov(p,y)/Var(p), a=mean(y)-b*mean(p). v25는 이미 b_opt=1.015로 거의 최적이었지만
v26은 feature가 41개 늘었으니 다시 확인한다.

holdout: CatBoost 학습에 쓰인 tr_i/es_i(time_split_es, 시간순 마지막 8%)의 es_i.
  CatBoost 입장에서는 완전히 out-of-sample이다. HGB는 fit(X,y) 전체에 early_stopping=True로
  내부적으로 자체 랜덤 검증셋을 따로 떼어 쓰므로 es_i 일부가 HGB 학습에 섞였을 수 있어
  완벽히 깨끗하진 않지만(50:50 블렌드 중 절반), 실용적으로 충분히 신뢰할 수 있는 근사다.

계수는 반드시 이 holdout에서만 추정한다 (in-sample로 하면 optimistic bias로 과대적합).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd

from arsenal_entropy import K_ARSENAL  # noqa (모듈 로드 순서 안정화용, 사용 안 함)
from count_split import build_count_table, transform_count, K_COUNT
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role
from career_volatility import build_volatility_table, transform_volatility, K_VOL
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from phase2_common import time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import build_platoon_table, transform_platoon, K_PLATOON
from trackman_profile import build_trackman_profile, transform_trackman

DATA_PATH = "../data/train.csv"
ARTIFACT_PATH = "../submit/model/model_artifacts_v26.pkl"
TM_CACHE = "phase64_trackman_profile.parquet"

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("아티팩트 로드...")
art = joblib.load(ARTIFACT_PATH)
hgb, cb = art["hgb"], art["cat"]
feature_order = art["feature_order"]
w_hgb, w_cat = art["w_hgb"], art["w_cat"]

log("피처 재구성 (train_final_v26.py와 동일 파이프라인)...")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
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

pt = build_platoon_table(df)
X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
it, io = build_inning_table(df), build_inning_offset(df)
X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
ctb = build_count_table(df)
X_cnt = transform_count(df, ctb, prior, sr, k=K_COUNT).reset_index(drop=True)

matched = build_matched(df)
pt_tables = build_pitchtype_tables(matched, sr)
X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)

gr = build_global_rates(df)
ly_tbl = build_lastyear_table(df)
X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)

vol_tbl = build_volatility_table(se)
X_vol = transform_volatility(df, vol_tbl, sr, k=K_VOL).reset_index(drop=True)

role_tbl = build_role_table(df)
X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)

base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                        base_middle).reset_index(drop=True)

prof = pd.read_parquet(TM_CACHE)
X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)

X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
C = add_crosses(X)
X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm], axis=1)
X = X[feature_order].astype(np.float64)
log(f"피처 {X.shape[1]}개 재구성 완료")

tr_i, es_i = time_split_es(len(X))
X_es, y_es = X.iloc[es_i], y[es_i]
log(f"holdout(es_i) n={len(es_i):,} (CatBoost 순수 OOS, HGB는 근사적으로만 OOS)")

p_es = w_hgb * hgb.predict_proba(X_es)[:, 1] + w_cat * cb.predict_proba(X_es)[:, 1]


def bss(y, p):
    r = y.mean()
    return max(0.0, 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


score_raw = bss(y_es, p_es)

# affine 최적 재보정: b = Cov(p,y)/Var(p), a = mean(y) - b*mean(p)
cov = np.mean((p_es - p_es.mean()) * (y_es - y_es.mean()))
var_p = p_es.var()
b_opt = cov / var_p if var_p > 0 else 1.0
a_opt = y_es.mean() - b_opt * p_es.mean()
p_cal = np.clip(a_opt + b_opt * p_es, 1e-6, 1 - 1e-6)
score_cal = bss(y_es, p_cal)

log(f"holdout score(원본)   = {score_raw:.1f}   bias={p_es.mean()-y_es.mean():+.4f}  b_opt={b_opt:.4f}")
log(f"holdout score(보정후) = {score_cal:.1f}   (delta {score_cal-score_raw:+.1f})")
log(f"보정계수: a={a_opt:+.5f}  b={b_opt:.5f}")

art["calib_a"] = float(a_opt)
art["calib_b"] = float(b_opt)
joblib.dump(art, ARTIFACT_PATH)
log(f"저장 완료: {ARTIFACT_PATH} (calib_a/calib_b 추가됨)")
