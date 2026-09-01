"""v72 = v66 + 투수실력잔차 부가보정(codex idea58 recipe, 프로덕션 재현).

핵심 차이: 지금까지의 모든 aux head는 "확률 멤버로 가중평균"됐다. 이건 다르다.
    p_final = p_v66 + ALPHA * residual_prediction(X_78)
residual_prediction은 확률이 아니라 부호 있는 보정값이고, 학습 타깃도 확률 y가
아니라 y - 투수시즌LOO평균(비축소, n>=20 행만)이다. 입력도 78피처(game_context+
batter_matchup+environment_team+trackman)로 pitcher_ability/sample_reliability를
명시적으로 뺐다.

codex의 fold A/C 검증(idea58_orthogonal_residual.py, idea58_A/C_alpha_grid.csv):
raw residual alpha=0.10 델타 A=+9.59 C=+13.27 (둘 다 양수, 이번 세션 신규축 중
fold A/C가 방향까지 일치한 유일 사례). 단, corr(PC1)=0.358(A)/0.676(C)라 PC1을
완전히 직교화하면 fold C 신호가 사라짐 -> "새 독립축"이 아니라 "v60a 투수실력축의
함수형/시간별 오차 보정"으로 해석. 그래도 로컬 재현성 자체는 이번 세션 최고 수준이라
프로덕션화한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
ALPHA = 0.10
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState", "PCG64")


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


def block(name: str) -> str:
    # idea57_feature_usage_blocks.py와 완전히 동일한 분류 규칙(codex 원본 그대로).
    reliability_exact = {
        "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n", "inseason_n",
        "platoon_n", "inning_n", "pt_n", "count_n", "ly_n", "bat_inseason_n",
        "bat_ly_n", "bplatoon_n", "role_n_app", "vol_n_seasons",
        "pitcher_id_count", "batter_id_count", "pitcher_team_id_count", "batter_team_id_count",
    }
    if name in reliability_exact or name.startswith("flag_") or name.endswith("_missing"):
        return "sample_reliability"
    if name.startswith("tm_"):
        return "trackman"
    if (name.startswith("bat_") or name.startswith("batter_") or name.startswith("asof_batter")
            or name.startswith("bplatoon") or name in {"x_p_over_b", "x_platoon_x_samehand"}):
        return "batter_matchup"
    context_exact = {
        "cat_top_bottom", "cat_base_state", "inning", "balls_before", "strikes_before",
        "outs_before", "run_top_before", "run_bot_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
        "runner_on_3b", "num_runners_on", "home_win_expectancy", "away_win_expectancy",
        "li", "pitcher_hand", "batter_hand", "same_hand", "count_state", "hand_matchup",
        "count_diff", "x_count_pressure", "role_x_inning",
    }
    if name in context_exact or name.startswith("x_ability_x_count") or name.startswith("x_ability_x_pressure"):
        return "game_context"
    if (name in {"season", "game_month", "game_dayofweek", "cat_game_type"}
            or "team_id" in name):
        return "environment_team"
    ability_prefixes = (
        "asof_pitcher_", "inseason_", "x_ability", "ability_", "platoon_", "inning_",
        "pt_", "ly_", "vol_", "role_", "form", "diff_", "x_exp", "x_rev", "x_mid",
        "x_kal", "x_prev", "x_ball_over_strike",
    )
    if name.startswith(ability_prefixes):
        return "pitcher_ability"
    return "other"


log("v66 아티팩트 로드...")
v66 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v66.pkl"))
log(f"  hgbs={len(v66['hgbs'])} cats={len(v66['cats'])}")

log("데이터 로드 + 피처 재구성...")
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
X = X[v66["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

allowed_blocks = {"game_context", "batter_matchup", "environment_team", "trackman"}
cols = [c for c in v66["feature_order"] if block(c) in allowed_blocks]
log(f"보정모델 입력피처 {len(cols)}개 (game_context/batter_matchup/environment_team/trackman)")
assert len(cols) == 78, f"기대 78개, 실제 {len(cols)}개 -- codex 재현과 불일치"
Xr = X[cols]

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("투수-시즌 LOO 실력(비축소) + 잔차 타겟 구성...")
pid = df["pitcher_id"].to_numpy()
seasons_arr = df["season"].to_numpy(np.float64)
sub = pd.DataFrame({"pid": pid, "season": seasons_arr, "y": y.astype(np.float64)})
ps = sub.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
sub = sub.join(ps, on=["pid", "season"])
n_arr = sub["n"].to_numpy(np.float64)
ability_loo = (sub["s"].to_numpy(np.float64) - y.astype(np.float64)) / np.maximum(n_arr - 1, 1)
resid = y.astype(np.float64) - ability_loo
fit_mask = n_arr >= 20
log(f"  잔차타겟 커버리지={fit_mask.mean():.2%}  mean={resid[fit_mask].mean():+.6f}  sd={resid[fit_mask].std():.4f}")

log("HGB 잔차보정모델 학습 (전체데이터, n>=20 행만)...")
model = HistGradientBoostingRegressor(
    loss="squared_error", max_iter=350, learning_rate=0.03, max_depth=6,
    max_leaf_nodes=31, l2_regularization=10.0, early_stopping=True,
    validation_fraction=0.10, n_iter_no_change=25, random_state=42,
)
ts = time.time()
model.fit(Xr.loc[fit_mask], resid[fit_mask], sample_weight=w[fit_mask])
log(f"  학습완료 n_iter={model.n_iter_} ({time.time()-ts:.0f}s)")
strip_rng(model)

common = dict(v66)
common["residcorr_model"] = model
common["residcorr_cols"] = cols
common["residcorr_alpha"] = ALPHA
out = os.path.join(OUT_DIR, "model_artifacts_v72.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
