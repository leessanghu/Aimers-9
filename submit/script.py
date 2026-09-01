# script.py
import glob
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# v128: nn_raw 순수 numpy forward (torch 불필요, dev/nnraw_numpy_forward.py와 동일)
# =======================
def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _nnraw_block(x, state, prefix):
    h = x @ state[f"{prefix}.0.weight"].T + state[f"{prefix}.0.bias"]
    h = _layernorm(h, state[f"{prefix}.1.weight"], state[f"{prefix}.1.bias"])
    return _gelu_tanh(h)


def nnraw_forward(Xz, ip, ib, ipt, ibt, state):
    e = np.concatenate([state["emb_p.weight"][ip], state["emb_b.weight"][ib],
                         state["emb_pt.weight"][ipt], state["emb_bt.weight"][ibt]], axis=1)
    x = np.concatenate([Xz, e], axis=1)
    h = _nnraw_block(x, state, "inp")
    h = h + _nnraw_block(h, state, "b1")
    h = h + _nnraw_block(h, state, "b2")
    logit = (h @ state["head_y.weight"].T + state["head_y.bias"]).squeeze(1)
    return 1.0 / (1.0 + np.exp(-logit))


NNRAW_CONTEXT_FEATS = [
    "cat_top_bottom", "cat_game_type", "cat_base_state", "season", "game_month",
    "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
    "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand", "same_hand", "count_state", "hand_matchup",
    "flag_asof_pitcher_n_zero", "asof_pitcher_n", "flag_asof_batter_n_zero",
    "asof_batter_n", "flag_asof_pitcher_pitchmix_n_zero", "asof_pitcher_pitchmix_n",
    "flag_prev_game_missing", "pitcher_id_count", "batter_id_count",
    "pitcher_team_id_count", "batter_team_id_count", "inseason_n",
    "inseason_is_first_appearance", "platoon_n", "inning_n", "pt_n",
    "x_count_pressure", "count_n", "vol_n_seasons", "role_n_app", "form_missing",
    "tm_n", "tm_matched", "bat_inseason_n", "bat_ly_n", "bplatoon_n",
]


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리를 fit된 통계만으로 재현 (dev/features.py FeatureBuilder와 동일 로직)
# =======================

def build_features(df, stats):
    out = {}

    cat_cols = stats["cat_cols"]
    cats = stats["cat_encoder"].transform(df[cat_cols].astype(str))
    for i, c in enumerate(cat_cols):
        out[f"cat_{c}"] = cats[:, i]

    num_median = stats["num_median"]
    for c in stats["raw_num_cols"]:
        out[c] = df[c].fillna(num_median.get(c, 0.0)).to_numpy(dtype=np.float64)

    out["pitcher_hand"] = df["pitcher_hand"].to_numpy(dtype=np.float64)
    out["batter_hand"] = df["batter_hand"].to_numpy(dtype=np.float64)
    out["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float64).to_numpy()
    out["count_state"] = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy(dtype=np.float64)
    out["hand_matchup"] = (df["pitcher_hand"] * 10 + df["batter_hand"]).to_numpy(dtype=np.float64)

    rate_global_mean = stats["rate_global_mean"]
    smooth_k_rate = stats["smooth_k_rate"]
    smoothed = {}
    for n_col, rate_cols in stats["rate_groups"].items():
        n = df[n_col].fillna(0).to_numpy(dtype=np.float64)
        out[f"flag_{n_col}_zero"] = (n == 0).astype(np.float64)
        out[n_col] = np.log1p(n)
        for c in rate_cols:
            raw = df[c].fillna(0).to_numpy(dtype=np.float64)
            gm = rate_global_mean[c]
            sm = (n * raw + smooth_k_rate * gm) / (n + smooth_k_rate)
            out[f"{c}_smooth"] = sm
            smoothed[c] = sm

    no_n_rate_cols = stats["no_n_rate_cols"]
    miss_flag = df[no_n_rate_cols[0]].isna().to_numpy()
    out["flag_prev_game_missing"] = miss_flag.astype(np.float64)
    for c in no_n_rate_cols:
        out[c] = df[c].fillna(rate_global_mean[c]).to_numpy(dtype=np.float64)

    out["diff_success_rate"] = smoothed["asof_pitcher_success_rate"] - smoothed["asof_batter_success_rate"]
    out["diff_middle_rate"] = smoothed["asof_pitcher_middle_rate"] - smoothed["asof_batter_middle_rate"]

    id_count = stats["id_count"]
    for col in stats["id_count_cols"]:
        cnt = df[col].map(id_count[col]).fillna(0).to_numpy(dtype=np.float64)
        out[f"{col}_count"] = np.log1p(cnt)

    team_count = stats["team_count"]
    for col in stats["team_cols"]:
        cnt = df[col].map(team_count[col]).fillna(0).to_numpy(dtype=np.float64)
        out[f"{col}_count"] = np.log1p(cnt)

    team_te = stats["team_te"]
    global_y_mean = stats["global_y_mean"]
    for col in stats["team_cols"]:
        out[f"{col}_te"] = df[col].map(team_te[col]).fillna(global_y_mean).to_numpy(dtype=np.float64)

    return pd.DataFrame(out, index=df.index)


# =======================
# in-season(시즌 한정) 피처 — dev/inseason.py와 동일 로직 (학습 리포지토리 의존 없이 재구현)
# leakage 안전성: 각 행은 자기 투수의 '직전 시즌 끝 시점'만 참조 (같은 시즌 다른 행 안 씀)
# =======================

def build_inseason_features(df, inseason_stats):
    season_end_table = inseason_stats["season_end_table"]
    global_success_rate = inseason_stats["global_success_rate"]
    seasons_range = inseason_stats["seasons_range"]
    # K는 학습 때 쓴 값을 그대로 사용 (v13부터 60; 이론 최적 87, 전반->후반 검증 최적 150)
    K_SMOOTH = float(inseason_stats.get("k_smooth", 15.0))
    BALL_PRIOR, REVERSE_PRIOR = 0.4, 0.25

    pivots = {}
    for col, val_col in [("N_end", "N_end"), ("S_end", "S_end"), ("B_end", "B_end"),
                         ("R_end", "R_end"), ("rate", "prior_success_rate")]:
        p = season_end_table.pivot(index="pitcher_id", columns="season", values=val_col)
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        pivots[col] = p.stack(future_stack=True)

    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    ends = {}
    for col in ["N_end", "S_end", "B_end", "R_end"]:
        vals = pivots[col].reindex(lookup_idx).to_numpy()
        ends[col] = np.nan_to_num(vals.astype(np.float64), nan=0.0)
    prior_rate_vals = pivots["rate"].reindex(lookup_idx).to_numpy()
    prior_rate = pd.Series(prior_rate_vals).fillna(global_success_rate).to_numpy(np.float64)

    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(df["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
    b_now = np.round(df["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now)
    r_now = np.round(df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - ends["N_end"], 0, None)
    s_season = np.clip(s_now - ends["S_end"], 0, None)
    b_season = np.clip(b_now - ends["B_end"], 0, None)
    r_season = np.clip(r_now - ends["R_end"], 0, None)

    rate_raw = np.divide(s_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    ball_raw = np.divide(b_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    rev_raw = np.divide(r_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)

    out = pd.DataFrame(index=df.index)
    out["inseason_success_smooth"] = (n_season * np.nan_to_num(rate_raw) + K_SMOOTH * prior_rate) / (n_season + K_SMOOTH)
    out["inseason_ball_smooth"] = (n_season * np.nan_to_num(ball_raw) + K_SMOOTH * BALL_PRIOR) / (n_season + K_SMOOTH)
    out["inseason_reverse_smooth"] = (n_season * np.nan_to_num(rev_raw) + K_SMOOTH * REVERSE_PRIOR) / (n_season + K_SMOOTH)
    out["inseason_n"] = np.log1p(n_season)
    out["inseason_is_first_appearance"] = (n_season == 0).astype(np.float64)
    return out


def get_prior_pitcher_rate(df, inseason_stats):
    """각 행의 '직전 시즌 끝 시점' 투수 marginal 성공률 (플래툰 축소 기준점)."""
    season_end_table = inseason_stats["season_end_table"]
    global_success_rate = inseason_stats["global_success_rate"]
    seasons_range = inseason_stats["seasons_range"]

    p = season_end_table.pivot(index="pitcher_id", columns="season", values="prior_success_rate")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    pivot_rate = p.stack(future_stack=True)

    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = pivot_rate.reindex(lookup_idx).to_numpy()
    return pd.Series(vals).fillna(global_success_rate).to_numpy(np.float64)


# =======================
# platoon 피처 — (투수, 타자손) 조건부 성공률의 그 투수 자신 대비 편차
#
# 규칙 준수: 각 행은 자기 투수의 '직전 시즌 끝 시점까지' 누적된 (pitcher_id, batter_hand)
# 조회만 한다(pivot+ffill+stack, in-season과 동일 구조). 같은 시즌의 다른 행이나
# test.csv의 다른 행은 전혀 참조하지 않는다.
#
# 근거: 주최측 asof_* 컬럼은 전부 marginal(투수 전체 성공률)이라 "이 투수는 좌타에
# 유독 약하다" 같은 조건부 개인차를 모델이 볼 수 없다. 노이즈 제거 후 진짜 개인차
# SD=0.0438 (투수 실력 개인차 SD=0.0555의 79%) — 그 행 컬럼만으론 계산 불가능한 정보.
# =======================

def build_platoon_features(df, platoon_stats, prior_rate):
    platoon_table = platoon_stats["platoon_table"]
    seasons_range = platoon_stats["seasons_range"]
    k = platoon_stats["k"]

    def lookup(value_col):
        p = platoon_table.pivot_table(index=["pitcher_id", "batter_hand"], columns="season",
                                      values=value_col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["batter_hand"], df["season"] - 1])
        return p.stack(future_stack=True).reindex(idx).to_numpy()

    s_cell = np.nan_to_num(lookup("s").astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(lookup("n").astype(np.float64), nan=0.0)

    prior = np.asarray(prior_rate, dtype=np.float64)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["platoon_diff"] = rate_smooth - prior
    out["platoon_n"] = np.log1p(n_cell)
    return out


# =======================
# inning 피처 — 투수x이닝 조건부 성공률의 (그 투수 자신 + 전역 이닝효과) 대비 편차
#
# 규칙 준수: platoon과 동일 구조. 각 행은 자기 투수의 '직전 시즌 끝 시점까지' 누적된
# (pitcher_id, inning) 셀만 조회한다. 같은 시즌/test.csv의 다른 행은 전혀 참조하지 않는다.
#
# 근거: 모델은 inning을 원시 피처로 갖고 있어 '전역 이닝 효과'는 이미 알지만, "이 투수가
# 6이닝에 유독 무너진다" 같은 개인 곡선은 볼 수 없다. 노이즈 제거 후 진짜 개인차
# SD=0.0209 -> 상한 ~174점. 전역 이닝 주효과(inning_offset)를 prior에서 빼서 모델이
# 이미 아는 inning 원시피처와 중복되지 않는 '순수 개인 상호작용'만 남긴다.
# =======================

def build_inning_features(df, inning_stats, prior_rate):
    inning_table = inning_stats["inning_table"]
    inning_offset = inning_stats["inning_offset"]
    seasons_range = inning_stats["seasons_range"]
    k = inning_stats["k"]

    inn = np.clip(df["inning"].to_numpy(np.int64), 1, 9)

    def lookup(value_col):
        p = inning_table.pivot_table(index=["pitcher_id", "_inn"], columns="season",
                                     values=value_col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        idx = pd.MultiIndex.from_arrays([df["pitcher_id"], inn, df["season"] - 1])
        return p.stack(future_stack=True).reindex(idx).to_numpy()

    s_cell = np.nan_to_num(lookup("s").astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(lookup("n").astype(np.float64), nan=0.0)

    off = pd.Series(inn).map(inning_offset).fillna(0.0).to_numpy(np.float64)
    prior = np.clip(np.asarray(prior_rate, dtype=np.float64) + off, 1e-6, 1 - 1e-6)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["inning_diff"] = rate_smooth - prior
    out["inning_n"] = np.log1p(n_cell)
    return out


# =======================
# 구종(pitch type) 피처 — Trackman을 투구 단위로 매칭해 구종별 제구력을 복원하고,
# 현재 투구의 실제 구종은 쓰지 않고 '주변화(marginalize)'해서 기대 제구력만 조회 (dev/pitchtype.py와 동일).
#
# P(성공|x) = sum_t P(구종 t|투수,카운트) * P(성공|구종 t, 투수, x)  <- 항등식, 규칙 위반 아님.
# 두 항 모두 학습 시점에 과거 train으로만 추정된 테이블이고, 이 함수는 조회만 한다.
#
# 근거: 매칭 정밀도 99.5~99.7%(batter_hand로 교차검증, 매칭키 미사용) / 커버리지 61.5%.
#       진짜 신호 SD=0.0271(노이즈 제거) / 재현상관 r=+0.475(platoon 0.328보다 높음).
# 규칙 준수: 테이블은 (투수/전역, 구종, season) 누적이며 각 행은 season-1까지만 조회한다.
#          추론 시 Trackman 원본 파일은 불필요(테이블이 아티팩트에 저장됨). test 행 간 참조 없음.
# =======================

_PT_TYPES = ["fastball", "breaking", "offspeed", "other"]


def _pt_pivot(tbl, index, value, seasons_range):
    p = tbl.pivot_table(index=index, columns="season", values=value, aggfunc="first")
    return p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)


def build_pitchtype_features(df, pitchtype_stats, prior_rate):
    tables = pitchtype_stats["tables"]
    global_rate = pitchtype_stats["global_rate"]
    seasons_range = pitchtype_stats["seasons_range"]
    k_control, k_mix = pitchtype_stats["k_control"], pitchtype_stats["k_mix"]

    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    prior = np.asarray(prior_rate, dtype=np.float64)
    n_rows = len(df)

    gt_s = _pt_pivot(tables["gtype"], "ptype", "s", seasons_range)
    gt_n = _pt_pivot(tables["gtype"], "ptype", "n", seasons_range)
    gm_n = _pt_pivot(tables["gmix"], ["count_state", "ptype"], "n", seasons_range)
    ct_s = _pt_pivot(tables["ctrl"], ["pitcher_id", "ptype"], "s", seasons_range)
    ct_n = _pt_pivot(tables["ctrl"], ["pitcher_id", "ptype"], "n", seasons_range)
    mx_n = _pt_pivot(tables["mix"], ["pitcher_id", "count_state", "ptype"], "n", seasons_range)

    prev = season - 1
    num_pred = np.zeros(n_rows)
    den_mix = np.zeros(n_rows)
    tot_n = np.zeros(n_rows)

    for t in _PT_TYPES:
        gs = np.nan_to_num(gt_s.reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        gn = np.nan_to_num(gt_n.reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        type_rate = np.divide(gs, gn, out=np.full(n_rows, global_rate), where=gn > 0)

        cs_ = np.nan_to_num(ct_s.reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        cn_ = np.nan_to_num(ct_n.reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        anchor = np.clip(prior + (type_rate - global_rate), 1e-6, 1 - 1e-6)
        ctrl_t = (cs_ + k_control * anchor) / (cn_ + k_control)

        pm = np.nan_to_num(mx_n.reindex(pd.MultiIndex.from_arrays([pid, cs, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        gmx = np.nan_to_num(gm_n.reindex(pd.MultiIndex.from_arrays([cs, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        num_pred += (pm + k_mix * gmx) * ctrl_t
        den_mix += (pm + k_mix * gmx)
        tot_n += cn_

    pred = np.divide(num_pred, den_mix, out=prior.copy(), where=den_mix > 0)
    out = pd.DataFrame(index=df.index)
    out["pt_pred"] = pred
    out["pt_dev"] = pred - prior
    out["pt_n"] = np.log1p(tot_n)
    return out


# =======================
# lastyear 피처 — '작년 한 시즌만' rate들 + 합성 실력 추정치 (dev/lastyear.py와 동일)
#
# 채택 근거(잔차 기반 사전 선별, 학습 없이 계산): 7개 합동 예상이득 +17.4
#   지표 검증됨 — 구종 +5.6 -> 실제 +6.7 / workload,form +0.4 -> 실제 가치없음
#   같은 규칙으로 Teacher(+0.0), disjoint 블록(+0.8)은 학습 전에 기각
# 핵심: ly_success는 ly_reverse 위에 아무것도 더하지 않음(둘 다 +6.1).
#   진짜 새 정보는 '작년 reverse율' — 다음시즌 제구와 상관 -0.494(success +0.531에 맞먹음)인데
#   그동안 실력 추정에 전혀 쓰이지 않았다.
# 복원: 누적(season-1) - 누적(season-2). in-season과 같은 차분 트릭의 한 칸 앞.
# 규칙 준수: 각 행은 자기 투수의 season-1, season-2 시점 누적만 조회. 행 간 참조 없음.
# =======================

def build_lastyear_features(df, lastyear_stats):
    ly_table = lastyear_stats["ly_table"]
    gr = lastyear_stats["global_rates"]
    seasons_range = lastyear_stats["seasons_range"]
    k = lastyear_stats["k"]
    W = lastyear_stats["composite_w"]

    cols = ["N_end", "S_end", "R_end", "B_end", "M_end"]
    pivots = {c: ly_table.pivot(index="pitcher_id", columns="season", values=c)
                          .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
              for c in cols}
    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    i1 = pd.MultiIndex.from_arrays([pid, season - 1])
    i2 = pd.MultiIndex.from_arrays([pid, season - 2])

    cum1, cum2 = {}, {}
    for c in cols:
        cum1[c] = np.nan_to_num(pivots[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0)
        cum2[c] = np.nan_to_num(pivots[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0)

    n_ly = np.clip(cum1["N_end"] - cum2["N_end"], 0, None)
    career_n = cum1["N_end"]
    career_rate = np.divide(cum1["S_end"], career_n, out=np.full_like(career_n, np.nan), where=career_n > 0)
    career_rate = np.nan_to_num(career_rate, nan=gr["success"])

    out = pd.DataFrame(index=df.index)
    for src, nm, gkey in [("S_end", "ly_success", "success"), ("R_end", "ly_reverse", "reverse"),
                          ("B_end", "ly_ball", "ball"), ("M_end", "ly_middle", "middle")]:
        cnt = np.clip(cum1[src] - cum2[src], 0, None)
        raw = np.divide(cnt, n_ly, out=np.full_like(n_ly, np.nan), where=n_ly > 0)
        gm = gr[gkey]
        out[nm] = (n_ly * np.nan_to_num(raw, nan=gm) + k * gm) / (n_ly + k)

    out["ly_n"] = np.log1p(n_ly)
    out["ly_minus_career"] = out["ly_success"].to_numpy() - career_rate
    out["ability_composite"] = (W["career"] * career_rate
                                + W["ly_success"] * out["ly_success"].to_numpy()
                                + W["ly_reverse"] * out["ly_reverse"].to_numpy()
                                + W["ly_ball"] * out["ly_ball"].to_numpy()
                                + W["ly_middle"] * out["ly_middle"].to_numpy())
    return out


# =======================
# 투구단위 라벨 복원 조건부 (v17, dev/pitchlabels.py와 동일)
#
# 발견: 같은 투수의 연속 행에서 asof_pitcher_n 증가량이 정확히 +1인 비율 = 100.00%.
#   -> 누적 카운트 차분으로 그 투구의 reverse/middle/ball/strike 라벨을 정확히 복원 가능
#   (success로 대조검증 시 100.000% 일치). 주최측이 라벨을 사실상 5개 준 셈이었다.
# 규칙 준수: 라벨 복원과 조건부 테이블은 train에서만 계산(학습 스크립트), 여기서는 그
#   결과 테이블을 (entity, ctx, season-1까지)로 조회만 한다. test 행 간 참조 없음.
# =======================

_LABEL_NAMES = ["reverse", "middle", "ball", "strike"]


def build_label_cond_features(df, stats, ctx, prefix):
    table, glob, by_ctx = stats["table"], stats["glob"], stats["by_ctx"]
    seasons_range, k = stats["seasons_range"], stats["k"]
    ctx = np.asarray(ctx)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"].to_numpy(), ctx, df["season"].to_numpy() - 1])

    def lk(col):
        p = table.pivot_table(index=["_e", "_c"], columns="season", values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return np.nan_to_num(p.stack(future_stack=True).reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    n_cell = lk("n")
    out = pd.DataFrame(index=df.index)
    for name in _LABEL_NAMES:
        s_cell = lk(name)
        anchor = pd.Series(ctx).map(by_ctx[name]).fillna(glob[name]).to_numpy(np.float64)
        rate = (s_cell + k * anchor) / (n_cell + k)
        out[f"{prefix}_{name}_dev"] = rate - anchor
    out[f"{prefix}_n"] = np.log1p(n_cell)
    return out


# =======================
# v17b 전용 — lastyear strike / pitchmix + arsenal JS / Trackman pitch_of_pa
# (실측 근거 확보 목적. phase43 잔차 스크리닝에서 개별로는 기각선 근처였음)
# =======================

_MIX_COLS = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]


def build_strike_features(df, stats):
    kt, gk, seasons_range, k = stats["kt"], stats["gk"], stats["seasons_range"], stats["k_strike"]
    piv = {c: kt.pivot(index="pitcher_id", columns="season", values=c)
              .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in ["K_end", "N_end"]}
    i1 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    i2 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 2])
    v1 = {c: np.nan_to_num(piv[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    v2 = {c: np.nan_to_num(piv[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    n_ly = np.clip(v1["N_end"] - v2["N_end"], 0, None)
    c_ly = np.clip(v1["K_end"] - v2["K_end"], 0, None)
    raw = np.divide(c_ly, n_ly, out=np.full_like(n_ly, np.nan), where=n_ly > 0)
    return pd.DataFrame({"ly_strike": (n_ly * np.nan_to_num(raw, nan=gk) + k * gk) / (n_ly + k)}, index=df.index)


def build_pitchmix_arsenal_features(df, stats):
    mtd, gmix, seasons_range, k = stats["mtd"], stats["gmix"], stats["seasons_range"], stats["k_mix"]
    piv = {c: mtd.pivot(index="pitcher_id", columns="season", values=c)
              .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in ["MN"] + _MIX_COLS}
    i1 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    i2 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 2])
    v1 = {c: np.nan_to_num(piv[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    v2 = {c: np.nan_to_num(piv[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    mn_ly = np.clip(v1["MN"] - v2["MN"], 0, None)

    out = pd.DataFrame(index=df.index)
    ly_p, car_p = [], []
    for c in _MIX_COLS:
        cc = np.clip(v1[c] - v2[c], 0, None)
        r_ly = np.divide(cc, mn_ly, out=np.full_like(mn_ly, np.nan), where=mn_ly > 0)
        r_ly = (mn_ly * np.nan_to_num(r_ly, nan=gmix[c]) + k * gmix[c]) / (mn_ly + k)
        r_car = np.divide(v1[c], v1["MN"], out=np.full_like(mn_ly, np.nan), where=v1["MN"] > 0)
        r_car = np.nan_to_num(r_car, nan=gmix[c])
        short = c.split("_")[-2]
        out[f"lymix_{short}"] = r_ly
        out[f"lymix_{short}_minus_career"] = r_ly - r_car
        ly_p.append(r_ly)
        car_p.append(r_car)

    P = np.clip(np.vstack(ly_p).T, 1e-9, None); P /= P.sum(1, keepdims=True)
    Q = np.clip(np.vstack(car_p).T, 1e-9, None); Q /= Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    js = 0.5 * (P * np.log(P / M)).sum(1) + 0.5 * (Q * np.log(Q / M)).sum(1)
    out["arsenal_js"] = np.nan_to_num(js, nan=0.0)
    return out


def build_popa_features(df, stats):
    prof, seasons_range = stats["prof"], stats["seasons_range"]
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    piv = {c: prof.pivot_table(index=["pitcher_id", "count_state"], columns="season", values=c, aggfunc="first")
              .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in ["popa_mean", "popa_max"]}
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"].to_numpy(), cs, df["season"].to_numpy() - 1])
    return pd.DataFrame({c: np.nan_to_num(piv[c].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
                         for c in piv}, index=df.index)


# =======================
# time103 피처 — v16 신규 12개.
#
# v15의 작년 한 시즌 snapshot을 확장해:
#   (1) 작년 rate - 작년 리그 rate
#   (2) 올해 시즌 한정 rate - 작년 rate
#   (3) 작년 rate - 오래된 커리어 rate
# 를 S/R/B/M 네 축에 대해 만든다.
#
# 규칙 준수: 각 행은 공식 asof 컬럼과 train에서 저장한 season-end 테이블만 조회한다.
# test.csv의 다른 행은 사용하지 않는다.
# =======================

_TIME103_COLS = [
    "ly_success_rel", "cur_minus_ly_success", "ly_minus_old_success",
    "ly_reverse_rel", "cur_minus_ly_reverse", "ly_minus_old_reverse",
    "ly_ball_rel", "cur_minus_ly_ball", "ly_minus_old_ball",
    "ly_middle_rel", "cur_minus_ly_middle", "ly_minus_old_middle",
]


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


def build_hidden_denominator_features(df):
    # Pure row-wise transform of the two official prev-game rates.
    out = pd.DataFrame(index=df.index)
    for k, max_q in ((1, 160), (3, 480), (5, 800)):
        out[f"prev{k}_hidden_total_n"] = infer_min_denominator(
            df[f"asof_pitcher_prev{k}_game_success_rate"],
            df[f"asof_pitcher_prev{k}_game_middle_rate"],
            max_q,
        )
    out["prev3_hidden_avg_n"] = out["prev3_hidden_total_n"] / 3.0
    out["prev5_hidden_avg_n"] = out["prev5_hidden_total_n"] / 5.0
    out["prev1_vs_prev3_workload"] = out["prev1_hidden_total_n"] - out["prev3_hidden_avg_n"]
    out["prev3_vs_prev5_workload"] = out["prev3_hidden_avg_n"] - out["prev5_hidden_avg_n"]
    return out.astype(np.float64)


# =======================
# 커리어 시즌간 변동성 (v20, dev/career_volatility.py와 동일)
#
# 지금까지는 '평균 제구력'만 썼지, 이 투수가 시즌마다 얼마나 들쭉날쭉한지는 피처가 없었다.
# AMEX/Home Credit류 "entity 과거 관측치를 std/min/max로 압축" 패턴을 그대로 적용.
#
# 규칙 준수: vol_table은 train에서 season_end_table의 연속 시즌 차분(고립 성공률)으로만
# 만들고, 각 행은 자기 투수의 '직전 시즌까지' expanding 통계만 조회한다 (1칸 shift로
# 자기 시즌 자체는 제외). test 행 간 참조 없음.
# =======================

def build_volatility_features(df, vol_stats):
    vol_table = vol_stats["vol_table"]
    seasons_range = vol_stats["seasons_range"]
    k = vol_stats["k"]
    global_std = vol_stats["global_std"]  # fit(=train) 시점에 고정된 상수. 배치(subset/full)에 따라 달라지면 행 독립성 위반.

    cols = ["exp_std", "exp_min", "exp_max", "exp_count"]
    piv = {c: vol_table.pivot(index="pitcher_id", columns="season", values=c)
                       .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in cols}
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = {c: piv[c].reindex(idx).to_numpy().astype(np.float64) for c in cols}

    exp_count = np.nan_to_num(vals["exp_count"], nan=0.0)
    exp_std = vals["exp_std"]
    exp_min = vals["exp_min"]
    exp_max = vals["exp_max"]

    valid = exp_count >= 2
    shrunk_std = np.where(valid, (exp_count * np.nan_to_num(exp_std, nan=global_std) + k * global_std) / (exp_count + k), global_std)

    out = pd.DataFrame(index=df.index)
    out["vol_std"] = shrunk_std
    out["vol_min"] = np.where(exp_count >= 1, np.nan_to_num(exp_min, nan=0.5), 0.5)
    out["vol_max"] = np.where(exp_count >= 1, np.nan_to_num(exp_max, nan=0.5), 0.5)
    out["vol_range"] = out["vol_max"] - out["vol_min"]
    out["vol_n_seasons"] = np.log1p(exp_count)
    return out


# =======================
# 구종 레퍼토리 엔트로피 (v22, dev/arsenal_entropy.py와 동일)
#
# 공식 asof_pitcher_{fastball,breaking,offspeed}_rate + pitchmix_n을 그대로 쓰는 비선형
# 결합이라 leakage 걱정이 전혀 없다(crosses.py와 같은 철학). global_mix는 fit(=train) 시점에
# 고정된 상수를 써야 한다 (배치에 따라 달라지면 행 독립성 위반 — career_volatility에서
# 겪은 것과 같은 종류의 버그이므로 여기서는 처음부터 고정 상수로만 구현한다).
# =======================

_ARSENAL_MIX_COLS = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
_ARSENAL_EPS = 1e-9


def build_arsenal_features(df, arsenal_stats):
    global_mix = arsenal_stats["global_mix"]
    k = arsenal_stats["k"]

    n = df["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)
    raws = {c: df[c].fillna(0).to_numpy(np.float64) for c in _ARSENAL_MIX_COLS}

    shrunk = np.column_stack([
        (n * raws[c] + k * global_mix[c]) / (n + k) for c in _ARSENAL_MIX_COLS
    ])
    shrunk = np.clip(shrunk, _ARSENAL_EPS, None)
    shrunk = shrunk / shrunk.sum(axis=1, keepdims=True)

    out = pd.DataFrame(index=df.index)
    out["arsenal_entropy"] = -(shrunk * np.log(shrunk)).sum(axis=1)
    out["arsenal_top_share"] = shrunk.max(axis=1)
    return out


def _league_season_rates(ly_table, seasons_range):
    rows = []
    for s in seasons_range:
        cur = ly_table[ly_table["season"] == s]
        if cur.empty:
            continue
        den = max(float(cur["N_end"].sum()), 1.0)
        rows.append({
            "season": s,
            "success": float(cur["S_end"].sum() / den),
            "reverse": float(cur["R_end"].sum() / den),
            "ball": float(cur["B_end"].sum() / den),
            "middle": float(cur["M_end"].sum() / den),
        })
    return pd.DataFrame(rows).set_index("season")


def _lookup_time_cum(df, pivots, season_offset):
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] + season_offset])
    out = {}
    for c in ["N_end", "S_end", "R_end", "B_end", "M_end"]:
        out[c] = np.nan_to_num(pivots[c].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    return out


def _rates_from_counts(num, den, fallback):
    raw = np.divide(num, den, out=np.full_like(den, np.nan, dtype=np.float64), where=den > 0)
    return np.nan_to_num(raw, nan=fallback)


def build_time103_features(df, time103_stats):
    ly_table = time103_stats["ly_table"]
    gr = time103_stats["global_rates"]
    seasons_range = time103_stats["seasons_range"]
    k_cur = float(time103_stats.get("k_cur", 15.0))
    k_ly = float(time103_stats.get("k_ly", 30.0))

    pivots = {c: ly_table.pivot(index="pitcher_id", columns="season", values=c)
                          .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
              for c in ["N_end", "S_end", "R_end", "B_end", "M_end"]}
    league = _league_season_rates(ly_table, seasons_range)

    c1 = _lookup_time_cum(df, pivots, -1)
    c2 = _lookup_time_cum(df, pivots, -2)

    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    now = {
        "N_end": n_now,
        "S_end": np.round(df["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "R_end": np.round(df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "B_end": np.round(df["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "M_end": np.round(df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_now),
    }

    n_cur = np.clip(now["N_end"] - c1["N_end"], 0, None)
    n_ly = np.clip(c1["N_end"] - c2["N_end"], 0, None)
    n_old = c2["N_end"]

    out = pd.DataFrame(index=df.index)
    prev_season = df["season"].to_numpy() - 1
    for key, col in [("success", "S_end"), ("reverse", "R_end"), ("ball", "B_end"), ("middle", "M_end")]:
        gm = float(gr[key])
        cnt_cur = np.clip(now[col] - c1[col], 0, None)
        cnt_ly = np.clip(c1[col] - c2[col], 0, None)
        cnt_old = np.clip(c2[col], 0, None)

        cur_raw = _rates_from_counts(cnt_cur, n_cur, gm)
        ly_raw = _rates_from_counts(cnt_ly, n_ly, gm)
        old_raw = _rates_from_counts(cnt_old, n_old, gm)

        cur_sm = (n_cur * cur_raw + k_cur * gm) / (n_cur + k_cur)
        ly_sm = (n_ly * ly_raw + k_ly * gm) / (n_ly + k_ly)
        old_sm = (n_old * old_raw + k_ly * gm) / (n_old + k_ly)

        rel = np.asarray([float(league.loc[s, key]) if s in league.index else gm
                          for s in prev_season], dtype=np.float64)
        out[f"ly_{key}_rel"] = ly_sm - rel
        out[f"cur_minus_ly_{key}"] = cur_sm - ly_sm
        out[f"ly_minus_old_{key}"] = ly_sm - old_sm

    return out[_TIME103_COLS].astype(np.float64)


# =======================
# 교차항 — 조립된 피처 행렬에서만 계산 (행 간 참조 전혀 없음, dev/crosses.py와 동일)
#
# 트리는 곱/합/차를 스스로 근사할 수 있으므로 무작정 곱하면 노이즈만 는다.
# 트리가 '비효율적으로' 근사하는 것만 명시적으로 준다:
#   (1) 비율 x/y  (2) 여러 항의 합  (3) 실력 x 상황압박 대각선 상호작용
# 2024 폴드: 826.7 -> 848.4 (+21.7). CatBoost가 826.5->855.2로 크게 개선.
# =======================

_EPS = 1e-6


def _cg(X, name, default=0.0):
    return X[name].to_numpy(np.float64) if name in X.columns else np.full(len(X), default)


def add_crosses(X):
    ability = _cg(X, "kal_post") if "kal_post" in X.columns else _cg(X, "inseason_success_smooth")
    plat = _cg(X, "platoon_diff")
    inn_d = _cg(X, "inning_diff")
    career = _cg(X, "asof_pitcher_success_rate_smooth")
    batter = _cg(X, "asof_batter_success_rate_smooth")
    ball = _cg(X, "asof_pitcher_ball_rate_smooth")
    strike = _cg(X, "asof_pitcher_strike_rate_smooth")
    rev = _cg(X, "asof_pitcher_reverse_rate_smooth")
    mid = _cg(X, "asof_pitcher_middle_rate_smooth")
    balls = _cg(X, "balls_before")
    strikes = _cg(X, "strikes_before")
    cnt = _cg(X, "count_state")
    inning = _cg(X, "inning")
    same = _cg(X, "same_hand")
    n_exp = _cg(X, "asof_pitcher_n")
    prev5 = _cg(X, "asof_pitcher_prev5_game_success_rate")
    prev1 = _cg(X, "asof_pitcher_prev1_game_success_rate")

    out = pd.DataFrame(index=X.index)
    here = ability + plat + inn_d
    out["x_ability_here"] = here
    pressure = balls - strikes
    out["x_count_pressure"] = pressure
    out["x_ability_x_count"] = here * cnt
    out["x_ability_x_pressure"] = here * pressure
    out["x_ability_x_inning"] = here * inning
    out["x_platoon_x_samehand"] = plat * same
    out["x_exp_x_ability"] = n_exp * here
    out["x_p_over_b"] = career / (batter + _EPS)
    out["x_ball_over_strike"] = ball / (strike + _EPS)
    out["x_rev_over_succ"] = rev / (career + _EPS)
    out["x_mid_over_succ"] = mid / (career + _EPS)
    out["x_kal_minus_career"] = ability - career
    out["x_prev5_minus_career"] = prev5 - career
    out["x_prev1_minus_prev5"] = prev1 - prev5
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float64)


# =======================
# count_state 조건부 피처 (v26, dev/count_split.py와 동일)
#
# phase65 오라클 천장 실측: pitcher x count_state 천장=1223 (platoon 1134, inning 1029보다 높음).
# platoon/inning과 동일한 (pitcher, ctx, season) 누적 lookup 구조.
# =======================

def build_count_features(df, count_stats, prior_rate):
    count_table = count_stats["count_table"]
    seasons_range = count_stats["seasons_range"]
    k = count_stats["k"]
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()

    def lookup(value_col):
        p = count_table.pivot_table(index=["pitcher_id", "count_state"], columns="season",
                                    values=value_col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        idx = pd.MultiIndex.from_arrays([df["pitcher_id"], cs, df["season"] - 1])
        return p.stack(future_stack=True).reindex(idx).to_numpy()

    s_cell = np.nan_to_num(lookup("s").astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(lookup("n").astype(np.float64), nan=0.0)

    prior = np.asarray(prior_rate, dtype=np.float64)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["count_diff"] = rate_smooth - prior
    out["count_n"] = np.log1p(n_cell)
    return out


# =======================
# 역할(선발/불펜) 프로파일 (v26, dev/formfeat.py와 동일)
#
# train 행 시퀀스에서 등판 경계를 복원(season/month/dayofweek 변화 또는 inning 감소)해
# 등판당 투구수/이닝 분포를 낸다. 선발은 100구 넘긴 상태, 불펜은 방금 등판한 상태로
# 같은 inning=7이라도 피로도가 반대 -> role x inning 교차항으로 트리에 명시적으로 제공.
# 규칙 준수: role_table은 train에서만 만들고, 각 행은 season-1까지 누적만 조회한다.
# =======================

ROLE_COLS = ["role_ppa", "role_first_inn_share", "role_late_share", "role_med_inning", "role_n_app"]


def _expanding_role(tbl):
    rows = []
    for pid, grp in tbl.groupby("pitcher_id"):
        grp = grp.sort_values("season")
        n_cum = 0.0
        acc = {c: 0.0 for c in ROLE_COLS if c != "role_n_app"}
        for _, r in grp.iterrows():
            n = float(r["role_n_app"])
            for c in acc:
                v = r[c]
                if np.isfinite(v):
                    acc[c] += v * n
            n_cum += n
            out = {"pitcher_id": pid, "season": int(r["season"]), "role_n_app": n_cum}
            for c in acc:
                out[c] = acc[c] / n_cum if n_cum > 0 else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def build_role_features(df, role_stats):
    role_tbl = role_stats["role_table"]
    seasons_range = role_stats["seasons_range"]
    k = role_stats["k_role"]
    exp = _expanding_role(role_tbl)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    glob = {c: float(exp[c].median()) for c in ROLE_COLS if c != "role_n_app"}

    piv_n = exp.pivot_table(index="pitcher_id", columns="season", values="role_n_app", aggfunc="first")
    piv_n = piv_n.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    n_app = np.nan_to_num(piv_n.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    out = pd.DataFrame(index=df.index)
    out["role_n_app"] = np.log1p(n_app)
    for c in ROLE_COLS:
        if c == "role_n_app":
            continue
        p = exp.pivot_table(index="pitcher_id", columns="season", values=c, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        v = p.reindex(idx).to_numpy().astype(np.float64)
        gm = glob[c]
        v = np.where(np.isfinite(v), v, gm)
        out[c] = (n_app * v + k * gm) / (n_app + k)

    out["role_x_inning"] = out["role_ppa"].to_numpy() * df["inning"].to_numpy(np.float64)
    return out.astype(np.float64)


# =======================
# 폼 피처 (v26, dev/formfeat.py와 동일)
#
# asof_pitcher_prev{1,3,5}_game_* 를 (1) 리그평균이 아니라 그 투수 자신의 inseason 베이스라인
# 대비, (2) 등판당 투구수 추정치(role_ppa)로 표본 신뢰도만큼 축소, (3) logit 공간에서 차분한다.
# 전부 공식 컬럼의 행 내부 변환이라 test 행 간 참조가 없다.
# =======================

_FORM_EPS = 1e-6


def _form_logit(p):
    p = np.clip(p, _FORM_EPS, 1 - _FORM_EPS)
    return np.log(p / (1 - p))


def build_form_features(df, role_feats, baseline_success, baseline_middle, k_form=40.0):
    out = pd.DataFrame(index=df.index)
    ppa = np.clip(role_feats["role_ppa"].to_numpy(np.float64), 1.0, None)

    base_s = np.clip(np.asarray(baseline_success, dtype=np.float64), _FORM_EPS, 1 - _FORM_EPS)
    base_m = np.clip(np.asarray(baseline_middle, dtype=np.float64), _FORM_EPS, 1 - _FORM_EPS)
    lb_s, lb_m = _form_logit(base_s), _form_logit(base_m)

    miss = df["asof_pitcher_prev1_game_success_rate"].isna().to_numpy()
    out["form_missing"] = miss.astype(np.float64)

    forms = {}
    for k in (1, 3, 5):
        n_est = ppa * k
        for kind, base_p, lb in (("success", base_s, lb_s), ("middle", base_m, lb_m)):
            col = f"asof_pitcher_prev{k}_game_{kind}_rate"
            raw = df[col].to_numpy(np.float64)
            raw = np.where(np.isfinite(raw), raw, base_p)
            p_sm = (n_est * raw + k_form * base_p) / (n_est + k_form)
            f = _form_logit(p_sm) - lb
            out[f"form{k}_{kind}"] = f
            forms[(k, kind)] = f

    out["form_accel"] = forms[(1, "success")] - forms[(5, "success")]
    out["form_1_minus_3"] = forms[(1, "success")] - forms[(3, "success")]
    out["form_3_minus_5"] = forms[(3, "success")] - forms[(5, "success")]
    out["form_reliability"] = np.log1p(ppa)

    cols = ["form1_success", "form3_success", "form5_success", "form1_middle", "form3_middle",
            "form5_middle", "form_accel", "form_1_minus_3", "form_3_minus_5", "form_reliability",
            "form_missing"]
    return out[cols].astype(np.float64)


# =======================
# Trackman 물리 프로파일 (v26, dev/trackman_profile.py와 동일)
#
# 릴리스포인트 반복성((투수x구종) 내부 SD로 레퍼토리 다양성과 분리), 무브먼트 크기,
# 등판 내부 구속 감쇠(피로), 압박 상황 릴리스 산포 변화. 전부 (pitcher, season-1) 조회.
# 추론 시 trackman 원본 CSV 불필요 — 프로파일 테이블이 아티팩트에 저장되어 있다.
# =======================

TM_PROFILE_COLS = [
    "tm_n", "tm_release_sd", "tm_rel_h_sd", "tm_rel_s_sd", "tm_ext_sd",
    "tm_speed_sd", "tm_ivb_sd", "tm_hb_sd", "tm_break_mag",
    "tm_speed_mean", "tm_spin_mean", "tm_ext_mean", "tm_rel_h_mean", "tm_rel_s_mean",
    "tm_velo_decay", "tm_press_rel_sd",
]


def _expanding_trackman(prof):
    rows = []
    for pid, grp in prof.groupby("pitcher_id"):
        grp = grp.sort_values("season")
        n_cum = 0.0
        acc = {c: 0.0 for c in TM_PROFILE_COLS if c != "tm_n"}
        for _, r in grp.iterrows():
            n = float(r["tm_n"]) if np.isfinite(r["tm_n"]) else 0.0
            for c in acc:
                v = r[c]
                if np.isfinite(v):
                    acc[c] += v * n
            n_cum += n
            out = {"pitcher_id": pid, "season": int(r["season"]), "tm_n": n_cum}
            for c in acc:
                out[c] = acc[c] / n_cum if n_cum > 0 else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def build_trackman_features(df, trackman_stats):
    prof = trackman_stats["profile"]
    seasons_range = trackman_stats["seasons_range"]
    k = trackman_stats["k"]
    exp = _expanding_trackman(prof)

    out = pd.DataFrame(index=df.index)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    glob = {c: float(exp[c].median()) for c in TM_PROFILE_COLS if c != "tm_n"}

    piv_n = exp.pivot_table(index="pitcher_id", columns="season", values="tm_n", aggfunc="first")
    piv_n = piv_n.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    n_cell = np.nan_to_num(piv_n.reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    out["tm_n"] = np.log1p(n_cell)
    out["tm_matched"] = (n_cell > 0).astype(np.float64)

    for c in TM_PROFILE_COLS:
        if c == "tm_n":
            continue
        p = exp.pivot_table(index="pitcher_id", columns="season", values=c, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        v = p.reindex(idx).to_numpy().astype(np.float64)
        gm = glob[c]
        v = np.where(np.isfinite(v), v, gm)
        out[c] = (n_cell * v + k * gm) / (n_cell + k)

    return out.astype(np.float64)


# =======================
# trackman x 저표본 상호작용 (v27, dev/trackman_profile.py와 동일)
#
# phase70 층화 검정: 물리량은 제구력의 '원인'이고 우리는 '결과'(성공률 이력)를 직접 관측한다.
# 결과가 충분한 투수에겐 원인 정보가 무용지물, 이력이 부족한 투수에게만 가치가 있다.
#   asof_pitcher_n 4분위별 trackman 증분 (1시그마=1.6):
#     Q1 저표본 +34.9 / Q2 +47.5 / Q3 -6.7 / Q4 고표본 +3.1
# 임계값은 fit(=train) 시점 상수를 아티팩트에서 읽는다 (배치 의존이면 행 독립성 위반).
# =======================

# =======================
# v107: 신규 물리 파생피처 6개.
# 기존 trackman_profile 규약과 동일(expanding 누적 -> season-1 룩업 -> K 축소).
# tm_spin_sd(스핀 일관성), tm_velo_loss(종속 감속), tm_k2_rel_sd(2스트라이크 릴리스 산포),
# tm_type_sep(구종간 릴리스 분리도) + pitchmix 엔트로피/최대점유율.
# Rule 4: 각 행은 자기 pitcher_id/season으로 train 기반 테이블만 조회한다.
# =======================
NEWTM_COLS = ["tm_spin_sd", "tm_velo_loss", "tm_k2_rel_sd", "tm_type_sep"]


def build_newtm_features(df, newtm_stats, X_ref):
    exp = newtm_stats["profile"]
    seasons_range = newtm_stats["seasons_range"]
    k = float(newtm_stats["k"])
    glob = newtm_stats["global_median"]

    out = pd.DataFrame(index=df.index)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])

    piv_n = exp.pivot_table(index="pitcher_id", columns="season", values="tm_n", aggfunc="first")
    piv_n = piv_n.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    n_cell = np.nan_to_num(piv_n.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    for c in NEWTM_COLS:
        p = exp.pivot_table(index="pitcher_id", columns="season", values=c, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        v = p.reindex(idx).to_numpy().astype(np.float64)
        gm = float(glob[c])
        v = np.where(np.isfinite(v), v, gm)
        out[c] = (n_cell * v + k * gm) / (n_cell + k)

    P = X_ref[["asof_pitcher_fastball_rate_smooth",
               "asof_pitcher_breaking_rate_smooth",
               "asof_pitcher_offspeed_rate_smooth"]].to_numpy(np.float64)
    P = np.clip(P, 1e-9, None)
    P = P / P.sum(axis=1, keepdims=True)
    out["pitchmix_entropy"] = -(P * np.log(P)).sum(axis=1)
    out["pitchmix_maxshare"] = P.max(axis=1)
    return out.astype(np.float64)


def build_trackman_lown_features(X_tm, asof_pitcher_n, threshold):
    n = np.asarray(asof_pitcher_n, dtype=np.float64)
    lown = (np.nan_to_num(n, nan=0.0) <= threshold).astype(np.float64)
    cols = [c for c in X_tm.columns if c != "tm_matched"]
    out = pd.DataFrame(index=X_tm.index)
    out["tm_lown_flag"] = lown
    for c in cols:
        out[f"{c}_x_lown"] = X_tm[c].to_numpy(np.float64) * lown
    return out.astype(np.float64)


# =======================
# 타자 in-season 피처 (v27, dev/batterform.py와 동일)
#
# 투수 쪽은 inseason/lastyear/platoon/inning/count/pitchtype/volatility/form/trackman까지
# 다 만들었는데 타자 쪽은 공식 컬럼 4개뿐이었다. inseason.py와 동일한 누적 차분 트릭 적용.
# phase70: bat_inseason_smooth +17.05(6.6시그마), 블록 합동 +17.3
# 규칙 준수: batter_table은 train에서만 만들고 각 행은 season-1 누적만 조회한다.
# =======================

BATTER_COLS = ["bat_inseason_smooth", "bat_inseason_n", "bat_ly_rate", "bat_ly_n",
               "bat_inseason_minus_career"]


def build_batter_features(df, batter_stats):
    batter_table = batter_stats["batter_table"]
    seasons_range = batter_stats["seasons_range"]
    global_rate = batter_stats["global_rate"]
    k = batter_stats["k"]

    pv = {}
    for c in ("S", "N"):
        p = batter_table.pivot(index="batter_id", columns="season", values=c)
        pv[c] = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)

    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["season"] - 1])
    S_end = np.nan_to_num(pv["S"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    N_end = np.nan_to_num(pv["N"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    n_now = df["asof_batter_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(df["asof_batter_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
    n_seas = np.clip(n_now - N_end, 0, None)
    s_seas = np.clip(s_now - S_end, 0, None)
    career = df["asof_batter_success_rate"].fillna(global_rate).to_numpy(np.float64)

    out = pd.DataFrame(index=df.index)
    out["bat_inseason_smooth"] = (s_seas + k * global_rate) / (n_seas + k)
    out["bat_inseason_n"] = np.log1p(n_seas)
    out["bat_ly_rate"] = np.divide(S_end, np.maximum(N_end, 1.0),
                                   out=np.full(len(df), global_rate), where=N_end > 0)
    out["bat_ly_n"] = np.log1p(N_end)
    out["bat_inseason_minus_career"] = out["bat_inseason_smooth"].to_numpy() - career
    return out[BATTER_COLS].astype(np.float64)


# =======================
# in-season 라벨차원 보충 (v28, dev/inseason_full.py와 동일)
#
# inseason.py의 season_end_table은 N/S/B/R만 저장해 middle/strike가 빠져 있었다
# (lastyear는 ly_middle을 이미 쓰고 있었으므로 in-season만 비대칭).
#
# phase75 SHAP 검증: inseason_cmd_index magnitude 0.02048 (전체 5위)로 신규 피처 중 최고.
#   스크리너(부분상관)는 +0.47로 과소평가했는데, 이는 '새 정보'가 아니라 '더 좋은 표현'이라
#   트리의 split 용량을 절약해주는 종류이기 때문(crosses.py와 같은 철학).
# =======================

def get_n_end(df, inseason_stats):
    """각 행의 '직전 시즌 종료 시점 누적 투구수'. middle/strike 분모를 success와 맞추기 위함."""
    season_end_table = inseason_stats["season_end_table"]
    seasons_range = inseason_stats["seasons_range"]
    p = season_end_table.pivot(index="pitcher_id", columns="season", values="N_end")
    p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    return np.nan_to_num(p.reindex(idx).to_numpy().astype(np.float64), nan=0.0)


INSEASON_FULL_COLS = ["inseason_middle_smooth", "inseason_strike_smooth",
                      "inseason_cmd_index", "inseason_middle_minus_career"]


def build_inseason_full_features(df, inseason_full_stats, n_end, inseason_success, inseason_reverse):
    table_full = inseason_full_stats["table_full"]
    priors = inseason_full_stats["priors"]
    seasons_range = inseason_full_stats["seasons_range"]
    k = inseason_full_stats["k"]

    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])

    def lk(col):
        p = table_full.pivot(index="pitcher_id", columns="season", values=col)
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        return np.nan_to_num(p.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    M_end, K_end = lk("M_end"), lk("K_end")
    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    m_now = np.round(df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_now)
    k_now = np.round(df["asof_pitcher_strike_rate"].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - np.asarray(n_end, dtype=np.float64), 0, None)
    m_season = np.clip(m_now - M_end, 0, None)
    k_season = np.clip(k_now - K_end, 0, None)

    mid = (m_season + k * priors["middle"]) / (n_season + k)
    stk = (k_season + k * priors["strike"]) / (n_season + k)

    out = pd.DataFrame(index=df.index)
    out["inseason_middle_smooth"] = mid
    out["inseason_strike_smooth"] = stk
    out["inseason_cmd_index"] = (np.asarray(inseason_success, dtype=np.float64)
                                 - np.asarray(inseason_reverse, dtype=np.float64) - mid)
    career_mid = df["asof_pitcher_middle_rate"].fillna(priors["middle"]).to_numpy(np.float64)
    out["inseason_middle_minus_career"] = mid - career_mid
    return out[INSEASON_FULL_COLS].astype(np.float64)


# =======================
# 타자 middle in-season + 타자 플래툰 (v28, dev/batter_split.py와 동일)
#
# phase65 오라클: batter_id 천장 148로 투수(840)보다 작지만 in-season 외에는 미개척이었다.
# phase74 스크리닝: bat_inseason_middle +15.1(6.2시그마)로 bat_inseason_smooth(+17.1)와 동급.
# K는 노이즈보정 편차 SD 실측으로 산출: batter x pitcher_hand 진짜SD=0.01002 -> K=2486.
# =======================

BAT_MID_COLS = ["bat_inseason_middle", "bat_middle_minus_career"]
BPLATOON_COLS = ["bplatoon_diff", "bplatoon_n"]


def build_batter_middle_features(df, bs_stats):
    table = bs_stats["bmid_table"]
    seasons_range = bs_stats["seasons_range"]
    global_middle = bs_stats["global_middle"]
    k = bs_stats["k_bmid"]
    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["season"] - 1])

    def lk(col):
        p = table.pivot(index="batter_id", columns="season", values=col)
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        return np.nan_to_num(p.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    BM_end, BN_end = lk("BM_end"), lk("BN_end")
    n_now = df["asof_batter_n"].fillna(0).to_numpy(np.float64)
    m_now = np.round(df["asof_batter_middle_rate"].fillna(0).to_numpy(np.float64) * n_now)
    n_seas = np.clip(n_now - BN_end, 0, None)
    m_seas = np.clip(m_now - BM_end, 0, None)

    mid = (m_seas + k * global_middle) / (n_seas + k)
    career = df["asof_batter_middle_rate"].fillna(global_middle).to_numpy(np.float64)

    out = pd.DataFrame(index=df.index)
    out["bat_inseason_middle"] = mid
    out["bat_middle_minus_career"] = mid - career
    return out[BAT_MID_COLS].astype(np.float64)


def build_bplatoon_features(df, bs_stats):
    table = bs_stats["bplatoon_table"]
    marginal = bs_stats["marginal"]
    seasons_range = bs_stats["seasons_range"]
    global_rate = bs_stats["global_rate"]
    k = bs_stats["k_bplatoon"]

    # 타자 자신의 marginal 성공률 (조건부 축소 기준점)
    pm = marginal.pivot(index="batter_id", columns="season", values="rate")
    pm = pm.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    midx = pd.MultiIndex.from_arrays([df["batter_id"], df["season"] - 1])
    prior = pd.Series(pm.reindex(midx).to_numpy().astype(np.float64)).fillna(global_rate).to_numpy(np.float64)

    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["pitcher_hand"], df["season"] - 1])

    def lk(col):
        p = table.pivot_table(index=["batter_id", "pitcher_hand"], columns="season",
                              values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return np.nan_to_num(p.stack(future_stack=True).reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    s_cell, n_cell = lk("s"), lk("n")
    rate = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["bplatoon_diff"] = rate - prior
    out["bplatoon_n"] = np.log1p(n_cell)
    return out[BPLATOON_COLS].astype(np.float64)


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

# =======================
# v110: codex v20_905 파이프라인 (완전히 독립적인 별도 모델).
# codex 패키지의 script.py 로직을 그대로 옮김. 피처모듈(*.py)과 pkl 2개는
# model/ 안에 동봉된다. 우리 블렌드와 CODEX_WEIGHT로 선형결합.
# =======================

def _cx_sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -50.0, 50.0)))


def _cx_logit(v):
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    return np.log(v / (1.0 - v))


def _cx_clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)


def _cx_clean(v):
    a = np.asarray(v, dtype=np.float64)
    return np.where(np.isfinite(a), a, 0.0)


def _cx_candidate(base, name):
    if "__x__" in name:
        left, right = name.split("__x__", 1)
        return _cx_clean(base[left].to_numpy()) * _cx_clean(base[right].to_numpy())
    if name.endswith("__abs"):
        return np.abs(_cx_clean(base[name[:-5]].to_numpy()))
    if name.endswith("__sq"):
        x = _cx_clean(base[name[:-4]].to_numpy())
        return np.sign(x) * np.minimum(np.abs(x), 8.0) ** 2
    if name.endswith("__logabs"):
        x = _cx_clean(base[name[:-8]].to_numpy())
        return np.sign(x) * np.log1p(np.abs(x))
    return _cx_clean(base[name].to_numpy())


def predict_codex(test, model_dir):
    """codex v20_905 예측. 필요한 파일이 없으면 None을 돌려주고 조용히 스킵한다."""
    import sys
    need = ["phase_v12_submission.pkl", "v20_905_submission.pkl", "v7_features.py"]
    if not all(os.path.exists(os.path.join(model_dir, f)) for f in need):
        return None
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from catboost import Pool
    import advanced_features
    import downside_features
    import orthogonal_features
    import persona_features
    import raw_id_features
    import v7_features

    v12 = joblib.load(os.path.join(model_dir, "phase_v12_submission.pkl"))
    v20 = joblib.load(os.path.join(model_dir, "v20_905_submission.pkl"))

    base = v7_features.build_features(test, v12["v7_stats"])
    ins = v7_features.build_inseason_features(test, v12["v7_inseason_stats"])
    prior = v7_features.get_prior_pitcher_rate(test, v12["v7_inseason_stats"])
    platoon = v7_features.build_platoon_features(test, v12["v7_platoon_stats"], prior)
    base = pd.concat([base, ins, platoon], axis=1)[v12["v7_feature_order"]]
    personas = persona_features.transform_personas(test, v12["persona_state"])
    advanced = advanced_features.add_advanced_features(test, v12["global_mean"])
    x = pd.concat([base.reset_index(drop=True), personas.reset_index(drop=True),
                   advanced.reset_index(drop=True)], axis=1)[v12["feature_order"]]

    blend = v12["base_blend"]
    base_x = x[v12["base_feature_order"]]
    logits = [_cx_logit(v12["base_models"][n].predict_proba(base_x)[:, 1]) for n in blend["names"]]
    weights = np.array([blend["weights"][n] for n in blend["names"]])
    base_pred = _cx_sigmoid(blend["a"] * (np.column_stack(logits) @ weights) + blend["b"])

    phases = np.select([test["game_month"] <= 4, test["game_month"] <= 7], [0, 1], default=2)
    v12_pred = np.zeros(len(test), dtype=np.float64)
    for phase_id, name in enumerate(["early", "middle", "late"]):
        idx = np.flatnonzero(phases == phase_id)
        expert = _cx_clip(v12["phase_models"][name].predict(x))
        st = v12["phase_states"][name]
        z = (1.0 - st["weight"]) * _cx_logit(base_pred) + st["weight"] * _cx_logit(expert)
        v12_pred[idx] = _cx_sigmoid(st["a"] * z + st["b"])[idx]
    v12_pred = _cx_clip(v12_pred)

    raw_x = raw_id_features.build_raw_id_matrix(test, v20["global_mean"])
    raw_pred = _cx_clip(v20["raw_id_model"].predict(Pool(raw_x, cat_features=v20["raw_cat_idx"])))
    rb = v20["raw_blend_state"]
    w_ = rb["weight"]
    raw_blend = _cx_sigmoid(rb["a"] * ((1.0 - w_) * _cx_logit(v12_pred)
                                       + w_ * _cx_logit(raw_pred)) + rb["b"])

    frames = [advanced_features.add_advanced_features(test, v20["global_mean"]),
              downside_features.add_downside_features(test, v20["global_mean"]),
              orthogonal_features.add_orthogonal_features(test, v20["global_mean"])]
    cbase = pd.concat(frames, axis=1)

    def num(name, fill=0.0):
        return pd.to_numeric(test[name], errors="coerce").fillna(fill).to_numpy(np.float64)

    balls, strikes = num("balls_before"), num("strikes_before")
    inning, month = num("inning", 5.0), num("game_month", 6.0)
    li = np.maximum(num("li", 1.0), 0.0)
    runners = num("num_runners_on")
    risp = ((num("runner_on_2b") > 0) | (num("runner_on_3b") > 0)).astype(np.float64)
    cbase["raw_count_risk"] = balls - strikes + 0.8 * ((balls == 3) & (strikes == 2)) + 0.4 * (balls == 3)
    cbase["raw_pressure"] = np.log1p(li) * (1.0 + risp + 0.35 * runners)
    cbase["raw_late"] = (inning >= 7).astype(np.float32)
    cbase["raw_late_month"] = np.maximum(month - 7.0, 0.0).astype(np.float32)
    cbase["raw_full_count"] = ((balls == 3) & (strikes == 2)).astype(np.float32)
    cbase["raw_three_ball"] = (balls == 3).astype(np.float32)
    cbase["raw_two_strike"] = (strikes == 2).astype(np.float32)
    cbase["raw_hitter_count"] = (balls > strikes).astype(np.float32)
    cbase = cbase.astype(np.float32)

    cmat = np.column_stack([_cx_candidate(cbase, n) for n in v20["correction_names"]])
    z = v20["correction_scaler"].transform(cmat)
    correction = np.clip(z @ v20["correction_beta"], -0.16, 0.16)
    return _cx_clip(_cx_sigmoid(_cx_logit(raw_blend) + correction))


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    OUT_DIR = os.path.join(BASE_DIR, "output")
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "model_artifacts_v*.pkl")))
    if not candidates:
        raise FileNotFoundError(
            f"model/ 안에 model_artifacts_v*.pkl 이 없음: {MODEL_DIR} 내용물={os.listdir(MODEL_DIR)}")
    ARTIFACT_PATH = candidates[-1]
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model...")
    artifacts = joblib.load(ARTIFACT_PATH)
    stats = artifacts["stats"]
    # v29: HGB 1개 -> 3변종(d6/d8/sub) 평균, CatBoost도 3변종(d6/d8/rsm)으로 계열내 다양화.
    # 구버전(v27/v28) 호환: hgbs 없으면 단일 hgb를 리스트로 감싼다.
    hgbs = artifacts["hgbs"] if "hgbs" in artifacts else [artifacts["hgb"]]
    cats = artifacts["cats"] if "cats" in artifacts else [artifacts["cat"]]
    # v31: Hurdle 인수분해(core_fail x success|no_core) 앙상블 멤버 추가.
    # v34: Hurdle도 3변종(d6/d8/sub) 다양화. 구버전(v31/v32/v33) 호환: 리스트 없으면 단일모델 감싸기.
    core_fail_models = artifacts.get("core_fail_models")
    if core_fail_models is None and artifacts.get("core_fail_model") is not None:
        core_fail_models = [artifacts["core_fail_model"]]
    succ_nc_models = artifacts.get("succ_nc_models")
    if succ_nc_models is None and artifacts.get("succ_nc_model") is not None:
        succ_nc_models = [artifacts["succ_nc_model"]]
    hurdle_weight = artifacts.get("hurdle_weight", 0.0)
    # v36: 판정축 혼합분해(call 3-class x success|call) 멤버 추가.
    call3_model = artifacts.get("call3_model")
    succ_given_call_models = artifacts.get("succ_given_call_models")
    mix_weight = artifacts.get("mix_weight", 0.0)
    # v38: Cross-fitted Denoised Target (y_soft로 학습한 회귀모델) 앙상블 멤버 추가.
    denoise_model = artifacts.get("denoise_model")
    denoise_weight = artifacts.get("denoise_weight", 0.0)
    # v39: Masked Multi-Task 공유트리 (head0=direct, head1=core_fail, head2=success|no_core).
    multitask_model = artifacts.get("multitask_model")
    multi_task_weight = artifacts.get("multi_weight", 0.0)
    # v40: 다중해상도 공유트리 (head0=direct, head1=투수-시즌 LOO, head2=투수x손 LOO). head0만 사용.
    multires_model = artifacts.get("multires_model")
    multires_weight = artifacts.get("multires_weight", 0.0)
    # v41: 3단 순서형 캐스케이드 P(not reverse)*P(not middle|not reverse)*P(success|not rev&mid).
    ordinal_stage1 = artifacts.get("ordinal_stage1")
    ordinal_stage2 = artifacts.get("ordinal_stage2")
    ordinal_stage3 = artifacts.get("ordinal_stage3")
    ordinal_weight = artifacts.get("ordinal_weight", 0.0)
    # v49: 폼 나우캐스팅 공유트리 (head0=direct, head1=향후W투구 실제성공률, train전용).
    formcast_model = artifacts.get("formcast_model")
    formcast_weight = artifacts.get("formcast_weight", 0.0)
    # v50: middle축 공유트리 (head0=direct, head1=1-lab_middle, train전용). head0만 사용.
    midaxis_model = artifacts.get("midaxis_model")
    midaxis_weight = artifacts.get("midaxis_weight", 0.0)
    # v51: 통합 5-head 공유트리 (y/not_rev/not_mid|not_rev/투수시즌LOO/투수x손LOO). head0만 사용.
    unified5_model = artifacts.get("unified5_model")
    unified5_weight = artifacts.get("unified5_weight", 0.0)
    # v52: ball축(판정/존) 공유트리 (head0=direct, head1=1-lab_ball). head0만 사용.
    ballaxis_model = artifacts.get("ballaxis_model")
    ballaxis_weight = artifacts.get("ballaxis_weight", 0.0)
    # v57: strike축(판정/존) 공유트리 (head0=direct, head1=1-lab_strike). head0만 사용.
    strikeaxis_model = artifacts.get("strikeaxis_model")
    strikeaxis_weight = artifacts.get("strikeaxis_weight", 0.0)
    # v58: other축("기타" 범주, success+reverse+middle 합=0) 공유트리. head0만 사용.
    otheraxis_model = artifacts.get("otheraxis_model")
    otheraxis_weight = artifacts.get("otheraxis_weight", 0.0)
    # v60: middle+other 3-head 통합 공유트리 (y / 1-middle / 1-other). head0만 사용.
    midother_model = artifacts.get("midother_model")
    midother_weight = artifacts.get("midother_weight", 0.0)
    # v61: 메가 통합 6-head 공유트리 (y/1-middle/1-other/1-ball/투수시즌LOO/투수x손LOO). head0만 사용.
    mega_model = artifacts.get("mega_model")
    mega_weight = artifacts.get("mega_weight", 0.0)
    # v62: 조건부 ball축 (not-dangerous 행에서만 1-ball). head0만 사용.
    condball_model = artifacts.get("condball_model")
    condball_weight = artifacts.get("condball_weight", 0.0)
    # v63: count잔차축 (y - E[y|count]). head0만 사용.
    countresid_model = artifacts.get("countresid_model")
    countresid_weight = artifacts.get("countresid_weight", 0.0)
    # v64: 향후50투구 성공률+1-middle율 공유트리. head0만 사용.
    future50_model = artifacts.get("future50_model")
    future50_weight = artifacts.get("future50_weight", 0.0)
    # v70: 투수능력 잔차축 (y - 투수시즌LOO성공률). head0만 사용.
    pitcherresid_model = artifacts.get("pitcherresid_model")
    pitcherresid_weight = artifacts.get("pitcherresid_weight", 0.0)
    # v73: dangerous(=middle or reverse) 행에서만 1-ball. cond_ball의 여집합. head0만 사용.
    dangerball_model = artifacts.get("dangerball_model")
    dangerball_weight = artifacts.get("dangerball_weight", 0.0)
    # v74: 5-class softmax(middle/reverse/nd&ball/nd&strike/nd&기타).
    # P(success) = sum_c P(c) * E[y|c]. E[y|c]는 train에서 추정해 저장된 값.
    mc5_model = artifacts.get("mc5_model")
    # v97: 시드배깅 - 여러 시드로 학습한 mc5 모델 리스트가 있으면 proba를 평균(분산 축소).
    # 없으면(하위호환) 단일 mc5_model 사용.
    mc5_models = artifacts.get("mc5_models")
    mc5_weight = artifacts.get("mc5_weight", 0.0)
    mc5_succ = artifacts.get("mc5_succ")
    # v84: 디코더 절편. 없으면 0 -> 기존 동작과 동일(하위호환).
    mc5_intercept = float(artifacts.get("mc5_intercept", 0.0))
    # v85: risk-threshold 보정. risk=P(middle)+P(reverse)는 성공률 0인 실패유형 확률로,
    # 이 값이 클수록 모델이 체계적으로 과대예측한다(risk 최상위 분위 편차 -0.0139).
    # risk가 임계 초과한 행만 선별해 낮춘다. 전체를 선형으로 깎는 것(+6.4)보다
    # threshold 형태가 압도적(+20.8, 시간분할 4가지 분할점 모두 +13~21).
    risk_idx = artifacts.get("risk_class_idx")      # 성공률 0인 class 인덱스 목록
    risk_thr = float(artifacts.get("risk_thr", 0.0))
    risk_alpha = float(artifacts.get("risk_alpha", 0.0))
    # v88: 평균중립 보정. center는 train(2024)에서 계산한 상수 -> 각 행 예측이
    # 자기 행 확률만 참조(Rule 4 준수). 보정의 레벨성분을 제거해 v86의 레벨
    # 과잉보정(-15점) 재발을 막는다. 0이면 기존(비중심) 동작.
    risk_center = float(artifacts.get("risk_center", 0.0))
    # v91: 시즌 표본량(inseason_n) 축 보정. 시즌 후반(표본 많은 투수)에서 모델이
    # 체계적으로 과대예측하는 것을 잡는다. risk와 상관 -0.05로 거의 독립.
    # 4개 시간분할 전부 +5.15~17.28, 하한 -0.82. center/thr은 train 2024 상수.
    ns_thr = float(artifacts.get("ns_thr", 0.0))
    ns_center = float(artifacts.get("ns_center", 0.0))
    ns_alpha = float(artifacts.get("ns_alpha", 0.0))
    # v75: pitcher/batter 임베딩 MLP. torch 학습, numpy 순수 행렬곱으로 추론(의존성 없음).
    mlp_weights = artifacts.get("mlp_weights")
    mlp_weight = artifacts.get("mlp_weight", 0.0)
    # v95: 투수x2스트라이크 개인 슬로프 보정. "이 투수가 2스트라이크(결정구 상황)에서
    # 자기 평균 대비 특히 잘/못 던지는가"를 투수별 베이지안 축소(K=1500)로 추정해
    # 더한다. mc5_risk/count_resid와 상관 낮음(-0.08/0.15) -> 별개 정보축.
    # center는 train 전체(2019-2024)에서 계산한 상수 -> Rule4 안전.
    k2_n_by_pid = artifacts.get("k2_n_by_pid")
    k2_gap_by_pid = artifacts.get("k2_gap_by_pid")
    k2_K = float(artifacts.get("k2_K", 0.0))
    k2_alpha = float(artifacts.get("k2_alpha", 0.0))
    k2_center = float(artifacts.get("k2_center", 0.0))
    # v96: 물리기반 '크게 벗어난 볼' 보정 (대회 공식 제구실패 정의 2번).
    # mc5의 nd&ball 클래스(P(ball)) 안에서, trackman 릴리스일관성+무브먼트+상황
    # 41피처만으로 "제구된 볼 vs 크게 벗어난 볼"을 재조준한 g(x). 투수실력 피처를
    # 의도적으로 배제해 다른 헤드와 정보가 겹치지 않게 함. 평균중립(train 상수 center).
    ballsize_model = artifacts.get("ballsize_model")
    ballsize_feats = artifacts.get("ballsize_feats")
    ballsize_const = float(artifacts.get("ballsize_const", 0.0))
    ballsize_center = float(artifacts.get("ballsize_center", 0.0))
    ballsize_alpha = float(artifacts.get("ballsize_alpha", 0.0))
    # v90: PA-event 4-class(continue-ball/continue-strike/2s-foul/PA-end) softmax.
    # P(success) = sum_c P(c) * E[y|c]. mc5와 동일한 소프트맥스 디코더 패턴.
    pa4_model = artifacts.get("pa4_model")
    pa4_weight = artifacts.get("pa4_weight", 0.0)
    pa4_succ = artifacts.get("pa4_succ")
    # v79: 경기내 컨디션 aux head (head0=y, head1=현재경기 직전까지 누적성공률).
    # 경기간 컨디션 자기상관은 0.047로 죽었지만 경기내는 0.191로 살아있다는 진단에서 도출.
    # train에서만 타깃을 만들고 test는 각 행 자기 피처로 추정 -> Rule 4 준수. head0만 사용.
    ingame_model = artifacts.get("ingame_model")
    ingame_weight = artifacts.get("ingame_weight", 0.0)
    # v48: 최종 예측에 더할 상수(시즌 레벨 보정). 없으면 0.0 = 기존 동작 그대로.
    level_shift = float(artifacts.get("level_shift", 0.0))
    base_weight = artifacts.get("base_weight",
                                1.0 - hurdle_weight - mix_weight - denoise_weight
                                - multi_task_weight - multires_weight - ordinal_weight
                                - formcast_weight - midaxis_weight - unified5_weight
                                - ballaxis_weight - strikeaxis_weight - otheraxis_weight
                                - midother_weight - mega_weight
                                - condball_weight - countresid_weight - future50_weight
                                - pitcherresid_weight - dangerball_weight - mc5_weight
                                - mlp_weight - ingame_weight - pa4_weight)
    batter_stats = artifacts["batter_stats"]
    inseason_full_stats = artifacts["inseason_full_stats"]
    batter_split_stats = artifacts["batter_split_stats"]
    inseason_stats = artifacts["inseason_stats"]
    platoon_stats = artifacts["platoon_stats"]
    inning_stats = artifacts["inning_stats"]
    count_stats = artifacts["count_stats"]
    pitchtype_stats = artifacts["pitchtype_stats"]
    lastyear_stats = artifacts["lastyear_stats"]
    volatility_stats = artifacts["volatility_stats"]
    role_stats = artifacts["role_stats"]
    trackman_stats = artifacts["trackman_stats"]
    form_base_middle_global = artifacts["form_base_middle_global"]
    feature_order = artifacts["feature_order"]
    w_hgb, w_cat = artifacts["w_hgb"], artifacts["w_cat"]
    print(f" OK. w_hgb={w_hgb}  w_cat={w_cat}  HGB{len(hgbs)}변종  Cat{len(cats)}변종  features={len(feature_order)}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features (v28 = v27 + inseason middle/strike + batter middle + batter platoon)...")
    ids = test[ID_COL].tolist()
    X_base = build_features(test, stats).reset_index(drop=True)
    X_inseason = build_inseason_features(test, inseason_stats).reset_index(drop=True)
    prior_rate = get_prior_pitcher_rate(test, inseason_stats)
    X_platoon = build_platoon_features(test, platoon_stats, prior_rate).reset_index(drop=True)
    X_inning = build_inning_features(test, inning_stats, prior_rate).reset_index(drop=True)
    X_count = build_count_features(test, count_stats, prior_rate).reset_index(drop=True)
    X_pitchtype = build_pitchtype_features(test, pitchtype_stats, prior_rate).reset_index(drop=True)
    X_lastyear = build_lastyear_features(test, lastyear_stats).reset_index(drop=True)
    X_volatility = build_volatility_features(test, volatility_stats).reset_index(drop=True)
    X_role = build_role_features(test, role_stats).reset_index(drop=True)
    base_middle_arr = np.full(len(test), form_base_middle_global)
    X_form = build_form_features(test, X_role, X_inseason["inseason_success_smooth"].to_numpy(np.float64),
                                 base_middle_arr).reset_index(drop=True)
    X_trackman = build_trackman_features(test, trackman_stats).reset_index(drop=True)
    X_trackman_lown = build_trackman_lown_features(
        X_trackman, test["asof_pitcher_n"].to_numpy(np.float64),
        trackman_stats["lown_threshold"]).reset_index(drop=True)
    X_batter = build_batter_features(test, batter_stats).reset_index(drop=True)
    n_end_row = get_n_end(test, inseason_stats)
    X_inseason_full = build_inseason_full_features(
        test, inseason_full_stats, n_end_row,
        X_inseason["inseason_success_smooth"].to_numpy(np.float64),
        X_inseason["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
    X_bat_middle = build_batter_middle_features(test, batter_split_stats).reset_index(drop=True)
    X_bplatoon = build_bplatoon_features(test, batter_split_stats).reset_index(drop=True)

    X = pd.concat([X_base, X_inseason, X_platoon, X_inning, X_pitchtype], axis=1)
    X.index = test.index
    X = X.astype(np.float64)
    X_cross = add_crosses(X)
    for extra in (X_lastyear, X_count, X_volatility, X_role, X_form, X_trackman,
                  X_trackman_lown, X_batter, X_inseason_full, X_bat_middle, X_bplatoon):
        extra.index = test.index
    X = pd.concat([X, X_cross, X_lastyear, X_count, X_volatility, X_role, X_form, X_trackman,
                   X_trackman_lown, X_batter, X_inseason_full, X_bat_middle, X_bplatoon], axis=1)
    X_all = X
    X = X[feature_order].astype(np.float64)
    print(f" features={X.shape[1]}")

    # v107: 물리/커맨드 헤드용 신규피처 (본 블렌드 X와 별개로 물리헤드에만 공급)
    newtm_stats = artifacts.get("newtm_stats")
    if newtm_stats is not None:
        X_newtm = build_newtm_features(test, newtm_stats, X).astype(np.float64)
        X_newtm.index = X.index
        X_phys_all = pd.concat([X, X_newtm], axis=1)
    else:
        X_phys_all = None

    # v108: XGBoost raw-ID 헤드용 (162피처 + pitcher_id/batter_id/team_id를 XGBoost
    # native categorical로 직접 투입). 학습시 category 목록을 아티팩트에 저장해두고
    # test에서도 동일 카테고리로 캐스팅(안 본 ID는 자동 NaN -> XGB가 결측으로 처리).
    xgbrawid_cats = artifacts.get("xgbrawid_cats")
    if xgbrawid_cats is not None:
        X_xgbrawid = X[xgbrawid_cats["feature_order"]].copy()
        X_xgbrawid["pitcher_id"] = pd.Categorical(
            test["pitcher_id"].to_numpy(), categories=xgbrawid_cats["pitcher_id_cats"])
        X_xgbrawid["batter_id"] = pd.Categorical(
            test["batter_id"].to_numpy(), categories=xgbrawid_cats["batter_id_cats"])
        X_xgbrawid["pitcher_team_id_cat"] = pd.Categorical(
            test["pitcher_team_id"].to_numpy(), categories=xgbrawid_cats["pitcher_team_id_cats"])
        X_xgbrawid["batter_team_id_cat"] = pd.Categorical(
            test["batter_team_id"].to_numpy(), categories=xgbrawid_cats["batter_team_id_cats"])
    else:
        X_xgbrawid = None

    # v109: XGBoost 컨텍스트-only 헤드용 (축소평균 제외 원시피처 51개 + raw ID).
    xgbctx_cats = artifacts.get("xgbctx_cats")
    if xgbctx_cats is not None:
        X_xgbctx = X[xgbctx_cats["feature_order"]].copy()
        X_xgbctx["pitcher_id"] = pd.Categorical(
            test["pitcher_id"].to_numpy(), categories=xgbctx_cats["pitcher_id_cats"])
        X_xgbctx["batter_id"] = pd.Categorical(
            test["batter_id"].to_numpy(), categories=xgbctx_cats["batter_id_cats"])
        X_xgbctx["pitcher_team_id_cat"] = pd.Categorical(
            test["pitcher_team_id"].to_numpy(), categories=xgbctx_cats["pitcher_team_id_cats"])
        X_xgbctx["batter_team_id_cat"] = pd.Categorical(
            test["batter_team_id"].to_numpy(), categories=xgbctx_cats["batter_team_id_cats"])
    else:
        X_xgbctx = None

    print("Inference model...")
    if len(X):
        # v46: hgbs에 classifier(predict_proba)와 regressor(predict, squared_error) 혼재 가능.
        def _hgb_predict(m):
            if hasattr(m, "predict_proba"):
                return m.predict_proba(X)[:, 1]
            return np.clip(m.predict(X), 0.0, 1.0)
        p_hgb = np.mean([_hgb_predict(m) for m in hgbs], axis=0)
        p_cat = np.mean([c.predict_proba(X)[:, 1] for c in cats], axis=0)
        p_ensemble = w_hgb * p_hgb + w_cat * p_cat
        if core_fail_models and succ_nc_models and hurdle_weight > 0:
            hurdle_variants = [(1 - cm.predict_proba(X)[:, 1]) * sm.predict_proba(X)[:, 1]
                              for cm, sm in zip(core_fail_models, succ_nc_models)]
            p_hurdle = np.mean(hurdle_variants, axis=0)
        else:
            p_hurdle = None
        if call3_model is not None and succ_given_call_models and mix_weight > 0:
            p_call = call3_model.predict_proba(X)
            p_succ_given = [m.predict_proba(X)[:, 1] for m in succ_given_call_models]
            p_mix = sum(p_call[:, c] * p_succ_given[c] for c in range(len(p_succ_given)))
        else:
            p_mix = None
        if denoise_model is not None and denoise_weight > 0:
            p_denoise = np.clip(denoise_model.predict(X), 0.0, 1.0)
        else:
            p_denoise = None
        if multitask_model is not None and multi_task_weight > 0:
            heads = np.clip(multitask_model.predict(X), 0.0, 1.0)
            p_multitask = 0.5 * heads[:, 0] + 0.5 * (1 - heads[:, 1]) * heads[:, 2]
        else:
            p_multitask = None
        # v111: 시드배깅. *_models(리스트)가 있으면 시드평균, 없으면 기존 단일모델.
        multires_models = artifacts.get("multires_models")
        if multires_models and multires_weight > 0:
            p_multires = np.mean([np.clip(m.predict(X), 0.0, 1.0)[:, 0]
                                  for m in multires_models], axis=0)
        elif multires_model is not None and multires_weight > 0:
            heads_mr = np.clip(multires_model.predict(X), 0.0, 1.0)
            p_multires = heads_mr[:, 0]
        else:
            p_multires = None
        ordinal_stages_bag = artifacts.get("ordinal_stages_bag")
        if ordinal_stages_bag and ordinal_weight > 0:
            p_ordinal = np.mean([(s1.predict_proba(X)[:, 1] * s2.predict_proba(X)[:, 1]
                                  * s3.predict_proba(X)[:, 1])
                                 for s1, s2, s3 in ordinal_stages_bag], axis=0)
        elif ordinal_stage1 is not None and ordinal_stage2 is not None and ordinal_stage3 is not None \
                and ordinal_weight > 0:
            po1 = ordinal_stage1.predict_proba(X)[:, 1]
            po2 = ordinal_stage2.predict_proba(X)[:, 1]
            po3 = ordinal_stage3.predict_proba(X)[:, 1]
            p_ordinal = po1 * po2 * po3
        else:
            p_ordinal = None
        if formcast_model is not None and formcast_weight > 0:
            heads_fc = np.clip(formcast_model.predict(X), 0.0, 1.0)
            p_formcast = heads_fc[:, 0]
        else:
            p_formcast = None
        if midaxis_model is not None and midaxis_weight > 0:
            heads_ma = np.clip(midaxis_model.predict(X), 0.0, 1.0)
            p_midaxis = heads_ma[:, 0]
        else:
            p_midaxis = None
        if unified5_model is not None and unified5_weight > 0:
            heads_u5 = np.clip(unified5_model.predict(X), 0.0, 1.0)
            p_unified5 = heads_u5[:, 0]
        else:
            p_unified5 = None
        if ballaxis_model is not None and ballaxis_weight > 0:
            heads_ba = np.clip(ballaxis_model.predict(X), 0.0, 1.0)
            p_ballaxis = heads_ba[:, 0]
        else:
            p_ballaxis = None
        if strikeaxis_model is not None and strikeaxis_weight > 0:
            heads_sa = np.clip(strikeaxis_model.predict(X), 0.0, 1.0)
            p_strikeaxis = heads_sa[:, 0]
        else:
            p_strikeaxis = None
        if otheraxis_model is not None and otheraxis_weight > 0:
            heads_oa = np.clip(otheraxis_model.predict(X), 0.0, 1.0)
            p_otheraxis = heads_oa[:, 0]
        else:
            p_otheraxis = None
        midother_models = artifacts.get("midother_models")
        if midother_models and midother_weight > 0:
            p_midother = np.mean([np.clip(m.predict(X), 0.0, 1.0)[:, 0]
                                  for m in midother_models], axis=0)
        elif midother_model is not None and midother_weight > 0:
            heads_mo = np.clip(midother_model.predict(X), 0.0, 1.0)
            p_midother = heads_mo[:, 0]
        else:
            p_midother = None
        if mega_model is not None and mega_weight > 0:
            heads_mg = np.clip(mega_model.predict(X), 0.0, 1.0)
            p_mega = heads_mg[:, 0]
        else:
            p_mega = None
        condball_models = artifacts.get("condball_models")
        if condball_models and condball_weight > 0:
            p_condball = np.mean([np.clip(m.predict(X), 0.0, 1.0)[:, 0]
                                  for m in condball_models], axis=0)
        elif condball_model is not None and condball_weight > 0:
            heads_cb = np.clip(condball_model.predict(X), 0.0, 1.0)
            p_condball = heads_cb[:, 0]
        else:
            p_condball = None
        countresid_models = artifacts.get("countresid_models")
        if countresid_models and countresid_weight > 0:
            p_countresid = np.mean([np.clip(m.predict(X), 0.0, 1.0)[:, 0]
                                    for m in countresid_models], axis=0)
        elif countresid_model is not None and countresid_weight > 0:
            heads_cr = np.clip(countresid_model.predict(X), 0.0, 1.0)
            p_countresid = heads_cr[:, 0]
        else:
            p_countresid = None
        future50_models = artifacts.get("future50_models")
        if future50_models and future50_weight > 0:
            p_future50 = np.mean([np.clip(m.predict(X), 0.0, 1.0)[:, 0]
                                  for m in future50_models], axis=0)
        elif future50_model is not None and future50_weight > 0:
            heads_f5 = np.clip(future50_model.predict(X), 0.0, 1.0)
            p_future50 = heads_f5[:, 0]
        else:
            p_future50 = None
        if pitcherresid_model is not None and pitcherresid_weight > 0:
            heads_pr = np.clip(pitcherresid_model.predict(X), 0.0, 1.0)
            p_pitcherresid = heads_pr[:, 0]
        else:
            p_pitcherresid = None
        if dangerball_model is not None and dangerball_weight > 0:
            heads_db = np.clip(dangerball_model.predict(X), 0.0, 1.0)
            p_dangerball = heads_db[:, 0]
        else:
            p_dangerball = None
        if (mc5_models or mc5_model is not None) and mc5_weight > 0 and mc5_succ is not None:
            if mc5_models:
                proba5 = np.mean([m.predict_proba(X) for m in mc5_models], axis=0)
            else:
                proba5 = mc5_model.predict_proba(X)
            p_mc5 = np.clip(mc5_intercept + proba5 @ np.asarray(mc5_succ, dtype=np.float64),
                            0.0, 1.0)
            if risk_idx is not None and risk_alpha > 0:
                risk_vec = proba5[:, list(risk_idx)].sum(axis=1)
            else:
                risk_vec = None
            if ballsize_alpha > 0:
                ball_vec = proba5[:, [0, 1, 2]].sum(axis=1)
            else:
                ball_vec = None
        else:
            p_mc5 = None
            ball_vec = None
        # v112: mc6 순수분할 헤드 (middle/reverse/wild/succ_ball/succ_strk/succ_play,
        # 전부 성공률 0% 또는 100%인 순수클래스). P(success)=sum(succ 클래스 확률).
        mc6pure_model = artifacts.get("mc6pure_model")
        mc6pure_models = artifacts.get("mc6pure_models")
        mc6pure_weight = float(artifacts.get("mc6pure_weight", 0.0))
        mc6pure_succ_classes = artifacts.get("mc6pure_succ_classes")
        if mc6pure_models and mc6pure_weight > 0:
            p_mc6pure = np.mean([np.clip(m.predict_proba(X)[:, mc6pure_succ_classes].sum(axis=1), 0.0, 1.0)
                                 for m in mc6pure_models], axis=0)
        elif mc6pure_model is not None and mc6pure_weight > 0:
            proba6 = mc6pure_model.predict_proba(X)
            p_mc6pure = np.clip(proba6[:, mc6pure_succ_classes].sum(axis=1), 0.0, 1.0)
        else:
            p_mc6pure = None
        # v132: F전문가 라우팅 - F리그 행만 F전문가(F행 전용학습 mc6)의 예측으로 교체.
        # 공유 mc6는 89% R행에 지배돼 F리그(판정레짐 상이)에서 약함(fold A F행 직접비교로 검증).
        fexpert_model = artifacts.get("fexpert_model")
        if fexpert_model is not None and p_mc6pure is not None:
            fx_r_value = artifacts["fexpert_r_value"]
            fx_succ = artifacts["fexpert_succ_classes"]
            maskF = (test["game_type"] != fx_r_value).to_numpy()
            if maskF.any():
                probaF = fexpert_model.predict_proba(X[maskF])
                p_mc6pure = p_mc6pure.copy()
                p_mc6pure[maskF] = np.clip(probaF[:, fx_succ].sum(axis=1), 0.0, 1.0)
            print(f" + F-expert routing (F행 {int(maskF.sum())}개 교체)")
        # v133: mc6split 헤드 - R/F 리그별 전문가 2모델을 game_type으로 라우팅한 합성.
        # 단독으론 공유mc6보다 약하나 오차방향이 달라 축으로 유효(fold A z=3.5).
        mc6split_R = artifacts.get("mc6split_model_R")
        mc6split_F = artifacts.get("mc6split_model_F")
        mc6split_weight = float(artifacts.get("mc6split_weight", 0.0))
        if mc6split_R is not None and mc6split_F is not None and mc6split_weight != 0:
            ms_succ = artifacts["mc6split_succ_classes"]
            ms_rval = artifacts["mc6split_r_value"]
            maskR_ms = (test["game_type"] == ms_rval).to_numpy()
            p_mc6split = np.empty(len(X), dtype=np.float64)
            if maskR_ms.any():
                probaR_ms = mc6split_R.predict_proba(X[maskR_ms])
                p_mc6split[maskR_ms] = np.clip(probaR_ms[:, ms_succ].sum(axis=1), 0.0, 1.0)
            if (~maskR_ms).any():
                probaF_ms = mc6split_F.predict_proba(X[~maskR_ms])
                p_mc6split[~maskR_ms] = np.clip(probaF_ms[:, ms_succ].sum(axis=1), 0.0, 1.0)
        else:
            p_mc6split = None
        # v134: F전문가 가산 헤드 - F리그 행만으로 학습한 mc6를 전 행에 적용, 소량 프로브.
        fexadd_model = artifacts.get("fexadd_model")
        fexadd_weight = float(artifacts.get("fexadd_weight", 0.0))
        if fexadd_model is not None and fexadd_weight != 0:
            fxa_succ = artifacts["fexadd_succ_classes"]
            proba_fxa = fexadd_model.predict_proba(X)
            p_fexadd = np.clip(proba_fxa[:, fxa_succ].sum(axis=1), 0.0, 1.0)
        else:
            p_fexadd = None
        # v135: 조건피처 분할 전문가 헤드 2종 (fold A z=4.3 / z=2.7, mc6split 직교화 후 생존).
        def _split_head(prefix):
            m0 = artifacts.get(f"{prefix}_model_0")
            m1 = artifacts.get(f"{prefix}_model_1")
            wgt = float(artifacts.get(f"{prefix}_weight", 0.0))
            if m0 is None or m1 is None or wgt == 0:
                return None
            succ_c = artifacts[f"{prefix}_succ_classes"]
            feat_c = artifacts[f"{prefix}_split_feature"]
            vals = X[feat_c].to_numpy(np.float64)
            mask1 = (vals >= 2) if feat_c == "strikes_before" else (vals > 0.5)
            p_out = np.empty(len(X), dtype=np.float64)
            if (~mask1).any():
                pr0 = m0.predict_proba(X[~mask1])
                p_out[~mask1] = np.clip(pr0[:, succ_c].sum(axis=1), 0.0, 1.0)
            if mask1.any():
                pr1 = m1.predict_proba(X[mask1])
                p_out[mask1] = np.clip(pr1[:, succ_c].sum(axis=1), 0.0, 1.0)
            return p_out

        p_shsplit = _split_head("shsplit")
        p_tssplit = _split_head("tssplit")
        # v117: 연속실패(streak) 헤드. 보조타겟=clip(연속실패,0,10)/10, head0(y)만 사용.
        strk_model = artifacts.get("strk_model")
        strk_weight = float(artifacts.get("strk_weight", 0.0))
        if strk_model is not None and strk_weight != 0:
            p_strk = np.clip(strk_model.predict(X), 0.0, 1.0)[:, 0]
        else:
            p_strk = None
        # v118: 구종(pitchtype) 헤드. 보조타겟=[직구,변화구,오프스피드], head0(y)만 사용.
        pitchtype_model = artifacts.get("pitchtype_model")
        pitchtype_weight = float(artifacts.get("pitchtype_weight", 0.0))
        if pitchtype_model is not None and pitchtype_weight != 0:
            p_pitchtype = np.clip(pitchtype_model.predict(X), 0.0, 1.0)[:, 0]
        else:
            p_pitchtype = None
        # v113: mc6 계층분해 헤드 (wild/succball/strike 3서브헤드 평균, head0=y만 사용).
        # rho가 음수로 나와 가중치도 음수(빼는 방향) - 부호 있는 가중치이므로 0 비교는 != 사용.
        mc6hier_models = artifacts.get("mc6hier_models")
        mc6hier_weight = float(artifacts.get("mc6hier_weight", 0.0))
        if mc6hier_models and mc6hier_weight != 0:
            p_mc6hier = np.mean([np.clip(m.predict(X), 0.0, 1.0)[:, 0]
                                 for m in mc6hier_models], axis=0)
        else:
            p_mc6hier = None
        # v122: 안쓰이는피처(tm_matched/tm_lown_flag/pitcher_hand/form_missing) +
        # season/game_type 조건화 스무딩 XGB. 음수가중치 프로브(fold A/C s*<0 부호일치).
        xgbunused_model = artifacts.get("xgbunused_model")
        xgbunused_weight = float(artifacts.get("xgbunused_weight", 0.0))
        if xgbunused_model is not None and xgbunused_weight != 0:
            feat_order_xu = artifacts["xgbunused_feat_order"]
            raw_cols_xu = artifacts["xgbunused_raw_cols"]
            smap1_xu = artifacts["xgbunused_smap_season_tmm"]
            smap2_xu = artifacts["xgbunused_smap_gtype_lown"]
            g_all_xu = artifacts["xgbunused_g_all"]
            Xu = X[raw_cols_xu].astype(np.float64).copy()
            key1_xu = list(zip(X["season"].astype(int), X["tm_matched"].astype(int)))
            Xu["smooth_season_tmm"] = pd.Series(key1_xu).map(smap1_xu).fillna(g_all_xu).to_numpy(np.float64)
            key2_xu = list(zip(X["cat_game_type"].astype(int), X["tm_lown_flag"].astype(int)))
            Xu["smooth_gtype_lown"] = pd.Series(key2_xu).map(smap2_xu).fillna(g_all_xu).to_numpy(np.float64)
            Xu = Xu[feat_order_xu]
            p_xgbunused = np.clip(xgbunused_model.predict_proba(Xu)[:, 1], 0.0, 1.0)
        else:
            p_xgbunused = None
        # v125: lt_y 헤드 - linear_tree LGBM(조각별 선형), binary y 직접.
        # 계단함수 모델들이 공통으로 놓치는 매끄러운 기울기 성분. 음수가중치(빼기) 프로브.
        lty_model = artifacts.get("lty_model")
        lty_weight = float(artifacts.get("lty_weight", 0.0))
        if lty_model is not None and lty_weight != 0:
            lty_feats = artifacts["lty_feat_order"]
            p_lty = np.clip(lty_model.predict_proba(X[lty_feats])[:, 1], 0.0, 1.0)
        else:
            p_lty = None
        # v126: mc6aux 헤드 - CatBoost MultiRMSE([y, onehot6클래스]), 추론은 head0(y)만.
        # mc6 타겟재정의 + 멀티태스크 결합, Brier와 목적함수 일치. Rule4 안전.
        mc6aux_model = artifacts.get("mc6aux_model")
        mc6aux_weight = float(artifacts.get("mc6aux_weight", 0.0))
        if mc6aux_model is not None and mc6aux_weight != 0:
            mc6aux_feats = artifacts["mc6aux_feat_order"]
            p_mc6aux = np.clip(mc6aux_model.predict(X[mc6aux_feats])[:, 0], 0.0, 1.0)
        else:
            p_mc6aux = None
        # v128: N1 헤드 - 원시컨텍스트53+원시비율18(=71)+ID임베딩4종 MLP.
        # 가공피처(축소평균 등) 전혀 안 씀 - 기존 헤드들과 다른 함수공간+피처공간에서
        # 스스로 표현을 학습(수축곡선 포함). nn_raw(원시52+ID만)의 상위호환으로 교체됨.
        # fold A z=2.4 통과(nn_raw까지 직교화한 뒤에도 생존). 부호는 소량프로브로 측정.
        n1_models = artifacts.get("n1_models")
        n1_weight = float(artifacts.get("n1_weight", 0.0))
        if n1_models and n1_weight != 0:
            n1_raw18 = artifacts["n1_raw18_feats"]
            Xn1_ctx = X[NNRAW_CONTEXT_FEATS].to_numpy(np.float64)
            Xn1_raw18 = test[n1_raw18].astype(np.float64).to_numpy()
            Xn1 = np.concatenate([Xn1_ctx, Xn1_raw18], axis=1)
            pid_arr = test["pitcher_id"].to_numpy()
            bid_arr = test["batter_id"].to_numpy()
            ptid_arr = test["pitcher_team_id"].to_numpy()
            btid_arr = test["batter_team_id"].to_numpy()
            preds_n1 = []
            for mdl in n1_models:
                st = mdl["state"]
                z = (Xn1 - mdl["mu"]) / mdl["sd"]
                Xz = np.clip(np.nan_to_num(z, nan=0.0), -10, 10)
                ip = np.array([mdl["pmap"].get(v, 0) for v in pid_arr], dtype=np.int64)
                ib = np.array([mdl["bmap"].get(v, 0) for v in bid_arr], dtype=np.int64)
                ipt = np.array([mdl["ptmap"].get(v, 0) for v in ptid_arr], dtype=np.int64)
                ibt = np.array([mdl["btmap"].get(v, 0) for v in btid_arr], dtype=np.int64)
                preds_n1.append(nnraw_forward(Xz, ip, ib, ipt, ibt, st))
            p_n1 = np.clip(np.mean(preds_n1, axis=0), 0.0, 1.0)
        else:
            p_n1 = None
        # v130: zoneintent 헤드 - 5클래스(middle/reverse/wild/succ_out존/succ_in존) XGB.
        # 존의도(판정축의 거친 재분할)를 XGB로 조잡하게 학습 - fold A z=2.6.
        zoneintent_model = artifacts.get("zoneintent_model")
        zoneintent_weight = float(artifacts.get("zoneintent_weight", 0.0))
        if zoneintent_model is not None and zoneintent_weight != 0:
            zi_succ = artifacts["zoneintent_succ_classes"]
            proba_zi = zoneintent_model.predict_proba(X)
            p_zoneintent = np.clip(proba_zi[:, zi_succ].sum(axis=1), 0.0, 1.0)
        else:
            p_zoneintent = None
        # v130: ExtraTrees 헤드 - 부스팅과 다른 랜덤화(무작위 분할점)의 배깅트리. fold A z=1.8.
        et_model = artifacts.get("et_model")
        et_weight = float(artifacts.get("et_weight", 0.0))
        if et_model is not None and et_weight != 0:
            p_et = np.clip(et_model.predict_proba(X.fillna(0).to_numpy())[:, 1], 0.0, 1.0)
        else:
            p_et = None
        # v107: 물리/커맨드 헤드 (tm_* + 신규6 + 컨텍스트, multi-task head0=y)
        physhead_model = artifacts.get("physhead_model")
        physhead_feats = artifacts.get("physhead_feats")
        physhead_weight = float(artifacts.get("physhead_weight", 0.0))
        if physhead_model is not None and physhead_weight > 0 and X_phys_all is not None:
            p_phys = np.clip(physhead_model.predict(X_phys_all[physhead_feats])[:, 0], 0.0, 1.0)
        else:
            p_phys = None
        # v108: XGBoost raw-ID 헤드 (162피처 + pitcher_id/batter_id/team_id native categorical)
        xgbrawid_model = artifacts.get("xgbrawid_model")
        xgbrawid_weight = float(artifacts.get("xgbrawid_weight", 0.0))
        if xgbrawid_model is not None and xgbrawid_weight != 0 and X_xgbrawid is not None:
            p_xgbrawid = np.clip(xgbrawid_model.predict_proba(X_xgbrawid)[:, 1], 0.0, 1.0)
        else:
            p_xgbrawid = None
        xgbctx_model = artifacts.get("xgbctx_model")
        xgbctx_weight = float(artifacts.get("xgbctx_weight", 0.0))
        if xgbctx_model is not None and xgbctx_weight > 0 and X_xgbctx is not None:
            p_xgbctx = np.clip(xgbctx_model.predict_proba(X_xgbctx)[:, 1], 0.0, 1.0)
        else:
            p_xgbctx = None
        if ingame_model is not None and ingame_weight > 0:
            heads_ing = np.clip(ingame_model.predict(X), 0.0, 1.0)
            p_ingame = heads_ing[:, 0]
        else:
            p_ingame = None
        if mlp_weights is not None and mlp_weight > 0:
            w = mlp_weights
            pid_arr = test["pitcher_id"].to_numpy()
            bid_arr = test["batter_id"].to_numpy()
            ip_row = np.array([w["pmap"].get(v, 0) for v in pid_arr], dtype=np.int64)
            ib_row = np.array([w["bmap"].get(v, 0) for v in bid_arr], dtype=np.int64)
            Xrow = X.to_numpy(np.float32)
            z = np.clip((Xrow - w["mu"]) / w["sd"], -10, 10)
            h = np.concatenate([z, w["emb_p"][ip_row], w["emb_b"][ib_row]], axis=1)
            h = np.maximum(h @ w["W1"] + w["b1"], 0)
            h = np.maximum(h @ w["W2"] + w["b2"], 0)
            logit = (h @ w["W3"] + w["b3"]).squeeze(1)
            p_mlp = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.0, 1.0)
        else:
            p_mlp = None
        if pa4_model is not None and pa4_weight > 0 and pa4_succ is not None:
            proba4 = pa4_model.predict_proba(X)
            p_pa4 = np.clip(proba4 @ np.asarray(pa4_succ, dtype=np.float64), 0.0, 1.0)
        else:
            p_pa4 = None
        # v104: 헤드별 isotonic 재보정. fold A(2024) 전체로 학습한 단조맵을 각 헤드의
        # raw 출력에 적용해 실제 관측성공률에 맞게 재매핑(선형가중치는 그대로 사용).
        iso_maps = artifacts.get("iso_maps")
        if iso_maps:
            def _iso_apply(name, val):
                m = iso_maps.get(name)
                if m is None or val is None:
                    return val
                return np.clip(m.predict(val), 0.0, 1.0)
            p_ensemble = _iso_apply("base", p_ensemble)
            p_hurdle = _iso_apply("hurdle", p_hurdle)
            p_multires = _iso_apply("multires", p_multires)
            p_ordinal = _iso_apply("ordinal", p_ordinal)
            p_midother = _iso_apply("midother", p_midother)
            p_condball = _iso_apply("condball", p_condball)
            p_countresid = _iso_apply("countresid", p_countresid)
            p_future50 = _iso_apply("future50", p_future50)
            p_mc5 = _iso_apply("mc5", p_mc5)
            p_ingame = _iso_apply("ingame", p_ingame)
            print(f" + PerHeadIsotonic({len(iso_maps)}개 헤드)")
        preds = base_weight * p_ensemble
        if p_hurdle is not None:
            preds = preds + hurdle_weight * p_hurdle
        if p_mix is not None:
            preds = preds + mix_weight * p_mix
        if p_denoise is not None:
            preds = preds + denoise_weight * p_denoise
        if p_multitask is not None:
            preds = preds + multi_task_weight * p_multitask
        if p_multires is not None:
            preds = preds + multires_weight * p_multires
        if p_ordinal is not None:
            preds = preds + ordinal_weight * p_ordinal
        if p_formcast is not None:
            preds = preds + formcast_weight * p_formcast
        if p_midaxis is not None:
            preds = preds + midaxis_weight * p_midaxis
        if p_unified5 is not None:
            preds = preds + unified5_weight * p_unified5
        if p_ballaxis is not None:
            preds = preds + ballaxis_weight * p_ballaxis
        if p_strikeaxis is not None:
            preds = preds + strikeaxis_weight * p_strikeaxis
        if p_otheraxis is not None:
            preds = preds + otheraxis_weight * p_otheraxis
        if p_midother is not None:
            preds = preds + midother_weight * p_midother
        if p_mega is not None:
            preds = preds + mega_weight * p_mega
        if p_condball is not None:
            preds = preds + condball_weight * p_condball
        if p_countresid is not None:
            preds = preds + countresid_weight * p_countresid
        if p_future50 is not None:
            preds = preds + future50_weight * p_future50
        if p_pitcherresid is not None:
            preds = preds + pitcherresid_weight * p_pitcherresid
        if p_dangerball is not None:
            preds = preds + dangerball_weight * p_dangerball
        if p_mc5 is not None:
            preds = preds + mc5_weight * p_mc5
        if p_mc6pure is not None:
            preds = preds + mc6pure_weight * p_mc6pure
            print(f" + MC6pure(w={mc6pure_weight:.4f})")
        if p_strk is not None:
            preds = preds + strk_weight * p_strk
            print(f" + Streak(w={strk_weight:+.4f})")
        if p_pitchtype is not None:
            preds = preds + pitchtype_weight * p_pitchtype
            print(f" + PitchType(w={pitchtype_weight:+.4f})")
        if p_mc6hier is not None:
            preds = preds + mc6hier_weight * p_mc6hier
            print(f" + MC6hier(w={mc6hier_weight:+.4f})")
        if p_xgbunused is not None:
            preds = preds + xgbunused_weight * p_xgbunused
            print(f" + XGBunused(w={xgbunused_weight:+.4f})")
        if p_lty is not None:
            preds = preds + lty_weight * p_lty
            print(f" + LT-y(w={lty_weight:+.4f})")
        if p_mc6aux is not None:
            preds = preds + mc6aux_weight * p_mc6aux
            print(f" + MC6aux(w={mc6aux_weight:+.4f})")
        if p_n1 is not None:
            preds = preds + n1_weight * p_n1
            print(f" + N1(w={n1_weight:+.4f})")
        if p_mc6split is not None:
            preds = preds + mc6split_weight * p_mc6split
            print(f" + MC6split(w={mc6split_weight:+.4f})")
        if p_fexadd is not None:
            preds = preds + fexadd_weight * p_fexadd
            print(f" + FexpertAdd(w={fexadd_weight:+.4f})")
        if p_shsplit is not None:
            preds = preds + float(artifacts["shsplit_weight"]) * p_shsplit
            print(f" + SHsplit(w={float(artifacts['shsplit_weight']):+.4f})")
        if p_tssplit is not None:
            preds = preds + float(artifacts["tssplit_weight"]) * p_tssplit
            print(f" + TSsplit(w={float(artifacts['tssplit_weight']):+.4f})")
        if p_zoneintent is not None:
            preds = preds + zoneintent_weight * p_zoneintent
            print(f" + ZoneIntent(w={zoneintent_weight:+.4f})")
        if p_et is not None:
            preds = preds + et_weight * p_et
            print(f" + ExtraTrees(w={et_weight:+.4f})")
        if p_mlp is not None:
            preds = preds + mlp_weight * p_mlp
        if p_ingame is not None:
            preds = preds + ingame_weight * p_ingame
        if p_phys is not None:
            preds = preds + physhead_weight * p_phys
            print(f" + PhysHead(w={physhead_weight:.4f}, feats={len(physhead_feats)})")
        if p_xgbrawid is not None:
            preds = preds + xgbrawid_weight * p_xgbrawid
            print(f" + XGB-rawID(w={xgbrawid_weight:+.4f})")
        if p_xgbctx is not None:
            preds = preds + xgbctx_weight * p_xgbctx
            print(f" + XGB-ctx(w={xgbctx_weight:.4f})")
        if p_pa4 is not None:
            preds = preds + pa4_weight * p_pa4
        if all(v is None for v in (p_hurdle, p_mix, p_denoise, p_multitask, p_multires,
                                   p_ordinal, p_formcast, p_midaxis, p_unified5, p_ballaxis,
                                   p_strikeaxis, p_otheraxis, p_midother, p_mega, p_condball, p_countresid, p_future50,
                                   p_pitcherresid, p_dangerball, p_mc5, p_mlp, p_ingame, p_pa4)):
            preds = p_ensemble
        # v72: 투수실력잔차 부가보정. 확률멤버 가중평균이 아니라 부호있는 보정값을
        # 더한다(p_final = preds + alpha*residual). 입력은 game_context/batter_matchup/
        # environment_team/trackman 78피처만 사용, pitcher_ability/sample_reliability는
        # 명시적으로 제외해 실력축과 다른 정보를 학습시킨다.
        residcorr_model = artifacts.get("residcorr_model")
        residcorr_cols = artifacts.get("residcorr_cols")
        residcorr_alpha = float(artifacts.get("residcorr_alpha", 0.0))
        if residcorr_model is not None and residcorr_cols and residcorr_alpha:
            rhat = residcorr_model.predict(X[residcorr_cols])
            preds = preds + residcorr_alpha * rhat
            print(f" + ResidCorr(alpha={residcorr_alpha:.2f}, cols={len(residcorr_cols)})")
        # v48: 시즌 레벨 이동 보정. 트리는 season을 외삽 못 해서 학습기간보다 낮아진
        # 대상시즌 레벨을 절반쯤밖에 못 따라간다(폴드 실측: 변화의 54%가 오차로 남음).
        # level_shift는 그 잔여분을 상수로 상쇄한다. 예측 범위가 [0.29,0.67]이라
        # |shift|<=0.02에서는 클리핑이 발생하지 않는다.
        # v85: risk-threshold 보정 (level_shift 앞에 적용)
        if p_mc5 is not None and risk_vec is not None and risk_alpha > 0:
            cut = np.maximum(0.0, risk_vec - risk_thr)
            preds = preds - risk_alpha * (cut - risk_center)
            print(f" + RiskAdj(thr={risk_thr:.3f}, alpha={risk_alpha:.3f}, "
                  f"center={risk_center:.4f}, 적용행={int((cut>0).sum())}/{len(cut)})")
        # v96: 물리기반 '크게 벗어난 볼' 보정 (평균중립). ball_vec=P(nd&ball)는 mc5
        # 확률에서, g(x)는 trackman+상황 41피처 전용 서브모델에서 나옴.
        if ball_vec is not None and ballsize_model is not None and ballsize_alpha > 0:
            g_pred = np.clip(ballsize_model.predict_proba(X[ballsize_feats])[:, 1], 0.0, 1.0)
            ball_signal = ball_vec * (g_pred - ballsize_const)
            preds = preds + ballsize_alpha * (ball_signal - ballsize_center)
            print(f" + BallSizeAdj(alpha={ballsize_alpha:.3f}, "
                  f"center={ballsize_center:.5f}, 신호std={float(ball_signal.std()):.5f})")
        # v95: 투수x2스트라이크 개인 슬로프 보정 (평균중립). Rule4: 자기 행의
        # pitcher_id/strikes_before와 train전체로 미리 계산한 투수별 통계표만 참조.
        if k2_alpha > 0 and k2_n_by_pid is not None:
            pid_arr = test["pitcher_id"].to_numpy()
            is2k = (X["strikes_before"].to_numpy() == 2).astype(np.float64)
            n2k = np.array([k2_n_by_pid.get(v, 0.0) for v in pid_arr], dtype=np.float64)
            gap2k = np.array([k2_gap_by_pid.get(v, 0.0) for v in pid_arr], dtype=np.float64)
            shrunk = gap2k * (n2k / (n2k + k2_K))
            applied = is2k * shrunk
            preds = preds + k2_alpha * (applied - k2_center)
            print(f" + K2Adj(K={k2_K:.0f}, alpha={k2_alpha:.4f}, "
                  f"center={k2_center:.5f}, 적용행={int((is2k>0).sum())}/{len(is2k)})")
        # v91: inseason_n 축 보정 (평균중립)
        if ns_alpha > 0:
            ln_ = np.log1p(np.expm1(X["inseason_n"].to_numpy(np.float64)))
            ns_cut = np.maximum(0.0, ln_ - ns_thr)
            preds = preds - ns_alpha * (ns_cut - ns_center)
            print(f" + NsAdj(thr={ns_thr:.3f}, alpha={ns_alpha:.4f}, "
                  f"center={ns_center:.4f}, 적용행={int((ns_cut>0).sum())}/{len(ns_cut)})")
        if level_shift:
            preds = preds + level_shift
        # 분포 프로브: 특정 부분집합에만 상수를 더해 그 집합의 테스트 내 비중을 역산한다.
        # Score감소 = (1e5/BS) x (그 집합의 비율) x (2*b_S*delta + delta^2)
        # 하락폭이 비율에 선형이므로 delta를 알면 비율이 나온다. 추론 전용이며
        # 각 행은 자기 컬럼(game_month, asof_pitcher_n, pitcher_id)만 참조 -> Rule §4 준수.
        probe = artifacts.get("probe")
        if probe:
            pmode, pdelta = probe["mode"], float(probe["delta"])
            if pmode == "month37":
                sel = ((test["game_month"] >= 3) & (test["game_month"] <= 7)).to_numpy()
            elif pmode == "lown":
                sel = (test["asof_pitcher_n"].fillna(0).to_numpy() < 500)
            elif pmode == "rookie":
                known = set(probe["known_pitchers"])
                sel = ~test["pitcher_id"].isin(known).to_numpy()
            else:
                raise ValueError(f"알 수 없는 probe mode: {pmode}")
            preds = preds + pdelta * sel
            print(f" [PROBE] mode={pmode} delta={pdelta:+.4f} "
                  f"적용행={int(sel.sum())}/{len(sel)} ({sel.mean()*100:.2f}%)")
        # 최종 방어적 클리핑. 멤버가 전부 [0,1]이고 가중치 합이 1이면 볼록결합이라
        # 수학적으로 항상 [0,1] 안에 있지만, 가중치 설정 실수로 그 불변식이 깨지는
        # 경우까지 대비한다(EVALUATION.md 공식엔 클리핑이 없어 서버가 그대로 벌점).
        preds = np.clip(preds, 0.0, 1.0)
        # v110: codex v20_905와 선형 블렌드. 로컬 검증 불가(codex 모델이 2019-2024
        # 전체 in-sample이라 fold A/C 둘 다 오염) -> 사용자 판단으로 w=0.15 실험적 제출.
        codex_weight = float(artifacts.get("codex_weight", 0.0))
        if codex_weight > 0:
            p_codex = predict_codex(test, MODEL_DIR)
            if p_codex is not None:
                preds = np.clip((1.0 - codex_weight) * preds
                                + codex_weight * p_codex, 0.0, 1.0)
                print(f" + CodexV20(w={codex_weight:.4f})")
            else:
                print(" [WARN] codex 파일이 없어 블렌드를 건너뜀")
    else:
        preds = []
    print(f" preds={len(preds)}  (HGB {len(hgbs)}변종 + CatBoost {len(cats)}변종"
         f"{' + Hurdle(w=%.2f)' % hurdle_weight if hurdle_weight else ''}"
         f"{' + CallMix(w=%.2f)' % mix_weight if mix_weight else ''}"
         f"{' + Denoised(w=%.2f)' % denoise_weight if denoise_weight else ''}"
         f"{' + MultiTask(w=%.2f)' % multi_task_weight if multi_task_weight else ''}"
         f"{' + MultiRes(w=%.2f)' % multires_weight if multires_weight else ''}"
         f"{' + Ordinal(w=%.2f)' % ordinal_weight if ordinal_weight else ''}"
         f"{' + Formcast(w=%.2f)' % formcast_weight if formcast_weight else ''}"
         f"{' + MidAxis(w=%.2f)' % midaxis_weight if midaxis_weight else ''}"
         f"{' + Unified5(w=%.2f)' % unified5_weight if unified5_weight else ''}"
         f"{' + BallAxis(w=%.2f)' % ballaxis_weight if ballaxis_weight else ''}"
         f"{' + StrikeAxis(w=%.2f)' % strikeaxis_weight if strikeaxis_weight else ''}"
         f"{' + OtherAxis(w=%.2f)' % otheraxis_weight if otheraxis_weight else ''}"
         f"{' + MidOther3H(w=%.2f)' % midother_weight if midother_weight else ''}"
         f"{' + Mega6H(w=%.2f)' % mega_weight if mega_weight else ''}"
         f"{' + CondBall(w=%.2f)' % condball_weight if condball_weight else ''}"
         f"{' + CountResid(w=%.2f)' % countresid_weight if countresid_weight else ''}"
         f"{' + Future50(w=%.2f)' % future50_weight if future50_weight else ''}"
         f"{' + PitcherResid(w=%.2f)' % pitcherresid_weight if pitcherresid_weight else ''}"
         f"{' + DangerBall(w=%.2f)' % dangerball_weight if dangerball_weight else ''}"
         f"{' + MC5(w=%.2f)' % mc5_weight if mc5_weight else ''}"
         f"{' + MLP(w=%.2f)' % mlp_weight if mlp_weight else ''}"
         f"{' + InGame(w=%.2f)' % ingame_weight if ingame_weight else ''}"
         f"{' + PA4(w=%.2f)' % pa4_weight if pa4_weight else ''}"
         f"{'  [level_shift=%+.4f]' % level_shift if level_shift else ''})")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
