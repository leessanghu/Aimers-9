# script.py
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


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

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    OUT_DIR = os.path.join(BASE_DIR, "output")
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    ARTIFACT_PATH = os.path.join(MODEL_DIR, "model_artifacts_v17b.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model...")
    artifacts = joblib.load(ARTIFACT_PATH)
    hgb, cat, stats = artifacts["hgb"], artifacts["cat"], artifacts["stats"]
    inseason_stats = artifacts["inseason_stats"]
    platoon_stats = artifacts["platoon_stats"]
    inning_stats = artifacts["inning_stats"]
    pitchtype_stats = artifacts["pitchtype_stats"]
    lastyear_stats = artifacts["lastyear_stats"]
    labels_c_stats = artifacts["labels_c_stats"]
    labels_i_stats = artifacts["labels_i_stats"]
    extra_cde_stats = artifacts["extra_cde_stats"]
    extra_f_stats = artifacts["extra_f_stats"]
    feature_order = artifacts["feature_order"]
    w_hgb, w_cat = artifacts["w_hgb"], artifacts["w_cat"]
    print(f" OK. w_hgb={w_hgb}  w_cat={w_cat}  features={len(feature_order)}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features (v17a 101 + 10 extra: strike/pitchmix/JS/pitch_of_pa)...")
    ids = test[ID_COL].tolist()
    X_base = build_features(test, stats).reset_index(drop=True)
    X_inseason = build_inseason_features(test, inseason_stats).reset_index(drop=True)
    prior_rate = get_prior_pitcher_rate(test, inseason_stats)
    X_platoon = build_platoon_features(test, platoon_stats, prior_rate).reset_index(drop=True)
    X_inning = build_inning_features(test, inning_stats, prior_rate).reset_index(drop=True)
    X_pitchtype = build_pitchtype_features(test, pitchtype_stats, prior_rate).reset_index(drop=True)
    X_lastyear = build_lastyear_features(test, lastyear_stats).reset_index(drop=True)
    cs = (test["balls_before"] * 4 + test["strikes_before"]).to_numpy()
    inn = np.clip(test["inning"].to_numpy(np.int64), 1, 9)
    X_lc = build_label_cond_features(test, labels_c_stats, cs, "lc").reset_index(drop=True)
    X_li = build_label_cond_features(test, labels_i_stats, inn, "li").reset_index(drop=True)
    X_strike = build_strike_features(test, extra_cde_stats).reset_index(drop=True)
    X_mix = build_pitchmix_arsenal_features(test, extra_cde_stats).reset_index(drop=True)
    X_popa = build_popa_features(test, extra_f_stats).reset_index(drop=True)
    X = pd.concat([X_base, X_inseason, X_platoon, X_inning, X_pitchtype], axis=1)
    X.index = test.index
    X = X.astype(np.float64)
    X_cross = add_crosses(X)
    for frame in (X_lastyear, X_lc, X_li, X_strike, X_mix, X_popa):
        frame.index = test.index
    X = pd.concat([X, X_cross, X_lastyear, X_lc, X_li, X_strike, X_mix, X_popa], axis=1)
    X = X[feature_order].astype(np.float64)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    if len(X):
        preds = w_hgb * hgb.predict_proba(X)[:, 1] + w_cat * cat.predict_proba(X)[:, 1]
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
