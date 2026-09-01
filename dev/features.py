"""피처 엔지니어링 — 시점 안전(leakage-safe) 통계는 전부 '주어진 fit_df'에서만 계산한다.

설계 원칙
  - fit()에 넘기는 데이터가 통계의 유일한 원천이다. valid/test 변환 시에는 fit 시점에
    저장해 둔 통계만 재사용하고, valid/test 자신의 값은 절대 통계 계산에 넣지 않는다.
  - 학습 데이터 자기 자신에게 타깃 인코딩을 적용할 때는 K-fold OOF로 만들어
    (자기 행이 자기 통계에 들어가는) fold 내부 leakage를 막는다.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

TARGET_COL = "control_success"

CAT_COLS = ["top_bottom", "game_type", "base_state"]

RAW_NUM_COLS = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
]

# n(표본 수) 컬럼 -> 그 n으로 스무딩할 rate 컬럼들
RATE_GROUPS = {
    "asof_pitcher_n": [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    ],
    "asof_batter_n": ["asof_batter_success_rate", "asof_batter_middle_rate"],
    "asof_pitcher_pitchmix_n": [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ],
}
N_COLS = list(RATE_GROUPS.keys())

# n(표본 수) 컬럼이 없는 rate들 (prev1/3/5_game 계열) — 평균 대치 + 결측 플래그만 적용
NO_N_RATE_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]

TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
ID_COUNT_COLS = ["pitcher_id", "batter_id"]

SMOOTH_K_RATE = 20      # asof rate 스무딩 강도 (asof_*_n 기준)
# Optional per-rate empirical-Bayes strengths. Empty by default, preserving every
# existing training/submission artifact. Experiments may populate this mapping.
SMOOTH_K_BY_RATE = {}
SMOOTH_K_TEAM = 50      # team target encoding 스무딩 강도
N_OOF_FOLDS = 5

# Phase 2 ablation 대상 추가 피처 (모두 fit_df만으로 계산 -> valid/test 누수 없음)
EXTRA_FEATURE_NAMES = [
    "count_asof_success", "count_asof_ball", "count_asof_reverse",
    "diff_success_prev5", "diff_prev1_prev5",
    "pitcher_id_season_count", "batter_id_season_count",
    "pitcher_team_id_season_count", "batter_team_id_season_count",
]
SEASON_COUNT_COLS = ID_COUNT_COLS + TEAM_COLS

# 위험 피처: *_season_count는 fit_df(=train fold)의 "가장 최근 시즌" 등장 횟수를 계산해
# train fold 전체(모든 과거 시즌 행 포함)에 동일하게 매핑한다. 즉 2019년 행에도 2023년
# 등장 횟수가 붙어서, valid/test 누수는 아니지만 train 내부적으로는 미래 정보가 섞인다.
# row별 previous-season count / expanding count로 바꾸기 전까지는 ablation 결과를
# 신뢰하지 말 것 — phase2_feature_ablation.py가 이 목록을 risk=True로 표시한다.
RISKY_EXTRA_FEATURES = {
    "pitcher_id_season_count", "batter_id_season_count",
    "pitcher_team_id_season_count", "batter_team_id_season_count",
}


class FeatureBuilder:
    def __init__(self, seed=42, include_raw_rates=False, extra_features=None, include_team_te=True,
                 team_te_mode="shuffled_kfold"):
        self.seed = seed
        self.include_raw_rates = include_raw_rates
        # extra_features: None -> 전부 비활성(기존 챔피언과 동일), set()/list -> 그 이름들만 활성
        self.extra_features = set(extra_features) if extra_features else set()
        # team_te: shuffled KFold OOF라 시간 구조와 안 맞을 수 있음 (importance도 약했음).
        # False로 두면 team_te_* 컬럼 자체를 안 만든다 (team_count는 target 미사용이라 계속 유지).
        self.include_team_te = include_team_te
        # team_te_mode: "shuffled_kfold"(기존, seed 의존) 또는 "expanding"(row_num 시간순,
        # seed와 완전 무관 — 각 행은 자기보다 row_num이 작은 행들의 팀 성공률만 본다).
        self.team_te_mode = team_te_mode

    # ---------- fit: 통계를 오직 df(=train fold)에서만 계산 ----------
    def fit(self, df):
        self.cat_encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.cat_encoder_.fit(df[CAT_COLS].astype(str))

        # rate 컬럼별 global mean (스무딩 fallback), NaN은 제외하고 계산
        self.rate_global_mean_ = {}
        for cols in RATE_GROUPS.values():
            for c in cols:
                self.rate_global_mean_[c] = float(df[c].mean(skipna=True))
        for c in NO_N_RATE_COLS:
            self.rate_global_mean_[c] = float(df[c].mean(skipna=True))

        # 숫자 결측 대치용 median (RAW_NUM_COLS는 거의 결측 없지만 방어적으로)
        self.num_median_ = df[RAW_NUM_COLS].median(numeric_only=True)

        # team target encoding (전체 df 기준 — valid/test 변환용)
        y = df[TARGET_COL]
        self.global_y_mean_ = float(y.mean())
        self.team_te_ = {}
        self.team_count_ = {}
        for col in TEAM_COLS:
            grp = df.groupby(col)[TARGET_COL].agg(["sum", "count"])
            te = (grp["sum"] + SMOOTH_K_TEAM * self.global_y_mean_) / (grp["count"] + SMOOTH_K_TEAM)
            self.team_te_[col] = te.to_dict()
            self.team_count_[col] = df[col].value_counts().to_dict()

        # pitcher_id / batter_id count encoding (target 미사용 — leakage 없음)
        self.id_count_ = {col: df[col].value_counts().to_dict() for col in ID_COUNT_COLS}

        # season별(가장 최근 시즌 한정) count encoding — "현재 활동량" 프록시.
        # fit_df(=train fold)의 최대 시즌만 사용해 정적 매핑을 만들고 valid/test엔 그대로 조회만 한다.
        last_season = df["season"].max()
        last_df = df[df["season"] == last_season]
        self.season_count_ = {col: last_df[col].value_counts().to_dict() for col in SEASON_COUNT_COLS}
        return self

    # ---------- 공통: rate 스무딩 + 파생 피처 (target 인코딩 제외) ----------
    def _base_transform(self, df):
        out = {}

        cats = self.cat_encoder_.transform(df[CAT_COLS].astype(str))
        for i, c in enumerate(CAT_COLS):
            out[f"cat_{c}"] = cats[:, i]

        for c in RAW_NUM_COLS:
            out[c] = df[c].fillna(self.num_median_.get(c, 0.0)).to_numpy(dtype=np.float64)

        # pitcher_hand / batter_hand: 이미 정수 코드라 그대로 수치로 사용
        out["pitcher_hand"] = df["pitcher_hand"].to_numpy(dtype=np.float64)
        out["batter_hand"] = df["batter_hand"].to_numpy(dtype=np.float64)
        out["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float64).to_numpy()

        # 교차 범주: 카운트 상태(balls x strikes, 12종) / 손잡이 매치업(4~6종)
        out["count_state"] = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy(dtype=np.float64)
        out["hand_matchup"] = (df["pitcher_hand"] * 10 + df["batter_hand"]).to_numpy(dtype=np.float64)

        smoothed = {}
        for n_col, rate_cols in RATE_GROUPS.items():
            n = df[n_col].fillna(0).to_numpy(dtype=np.float64)
            out[f"flag_{n_col}_zero"] = (n == 0).astype(np.float64)
            out[n_col] = np.log1p(n)
            for c in rate_cols:
                raw = df[c].fillna(0).to_numpy(dtype=np.float64)
                gm = self.rate_global_mean_[c]
                k = float(SMOOTH_K_BY_RATE.get(c, SMOOTH_K_RATE))
                sm = (n * raw + k * gm) / (n + k)
                out[f"{c}_smooth"] = sm
                smoothed[c] = sm
                if self.include_raw_rates:
                    out[c] = df[c].fillna(gm).to_numpy(dtype=np.float64)

        miss_flag = df[NO_N_RATE_COLS[0]].isna().to_numpy()
        out["flag_prev_game_missing"] = miss_flag.astype(np.float64)
        for c in NO_N_RATE_COLS:
            out[c] = df[c].fillna(self.rate_global_mean_[c]).to_numpy(dtype=np.float64)

        # 투수 vs 타자 이력 차이 (스무딩된 값으로 계산)
        out["diff_success_rate"] = smoothed["asof_pitcher_success_rate"] - smoothed["asof_batter_success_rate"]
        out["diff_middle_rate"] = smoothed["asof_pitcher_middle_rate"] - smoothed["asof_batter_middle_rate"]

        for col in ID_COUNT_COLS:
            cnt = df[col].map(self.id_count_[col]).fillna(0).to_numpy(dtype=np.float64)
            out[f"{col}_count"] = np.log1p(cnt)

        for col in TEAM_COLS:
            cnt = df[col].map(self.team_count_[col]).fillna(0).to_numpy(dtype=np.float64)
            out[f"{col}_count"] = np.log1p(cnt)

        ef = self.extra_features
        if ef & {"count_asof_success", "count_asof_ball", "count_asof_reverse"}:
            cs = out["count_state"]
            if "count_asof_success" in ef:
                out["count_asof_success"] = cs * smoothed["asof_pitcher_success_rate"]
            if "count_asof_ball" in ef:
                out["count_asof_ball"] = cs * smoothed["asof_pitcher_ball_rate"]
            if "count_asof_reverse" in ef:
                out["count_asof_reverse"] = cs * smoothed["asof_pitcher_reverse_rate"]
        if "diff_success_prev5" in ef:
            out["diff_success_prev5"] = (
                smoothed["asof_pitcher_success_rate"] - out["asof_pitcher_prev5_game_success_rate"])
        if "diff_prev1_prev5" in ef:
            out["diff_prev1_prev5"] = (
                out["asof_pitcher_prev1_game_success_rate"] - out["asof_pitcher_prev5_game_success_rate"])
        for col in SEASON_COUNT_COLS:
            name = f"{col}_season_count"
            if name in ef:
                cnt = df[col].map(self.season_count_[col]).fillna(0).to_numpy(dtype=np.float64)
                out[name] = np.log1p(cnt)

        return out

    # ---------- valid/test 변환: fit 시점 통계만 사용 ----------
    def transform(self, df):
        out = self._base_transform(df)
        if self.include_team_te:
            for col in TEAM_COLS:
                te_map = self.team_te_[col]
                out[f"{col}_te"] = df[col].map(te_map).fillna(self.global_y_mean_).to_numpy(dtype=np.float64)
        return pd.DataFrame(out, index=df.index)

    # ---------- train 자기 자신 변환: team target encoding만 K-fold OOF (또는 expanding) ----------
    def transform_train_oof(self, df):
        out = self._base_transform(df)

        if self.include_team_te:
            if self.team_te_mode == "expanding":
                if "row_num" not in df.columns:
                    raise ValueError("team_te_mode='expanding'은 row_num 컬럼이 필요합니다.")
                order = np.argsort(df["row_num"].to_numpy())
                y_ordered = df[TARGET_COL].to_numpy(dtype=np.float64)[order]
                for col in TEAM_COLS:
                    vals_ordered = df[col].to_numpy()[order]
                    s = pd.Series(y_ordered)
                    g = s.groupby(vals_ordered)
                    cumsum_incl = g.cumsum().to_numpy()
                    cumcount_prior = g.cumcount().to_numpy().astype(np.float64)
                    cumsum_prior = cumsum_incl - y_ordered
                    te_ordered = (cumsum_prior + SMOOTH_K_TEAM * self.global_y_mean_) / (cumcount_prior + SMOOTH_K_TEAM)
                    te = np.empty(len(df), dtype=np.float64)
                    te[order] = te_ordered
                    out[f"{col}_te"] = te
            else:
                for col in TEAM_COLS:
                    oof = np.full(len(df), np.nan)
                    kf = KFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=self.seed)
                    for tr_idx, ho_idx in kf.split(df):
                        sub = df.iloc[tr_idx]
                        grp = sub.groupby(col)[TARGET_COL].agg(["sum", "count"])
                        te = (grp["sum"] + SMOOTH_K_TEAM * self.global_y_mean_) / (grp["count"] + SMOOTH_K_TEAM)
                        te_map = te.to_dict()
                        ho_vals = df.iloc[ho_idx][col]
                        oof[ho_idx] = ho_vals.map(te_map).fillna(self.global_y_mean_).to_numpy()
                    out[f"{col}_te"] = oof

        return pd.DataFrame(out, index=df.index)


    # ---------- 추론용: fit 시점 통계를 순수 dict로 export (submit.zip의 script.py가 사용) ----------
    def export_stats(self):
        return {
            "cat_encoder": self.cat_encoder_,
            "cat_cols": CAT_COLS,
            "raw_num_cols": RAW_NUM_COLS,
            "num_median": self.num_median_.to_dict(),
            "rate_groups": RATE_GROUPS,
            "no_n_rate_cols": NO_N_RATE_COLS,
            "rate_global_mean": self.rate_global_mean_,
            "team_cols": TEAM_COLS,
            "team_te": self.team_te_,
            "team_count": self.team_count_,
            "id_count_cols": ID_COUNT_COLS,
            "id_count": self.id_count_,
            "global_y_mean": self.global_y_mean_,
            "smooth_k_rate": SMOOTH_K_RATE,
            "smooth_k_by_rate": dict(SMOOTH_K_BY_RATE),
            "extra_features": sorted(self.extra_features),
            "season_count_cols": SEASON_COUNT_COLS,
            "season_count": self.season_count_,
        }


def feature_names(builder_df):
    return list(builder_df.columns)
