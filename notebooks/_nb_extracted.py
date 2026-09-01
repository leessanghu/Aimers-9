import sys
from pathlib import Path

try:
    import google.colab  # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = Path('/content/drive/MyDrive/Aimers9')
else:
    PROJECT_DIR = Path.cwd().resolve()
    if PROJECT_DIR.name == 'notebooks':
        PROJECT_DIR = PROJECT_DIR.parent

DATA_DIR = PROJECT_DIR / 'data'
PRED_DIR = PROJECT_DIR / 'dev' / 'phase3_preds'
PRED_DIR.mkdir(parents=True, exist_ok=True)

print("IN_COLAB:", IN_COLAB)
print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_DIR:", DATA_DIR)
print("PRED_DIR:", PRED_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)

# ===== dev/metrics.py =====
def brier_score(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y_true) ** 2))


def evaluate(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    bs = brier_score(y_true, p)
    r = float(y_true.mean())
    baseline_bs = r * (1 - r)
    bss = 1 - bs / baseline_bs if baseline_bs > 0 else float("nan")
    return {
        "n": len(y_true), "r": r, "brier_score": bs, "baseline_brier": baseline_bs,
        "bss": bss, "leaderboard_score": max(0.0, 100000 * bss),
    }

# ===== dev/features.py =====
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
NO_N_RATE_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]
TEAM_COLS = ["pitcher_team_id", "batter_team_id"]
ID_COUNT_COLS = ["pitcher_id", "batter_id"]
SMOOTH_K_RATE = 20
SMOOTH_K_TEAM = 50
N_OOF_FOLDS = 5
EXTRA_FEATURE_NAMES = [
    "count_asof_success", "count_asof_ball", "count_asof_reverse",
    "diff_success_prev5", "diff_prev1_prev5",
    "pitcher_id_season_count", "batter_id_season_count",
    "pitcher_team_id_season_count", "batter_team_id_season_count",
]
SEASON_COUNT_COLS = ID_COUNT_COLS + TEAM_COLS
RISKY_EXTRA_FEATURES = {
    "pitcher_id_season_count", "batter_id_season_count",
    "pitcher_team_id_season_count", "batter_team_id_season_count",
}


class FeatureBuilder:
    def __init__(self, seed=42, include_raw_rates=False, extra_features=None, include_team_te=True):
        self.seed = seed
        self.include_raw_rates = include_raw_rates
        self.extra_features = set(extra_features) if extra_features else set()
        self.include_team_te = include_team_te

    def fit(self, df):
        self.cat_encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.cat_encoder_.fit(df[CAT_COLS].astype(str))

        self.rate_global_mean_ = {}
        for cols in RATE_GROUPS.values():
            for c in cols:
                self.rate_global_mean_[c] = float(df[c].mean(skipna=True))
        for c in NO_N_RATE_COLS:
            self.rate_global_mean_[c] = float(df[c].mean(skipna=True))

        self.num_median_ = df[RAW_NUM_COLS].median(numeric_only=True)

        y = df[TARGET_COL]
        self.global_y_mean_ = float(y.mean())
        self.team_te_ = {}
        self.team_count_ = {}
        for col in TEAM_COLS:
            grp = df.groupby(col)[TARGET_COL].agg(["sum", "count"])
            te = (grp["sum"] + SMOOTH_K_TEAM * self.global_y_mean_) / (grp["count"] + SMOOTH_K_TEAM)
            self.team_te_[col] = te.to_dict()
            self.team_count_[col] = df[col].value_counts().to_dict()

        self.id_count_ = {col: df[col].value_counts().to_dict() for col in ID_COUNT_COLS}

        last_season = df["season"].max()
        last_df = df[df["season"] == last_season]
        self.season_count_ = {col: last_df[col].value_counts().to_dict() for col in SEASON_COUNT_COLS}
        return self

    def _base_transform(self, df):
        out = {}
        cats = self.cat_encoder_.transform(df[CAT_COLS].astype(str))
        for i, c in enumerate(CAT_COLS):
            out[f"cat_{c}"] = cats[:, i]

        for c in RAW_NUM_COLS:
            out[c] = df[c].fillna(self.num_median_.get(c, 0.0)).to_numpy(dtype=np.float64)

        out["pitcher_hand"] = df["pitcher_hand"].to_numpy(dtype=np.float64)
        out["batter_hand"] = df["batter_hand"].to_numpy(dtype=np.float64)
        out["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float64).to_numpy()
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
                sm = (n * raw + SMOOTH_K_RATE * gm) / (n + SMOOTH_K_RATE)
                out[f"{c}_smooth"] = sm
                smoothed[c] = sm
                if self.include_raw_rates:
                    out[c] = df[c].fillna(gm).to_numpy(dtype=np.float64)

        miss_flag = df[NO_N_RATE_COLS[0]].isna().to_numpy()
        out["flag_prev_game_missing"] = miss_flag.astype(np.float64)
        for c in NO_N_RATE_COLS:
            out[c] = df[c].fillna(self.rate_global_mean_[c]).to_numpy(dtype=np.float64)

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

    def transform(self, df):
        out = self._base_transform(df)
        if self.include_team_te:
            for col in TEAM_COLS:
                te_map = self.team_te_[col]
                out[f"{col}_te"] = df[col].map(te_map).fillna(self.global_y_mean_).to_numpy(dtype=np.float64)
        return pd.DataFrame(out, index=df.index)

    def transform_train_oof(self, df):
        out = self._base_transform(df)
        if self.include_team_te:
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

# ===== dev/phase2_common.py (필요한 함수만) =====
CAT_FEATURES = ["cat_top_bottom", "cat_game_type", "cat_base_state", "count_state", "hand_matchup"]


def build_fold(df, train_max_season, valid_season, extra_features=None, seed=42, include_team_te=True):
    train_fold = df[df["season"] <= train_max_season].reset_index(drop=True)
    valid_fold = df[df["season"] == valid_season].reset_index(drop=True)

    fb = FeatureBuilder(seed=seed, include_raw_rates=False, extra_features=extra_features,
                        include_team_te=include_team_te).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)
    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype(pd.CategoricalDtype(categories=X_train[c].cat.categories))

    known_pitchers = set(train_fold["pitcher_id"].unique())
    known_batters = set(train_fold["batter_id"].unique())
    seen_p = valid_fold["pitcher_id"].isin(known_pitchers).to_numpy()
    seen_b = valid_fold["batter_id"].isin(known_batters).to_numpy()

    return {
        "train_fold": train_fold, "valid_fold": valid_fold,
        "X_train": X_train, "X_valid": X_valid,
        "y_train": train_fold[TARGET_COL].to_numpy(), "y_valid": valid_fold[TARGET_COL].to_numpy(),
        "seen_pitcher_mask": seen_p, "seen_batter_mask": seen_b,
        "row_id": valid_fold["row_id"].to_numpy(),
    }


def time_split_es(n, frac=0.08):
    cut = int(n * (1 - frac))
    return np.arange(cut), np.arange(cut, n)

EXTRA_FEATURES = {"count_asof_ball", "diff_prev1_prev5"}
GATE_COLS = ["asof_pitcher_n", "flag_asof_pitcher_n_zero", "asof_batter_n", "flag_asof_batter_n_zero"]

FALLBACK_DEAD_LIST = [
    "asof_batter_n", "batter_id_count", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "diff_middle_rate", "asof_pitcher_middle_rate_smooth",
    "diff_success_rate", "run_top_before", "asof_pitcher_pitchmix_n", "game_dayofweek",
    "score_diff_home", "home_win_expectancy", "away_win_expectancy", "cat_top_bottom",
    "pitcher_hand", "strikes_before", "score_diff_pitcher_team", "runner_on_1b",
    "num_runners_on", "runner_on_3b", "runner_on_2b", "flag_prev_game_missing",
    "flag_asof_pitcher_pitchmix_n_zero", "flag_asof_batter_n_zero", "flag_asof_pitcher_n_zero",
    "batter_hand", "asof_batter_middle_rate_smooth", "run_bot_before",
    "asof_pitcher_prev1_game_middle_rate", "outs_before", "li",
    "asof_pitcher_breaking_rate_smooth", "run_total_before", "cat_base_state", "balls_before",
]

_dead_csv = PROJECT_DIR / 'dev' / 'dead_features_conservative_list.csv'
if _dead_csv.exists():
    DEAD_LIST = pd.read_csv(_dead_csv)["feature"].tolist()
    print(f"dead-feature 목록 로드: {_dead_csv} ({len(DEAD_LIST)}개)")
else:
    DEAD_LIST = FALLBACK_DEAD_LIST
    print(f"[fallback] {_dead_csv} 없음 -> 하드코딩된 목록 사용 ({len(DEAD_LIST)}개)")

DEAD_LIST_EXCL_SEASON = [c for c in DEAD_LIST if c != "season"]

FOLDS = [(2023, 2024)]  # SMOKE TEST override

# 학습 설정
D_MODEL = 64
DROPOUT = 0.2
ID_DROPOUT_P = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 4096 if DEVICE.type == 'cuda' else 1024
MAX_EPOCHS = 2  # SMOKE TEST override
PATIENCE = 1  # SMOKE TEST override
MIN_PLAYER_COUNT = 20  # vocab에 남길 최소 등장 횟수 (미만은 UNK)

df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
print(df.shape)

def build_vocab(series, min_count=MIN_PLAYER_COUNT):
    """0=UNK. min_count 미만으로 등장한 선수는 UNK로 묶는다."""
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return {v: i + 1 for i, v in enumerate(keep)}


def encode_ids(series, vocab):
    return series.map(vocab).fillna(0).astype(np.int64).to_numpy()


class PitchDataset(Dataset):
    """한 행 = 투구 하나. cont(연속형) + pid/bid(선수 ID) + gate_p/gate_b(게이트 입력) + y."""

    def __init__(self, cont, pid, bid, gate_p, gate_b, y):
        self.cont = torch.from_numpy(cont.astype(np.float32))
        self.pid = torch.from_numpy(pid.astype(np.int64))
        self.bid = torch.from_numpy(bid.astype(np.int64))
        self.gate_p = torch.from_numpy(gate_p.astype(np.float32))
        self.gate_b = torch.from_numpy(gate_b.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.cont[idx], self.pid[idx], self.bid[idx],
                self.gate_p[idx], self.gate_b[idx], self.y[idx])

class EmbeddingMLP(nn.Module):
    """plain: cont + pitcher_emb + batter_emb를 그대로 합산.
    gated : pitcher_emb/batter_emb에 콜드스타트 신호 기반 학습형 게이트(sigmoid)를 곱해서
            경험이 적은 선수일수록 ID 임베딩의 영향을 스스로 줄이도록 유도.
    """

    def __init__(self, n_cont, n_pitcher, n_batter, d_model=D_MODEL, dropout=DROPOUT, gated=False):
        super().__init__()
        self.gated = gated
        self.cont = nn.Linear(n_cont, d_model)
        self.pitcher_emb = nn.Embedding(n_pitcher, d_model, padding_idx=0)
        self.batter_emb = nn.Embedding(n_batter, d_model, padding_idx=0)
        if gated:
            self.gate_p = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            self.gate_b = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.body = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, cont, pid, bid, gate_p_in, gate_b_in):
        h = F.relu(self.cont(cont))
        pe = self.pitcher_emb(pid)
        be = self.batter_emb(bid)
        if self.gated:
            gp = torch.sigmoid(self.gate_p(gate_p_in))
            gb = torch.sigmoid(self.gate_b(gate_b_in))
            pe = pe * gp
            be = be * gb
        x = h + pe + be
        x = self.body(x)
        return self.head(x).squeeze(-1)  # logit


def id_dropout_(pid, bid, p, training):
    """학습 중에만, 확률 p로 pitcher/batter ID를 UNK(0)로 치환 -> ID에 대한 과의존 방지."""
    if not training or p <= 0:
        return pid, bid
    mask_p = torch.rand(pid.shape[0], device=pid.device) < p
    mask_b = torch.rand(bid.shape[0], device=bid.device) < p
    pid = pid.clone(); pid[mask_p] = 0
    bid = bid.clone(); bid[mask_b] = 0
    return pid, bid

def train_one_fold(df, train_max, valid_season, gated, verbose=True):
    fold = build_fold(df, train_max, valid_season, extra_features=EXTRA_FEATURES, seed=SEED,
                      include_team_te=False)
    train_fold, valid_fold = fold["train_fold"], fold["valid_fold"]
    X_train_full, X_valid_full = fold["X_train"], fold["X_valid"]

    # 게이트 입력 (죽은 피처 목록에 있어도 게이트 전용으로는 유효한 콜드스타트 신호)
    gate_train_all = X_train_full[GATE_COLS].to_numpy(dtype=np.float32)
    gate_valid_all = X_valid_full[GATE_COLS].to_numpy(dtype=np.float32)
    gate_p_train, gate_b_train = gate_train_all[:, :2], gate_train_all[:, 2:]
    gate_p_valid, gate_b_valid = gate_valid_all[:, :2], gate_valid_all[:, 2:]

    # 연속형 본 피처: Phase 2에서 검증된 안전한 35개 제거판
    cont_cols = [c for c in X_train_full.columns if c not in DEAD_LIST_EXCL_SEASON]
    X_train = X_train_full[cont_cols].to_numpy(dtype=np.float32)
    X_valid = X_valid_full[cont_cols].to_numpy(dtype=np.float32)

    # 표준화 (train 통계만 사용)
    mean, std = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std

    p_vocab = build_vocab(train_fold["pitcher_id"])
    b_vocab = build_vocab(train_fold["batter_id"])
    pid_train = encode_ids(train_fold["pitcher_id"], p_vocab)
    bid_train = encode_ids(train_fold["batter_id"], b_vocab)
    pid_valid = encode_ids(valid_fold["pitcher_id"], p_vocab)
    bid_valid = encode_ids(valid_fold["batter_id"], b_vocab)

    y_train, y_valid = fold["y_train"].astype(np.float32), fold["y_valid"].astype(np.float32)

    tr_idx, es_idx = time_split_es(len(y_train))  # train 내부 마지막 8% = early stopping용 (시간순)

    ds_train = PitchDataset(X_train[tr_idx], pid_train[tr_idx], bid_train[tr_idx],
                            gate_p_train[tr_idx], gate_b_train[tr_idx], y_train[tr_idx])
    ds_es = PitchDataset(X_train[es_idx], pid_train[es_idx], bid_train[es_idx],
                         gate_p_train[es_idx], gate_b_train[es_idx], y_train[es_idx])
    ds_valid = PitchDataset(X_valid, pid_valid, bid_valid, gate_p_valid, gate_b_valid, y_valid)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_es = DataLoader(ds_es, batch_size=BATCH_SIZE * 2, shuffle=False)
    dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE * 2, shuffle=False)

    model = EmbeddingMLP(X_train.shape[1], len(p_vocab) + 1, len(b_vocab) + 1, gated=gated).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    def run_eval(loader):
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for cont, pid, bid, gp, gb, y in loader:
                cont, pid, bid = cont.to(DEVICE), pid.to(DEVICE), bid.to(DEVICE)
                gp, gb = gp.to(DEVICE), gb.to(DEVICE)
                logit = model(cont, pid, bid, gp, gb)
                preds.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(preds), np.concatenate(ys)

    best_bs, best_state, since_best = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for cont, pid, bid, gp, gb, y in dl_train:
            cont, pid, bid = cont.to(DEVICE), pid.to(DEVICE), bid.to(DEVICE)
            gp, gb, y = gp.to(DEVICE), gb.to(DEVICE), y.to(DEVICE)
            pid, bid = id_dropout_(pid, bid, ID_DROPOUT_P, training=True)
            opt.zero_grad()
            logit = model(cont, pid, bid, gp, gb)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()

        p_es, y_es = run_eval(dl_es)
        es_bs = evaluate(y_es, p_es)["brier_score"]
        if verbose:
            print(f"  epoch{epoch:2d}  es_brier={es_bs:.6f}  es_bss={evaluate(y_es, p_es)['bss']:.5f}")
        if es_bs < best_bs - 1e-5:
            best_bs, since_best = es_bs, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= PATIENCE:
                if verbose:
                    print(f"  early stop @ epoch{epoch} (best es_brier={best_bs:.6f})")
                break

    model.load_state_dict(best_state)
    p_valid, y_valid_check = run_eval(dl_valid)
    assert np.array_equal(y_valid_check, y_valid)

    m = evaluate(y_valid, p_valid)
    for name, mask in [("pitcher", fold["seen_pitcher_mask"]), ("batter", fold["seen_batter_mask"])]:
        for tag, mm in [("seen", mask), ("unseen", ~mask)]:
            if mm.sum() > 0:
                m[f"{name}_{tag}_bss"] = evaluate(y_valid[mm], p_valid[mm])["bss"]
                m[f"{name}_{tag}_n"] = int(mm.sum())

    return {"row_id": fold["row_id"], "y_valid": y_valid, "pred": p_valid, "metrics": m, "model": model}

import time

all_metrics = []

for train_max, valid_season in FOLDS:
    for gated in [False, True]:
        tag = "gated" if gated else "plain"
        print(f"\n===== fold valid={valid_season}  variant={tag} =====")
        t0 = time.time()
        result = train_one_fold(df, train_max, valid_season, gated=gated)
        m = result["metrics"]
        print(f"  [{tag}] valid BSS={m['bss']:.6f}  score={m['leaderboard_score']:.1f}  "
              f"seen_p={m.get('pitcher_seen_bss')}  unseen_p={m.get('pitcher_unseen_bss')}  "
              f"seen_b={m.get('batter_seen_bss')}  unseen_b={m.get('batter_unseen_bss')}  "
              f"({time.time()-t0:.0f}s)")

        out = pd.DataFrame({"row_id": result["row_id"], "y_valid": result["y_valid"], "pred": result["pred"]})
        out_path = PRED_DIR / f"fold_{valid_season}_pred_embmlp_{tag}.csv"
        out.to_csv(out_path, index=False)
        print(f"  저장: {out_path}")

        m2 = dict(m)
        m2.update({"valid_season": valid_season, "variant": tag})
        all_metrics.append(m2)

summary = pd.DataFrame(all_metrics)
summary.to_csv(PRED_DIR / "phase3_embmlp_summary.csv", index=False, encoding="utf-8")
print("\n===== 요약 =====")
print(summary.to_string(index=False))