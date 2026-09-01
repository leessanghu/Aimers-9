"""노트북에 넣기 전 로직 검증용 스모크 테스트 스크립트 (작은 subset, CPU, 짧은 epoch).
이 파일은 검증용이며 실제 산출물이 아님 — 통과하면 동일 로직을 ipynb 셀로 옮긴다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from features import FeatureBuilder, TARGET_COL
from phase2_common import build_fold, time_split_es
from metrics import evaluate

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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
DEAD_LIST_EXCL_SEASON = [c for c in FALLBACK_DEAD_LIST if c != "season"]


def build_vocab(series, min_count=20):
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return {v: i + 1 for i, v in enumerate(keep)}  # 0=UNK


def encode_ids(series, vocab):
    return series.map(vocab).fillna(0).astype(np.int64).to_numpy()


class PitchDataset(Dataset):
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
        return self.cont[idx], self.pid[idx], self.bid[idx], self.gate_p[idx], self.gate_b[idx], self.y[idx]


class EmbeddingMLP(nn.Module):
    def __init__(self, n_cont, n_pitcher, n_batter, d_model=64, dropout=0.2, gated=False):
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
        return self.head(x).squeeze(-1)


def id_dropout_(pid, bid, p, training):
    if not training or p <= 0:
        return pid, bid
    mask_p = torch.rand(pid.shape[0], device=pid.device) < p
    mask_b = torch.rand(bid.shape[0], device=bid.device) < p
    pid = pid.clone()
    bid = bid.clone()
    pid[mask_p] = 0
    bid[mask_b] = 0
    return pid, bid


def train_variant(df, train_max, valid_season, gated, max_epochs=2, subset=20000):
    """스모크 테스트: subset 행만 써서 아주 짧게 학습."""
    fold = build_fold(df, train_max, valid_season, extra_features=EXTRA_FEATURES, seed=SEED,
                      include_team_te=False)
    train_fold, valid_fold = fold["train_fold"], fold["valid_fold"]
    X_train_full, X_valid_full = fold["X_train"], fold["X_valid"]

    gate_train = X_train_full[GATE_COLS].to_numpy(dtype=np.float32)
    gate_valid = X_valid_full[GATE_COLS].to_numpy(dtype=np.float32)
    gate_p_train, gate_b_train = gate_train[:, :2], gate_train[:, 2:]
    gate_p_valid, gate_b_valid = gate_valid[:, :2], gate_valid[:, 2:]

    cont_cols = [c for c in X_train_full.columns if c not in DEAD_LIST_EXCL_SEASON]
    X_train = X_train_full[cont_cols].to_numpy(dtype=np.float32)
    X_valid = X_valid_full[cont_cols].to_numpy(dtype=np.float32)

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

    # 스모크 테스트: 앞부분만 subset
    n = min(subset, len(y_train))
    idx = np.arange(n)
    tr_idx, es_idx = time_split_es(n)

    ds_train = PitchDataset(X_train[idx][tr_idx], pid_train[idx][tr_idx], bid_train[idx][tr_idx],
                            gate_p_train[idx][tr_idx], gate_b_train[idx][tr_idx], y_train[idx][tr_idx])
    ds_es = PitchDataset(X_train[idx][es_idx], pid_train[idx][es_idx], bid_train[idx][es_idx],
                         gate_p_train[idx][es_idx], gate_b_train[idx][es_idx], y_train[idx][es_idx])
    dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)
    dl_es = DataLoader(ds_es, batch_size=2048, shuffle=False)

    device = "cpu"
    model = EmbeddingMLP(X_train.shape[1], len(p_vocab) + 1, len(b_vocab) + 1, gated=gated).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(max_epochs):
        model.train()
        for cont, pid, bid, gp, gb, y in dl_train:
            pid, bid = id_dropout_(pid, bid, 0.15, training=True)
            opt.zero_grad()
            logit = model(cont, pid, bid, gp, gb)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for cont, pid, bid, gp, gb, y in dl_es:
                logit = model(cont, pid, bid, gp, gb)
                preds.append(torch.sigmoid(logit).numpy())
                ys.append(y.numpy())
        p_es, y_es = np.concatenate(preds), np.concatenate(ys)
        print(f"  epoch{epoch} es_bss={evaluate(y_es, p_es)['bss']:.5f}", flush=True)

    # valid 전체 subset(스모크용)로 예측
    n_v = min(subset, len(y_valid))
    ds_valid = PitchDataset(X_valid[:n_v], pid_valid[:n_v], bid_valid[:n_v],
                            gate_p_valid[:n_v], gate_b_valid[:n_v], y_valid[:n_v])
    dl_valid = DataLoader(ds_valid, batch_size=4096, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for cont, pid, bid, gp, gb, y in dl_valid:
            preds.append(torch.sigmoid(model(cont, pid, bid, gp, gb)).numpy())
    p_valid = np.concatenate(preds)
    m = evaluate(y_valid[:n_v], p_valid)
    print(f"  [{('gated' if gated else 'plain')}] valid(subset) BSS={m['bss']:.5f} score={m['leaderboard_score']:.1f}",
          flush=True)
    return m


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    print("df loaded", df.shape, flush=True)

    for gated in [False, True]:
        print(f"\n--- gated={gated} ---", flush=True)
        train_variant(df, 2023, 2024, gated=gated, max_epochs=2, subset=20000)

    print(f"\n스모크 테스트 총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
