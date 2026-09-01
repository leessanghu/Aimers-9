"""GPU embedding MLP experiments with leakage-safe rolling validation.

This module is intentionally self-contained so it can be pasted into or imported
from the Colab notebook. It uses only official row features and never builds
statistics from a validation season.
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET = "control_success"
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]

RAW_NUM_COLS = [
    "season", "game_month", "game_dayofweek", "inning", "balls_before",
    "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
]
RATE_GROUPS = {
    "asof_pitcher_n": [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
    ],
    "asof_batter_n": ["asof_batter_success_rate", "asof_batter_middle_rate"],
    "asof_pitcher_pitchmix_n": [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],
}
NO_N_RATE_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]
CAT_COLS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "top_bottom", "game_type", "base_state", "count_state", "hand_matchup",
]
ID_CAT_POSITIONS = (0, 1)


@dataclass
class TrainConfig:
    batch_size: int = 8192
    max_epochs: int = 28
    patience: int = 5
    lr: float = 1.8e-3
    weight_decay: float = 2e-4
    hidden_dim: int = 256
    blocks: int = 3
    dropout: float = 0.12
    id_dropout: float = 0.06
    cat_dropout: float = 0.01
    es_tail_fraction: float = 0.20
    min_delta: float = 2e-6
    num_workers: int = 0
    use_amp: bool = True
    mean_penalty: float = 0.0
    init_output_bias: bool = True
    target_prior_mode: str = "linear_trend"
    target_mean: float | None = None


EXPERIMENTS = [
    {"name": "embmlp_plain_v2", "context_gate": False, "loss": "brier", "seeds": [42]},
    {
        "name": "embmlp_hybrid_prior",
        "context_gate": False,
        "loss": "hybrid",
        "seeds": [42, 2026],
        "cfg": {"dropout": 0.16, "id_dropout": 0.10, "cat_dropout": 0.02, "lr": 1.4e-3},
    },
    {
        "name": "embmlp_meanreg_trend",
        "context_gate": False,
        "loss": "brier",
        "seeds": [42],
        "cfg": {"dropout": 0.18, "id_dropout": 0.12, "cat_dropout": 0.02, "mean_penalty": 0.04},
    },
    {
        "name": "embmlp_gated_hybrid",
        "context_gate": True,
        "loss": "hybrid",
        "seeds": [42],
        "cfg": {"dropout": 0.18, "id_dropout": 0.10, "cat_dropout": 0.02, "lr": 1.2e-3},
    },
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def bss_score(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    bs = float(np.mean((pred - y) ** 2))
    base = float(y.mean() * (1.0 - y.mean()))
    bss = 1.0 - bs / base
    return {"brier": bs, "bss": bss, "score": max(0.0, 100000.0 * bss)}


class FoldPreprocessor:
    """Fit-only-on-past numeric statistics and categorical vocabularies."""

    def __init__(self, smooth_k: float = 20.0):
        self.smooth_k = smooth_k

    @staticmethod
    def _cat_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in CAT_COLS[:-2]:
            out[col] = df[col].astype("string").fillna("<NA>")
        out["count_state"] = (
            df["balls_before"].astype("Int64").astype("string") + "-" +
            df["strikes_before"].astype("Int64").astype("string")
        )
        out["hand_matchup"] = (
            df["pitcher_hand"].astype("Int64").astype("string") + "-" +
            df["batter_hand"].astype("Int64").astype("string")
        )
        return out

    def fit(self, df: pd.DataFrame) -> "FoldPreprocessor":
        self.num_median = df[RAW_NUM_COLS].median(numeric_only=True).to_dict()
        rate_cols = [c for cols in RATE_GROUPS.values() for c in cols] + NO_N_RATE_COLS
        self.rate_mean = {c: float(df[c].mean(skipna=True)) for c in rate_cols}
        cats = self._cat_frame(df)
        self.cat_maps = {}
        self.cardinalities = []
        for col in CAT_COLS:
            values = pd.Index(cats[col].dropna().unique())
            self.cat_maps[col] = {value: i + 1 for i, value in enumerate(values)}
            self.cardinalities.append(len(values) + 1)

        raw = self._numeric_matrix(df)
        self.num_mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.num_std = raw.std(axis=0, dtype=np.float64).astype(np.float32)
        self.num_std[self.num_std < 1e-5] = 1.0
        return self

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        out: dict[str, np.ndarray] = {}
        for col in RAW_NUM_COLS:
            out[col] = df[col].fillna(self.num_median.get(col, 0.0)).to_numpy(np.float32)
        p_hand = df["pitcher_hand"].fillna(0).to_numpy(np.float32)
        b_hand = df["batter_hand"].fillna(0).to_numpy(np.float32)
        out["pitcher_hand"] = p_hand
        out["batter_hand"] = b_hand
        out["same_hand"] = (p_hand == b_hand).astype(np.float32)

        smoothed = {}
        for n_col, rate_cols in RATE_GROUPS.items():
            n = df[n_col].fillna(0).to_numpy(np.float32)
            out[f"log1p_{n_col}"] = np.log1p(n)
            out[f"flag_{n_col}_zero"] = (n == 0).astype(np.float32)
            for col in rate_cols:
                raw = df[col].fillna(0).to_numpy(np.float32)
                sm = (n * raw + self.smooth_k * self.rate_mean[col]) / (n + self.smooth_k)
                out[f"{col}_smooth"] = sm.astype(np.float32)
                smoothed[col] = sm

        missing = df[NO_N_RATE_COLS[0]].isna().to_numpy(np.float32)
        out["flag_prev_game_missing"] = missing
        for col in NO_N_RATE_COLS:
            out[col] = df[col].fillna(self.rate_mean[col]).to_numpy(np.float32)
        out["diff_success_rate"] = (
            smoothed["asof_pitcher_success_rate"] - smoothed["asof_batter_success_rate"]
        ).astype(np.float32)
        out["diff_middle_rate"] = (
            smoothed["asof_pitcher_middle_rate"] - smoothed["asof_batter_middle_rate"]
        ).astype(np.float32)
        self.numeric_names = list(out)
        return np.column_stack(list(out.values())).astype(np.float32, copy=False)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_num = self._numeric_matrix(df)
        x_num = np.clip((x_num - self.num_mean) / self.num_std, -10.0, 10.0).astype(np.float32)
        cats = self._cat_frame(df)
        x_cat = np.zeros((len(df), len(CAT_COLS)), dtype=np.int64)
        for j, col in enumerate(CAT_COLS):
            x_cat[:, j] = cats[col].map(self.cat_maps[col]).fillna(0).to_numpy(np.int64)
        return x_num, x_cat


def embedding_dim(cardinality: int) -> int:
    return min(32, max(4, int(round(1.6 * cardinality ** 0.56))))


def experiment_config(base_cfg: TrainConfig, experiment: dict) -> TrainConfig:
    overrides = experiment.get("cfg") or {}
    if not overrides:
        return base_cfg
    allowed = set(TrainConfig.__dataclass_fields__)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown TrainConfig override(s) for {experiment['name']}: {unknown}")
    return replace(base_cfg, **overrides)


def target_prior(df: pd.DataFrame, target_season: int, mode: str) -> float:
    y_mean = float(df[TARGET].mean())
    if mode == "train_mean":
        return y_mean
    season_mean = df.groupby("season")[TARGET].mean().sort_index()
    if mode == "last_season":
        return float(season_mean.iloc[-1])
    if mode != "linear_trend":
        raise ValueError(f"Unknown target_prior_mode: {mode}")
    if len(season_mean) < 2:
        return y_mean
    x = season_mean.index.to_numpy(np.float64)
    y = season_mean.to_numpy(np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = float(slope * target_season + intercept)
    lo = max(1e-4, float(y.min()) - 0.03)
    hi = min(1.0 - 1e-4, float(y.max()) + 0.03)
    return float(np.clip(pred, lo, hi))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class EmbeddingMLP(nn.Module):
    def __init__(self, n_num, cardinalities, cfg: TrainConfig, context_gate=False, output_bias=None):
        super().__init__()
        dims = [embedding_dim(c) for c in cardinalities]
        # Index 0 is a learned unknown category. ID dropout actively trains it,
        # which matters because new pitchers are common in the next season.
        self.embeddings = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, dims)])
        cat_dim = sum(dims)
        self.context_gate = context_gate
        self.gate = nn.Sequential(nn.Linear(cat_dim, n_num), nn.Sigmoid()) if context_gate else None
        self.input = nn.Linear(n_num + cat_dim, cfg.hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.blocks)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 1))
        if output_bias is not None:
            with torch.no_grad():
                p = float(np.clip(output_bias, 1e-5, 1.0 - 1e-5))
                self.head[-1].bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, x_num, x_cat):
        cat = torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)], dim=1)
        if self.context_gate:
            x_num = x_num * (0.5 + self.gate(cat))
        x = self.input(torch.cat([x_num, cat], dim=1))
        return self.head(self.blocks(x)).squeeze(1)


def make_loader(x_num, x_cat, y=None, batch_size=8192, shuffle=False, workers=0):
    tensors = [torch.from_numpy(x_num), torch.from_numpy(x_cat)]
    if y is not None:
        tensors.append(torch.from_numpy(np.asarray(y, dtype=np.float32)))
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )


def apply_category_dropout(x_cat, cfg: TrainConfig):
    x = x_cat.clone()
    if cfg.id_dropout > 0:
        mask = torch.rand((len(x), len(ID_CAT_POSITIONS)), device=x.device) < cfg.id_dropout
        for k, pos in enumerate(ID_CAT_POSITIONS):
            x[:, pos] = x[:, pos].masked_fill(mask[:, k], 0)
    if cfg.cat_dropout > 0:
        mask = torch.rand(x.shape, device=x.device) < cfg.cat_dropout
        x = x.masked_fill(mask, 0)
    return x


def loss_value(logits, y, mode, target_mean=None, mean_penalty=0.0):
    prob = torch.sigmoid(logits)
    brier = torch.mean((prob - y) ** 2)
    if mean_penalty > 0.0 and target_mean is not None:
        target = torch.as_tensor(float(target_mean), dtype=prob.dtype, device=prob.device)
        brier = brier + mean_penalty * (prob.mean() - target).pow(2)
    if mode == "brier":
        return brier
    if mode == "hybrid":
        return 0.75 * brier + 0.25 * nn.functional.binary_cross_entropy_with_logits(logits, y)
    raise ValueError(f"Unknown loss: {mode}")


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    parts = []
    for batch in loader:
        x_num, x_cat = batch[:2]
        logits = model(x_num.to(device, non_blocking=True), x_cat.to(device, non_blocking=True))
        parts.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(parts)


def fit_epochs(model, train_loader, valid_loader, y_valid, cfg, loss_mode, device,
               epochs, early_stop):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Keep the refit learning-rate trajectory identical to the epoch-selection
    # stage through best_epoch; only the stopping point changes.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.max_epochs)
    )
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_state, best_bs, best_epoch, stale = None, math.inf, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = apply_category_dropout(x_cat.to(device, non_blocking=True), cfg)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = loss_value(
                    model(x_num, x_cat), y, loss_mode,
                    target_mean=getattr(cfg, "target_mean", None),
                    mean_penalty=cfg.mean_penalty,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()

        if valid_loader is None:
            print(f"    epoch {epoch:02d}/{epochs} train_loss={running/seen:.6f}", flush=True)
            continue
        valid_pred = predict(model, valid_loader, device)
        valid_bs = float(np.mean((valid_pred - y_valid) ** 2))
        print(f"    epoch {epoch:02d} train_loss={running/seen:.6f} es_brier={valid_bs:.6f}", flush=True)
        if valid_bs < best_bs - cfg.min_delta:
            best_bs, best_epoch, stale = valid_bs, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if early_stop and stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_bs


def internal_time_split(df: pd.DataFrame, train_max: int, tail_fraction: float):
    current = np.flatnonzero(df["season"].to_numpy() == train_max)
    cut = max(1, int(len(current) * (1.0 - tail_fraction)))
    es_idx = current[cut:]
    fit_mask = np.ones(len(df), dtype=bool)
    fit_mask[es_idx] = False
    return np.flatnonzero(fit_mask), es_idx


def train_fold(train_fold, valid_fold, train_max, experiment, seed, cfg, device):
    seed_everything(seed)
    cfg = experiment_config(cfg, experiment)
    fit_idx, es_idx = internal_time_split(train_fold, train_max, cfg.es_tail_fraction)

    # Stage 1 selects epoch count without looking at the external validation year.
    prep_es = FoldPreprocessor().fit(train_fold.iloc[fit_idx])
    fit_num, fit_cat = prep_es.transform(train_fold.iloc[fit_idx])
    es_num, es_cat = prep_es.transform(train_fold.iloc[es_idx])
    fit_loader = make_loader(fit_num, fit_cat, train_fold.iloc[fit_idx][TARGET].to_numpy(),
                             cfg.batch_size, True, cfg.num_workers)
    es_loader = make_loader(es_num, es_cat, None, cfg.batch_size, False, cfg.num_workers)
    es_prior = target_prior(train_fold.iloc[fit_idx], train_max, cfg.target_prior_mode)
    cfg_es = replace(cfg, target_mean=es_prior)
    model = EmbeddingMLP(
        fit_num.shape[1], prep_es.cardinalities, cfg_es, experiment["context_gate"],
        output_bias=cfg_es.target_mean if cfg_es.init_output_bias else None,
    ).to(device)
    best_epoch, es_bs = fit_epochs(
        model, fit_loader, es_loader, train_fold.iloc[es_idx][TARGET].to_numpy(np.float32),
        cfg_es, experiment["loss"], device, cfg_es.max_epochs, True,
    )
    del model, prep_es, fit_num, fit_cat, es_num, es_cat, fit_loader, es_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 2 refits preprocessing and model on every allowed row for the chosen epochs.
    seed_everything(seed)
    prep = FoldPreprocessor().fit(train_fold)
    train_num, train_cat = prep.transform(train_fold)
    valid_num, valid_cat = prep.transform(valid_fold)
    train_loader = make_loader(train_num, train_cat, train_fold[TARGET].to_numpy(),
                               cfg.batch_size, True, cfg.num_workers)
    valid_loader = make_loader(valid_num, valid_cat, None, cfg.batch_size, False, cfg.num_workers)
    full_prior = target_prior(train_fold, int(valid_fold["season"].iloc[0]), cfg.target_prior_mode)
    cfg_full = replace(cfg, target_mean=full_prior)
    model = EmbeddingMLP(
        train_num.shape[1], prep.cardinalities, cfg_full, experiment["context_gate"],
        output_bias=cfg_full.target_mean if cfg_full.init_output_bias else None,
    ).to(device)
    fit_epochs(model, train_loader, None, None, cfg_full, experiment["loss"], device,
               max(2, best_epoch), False)
    pred = predict(model, valid_loader, device)
    unknown = {
        "pitcher_seen": valid_cat[:, 0] != 0,
        "batter_seen": valid_cat[:, 1] != 0,
    }
    return pred, best_epoch, es_bs, unknown


def run_experiments(train: pd.DataFrame, pred_dir: Path, cfg=None, experiments=None):
    cfg = cfg or TrainConfig()
    experiments = experiments or EXPERIMENTS
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Phase 3. Select a Colab GPU runtime.")
    device = torch.device("cuda")
    summary = []

    for train_max, valid_season in FOLDS:
        train_fold = train.loc[train["season"] <= train_max].reset_index(drop=True)
        valid_fold = train.loc[train["season"] == valid_season].reset_index(drop=True)
        y_valid = valid_fold[TARGET].to_numpy(np.float32)
        print(f"\n===== train<={train_max} -> valid={valid_season} =====", flush=True)
        for experiment in experiments:
            seed_preds, epoch_list = [], []
            started = time.time()
            unknown = None
            for seed in experiment["seeds"]:
                print(f"  {experiment['name']} seed={seed}", flush=True)
                pred, best_epoch, es_bs, unknown = train_fold_fn(
                    train_fold, valid_fold, train_max, experiment, seed, cfg, device
                )
                seed_preds.append(pred)
                epoch_list.append(best_epoch)
                print(f"    selected_epoch={best_epoch} internal_es_brier={es_bs:.6f}", flush=True)
            pred = np.mean(seed_preds, axis=0)
            metrics = bss_score(y_valid, pred)
            row = {
                "model": experiment["name"], "valid_season": valid_season,
                "seeds": str(experiment["seeds"]), "epochs": str(epoch_list),
                "target_mean": float(y_valid.mean()), "pred_mean": float(pred.mean()),
                "seconds": round(time.time() - started, 1), **metrics,
            }
            for label, mask in unknown.items():
                row[f"{label}_n"] = int(mask.sum())
                row[f"{label}_bss"] = bss_score(y_valid[mask], pred[mask])["bss"]
                if (~mask).any():
                    row[f"{label.replace('seen', 'unseen')}_n"] = int((~mask).sum())
                    row[f"{label.replace('seen', 'unseen')}_bss"] = bss_score(y_valid[~mask], pred[~mask])["bss"]
            summary.append(row)
            out = pd.DataFrame({
                "row_id": valid_fold["row_id"].to_numpy(), "y_valid": y_valid,
                f"pred_{experiment['name']}": pred,
            })
            path = pred_dir / f"fold_{valid_season}_pred_{experiment['name']}.csv"
            out.to_csv(path, index=False)
            print(f"  {experiment['name']} BSS={metrics['bss']:.6f} score={metrics['score']:.1f} saved={path}")
            del seed_preds
            gc.collect()
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(pred_dir / "phase3_embedding_summary.csv", index=False)
    with open(pred_dir / "phase3_embedding_config.json", "w", encoding="utf-8") as f:
        json.dump({"train": asdict(cfg), "experiments": experiments}, f, ensure_ascii=True, indent=2)
    return summary_df


# Alias avoids shadowing by the train_fold dataframe in run_experiments.
train_fold_fn = train_fold
