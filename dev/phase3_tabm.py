"""Leakage-safe rolling validation for TabM and TabM with PWL embeddings."""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import rtdl_num_embeddings
import tabm

from phase3_embedding_mlp import (
    FOLDS,
    TARGET,
    FoldPreprocessor,
    bss_score,
    internal_time_split,
    make_loader,
    seed_everything,
)


@dataclass
class TabMTrainConfig:
    batch_size: int = 1024
    eval_batch_size: int = 4096
    max_epochs: int = 40
    patience: int = 7
    es_tail_fraction: float = 0.20
    min_delta: float = 2e-6
    id_dropout: float = 0.05
    cat_dropout: float = 0.01
    gradient_clip: float = 1.0
    num_workers: int = 0
    use_amp: bool = True
    bin_sample_size: int = 400_000
    save_member_predictions: bool = True


EXPERIMENTS = [
    {
        "name": "tabm_base",
        "embedding": "none",
        "seeds": [42],
        "k": 32,
        "n_blocks": 3,
        "d_block": 512,
        "dropout": 0.10,
        "lr": 1.5e-3,
        "weight_decay": 3e-4,
    },
    {
        "name": "tabm_pwl",
        "embedding": "pwl",
        "seeds": [42],
        "k": 32,
        "n_blocks": 2,
        "d_block": 512,
        "dropout": 0.10,
        "lr": 1.5e-3,
        "weight_decay": 3e-4,
        "n_bins": 48,
        "d_embedding": 16,
    },
]


def _category_dropout(x_cat: torch.Tensor, cfg: TabMTrainConfig) -> torch.Tensor:
    x = x_cat.clone()
    if cfg.id_dropout:
        id_mask = torch.rand((len(x), 2), device=x.device) < cfg.id_dropout
        x[:, :2] = x[:, :2].masked_fill(id_mask, 0)
    if cfg.cat_dropout:
        x = x.masked_fill(torch.rand(x.shape, device=x.device) < cfg.cat_dropout, 0)
    return x


def _make_num_embeddings(x_num: np.ndarray, experiment: dict, cfg: TabMTrainConfig, seed: int):
    if experiment["embedding"] == "none":
        return None
    if experiment["embedding"] != "pwl":
        raise ValueError(f"Unknown numerical embedding: {experiment['embedding']}")

    rng = np.random.default_rng(seed)
    if len(x_num) > cfg.bin_sample_size:
        idx = np.sort(rng.choice(len(x_num), cfg.bin_sample_size, replace=False))
        bin_source = x_num[idx]
    else:
        bin_source = x_num
    bins = rtdl_num_embeddings.compute_bins(
        torch.from_numpy(np.ascontiguousarray(bin_source)),
        n_bins=experiment["n_bins"],
    )
    return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bins,
        d_embedding=experiment["d_embedding"],
        activation=False,
        version="B",
    )


def make_model(x_num: np.ndarray, cardinalities: list[int], experiment: dict,
               cfg: TabMTrainConfig, seed: int, device: torch.device):
    num_embeddings = _make_num_embeddings(x_num, experiment, cfg, seed)
    model = tabm.TabM.make(
        n_num_features=x_num.shape[1],
        cat_cardinalities=cardinalities,
        d_out=1,
        num_embeddings=num_embeddings,
        arch_type="tabm",
        k=experiment["k"],
        n_blocks=experiment["n_blocks"],
        d_block=experiment["d_block"],
        dropout=experiment["dropout"],
    )
    return model.to(device)


def _amp_settings(cfg: TabMTrainConfig, device: torch.device):
    enabled = cfg.use_amp and device.type == "cuda"
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype


@torch.inference_mode()
def predict_members(model, loader, cfg: TabMTrainConfig, device: torch.device):
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(cfg, device)
    parts = []
    for batch in loader:
        x_num, x_cat = batch[:2]
        with torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(
                x_num.to(device, non_blocking=True),
                x_cat.to(device, non_blocking=True),
            ).squeeze(-1)
        parts.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(parts, axis=0)


def fit_epochs(model, train_loader, valid_loader, y_valid, experiment, cfg,
               device, epochs, early_stop):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment["lr"], weight_decay=experiment["weight_decay"]
    )
    amp_enabled, amp_dtype = _amp_settings(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    best_bs, best_epoch, stale = math.inf, epochs, 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = _category_dropout(x_cat.to(device, non_blocking=True), cfg)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
                logits = model(x_num, x_cat).squeeze(-1)
            # Each of the k members is optimized independently. Averaging here is incorrect.
            prob = torch.sigmoid(logits.float())
            loss = torch.mean((prob - y[:, None]) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(y)
            n_seen += len(y)

        if valid_loader is None:
            print(f"    epoch {epoch:02d}/{epochs} train_brier={loss_sum/n_seen:.6f}", flush=True)
            continue

        member_pred = predict_members(model, valid_loader, cfg, device)
        ensemble_pred = member_pred.mean(axis=1)
        valid_bs = float(np.mean((ensemble_pred - y_valid) ** 2))
        member_std = float(member_pred.std(axis=1).mean())
        print(
            f"    epoch {epoch:02d} train_brier={loss_sum/n_seen:.6f} "
            f"es_brier={valid_bs:.6f} member_spread={member_std:.5f}",
            flush=True,
        )
        if valid_bs < best_bs - cfg.min_delta:
            best_bs, best_epoch, stale = valid_bs, epoch, 0
        else:
            stale += 1
            if early_stop and stale >= cfg.patience:
                break
    return best_epoch, best_bs


def train_one_fold(train_df, valid_df, train_max, experiment, seed, cfg, device):
    seed_everything(seed)
    fit_idx, es_idx = internal_time_split(train_df, train_max, cfg.es_tail_fraction)

    # Epoch selection sees only an internal tail of the latest allowed season.
    prep_es = FoldPreprocessor().fit(train_df.iloc[fit_idx])
    fit_num, fit_cat = prep_es.transform(train_df.iloc[fit_idx])
    es_num, es_cat = prep_es.transform(train_df.iloc[es_idx])
    fit_loader = make_loader(
        fit_num, fit_cat, train_df.iloc[fit_idx][TARGET].to_numpy(),
        cfg.batch_size, True, cfg.num_workers,
    )
    es_loader = make_loader(
        es_num, es_cat, None, cfg.eval_batch_size, False, cfg.num_workers,
    )
    model = make_model(fit_num, prep_es.cardinalities, experiment, cfg, seed, device)
    best_epoch, es_bs = fit_epochs(
        model, fit_loader, es_loader, train_df.iloc[es_idx][TARGET].to_numpy(np.float32),
        experiment, cfg, device, cfg.max_epochs, True,
    )
    del model, prep_es, fit_num, fit_cat, es_num, es_cat, fit_loader, es_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Refit from scratch on all past rows for exactly the selected number of epochs.
    seed_everything(seed)
    prep = FoldPreprocessor().fit(train_df)
    train_num, train_cat = prep.transform(train_df)
    valid_num, valid_cat = prep.transform(valid_df)
    train_loader = make_loader(
        train_num, train_cat, train_df[TARGET].to_numpy(),
        cfg.batch_size, True, cfg.num_workers,
    )
    valid_loader = make_loader(
        valid_num, valid_cat, None, cfg.eval_batch_size, False, cfg.num_workers,
    )
    model = make_model(train_num, prep.cardinalities, experiment, cfg, seed, device)
    fit_epochs(
        model, train_loader, None, None, experiment, cfg, device,
        max(1, best_epoch), False,
    )
    members = predict_members(model, valid_loader, cfg, device)
    seen = {"pitcher_seen": valid_cat[:, 0] != 0, "batter_seen": valid_cat[:, 1] != 0}
    del model, prep, train_num, train_cat, valid_num, valid_cat, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    return members, best_epoch, es_bs, seen


def run_tabm_experiments(train: pd.DataFrame, pred_dir: Path, cfg=None, experiments=None):
    cfg = cfg or TabMTrainConfig()
    experiments = experiments or EXPERIMENTS
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before starting TabM.")
    device = torch.device("cuda")
    rows = []

    for train_max, valid_season in FOLDS:
        train_df = train.loc[train["season"] <= train_max].reset_index(drop=True)
        valid_df = train.loc[train["season"] == valid_season].reset_index(drop=True)
        y_valid = valid_df[TARGET].to_numpy(np.float32)
        print(f"\n===== TabM train<={train_max} -> valid={valid_season} =====", flush=True)

        for experiment in experiments:
            started = time.time()
            seed_ensembles, epochs = [], []
            member_bss_means, member_bss_maxes, member_spreads = [], [], []
            seen = None
            for seed in experiment["seeds"]:
                print(f"  {experiment['name']} seed={seed}", flush=True)
                members, best_epoch, es_bs, seen = train_one_fold(
                    train_df, valid_df, train_max, experiment, seed, cfg, device
                )
                seed_ensembles.append(members.mean(axis=1))
                epochs.append(best_epoch)
                if cfg.save_member_predictions:
                    np.savez_compressed(
                        pred_dir / f"fold_{valid_season}_{experiment['name']}_seed{seed}_members.npz",
                        pred_members=members.astype(np.float32),
                        y_valid=y_valid,
                    )
                member_bss = np.array([bss_score(y_valid, members[:, j])["bss"] for j in range(members.shape[1])])
                member_bss_means.append(float(member_bss.mean()))
                member_bss_maxes.append(float(member_bss.max()))
                member_spreads.append(float(members.std(axis=1).mean()))
                print(
                    f"    selected_epoch={best_epoch} es_brier={es_bs:.6f} "
                    f"member_bss_mean={member_bss.mean():.6f} max={member_bss.max():.6f}",
                    flush=True,
                )
                del members
                gc.collect()

            pred = np.mean(seed_ensembles, axis=0)
            metrics = bss_score(y_valid, pred)
            row = {
                "model": experiment["name"],
                "valid_season": valid_season,
                "seeds": str(experiment["seeds"]),
                "epochs": str(epochs),
                "target_mean": float(y_valid.mean()),
                "pred_mean": float(pred.mean()),
                "calib_diff": float(pred.mean() - y_valid.mean()),
                "member_bss_mean": float(np.mean(member_bss_means)),
                "member_bss_max": float(np.max(member_bss_maxes)),
                "member_spread": float(np.mean(member_spreads)),
                "ensemble_gain_vs_member_mean": float(
                    metrics["bss"] - np.mean(member_bss_means)
                ),
                "seconds": round(time.time() - started, 1),
                **metrics,
            }
            for label, mask in seen.items():
                row[f"{label}_n"] = int(mask.sum())
                row[f"{label}_bss"] = bss_score(y_valid[mask], pred[mask])["bss"]
                if (~mask).any():
                    unseen = label.replace("seen", "unseen")
                    row[f"{unseen}_n"] = int((~mask).sum())
                    row[f"{unseen}_bss"] = bss_score(y_valid[~mask], pred[~mask])["bss"]
            rows.append(row)

            pd.DataFrame({
                "row_id": valid_df["row_id"].to_numpy(),
                "y_valid": y_valid,
                f"pred_{experiment['name']}": pred,
            }).to_csv(
                pred_dir / f"fold_{valid_season}_pred_{experiment['name']}.csv", index=False
            )
            print(
                f"  {experiment['name']} BSS={metrics['bss']:.6f} "
                f"score={metrics['score']:.1f}", flush=True
            )
            del seed_ensembles
            gc.collect()

    summary = pd.DataFrame(rows)
    summary.to_csv(pred_dir / "phase3_tabm_summary.csv", index=False)
    with open(pred_dir / "phase3_tabm_config.json", "w", encoding="utf-8") as f:
        json.dump({"train": asdict(cfg), "experiments": experiments}, f, indent=2)
    return summary
