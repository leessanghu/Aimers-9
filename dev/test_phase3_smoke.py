"""Small CPU smoke test for the Colab Phase 3 training core."""

import base64
import gzip
import json
from pathlib import Path

import pandas as pd
import torch

from phase3_embedding_mlp import (
    EmbeddingMLP,
    FoldPreprocessor,
    TrainConfig,
    fit_epochs,
    make_loader,
    predict,
)


ROOT = Path(__file__).resolve().parents[1]


def assert_notebook_embedded_core_is_current(notebook_name):
    notebook = json.loads((ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<notebook-cell-{i}>", "exec")
    cell_source = "".join(next(
        cell["source"] for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and any("_embedded_core =" in line for line in cell["source"])
    ))
    encoded_line = next(line for line in cell_source.splitlines() if line.startswith("_embedded_core ="))
    encoded = encoded_line.split("'", 2)[1]
    embedded = gzip.decompress(base64.b64decode(encoded))
    assert embedded == (ROOT / "dev" / "phase3_embedding_mlp.py").read_bytes()


def test_embedded_core_is_current():
    assert_notebook_embedded_core_is_current("colab_gpu_phase3.ipynb")
    assert_notebook_embedded_core_is_current("phase3_embedding_mlp.ipynb")


def test_preprocess_forward_and_backward():
    df = pd.read_csv(ROOT / "data" / "train.csv", nrows=6000, encoding="utf-8-sig")
    fit_df, valid_df = df.iloc[:4500], df.iloc[4500:]
    prep = FoldPreprocessor().fit(fit_df)
    train_num, train_cat = prep.transform(fit_df)
    valid_num, valid_cat = prep.transform(valid_df)
    cfg = TrainConfig(
        batch_size=1024, max_epochs=1, patience=1,
        hidden_dim=64, blocks=1, use_amp=False,
    )
    model = EmbeddingMLP(train_num.shape[1], prep.cardinalities, cfg, context_gate=True)
    train_loader = make_loader(
        train_num, train_cat, fit_df["control_success"].to_numpy(),
        cfg.batch_size, True, 0,
    )
    valid_loader = make_loader(valid_num, valid_cat, None, cfg.batch_size, False, 0)
    fit_epochs(
        model, train_loader, None, None, cfg, "brier", torch.device("cpu"), 1, False,
    )
    pred = predict(model, valid_loader, torch.device("cpu"))
    assert pred.shape == (len(valid_df),)
    assert 0.0 < pred.min() <= pred.max() < 1.0


if __name__ == "__main__":
    test_embedded_core_is_current()
    test_preprocess_forward_and_backward()
    print("phase3 smoke test: OK")
