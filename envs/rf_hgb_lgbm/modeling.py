from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from phase2_common import time_split_es  # noqa: E402

SEED = 42

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=200,
    max_features="sqrt",
    n_jobs=6,
    random_state=SEED,
)

HGB_PARAMS = dict(
    max_depth=6,
    max_leaf_nodes=31,
    max_iter=700,
    learning_rate=0.03,
    l2_regularization=5.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=SEED,
)

LGBM_COMMON = dict(
    n_estimators=4000,
    subsample_freq=1,
    random_state=SEED,
    n_jobs=6,
    verbosity=-1,
)

LGBM_CLS_PARAMS = dict(
    **LGBM_COMMON,
    learning_rate=0.02,
    num_leaves=63,
    min_child_samples=160,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=5.0,
)

LGBM_L2_PARAMS = dict(
    **LGBM_COMMON,
    num_leaves=64,
    max_depth=12,
    learning_rate=0.005571638320335239,
    min_child_samples=28,
    subsample=0.9017762093981382,
    colsample_bytree=0.5291780969405919,
    reg_alpha=0.07089938907781941,
    reg_lambda=0.009306216375166584,
    min_split_gain=0.4888649495163153,
    max_bin=127,
)


def _probe_lgbm(model, X, y, objective):
    tr_idx, es_idx = time_split_es(len(X))
    eval_metric = "binary_logloss" if objective == "classifier" else "l2"
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[es_idx], y[es_idx])],
        eval_metric=eval_metric,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    return max(1, int(model.best_iteration_))


def fit_lgbm_classifier_full(X, y):
    probe = LGBMClassifier(**LGBM_CLS_PARAMS)
    best_iter = _probe_lgbm(probe, X, y, "classifier")
    params = {**LGBM_CLS_PARAMS, "n_estimators": best_iter}
    model = LGBMClassifier(**params).fit(X, y)
    return model, best_iter


def fit_lgbm_l2_full(X, y):
    y64 = y.astype(np.float64)
    probe = LGBMRegressor(**LGBM_L2_PARAMS)
    best_iter = _probe_lgbm(probe, X, y64, "l2")
    params = {**LGBM_L2_PARAMS, "n_estimators": best_iter}
    model = LGBMRegressor(**params).fit(X, y64)
    return model, best_iter


def fit_hgb_full(X, y):
    probe = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    best_iter = max(1, int(probe.n_iter_))
    params = {
        **HGB_PARAMS,
        "max_iter": best_iter,
        "early_stopping": False,
        "validation_fraction": None,
    }
    model = HistGradientBoostingClassifier(**params).fit(X, y)
    return model, best_iter


def fit_rf_full(X, y):
    return RandomForestClassifier(**RF_PARAMS).fit(X, y)


def predict_models(models, X):
    return {
        "rf": models["rf"].predict_proba(X)[:, 1],
        "hgb": models["hgb"].predict_proba(X)[:, 1],
        "lgbm_cls": models["lgbm_cls"].predict_proba(X)[:, 1],
        "lgbm_l2": np.clip(models["lgbm_l2"].predict(X), 0.0, 1.0),
    }
