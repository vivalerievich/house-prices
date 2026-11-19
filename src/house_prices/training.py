"""Reproduce the notebook's fixed CatBoost model and median baseline."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from house_prices.data import IDENTIFIER, TARGET
from house_prices.features import build_engineered_pipeline

SEED = 42
CV_FOLDS = 5
HOLDOUT_SHARE = 0.2
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "rmsle": "neg_root_mean_squared_log_error",
}


def inverse_prices(log_prices: np.ndarray) -> np.ndarray:
    """Convert log1p predictions to nonnegative prices."""
    return np.expm1(np.maximum(log_prices, 0.0))


def build_model() -> TransformedTargetRegressor:
    """Use the CatBoost configuration selected in the full notebook study."""
    regressor = CatBoostRegressor(
        depth=4,
        iterations=600,
        l2_leaf_reg=3,
        learning_rate=0.1,
        subsample=0.8,
        loss_function="RMSE",
        bootstrap_type="Bernoulli",
        task_type="CPU",
        random_seed=SEED,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    pipeline = Pipeline(
        [("preparation", build_engineered_pipeline()), ("regressor", regressor)]
    )
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=inverse_prices
    )


def score_prices(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "rmsle": root_mean_squared_log_error(actual, predicted),
    }


def fit_and_evaluate(data: pd.DataFrame) -> tuple[TransformedTargetRegressor, dict]:
    """Evaluate two fixed estimators using identical folds and a shared holdout."""
    if len(data) < 13:
        raise ValueError(
            "At least 13 rows are required for five-fold CV and a holdout."
        )
    predictors = data.drop(columns=[IDENTIFIER, TARGET])
    train_x, holdout_x, train_y, holdout_y = train_test_split(
        predictors, data[TARGET], test_size=HOLDOUT_SHARE, random_state=SEED
    )
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    model = build_model()
    estimates = {}
    for name, estimator in (
        ("median_baseline", DummyRegressor(strategy="median")),
        ("catboost", model),
    ):
        scores = cross_validate(
            estimator, train_x, train_y, cv=cv, scoring=SCORING, error_score="raise"
        )
        cv_summary = {}
        for metric, scorer in SCORING.items():
            values = scores[f"test_{metric}"]
            if scorer.startswith("neg_"):
                values = -values
            cv_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
        estimator.fit(train_x, train_y)
        estimates[name] = {
            "cv": cv_summary,
            "holdout": score_prices(holdout_y, estimator.predict(holdout_x)),
        }

    return model, {
        "seed": SEED,
        "train_rows": len(train_x),
        "holdout_rows": len(holdout_x),
        "cv_folds": CV_FOLDS,
        "models": estimates,
    }
