"""Baseline transaction-price regression: training, evaluation, and prediction.

Data preparation (selecting training rows, mapping the apartment/
transaction schemas onto canonical features) lives in
regression_features.py -- this module only fits/evaluates/uses the model,
which is a simple, interpretable scikit-learn LinearRegression. No
ensembles, no neural networks, no agents.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config.settings import (
    REGRESSION_MODEL_VERSION,
    REGRESSION_RANDOM_STATE,
    REGRESSION_TEST_SIZE,
)
from src.pricing.pricing_utils import enforce_non_negative_predictions  # noqa: F401  (re-exported)
from src.pricing.regression_features import FEATURE_COLUMNS, TARGET_COLUMN


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Split X/y, fit a LinearRegression, and evaluate it on the held-out split.

    Returns {"model": fitted LinearRegression, "report": metadata/metrics
    dict} -- the report is the "model artifact" saved for reproducibility
    (see scripts/train_baseline_pricing_model.py).

    Coefficients are reported as the relationship this baseline model
    learned while holding the other included features constant -- not as
    a causal effect.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=REGRESSION_TEST_SIZE, random_state=REGRESSION_RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    coefficients = dict(zip(FEATURE_COLUMNS, (float(c) for c in model.coef_)))

    report = {
        "model_type": "LinearRegression",
        "model_version": REGRESSION_MODEL_VERSION,
        "training_features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "training_dataset_size": int(len(X)),
        "train_row_count": int(len(X_train)),
        "test_row_count": int(len(X_test)),
        "test_size": REGRESSION_TEST_SIZE,
        "random_state": REGRESSION_RANDOM_STATE,
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"model": model, "report": report}


def predict(model: LinearRegression, X: pd.DataFrame) -> pd.Series:
    """Predict prices for X using only the canonical FEATURE_COLUMNS.

    Any other columns present in X (ids, addresses, leakage fields, etc.)
    are ignored, never used as predictors.
    """
    values = model.predict(X[FEATURE_COLUMNS])
    return pd.Series(values, index=X.index, name="predicted_price")
