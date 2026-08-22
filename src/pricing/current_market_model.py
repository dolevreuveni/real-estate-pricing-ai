"""Current Market pricing model: training, evaluation, and prediction.

Data preparation (loading, filtering, and mapping the apartment/listing
schemas onto canonical features) lives in current_market_features.py --
this module only builds/fits/evaluates/uses the model.

An interpretable scikit-learn pipeline: numeric features pass through
unscaled (so coefficients stay directly interpretable in original units,
matching the historical regression model's convention) and categorical
features go through a OneHotEncoder. Still plain LinearRegression --
no ensembles, no neural networks.

This model is entirely separate from src/pricing/regression_model.py
(the historical GovMap/CBS regression) -- the two are independent
pricing signals, combined only in src/pricing/pricing_recommendation.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config.settings import (
    CURRENT_MARKET_DATA_TYPE,
    CURRENT_MARKET_MODEL_VERSION,
    REGRESSION_RANDOM_STATE,
    REGRESSION_TEST_SIZE,
)
from src.pricing.current_market_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("regression", LinearRegression())])


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, input_file: str = "") -> dict:
    """Split X/y, fit the pipeline, and evaluate it on the held-out split.

    Uses the same random_state/test_size convention as the historical
    regression model (src/config/settings.py REGRESSION_RANDOM_STATE /
    REGRESSION_TEST_SIZE) for a reproducible split.

    Returns {"model": fitted Pipeline, "report": metadata/metrics dict}.
    Because the training data is synthetic POC data, the report is
    explicitly labeled data_type="synthetic_poc" -- these metrics validate
    the pipeline, not real-world model accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=REGRESSION_TEST_SIZE, random_state=REGRESSION_RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    coefficients = dict(
        zip(feature_names, (float(c) for c in pipeline.named_steps["regression"].coef_))
    )

    report = {
        "model_type": "LinearRegression (ColumnTransformer: passthrough numeric + OneHotEncoder categorical)",
        "model_version": CURRENT_MARKET_MODEL_VERSION,
        "data_type": CURRENT_MARKET_DATA_TYPE,
        "input_file": str(input_file),
        "target": TARGET_COLUMN,
        "features": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_dataset_size": int(len(X)),
        "test_size": REGRESSION_TEST_SIZE,
        "random_state": REGRESSION_RANDOM_STATE,
        "coefficients": coefficients,
        "intercept": float(pipeline.named_steps["regression"].intercept_),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"model": pipeline, "report": report}


def predict(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Predict prices for X using only the canonical FEATURE_COLUMNS.

    Any other columns present in X are ignored, never used as predictors.
    """
    values = model.predict(X[FEATURE_COLUMNS])
    return pd.Series(values, index=X.index, name="predicted_price")
