"""Tests for regression_model. All data here is synthetic/local -- no
live GovMap or CBS access is required.
"""
import numpy as np
import pandas as pd
import pytest

from src.pricing.regression_features import FEATURE_COLUMNS, TARGET_COLUMN
from src.pricing.regression_model import (
    enforce_non_negative_predictions,
    predict,
    train_and_evaluate,
)


def _synthetic_training_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.uniform(40, 150, n)
    rooms = rng.integers(2, 6, n).astype(float)
    floor = rng.integers(0, 12, n).astype(float)
    price = 20_000 * area + 80_000 * rooms + 15_000 * floor + 1_000_000
    X = pd.DataFrame({"area_sqm": area, "rooms": rooms, "floor": floor})
    y = pd.Series(price, name=TARGET_COLUMN)
    return X, y


def test_model_trains_successfully_on_valid_synthetic_data():
    X, y = _synthetic_training_frame()
    result = train_and_evaluate(X, y)

    assert result["model"] is not None
    assert result["report"]["model_type"] == "LinearRegression"


def test_evaluation_metrics_are_produced():
    X, y = _synthetic_training_frame()
    result = train_and_evaluate(X, y)
    report = result["report"]

    for key in (
        "mae",
        "rmse",
        "r2",
        "intercept",
        "coefficients",
        "train_row_count",
        "test_row_count",
        "random_state",
        "training_dataset_size",
    ):
        assert key in report

    assert report["train_row_count"] + report["test_row_count"] == report["training_dataset_size"]
    assert report["r2"] > 0.99  # near-perfect linear fit, no noise
    assert set(report["coefficients"].keys()) == set(FEATURE_COLUMNS)


def test_coefficients_approximate_the_known_linear_relationship():
    X, y = _synthetic_training_frame(n=200)
    result = train_and_evaluate(X, y)
    coefs = result["report"]["coefficients"]

    assert coefs["area_sqm"] == pytest.approx(20_000, rel=0.05)
    assert coefs["rooms"] == pytest.approx(80_000, rel=0.05)
    assert coefs["floor"] == pytest.approx(15_000, rel=0.05)
    assert result["report"]["intercept"] == pytest.approx(1_000_000, rel=0.05)


def test_predict_returns_a_value_for_every_row():
    X, y = _synthetic_training_frame()
    result = train_and_evaluate(X, y)

    new_apartments = pd.DataFrame(
        {"area_sqm": [70.0, 100.0], "rooms": [3.0, 4.0], "floor": [1.0, 5.0]}
    )
    predictions = predict(result["model"], new_apartments)

    assert len(predictions) == 2
    assert predictions.notna().all()


def test_predict_ignores_extra_non_feature_columns():
    X, y = _synthetic_training_frame()
    result = train_and_evaluate(X, y)

    apartments_with_extra_columns = pd.DataFrame(
        {
            "area_sqm": [70.0],
            "rooms": [3.0],
            "floor": [1.0],
            "deal_id": [999],
            "original_price": [123456],
        }
    )
    predictions = predict(result["model"], apartments_with_extra_columns)
    assert len(predictions) == 1


def test_negative_predictions_are_never_silently_accepted():
    predictions = pd.Series([100_000.0, -50.0, 200_000.0], index=[0, 1, 2])
    cleaned = enforce_non_negative_predictions(predictions)

    assert cleaned.loc[0] == pytest.approx(100_000.0)
    assert np.isnan(cleaned.loc[1])
    assert cleaned.loc[2] == pytest.approx(200_000.0)
    # the row is never dropped -- it stays in the index, just unpriced
    assert list(cleaned.index) == [0, 1, 2]


def test_leakage_columns_are_not_used_as_predictors():
    forbidden = {
        "deal_id",
        "address",
        "transaction_date",
        "original_price",
        "original_price_per_sqm",
        "price_index_at_transaction",
        "current_price_index",
        "index_adjustment_factor",
        "adjusted_price_per_sqm",
        "source",
        "source_url",
        "exclusion_reason",
    }
    assert forbidden.isdisjoint(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
