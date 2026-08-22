"""Tests for current_market_model. Training data here is synthetic/local
-- no live network access is required. One test reads the already-
committed data/processed/apartments.csv (a local project file, not a
network call) to verify all 39 real target apartments can be mapped and
priced.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pricing.current_market_features import (
    apartments_to_market_feature_frame,
    listings_to_training_frame,
    select_training_rows,
)
from src.pricing.current_market_model import predict, train_and_evaluate
from src.pricing.pricing_utils import enforce_non_negative_predictions


def _synthetic_listings(n=150, seed=2):
    rng = np.random.default_rng(seed)
    area = rng.uniform(40, 160, n)
    rooms = rng.integers(2, 6, n).astype(float)
    floor = rng.integers(0, 15, n).astype(float)
    balcony = rng.uniform(0, 30, n)
    parking = rng.integers(0, 3, n).astype(float)
    storage = rng.uniform(0, 10, n)
    garden = rng.choice([0.0, 0.0, 0.0, 20.0], n)
    roof = rng.choice([0.0, 0.0, 0.0, 30.0], n)
    property_type = rng.choice(["Apartment", "Duplex", "Garden Apartment", "Penthouse"], n)
    market_segment = rng.choice(["Second Hand", "New Project"], n)
    balcony_direction = rng.choice(["North", "South", "East", "West"], n)
    is_top_floor = rng.choice([True, False], n)

    segment_premium = np.where(market_segment == "New Project", 400_000, 0)
    price = (
        30_000 * area
        + 60_000 * rooms
        + 20_000 * floor
        + 5_000 * balcony
        + 15_000 * parking
        + 3_000 * storage
        + 8_000 * garden
        + 6_000 * roof
        + segment_premium
        + 800_000
    )
    return pd.DataFrame(
        {
            "listing_id": [f"SYN-{i}" for i in range(n)],
            "area_sqm": area,
            "rooms": rooms,
            "floor": floor,
            "balcony_area_sqm": balcony,
            "parking_count": parking,
            "storage_area_sqm": storage,
            "garden_area_sqm": garden,
            "roof_area_sqm": roof,
            "property_type": property_type,
            "market_segment": market_segment,
            "balcony_direction": balcony_direction,
            "is_top_floor": is_top_floor,
            "asking_price": price,
        }
    )


def test_model_trains_successfully_on_synthetic_data():
    listings = _synthetic_listings()
    trainable = select_training_rows(listings)
    X, y = listings_to_training_frame(trainable)

    result = train_and_evaluate(X, y, input_file="synthetic_test.xlsx")

    assert result["model"] is not None
    assert result["report"]["model_type"].startswith("LinearRegression")
    assert result["report"]["data_type"] == "synthetic_poc"


def test_evaluation_metrics_are_produced():
    listings = _synthetic_listings()
    trainable = select_training_rows(listings)
    X, y = listings_to_training_frame(trainable)

    result = train_and_evaluate(X, y)
    report = result["report"]

    for key in ("mae", "rmse", "r2", "training_rows", "test_rows", "random_state", "features"):
        assert key in report
    assert report["training_rows"] + report["test_rows"] == len(X)
    assert report["r2"] > 0.9  # strong synthetic signal, near-perfect fit


def test_predict_all_39_real_project_apartments():
    apartments_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "apartments.csv"
    if not apartments_path.exists():
        pytest.skip("data/processed/apartments.csv not available")

    listings = _synthetic_listings()
    trainable = select_training_rows(listings)
    X, y = listings_to_training_frame(trainable)
    result = train_and_evaluate(X, y)

    apartments = pd.read_csv(apartments_path)
    assert len(apartments) == 39

    features = apartments_to_market_feature_frame(apartments)
    priceable = features[features["is_priceable"]]
    assert len(priceable) == 39  # real enriched apartment data has no missing fields

    raw_predictions = predict(result["model"], priceable)
    predictions = enforce_non_negative_predictions(raw_predictions)

    assert len(predictions) == 39
    assert predictions.notna().all()
    assert (predictions > 0).all()
