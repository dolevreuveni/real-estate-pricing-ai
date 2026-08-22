"""End-to-end synthetic test of the Feature #8 / #7.5 pipeline: train the
Current Market model, predict for target apartments, and combine with a
(synthetic, in-memory) historical signal into recommended_marketing_price.
Mirrors scripts/generate_pricing_recommendations.py without touching real
files or the network.
"""
import numpy as np
import pandas as pd

from src.config.settings import CURRENT_MARKET_DATA_TYPE
from src.pricing.current_market_features import (
    apartments_to_market_feature_frame,
    listings_to_training_frame,
    select_training_rows,
)
from src.pricing.current_market_model import predict, train_and_evaluate
from src.pricing.pricing_recommendation import combine_prices
from src.pricing.pricing_utils import enforce_non_negative_predictions

FINAL_COLUMNS = [
    "apartment_id",
    "rooms",
    "floor_min",
    "floor_max",
    "num_levels",
    "interior_area_sqm",
    "balcony_area_sqm",
    "balcony_direction",
    "directions",
    "parking_count",
    "storage_area_sqm",
    "garden_area_sqm",
    "roof_area_sqm",
    "is_top_floor",
    "property_type",
    "historical_base_price",
    "historical_base_price_per_sqm",
    "current_market_price",
    "current_market_price_per_sqm",
    "historical_weight",
    "current_market_weight",
    "recommended_marketing_price",
    "recommended_marketing_price_per_sqm",
    "historical_model_version",
    "current_market_model_version",
    "current_market_data_type",
    "pricing_status",
]


def _synthetic_listings(n=120, seed=3):
    rng = np.random.default_rng(seed)
    area = rng.uniform(40, 160, n)
    rooms = rng.integers(2, 6, n).astype(float)
    floor = rng.integers(0, 12, n).astype(float)
    balcony = rng.uniform(0, 30, n)
    parking = rng.integers(0, 3, n).astype(float)
    storage = rng.uniform(0, 10, n)
    garden = rng.choice([0.0, 0.0, 20.0], n)
    roof = rng.choice([0.0, 0.0, 30.0], n)
    property_type = rng.choice(["Apartment", "Duplex", "Garden Apartment"], n)
    market_segment = rng.choice(["Second Hand", "New Project"], n)
    balcony_direction = rng.choice(["North", "South", "East", "West"], n)
    is_top_floor = rng.choice([True, False], n)
    price = (
        28_000 * area
        + 65_000 * rooms
        + 18_000 * floor
        + 4_000 * balcony
        + 10_000 * parking
        + 2_000 * storage
        + 900_000
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


def _synthetic_apartments():
    return pd.DataFrame(
        [
            {
                "apartment_id": 1,
                "rooms": 3,
                "floor_min": 1,
                "floor_max": 1,
                "num_levels": 1,
                "interior_area_sqm": 70.0,
                "balcony_area_sqm": 12.0,
                "balcony_direction": "East",
                "directions": "מזרח",
                "parking_count": 1,
                "storage_area_sqm": 4.0,
                "garden_area_sqm": 0.0,
                "roof_area_sqm": 0.0,
                "is_top_floor": False,
                "property_type": "regular",
            },
            {
                "apartment_id": 2,
                "rooms": 6,
                "floor_min": 9,
                "floor_max": 11,
                "num_levels": 3,
                "interior_area_sqm": 255.2,
                "balcony_area_sqm": 94.6,
                "balcony_direction": "South-West",
                "directions": "מזרח",
                "parking_count": 2,
                "storage_area_sqm": 8.0,
                "garden_area_sqm": 0.0,
                "roof_area_sqm": 76.6,
                "is_top_floor": True,
                "property_type": "triplex",
            },
        ]
    )


def test_final_recommendation_schema_and_synthetic_data_type_label():
    listings = _synthetic_listings()
    trainable = select_training_rows(listings)
    X, y = listings_to_training_frame(trainable)
    fit = train_and_evaluate(X, y, input_file="synthetic_test.xlsx")
    model, report = fit["model"], fit["report"]

    apartments = _synthetic_apartments()
    market_features = apartments_to_market_feature_frame(apartments)
    assert market_features["is_priceable"].all()

    raw = predict(model, market_features)
    current_market_price = enforce_non_negative_predictions(raw)

    # synthetic historical signal, standing in for Feature #7's output
    historical_base_price = pd.Series([4_000_000.0, 12_000_000.0], index=apartments.index)

    result = apartments.copy()
    result["current_market_price"] = current_market_price
    result["current_market_price_per_sqm"] = (
        result["current_market_price"] / result["interior_area_sqm"]
    )
    result["historical_base_price"] = historical_base_price
    result["historical_base_price_per_sqm"] = (
        result["historical_base_price"] / result["interior_area_sqm"]
    )
    result["historical_weight"] = 0.7
    result["current_market_weight"] = 0.3
    result["historical_model_version"] = "baseline_linear_v1"
    result["current_market_model_version"] = report["model_version"]
    result["current_market_data_type"] = CURRENT_MARKET_DATA_TYPE
    result["pricing_status"] = "priced"
    result["recommended_marketing_price"] = [
        combine_prices(h, c)
        for h, c in zip(result["historical_base_price"], result["current_market_price"])
    ]
    result["recommended_marketing_price_per_sqm"] = (
        result["recommended_marketing_price"] / result["interior_area_sqm"]
    )

    final = result[FINAL_COLUMNS]

    assert set(FINAL_COLUMNS).issubset(set(final.columns))
    assert (final["current_market_data_type"] == "synthetic_poc").all()
    assert final["recommended_marketing_price"].notna().all()
    assert (final["recommended_marketing_price"] > 0).all()
    # enriched fields survive into the final output
    assert final["parking_count"].notna().all()
    assert final["balcony_direction"].notna().all()
    assert final["is_top_floor"].notna().all()

    # sanity: recommended price is the documented 70/30 blend
    expected = (
        final["historical_base_price"] * final["historical_weight"]
        + final["current_market_price"] * final["current_market_weight"]
    )
    assert (final["recommended_marketing_price"] - expected).abs().max() < 1e-6
