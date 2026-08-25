"""Tests for src/pricing/custom_apartment_pricing.py -- the orchestration
helper shared by the dashboard's Project Pricing and Price Simulator
pages. All training data here is synthetic/local; no network access.
"""
import numpy as np
import pandas as pd
import pytest

from src.config.settings import CURRENT_MARKET_WEIGHT, HISTORICAL_MARKET_WEIGHT
from src.pricing.current_market_features import FEATURE_COLUMNS as MARKET_FEATURE_COLUMNS
from src.pricing.current_market_model import train_and_evaluate as train_market_model
from src.pricing.custom_apartment_pricing import (
    build_pricing_breakdown,
    custom_apartment_to_historical_features,
    custom_apartment_to_market_features,
    predict_custom_apartment,
)
from src.pricing.regression_features import FEATURE_COLUMNS as HISTORICAL_FEATURE_COLUMNS
from src.pricing.regression_model import train_and_evaluate as train_historical_model


def _custom_apartment(**overrides):
    apartment = {
        "rooms": 3,
        "interior_area_sqm": 75.0,
        "floor": 5,
        "num_levels": 1,
        "balcony_area_sqm": 14.0,
        "balcony_direction": "South",
        "parking_count": 1,
        "storage_area_sqm": 5.0,
        "garden_area_sqm": 0.0,
        "roof_area_sqm": 0.0,
        "is_top_floor": False,
        "property_type": "Apartment",
    }
    apartment.update(overrides)
    return apartment


def test_custom_apartment_maps_to_historical_features_correctly():
    apartment = _custom_apartment()
    X = custom_apartment_to_historical_features(apartment)

    assert list(X.columns) == HISTORICAL_FEATURE_COLUMNS
    assert X.iloc[0]["area_sqm"] == 75.0
    assert X.iloc[0]["rooms"] == 3
    assert X.iloc[0]["floor"] == 5


def test_custom_apartment_does_not_send_parking_storage_to_historical_model():
    apartment = _custom_apartment()
    X = custom_apartment_to_historical_features(apartment)

    forbidden = {"parking_count", "storage_area_sqm", "balcony_area_sqm", "balcony_direction",
                 "garden_area_sqm", "roof_area_sqm", "property_type", "is_top_floor", "market_segment"}
    assert forbidden.isdisjoint(X.columns)
    assert set(X.columns) == {"area_sqm", "rooms", "floor"}


def test_custom_apartment_maps_to_market_features_correctly():
    apartment = _custom_apartment()
    X = custom_apartment_to_market_features(apartment)

    assert list(X.columns) == MARKET_FEATURE_COLUMNS
    assert X.iloc[0]["parking_count"] == 1
    assert X.iloc[0]["storage_area_sqm"] == 5.0
    assert X.iloc[0]["balcony_direction"] == "South"
    assert bool(X.iloc[0]["is_top_floor"]) is False


def test_custom_apartment_uses_richer_fields_for_current_market():
    apartment = _custom_apartment()
    X = custom_apartment_to_market_features(apartment)

    richer_fields = {"parking_count", "storage_area_sqm", "garden_area_sqm", "roof_area_sqm",
                      "balcony_direction", "is_top_floor", "market_segment"}
    assert richer_fields.issubset(set(X.columns))


def test_no_apartment_id_used_as_feature_anywhere():
    apartment = _custom_apartment()
    historical_X = custom_apartment_to_historical_features(apartment)
    market_X = custom_apartment_to_market_features(apartment)

    assert "apartment_id" not in historical_X.columns
    assert "apartment_id" not in market_X.columns


def test_no_target_leakage_predictors_introduced():
    apartment = _custom_apartment()
    historical_X = custom_apartment_to_historical_features(apartment)
    market_X = custom_apartment_to_market_features(apartment)

    leakage_fields = {
        "asking_price", "price_per_sqm", "original_price", "original_price_per_sqm",
        "adjusted_price", "adjusted_price_per_sqm", "listing_id", "source_url", "deal_id",
    }
    assert leakage_fields.isdisjoint(historical_X.columns)
    assert leakage_fields.isdisjoint(market_X.columns)


def _synthetic_historical_fit():
    rng = np.random.default_rng(0)
    n = 60
    area = rng.uniform(40, 150, n)
    rooms = rng.integers(2, 6, n).astype(float)
    floor = rng.integers(0, 12, n).astype(float)
    price = 20_000 * area + 80_000 * rooms + 15_000 * floor + 1_000_000
    X = pd.DataFrame({"area_sqm": area, "rooms": rooms, "floor": floor})
    y = pd.Series(price)
    return train_historical_model(X, y)


def _synthetic_market_fit():
    rng = np.random.default_rng(1)
    n = 150
    area = rng.uniform(40, 160, n)
    rooms = rng.integers(2, 6, n).astype(float)
    floor = rng.integers(0, 15, n).astype(float)
    balcony = rng.uniform(0, 30, n)
    parking = rng.integers(0, 3, n).astype(float)
    storage = rng.uniform(0, 10, n)
    garden = rng.choice([0.0, 0.0, 20.0], n)
    roof = rng.choice([0.0, 0.0, 30.0], n)
    property_type = rng.choice(["Apartment", "Duplex", "Garden Apartment", "Penthouse"], n)
    market_segment = rng.choice(["Second Hand", "New Project"], n)
    balcony_direction = rng.choice(["North", "South", "East", "West"], n)
    is_top_floor = rng.choice([True, False], n)
    price = (
        30_000 * area + 60_000 * rooms + 20_000 * floor + 5_000 * balcony
        + 15_000 * parking + 3_000 * storage + 8_000 * garden + 6_000 * roof + 800_000
    )
    X = pd.DataFrame({
        "area_sqm": area, "rooms": rooms, "floor": floor, "balcony_area_sqm": balcony,
        "parking_count": parking, "storage_area_sqm": storage, "garden_area_sqm": garden,
        "roof_area_sqm": roof, "property_type": property_type, "market_segment": market_segment,
        "balcony_direction": balcony_direction, "is_top_floor": is_top_floor,
    })
    y = pd.Series(price)
    return train_market_model(X, y)


def test_predict_custom_apartment_uses_same_predict_functions():
    historical_fit = _synthetic_historical_fit()
    market_fit = _synthetic_market_fit()
    apartment = _custom_apartment()

    predictions = predict_custom_apartment(apartment, historical_fit["model"], market_fit["model"])

    assert predictions["historical_base_price"] > 0
    assert predictions["current_market_price"] > 0


def test_pricing_blend_uses_existing_configured_weights():
    breakdown = build_pricing_breakdown(
        historical_base_price=4_000_000.0,
        current_market_price=6_000_000.0,
        interior_area_sqm=75.0,
    )
    expected = 4_000_000.0 * HISTORICAL_MARKET_WEIGHT + 6_000_000.0 * CURRENT_MARKET_WEIGHT
    assert breakdown["recommended_marketing_price"] == pytest.approx(expected)
    assert breakdown["historical_weight"] == HISTORICAL_MARKET_WEIGHT
    assert breakdown["current_market_weight"] == CURRENT_MARKET_WEIGHT


def test_zero_scenario_leaves_market_recommendation_unchanged():
    base = build_pricing_breakdown(4_000_000.0, 6_000_000.0, 75.0)
    scenario = build_pricing_breakdown(
        4_000_000.0, 6_000_000.0, 75.0,
        company_positioning_pct=0.0, sales_phase_pct=0.0,
        inventory_strategy_pct=0.0, manual_adjustment_pct=0.0, manual_adjustment_amount=0.0,
    )
    assert base["recommended_marketing_price"] == pytest.approx(scenario["recommended_marketing_price"])
    assert scenario["final_strategy_price"] == pytest.approx(scenario["recommended_marketing_price"])


def test_positive_scenario_adjustment_changes_final_price_but_not_recommendation():
    base = build_pricing_breakdown(4_000_000.0, 6_000_000.0, 75.0)
    scenario = build_pricing_breakdown(
        4_000_000.0, 6_000_000.0, 75.0, company_positioning_pct=0.02,
    )

    # market recommendation is identical -- only the final price moved
    assert base["recommended_marketing_price"] == pytest.approx(scenario["recommended_marketing_price"])
    assert scenario["final_strategy_price"] == pytest.approx(
        scenario["recommended_marketing_price"] * 1.02
    )
    assert scenario["final_strategy_price"] > base["final_strategy_price"]


def test_strategy_scenario_reuses_existing_strategy_logic():
    # manual amount + pct combined, matching apply_strategy_adjustment's formula exactly
    breakdown = build_pricing_breakdown(
        4_000_000.0, 6_000_000.0, 75.0,
        company_positioning_pct=0.01, sales_phase_pct=0.0, inventory_strategy_pct=0.0,
        manual_adjustment_pct=0.02, manual_adjustment_amount=10_000.0,
    )
    recommended = breakdown["recommended_marketing_price"]
    expected_final = recommended * (1 + 0.01 + 0.02) + 10_000.0
    assert breakdown["final_strategy_price"] == pytest.approx(expected_final)


def test_breakdown_price_per_sqm_uses_provided_area():
    breakdown = build_pricing_breakdown(4_000_000.0, 6_000_000.0, interior_area_sqm=80.0)
    expected_per_sqm = breakdown["final_strategy_price"] / 80.0
    assert breakdown["final_strategy_price_per_sqm"] == pytest.approx(expected_per_sqm)
