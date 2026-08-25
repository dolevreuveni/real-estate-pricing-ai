"""Tests for regression_features."""
import pandas as pd
import pytest

from src.pricing.regression_features import (
    FEATURE_COLUMNS,
    apartments_to_feature_frame,
    select_training_transactions,
    transactions_to_training_frame,
)


def _transaction_row(**overrides):
    row = {
        "deal_id": 1,
        "address": "הלסינקי 13, תל אביב-יפו",
        "transaction_date": pd.Timestamp("2024-01-01"),
        "rooms": 3.0,
        "area_sqm": 80.0,
        "floor": 2.0,
        "original_price": 4_000_000.0,
        "original_price_per_sqm": 50000.0,
        "adjusted_price": 4_500_000.0,
        "adjusted_price_per_sqm": 56250.0,
        "is_eligible_comparable": True,
        "exclusion_reason": None,
        # used_for_historical_model is now the single source of truth for
        # trainability (src/data/historical_transaction_enrichment.py) --
        # default these fixtures to "would pass" so existing test intent
        # (missing adjusted_price / missing predictor -> not trainable)
        # keeps working without re-deriving the full whitelist logic here.
        "used_for_historical_model": True,
        "historical_model_exclusion_reason": None,
    }
    row.update(overrides)
    return row


def test_only_eligible_transactions_are_used_for_training():
    df = pd.DataFrame(
        [
            _transaction_row(deal_id=1, is_eligible_comparable=True),
            _transaction_row(
                deal_id=2,
                is_eligible_comparable=False,
                used_for_historical_model=False,
                historical_model_exclusion_reason="not_eligible_comparable",
            ),
        ]
    )
    eligible, trainable = select_training_transactions(df)

    assert list(eligible["deal_id"]) == [1]
    assert list(trainable["deal_id"]) == [1]


def test_transactions_missing_adjusted_price_are_excluded_from_training():
    df = pd.DataFrame(
        [
            _transaction_row(deal_id=1, adjusted_price=4_500_000.0),
            _transaction_row(
                deal_id=2,
                adjusted_price=None,
                used_for_historical_model=False,
                historical_model_exclusion_reason="missing_required_model_fields",
            ),
        ]
    )
    eligible, trainable = select_training_transactions(df)

    # both remain "eligible" -- nothing is dropped from the source view
    assert set(eligible["deal_id"]) == {1, 2}
    # only the one with a usable adjusted_price is trainable
    assert list(trainable["deal_id"]) == [1]


def test_transactions_missing_a_predictor_are_excluded_from_training():
    df = pd.DataFrame(
        [
            _transaction_row(deal_id=1, floor=2.0),
            _transaction_row(
                deal_id=2,
                floor=None,
                used_for_historical_model=False,
                historical_model_exclusion_reason="missing_required_model_fields",
            ),
        ]
    )
    eligible, trainable = select_training_transactions(df)

    assert set(eligible["deal_id"]) == {1, 2}
    assert list(trainable["deal_id"]) == [1]


def test_select_training_transactions_requires_used_for_historical_model_column():
    df = pd.DataFrame([{"deal_id": 1, "is_eligible_comparable": True}])
    with pytest.raises(ValueError, match="used_for_historical_model"):
        select_training_transactions(df)


def test_transactions_to_training_frame_has_only_the_canonical_columns():
    df = pd.DataFrame([_transaction_row(deal_id=1)])
    _, trainable = select_training_transactions(df)
    X, y = transactions_to_training_frame(trainable)

    assert list(X.columns) == FEATURE_COLUMNS
    assert y.iloc[0] == pytest.approx(4_500_000.0)


def test_area_sqm_maps_to_interior_area_sqm():
    apartments = pd.DataFrame([{"interior_area_sqm": 85.0, "rooms": 3, "floor_min": 1}])
    features = apartments_to_feature_frame(apartments)
    assert features.loc[0, "area_sqm"] == 85.0


def test_floor_maps_to_floor_min_for_multilevel_apartments():
    # triplex spanning floors 9-11: floor_min is the baseline representation
    apartments = pd.DataFrame(
        [{"interior_area_sqm": 255.2, "rooms": 6, "floor_min": 9, "floor_max": 11}]
    )
    features = apartments_to_feature_frame(apartments)
    assert features.loc[0, "floor"] == 9


def test_apartment_missing_a_required_field_is_flagged_not_priceable():
    apartments = pd.DataFrame(
        [
            {"interior_area_sqm": 70.0, "rooms": 3, "floor_min": 1},
            {"interior_area_sqm": None, "rooms": 3, "floor_min": 1},
        ]
    )
    features = apartments_to_feature_frame(apartments)
    assert list(features["is_priceable"]) == [True, False]
