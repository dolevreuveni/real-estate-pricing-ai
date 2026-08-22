"""Tests for current_market_features."""
import pandas as pd
import pytest

from src.config.settings import TARGET_MARKET_SEGMENT
from src.pricing.current_market_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    apartments_to_market_feature_frame,
    listings_to_training_frame,
    select_training_rows,
)


def _listing_row(**overrides):
    row = {
        "listing_id": "SYN-0001",
        "market_segment": "Second Hand",
        "property_type": "Apartment",
        "rooms": 3,
        "area_sqm": 80.0,
        "floor": 2,
        "balcony_area_sqm": 10.0,
        "parking_count": 1,
        "storage_area_sqm": 4.0,
        "garden_area_sqm": 0.0,
        "roof_area_sqm": 0.0,
        "balcony_direction": "South",
        "is_top_floor": False,
        "asking_price": 4_500_000.0,
    }
    row.update(overrides)
    return row


def test_feature_whitelist_excludes_leakage_and_unmappable_fields():
    forbidden = {
        "listing_id",
        "listing_date",
        "source",
        "source_url",
        "address",
        "asking_price",
        "price_per_sqm",
        "project_name",
        "developer",
        "distance_from_project_km",
        "building_age",
        "condition",
        "elevator",
        "directions",
        "apartment_id",
    }
    assert forbidden.isdisjoint(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_feature_columns_are_exactly_numeric_plus_categorical():
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(NUMERIC_FEATURES) == {
        "area_sqm",
        "rooms",
        "floor",
        "balcony_area_sqm",
        "parking_count",
        "storage_area_sqm",
        "garden_area_sqm",
        "roof_area_sqm",
    }
    assert set(CATEGORICAL_FEATURES) == {
        "property_type",
        "market_segment",
        "balcony_direction",
        "is_top_floor",
    }


def test_select_training_rows_drops_rows_missing_required_fields():
    df = pd.DataFrame([_listing_row(listing_id="a"), _listing_row(listing_id="b", rooms=None)])
    trainable = select_training_rows(df)
    assert list(trainable["listing_id"]) == ["a"]


def test_listings_to_training_frame_uses_only_canonical_columns():
    df = pd.DataFrame([_listing_row()])
    trainable = select_training_rows(df)
    X, y = listings_to_training_frame(trainable)

    assert list(X.columns) == FEATURE_COLUMNS
    assert y.iloc[0] == pytest.approx(4_500_000.0)


def _apartment_row(**overrides):
    row = {
        "apartment_id": 1,
        "interior_area_sqm": 70.0,
        "rooms": 3,
        "floor_min": 1,
        "floor_max": 1,
        "balcony_area_sqm": 12.0,
        "parking_count": 1,
        "storage_area_sqm": 4.0,
        "garden_area_sqm": 0.0,
        "roof_area_sqm": 0.0,
        "balcony_direction": "East",
        "is_top_floor": False,
        "property_type": "regular",
    }
    row.update(overrides)
    return row


def test_apartment_area_floor_balcony_and_new_fields_are_mapped_correctly():
    apartments = pd.DataFrame(
        [
            _apartment_row(
                apartment_id=36,
                interior_area_sqm=255.2,
                rooms=6,
                floor_min=9,
                floor_max=11,
                balcony_area_sqm=94.6,
                parking_count=2,
                storage_area_sqm=8.0,
                roof_area_sqm=76.6,
                balcony_direction="South-West",
                is_top_floor=True,
                property_type="triplex",
            )
        ]
    )
    features = apartments_to_market_feature_frame(apartments)

    assert features.loc[0, "area_sqm"] == 255.2
    assert features.loc[0, "floor"] == 9
    assert features.loc[0, "balcony_area_sqm"] == 94.6
    assert features.loc[0, "parking_count"] == 2
    assert features.loc[0, "storage_area_sqm"] == 8.0
    assert features.loc[0, "roof_area_sqm"] == 76.6
    assert features.loc[0, "balcony_direction"] == "South-West"
    assert features.loc[0, "is_top_floor"] == True  # noqa: E712


def test_apartment_property_type_is_mapped_where_a_match_exists():
    apartments = pd.DataFrame([_apartment_row(property_type="garden")])
    features = apartments_to_market_feature_frame(apartments)
    assert features.loc[0, "property_type"] == "Garden Apartment"


def test_unmappable_triplex_property_type_passes_through_and_stays_priceable():
    apartments = pd.DataFrame([_apartment_row(property_type="triplex")])
    features = apartments_to_market_feature_frame(apartments)

    # no fabricated match -- the raw ("unmapped") value is used, and the
    # apartment is still considered priceable (the model handles an
    # unseen category via OneHotEncoder(handle_unknown="ignore"))
    assert features.loc[0, "property_type"] == "triplex"
    assert features.loc[0, "is_priceable"] == True  # noqa: E712


def test_market_segment_comes_from_config_not_apartment_data():
    apartments = pd.DataFrame([_apartment_row()])
    features = apartments_to_market_feature_frame(apartments)
    assert features.loc[0, "market_segment"] == TARGET_MARKET_SEGMENT


def test_apartment_missing_required_field_is_flagged_not_priceable():
    apartments = pd.DataFrame([_apartment_row(interior_area_sqm=None)])
    features = apartments_to_market_feature_frame(apartments)
    assert features.loc[0, "is_priceable"] == False  # noqa: E712


def test_apartment_missing_balcony_direction_is_flagged_not_priceable():
    apartments = pd.DataFrame([_apartment_row(balcony_direction=None)])
    features = apartments_to_market_feature_frame(apartments)
    assert features.loc[0, "is_priceable"] == False  # noqa: E712
