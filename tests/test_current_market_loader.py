"""Tests for current_market_loader."""
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import CURRENT_MARKET_INPUT_PATH
from src.data.current_market_loader import REQUIRED_COLUMNS, load_current_market_listings

SHEET_NAME = "Current_Market_Listings"


def test_default_input_path_is_the_updated_file():
    assert CURRENT_MARKET_INPUT_PATH.name == "current_market_500_updated.xlsx"


def _valid_row(**overrides):
    row = {
        "listing_id": "SYN-0001",
        "listing_date": "2026-08-01",
        "source": "Synthetic Yad2-like Listing",
        "source_url": "https://example.com/listings/SYN-0001",
        "market_segment": "Second Hand",
        "project_name": None,
        "developer": None,
        "address": "Jabotinsky 108, Tel Aviv",
        "city": "Tel Aviv",
        "neighborhood": "Kikar HaMedina / New North",
        "distance_from_project_km": 0.52,
        "property_type": "Apartment",
        "rooms": 3,
        "area_sqm": 80.0,
        "floor": 3,
        "total_floors": 6,
        "asking_price": 4_000_000,
        "price_per_sqm": 50_000.0,
        "building_year": 1998,
        "building_age": 28,
        "condition": "Renovated",
        "balcony_area_sqm": 10.0,
        "balcony_direction": "South",
        "parking_count": 1,
        "storage_area_sqm": 0.0,
        "elevator": True,
        "directions": "South, West",
        "garden_area_sqm": 0.0,
        "roof_area_sqm": 0.0,
        "is_top_floor": False,
    }
    row.update(overrides)
    return row


def _write_excel(path: Path, rows: list):
    df = pd.DataFrame(rows)
    df.to_excel(path, sheet_name=SHEET_NAME, index=False)


def test_valid_schema_loads_successfully(tmp_path: Path):
    path = tmp_path / "current_market.xlsx"
    _write_excel(path, [_valid_row()])

    df = load_current_market_listings(path, sheet_name=SHEET_NAME)

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["asking_price"] == 4_000_000


def test_missing_required_column_raises(tmp_path: Path):
    path = tmp_path / "current_market_missing_col.xlsx"
    row = _valid_row()
    del row["balcony_direction"]
    _write_excel(path, [row])

    with pytest.raises(ValueError, match="missing required column"):
        load_current_market_listings(path, sheet_name=SHEET_NAME)


def test_invalid_numeric_field_raises(tmp_path: Path):
    path = tmp_path / "current_market_bad_numeric.xlsx"
    _write_excel(path, [_valid_row(area_sqm="not_a_number")])

    with pytest.raises(ValueError, match="non-numeric"):
        load_current_market_listings(path, sheet_name=SHEET_NAME)
