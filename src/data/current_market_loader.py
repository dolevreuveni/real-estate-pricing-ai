"""Loading and validation for the manually supplied Current Market Excel
dataset.

IMPORTANT: data/external/current_market_500_updated.xlsx is SYNTHETIC POC
data representing the shape of data we expect to eventually receive from
a source such as Yad2, Madlan, or a developer/project feed. It must never
be labeled or treated as real scraped market data -- see
CURRENT_MARKET_DATA_TYPE in src/config/settings.py, which is stamped onto
every output this dataset produces downstream.

Feature #7.5 adds four columns on top of the original schema: directions,
garden_area_sqm, roof_area_sqm, is_top_floor. The input path is read from
CURRENT_MARKET_INPUT_PATH (src/config/settings.py) -- never hardcoded here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import CURRENT_MARKET_INPUT_PATH, CURRENT_MARKET_SHEET_NAME

REQUIRED_COLUMNS = [
    "listing_id",
    "listing_date",
    "source",
    "source_url",
    "market_segment",
    "project_name",
    "developer",
    "address",
    "city",
    "neighborhood",
    "distance_from_project_km",
    "property_type",
    "rooms",
    "area_sqm",
    "floor",
    "total_floors",
    "asking_price",
    "price_per_sqm",
    "building_year",
    "building_age",
    "condition",
    "balcony_area_sqm",
    "balcony_direction",
    "parking_count",
    "storage_area_sqm",
    "elevator",
    "directions",
    "garden_area_sqm",
    "roof_area_sqm",
    "is_top_floor",
]

NUMERIC_COLUMNS = [
    "distance_from_project_km",
    "rooms",
    "area_sqm",
    "floor",
    "total_floors",
    "asking_price",
    "price_per_sqm",
    "building_year",
    "building_age",
    "balcony_area_sqm",
    "parking_count",
    "storage_area_sqm",
    "garden_area_sqm",
    "roof_area_sqm",
]

DATE_COLUMNS = ["listing_date"]


def load_current_market_listings(
    path: str | Path = CURRENT_MARKET_INPUT_PATH,
    sheet_name: str = CURRENT_MARKET_SHEET_NAME,
) -> pd.DataFrame:
    """Load and validate the Current Market Excel dataset. Never invents
    missing values -- raises a clear error for a missing required column
    or a non-numeric/non-date value instead."""
    df = pd.read_excel(path, sheet_name=sheet_name)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"current_market_listings: missing required column(s) {missing} in {path} "
            f"(sheet '{sheet_name}'). Expected columns: {REQUIRED_COLUMNS}."
        )

    df = df[REQUIRED_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"current_market_listings: column '{column}' in {path} contains a "
                f"non-numeric value."
            ) from exc

    for column in DATE_COLUMNS:
        try:
            df[column] = pd.to_datetime(df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"current_market_listings: column '{column}' in {path} contains a value "
                f"that cannot be parsed as a date."
            ) from exc

    return df
