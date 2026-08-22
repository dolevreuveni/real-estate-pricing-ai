"""Loading and validation for external market-data CSV files.

Each loader reads a CSV with pandas, validates the required schema, parses
date/numeric columns, and returns a DataFrame. Empty (header-only) files
are valid and represent a dataset with no records yet. Invalid business
values (wrong types, missing columns) are never silently repaired --
loading raises a clear error instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import EXTERNAL_DATA_DIR

TRANSACTIONS_COLUMNS = [
    "address",
    "transaction_date",
    "rooms",
    "area_sqm",
    "floor",
    "original_price",
    "original_price_per_sqm",
    "price_index_at_transaction",
    "current_price_index",
    "index_adjustment_factor",
    "adjusted_price",
    "adjusted_price_per_sqm",
    "distance_from_project_km",
    "source",
    "source_url",
    "data_retrieved_at",
]
TRANSACTIONS_DATE_COLUMNS = ["transaction_date", "data_retrieved_at"]
TRANSACTIONS_NUMERIC_COLUMNS = [
    "rooms",
    "area_sqm",
    "floor",
    "original_price",
    "original_price_per_sqm",
    "price_index_at_transaction",
    "current_price_index",
    "index_adjustment_factor",
    "adjusted_price",
    "adjusted_price_per_sqm",
    "distance_from_project_km",
]

MARKET_LISTINGS_COLUMNS = [
    "address",
    "listing_date",
    "rooms",
    "area_sqm",
    "floor",
    "asking_price",
    "price_per_sqm",
    "distance_from_project_km",
    "source",
    "source_url",
    "data_retrieved_at",
]
MARKET_LISTINGS_DATE_COLUMNS = ["listing_date", "data_retrieved_at"]
MARKET_LISTINGS_NUMERIC_COLUMNS = [
    "rooms",
    "area_sqm",
    "floor",
    "asking_price",
    "price_per_sqm",
    "distance_from_project_km",
]

COMPETITOR_PROJECTS_COLUMNS = [
    "project_name",
    "address",
    "developer",
    "rooms",
    "area_sqm",
    "asking_price",
    "price_per_sqm",
    "property_type",
    "distance_from_project_km",
    "source",
    "source_url",
    "data_retrieved_at",
]
COMPETITOR_PROJECTS_DATE_COLUMNS = ["data_retrieved_at"]
COMPETITOR_PROJECTS_NUMERIC_COLUMNS = [
    "rooms",
    "area_sqm",
    "asking_price",
    "price_per_sqm",
    "distance_from_project_km",
]

TRANSACTIONS_PATH = EXTERNAL_DATA_DIR / "transactions.csv"
MARKET_LISTINGS_PATH = EXTERNAL_DATA_DIR / "market_listings.csv"
COMPETITOR_PROJECTS_PATH = EXTERNAL_DATA_DIR / "competitor_projects.csv"


def _load_csv_dataset(
    path: str | Path,
    required_columns: list,
    date_columns: list,
    numeric_columns: list,
    label: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required column(s) {missing} in {path}. "
            f"Expected columns: {required_columns}."
        )

    df = df[required_columns].copy()

    for column in numeric_columns:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{label}: column '{column}' in {path} contains a non-numeric value."
            ) from exc

    for column in date_columns:
        try:
            df[column] = pd.to_datetime(df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{label}: column '{column}' in {path} contains a value that "
                f"cannot be parsed as a date."
            ) from exc

    return df


def load_transactions(path: str | Path = TRANSACTIONS_PATH) -> pd.DataFrame:
    return _load_csv_dataset(
        path,
        TRANSACTIONS_COLUMNS,
        TRANSACTIONS_DATE_COLUMNS,
        TRANSACTIONS_NUMERIC_COLUMNS,
        "transactions",
    )


def load_market_listings(path: str | Path = MARKET_LISTINGS_PATH) -> pd.DataFrame:
    return _load_csv_dataset(
        path,
        MARKET_LISTINGS_COLUMNS,
        MARKET_LISTINGS_DATE_COLUMNS,
        MARKET_LISTINGS_NUMERIC_COLUMNS,
        "market_listings",
    )


def load_competitor_projects(path: str | Path = COMPETITOR_PROJECTS_PATH) -> pd.DataFrame:
    return _load_csv_dataset(
        path,
        COMPETITOR_PROJECTS_COLUMNS,
        COMPETITOR_PROJECTS_DATE_COLUMNS,
        COMPETITOR_PROJECTS_NUMERIC_COLUMNS,
        "competitor_projects",
    )
