"""Tests for market_data_loader."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.market_data_loader import (
    COMPETITOR_PROJECTS_COLUMNS,
    MARKET_LISTINGS_COLUMNS,
    TRANSACTIONS_COLUMNS,
    load_competitor_projects,
    load_market_listings,
    load_transactions,
)


def test_valid_empty_transactions_csv(tmp_path: Path):
    # A header-only file is a valid, empty transactions dataset. This is
    # tested in isolation against a temporary file rather than the real
    # data/external/transactions.csv, which Feature #5 intentionally
    # populates with real transactions and is not expected to be empty.
    csv_path = tmp_path / "transactions_empty.csv"
    csv_path.write_text(",".join(TRANSACTIONS_COLUMNS) + "\n", encoding="utf-8")

    df = load_transactions(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == TRANSACTIONS_COLUMNS


def test_valid_empty_market_listings_csv():
    df = load_market_listings()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == MARKET_LISTINGS_COLUMNS


def test_valid_empty_competitor_projects_csv():
    df = load_competitor_projects()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == COMPETITOR_PROJECTS_COLUMNS


def test_missing_required_column_raises(tmp_path: Path):
    csv_path = tmp_path / "transactions_missing_column.csv"
    csv_path.write_text(
        "address,transaction_date,rooms,area_sqm,original_price\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required column"):
        load_transactions(csv_path)


def test_invalid_numeric_field_raises(tmp_path: Path):
    header = ",".join(TRANSACTIONS_COLUMNS)
    row = (
        "Helsinki 24,2024-01-01,3,not_a_number,2,4000000,,500,600,,,,0.2,"
        "tax_authority,,2024-06-01"
    )
    csv_path = tmp_path / "transactions_bad_numeric.csv"
    csv_path.write_text(header + "\n" + row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-numeric"):
        load_transactions(csv_path)
