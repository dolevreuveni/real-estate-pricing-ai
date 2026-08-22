"""Tests for apartment_strategy_loader."""
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import APARTMENT_STRATEGY_ADJUSTMENTS_PATH
from src.data.apartment_strategy_loader import (
    REQUIRED_COLUMNS,
    load_apartment_strategy_adjustments,
)


def _write_csv(path: Path, rows: list):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_valid_schema_loads_successfully(tmp_path: Path):
    path = tmp_path / "strategy.csv"
    _write_csv(
        path,
        [
            {"apartment_id": 1, "manual_adjustment_pct": 0, "manual_adjustment_amount": 0, "strategy_note": ""},
            {"apartment_id": 2, "manual_adjustment_pct": 0.02, "manual_adjustment_amount": 0, "strategy_note": "premium"},
        ],
    )
    df = load_apartment_strategy_adjustments(path)

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 2


def test_missing_required_column_raises(tmp_path: Path):
    path = tmp_path / "strategy_missing_col.csv"
    _write_csv(
        path,
        [{"apartment_id": 1, "manual_adjustment_pct": 0, "strategy_note": ""}],
    )
    with pytest.raises(ValueError, match="missing required column"):
        load_apartment_strategy_adjustments(path)


def test_invalid_numeric_field_raises(tmp_path: Path):
    path = tmp_path / "strategy_bad_numeric.csv"
    _write_csv(
        path,
        [
            {
                "apartment_id": 1,
                "manual_adjustment_pct": "not_a_number",
                "manual_adjustment_amount": 0,
                "strategy_note": "",
            }
        ],
    )
    with pytest.raises(ValueError, match="non-numeric"):
        load_apartment_strategy_adjustments(path)


def test_duplicate_apartment_id_raises_at_load_time(tmp_path: Path):
    path = tmp_path / "strategy_dupes.csv"
    _write_csv(
        path,
        [
            {"apartment_id": 1, "manual_adjustment_pct": 0, "manual_adjustment_amount": 0, "strategy_note": ""},
            {"apartment_id": 1, "manual_adjustment_pct": 0.01, "manual_adjustment_amount": 0, "strategy_note": ""},
        ],
    )
    with pytest.raises(ValueError, match="duplicate apartment_id"):
        load_apartment_strategy_adjustments(path)


def test_real_apartment_strategy_file_has_all_39_neutral_rows():
    if not APARTMENT_STRATEGY_ADJUSTMENTS_PATH.exists():
        pytest.skip("apartment_strategy_adjustments.csv not available")

    df = load_apartment_strategy_adjustments()

    assert len(df) == 39
    assert set(df["apartment_id"]) == set(range(1, 40))
    assert (df["manual_adjustment_pct"] == 0).all()
    assert (df["manual_adjustment_amount"] == 0).all()
