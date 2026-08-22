"""Tests for apartment_reader normalization logic."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.apartment_reader import load_normalized_apartments, normalize_apartments

COLUMNS = [
    "מס' קומה",
    "מספר דירה",
    "מס' חדרים",
    'שטח דירה (מ"ר)',
    "שטח מרפסת",
    "כיווני אוויר",
    "הערות",
]


def _raw(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def test_regular_single_level_apartment():
    raw = _raw([
        [1, 101, 3, 69.0, 12.0, "מזרח", None],
    ])
    result = normalize_apartments(raw)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["apartment_id"] == 101
    assert row["num_levels"] == 1
    assert row["floor_min"] == 1
    assert row["floor_max"] == 1
    assert row["interior_area_sqm"] == pytest.approx(69.0)
    assert row["balcony_area_sqm"] == pytest.approx(12.0)
    assert row["property_type"] == "regular"


def test_multi_level_duplex_apartment_is_aggregated():
    raw = _raw([
        [9, 38, 7, 80.7, 19.6, "מערב", "דופלקס"],
        [10, 38, 7, 89.4, 20.0, None, "דופלקס"],
    ])
    result = normalize_apartments(raw)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["apartment_id"] == 38
    assert row["num_levels"] == 2
    assert row["floor_min"] == 9
    assert row["floor_max"] == 10
    assert row["interior_area_sqm"] == pytest.approx(170.1)
    assert row["balcony_area_sqm"] == pytest.approx(39.6)
    assert row["property_type"] == "duplex"


def test_summary_rows_are_excluded_and_not_required():
    raw = _raw([
        [9, 36, 6, 73.4, 12.0, "מזרח", "טריפלקס"],
        [10, 36, 6, 78.5, 20.0, "מזרח", "טריפלקס"],
        [11, 36, 6, 103.3, 62.6, "מזרח", "טריפלקס"],
        ['סה"כ', 36, 6, 255.2, None, "מזרח", "טריפלקס"],
        # apartment 37 has no summary row at all, and must still work
        [9, 37, 6, 73.7, 12.0, "מערב", "טריפלקס"],
        [10, 37, 6, 77.5, 20.0, "מערב", "טריפלקס"],
        [11, 37, 6, 108.9, 65.4, "מערב", "טריפלקס"],
    ])
    result = normalize_apartments(raw)

    # the summary row is not a separate apartment
    assert sorted(result["apartment_id"]) == [36, 37]

    apt36 = result[result["apartment_id"] == 36].iloc[0]
    apt37 = result[result["apartment_id"] == 37].iloc[0]
    assert apt36["num_levels"] == 3
    assert apt36["interior_area_sqm"] == pytest.approx(255.2)
    assert apt37["num_levels"] == 3
    assert apt37["interior_area_sqm"] == pytest.approx(260.1)
    assert apt36["property_type"] == "triplex"
    assert apt37["property_type"] == "triplex"


def test_duplicate_level_rows_are_not_double_counted():
    raw = _raw([
        [1, 5, 3, 69.0, 12.0, "מזרח", None],
        [1, 5, 3, 69.0, 12.0, "מזרח", None],  # accidental duplicate entry
    ])
    result = normalize_apartments(raw)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["num_levels"] == 1
    assert row["interior_area_sqm"] == pytest.approx(69.0)


def test_real_source_file_normalizes_to_expected_shape():
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Apartment_example.xlsx"
    if not raw_path.exists():
        pytest.skip("raw source file not available")

    result = load_normalized_apartments(raw_path)

    assert len(result) == 39
    assert set(result["apartment_id"]) == set(range(1, 40))

    garden = result[result["apartment_id"].isin([1, 2, 3])]
    assert (garden["property_type"] == "garden").all()

    apt36 = result[result["apartment_id"] == 36].iloc[0]
    assert apt36["num_levels"] == 3
    assert apt36["property_type"] == "triplex"
    assert apt36["interior_area_sqm"] == pytest.approx(255.2)

    # apartment 39 is a duplex with no "total" row in the source file
    apt39 = result[result["apartment_id"] == 39].iloc[0]
    assert apt39["num_levels"] == 2
    assert apt39["property_type"] == "duplex"
    assert apt39["interior_area_sqm"] == pytest.approx(158.6)
