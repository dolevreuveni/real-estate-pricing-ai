"""Tests for apartment_reader normalization logic."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.apartment_reader import (
    NORMALIZED_APARTMENT_COLUMNS,
    load_normalized_apartments,
    normalize_apartments,
)

COLUMNS = [
    "מס' קומה",
    "מספר דירה",
    "מס' חדרים",
    'שטח דירה (מ"ר)',
    "שטח מרפסת",
    "כיווני אוויר",
    "הערות",
]

# Feature #7.5: the same 7 required columns, plus the 6 optional
# enrichment columns.
ENRICHED_COLUMNS = COLUMNS + [
    "מספר חניות",
    "שטח מחסן",
    "כיוון מרפסת",
    "שטח גינה",
    "שטח גג מוצמד",
    "קומה אחרונה",
]


def _raw(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def _raw_enriched(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ENRICHED_COLUMNS)


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


# --- Feature #7.5: enrichment columns ---------------------------------


def test_new_enrichment_columns_are_read_correctly():
    raw = _raw_enriched(
        [
            [1, 101, 3, 69.0, 12.0, "מזרח", None, 1, 4.0, "מזרח", 0.0, 0.0, False],
        ]
    )
    result = normalize_apartments(raw)

    row = result.iloc[0]
    assert row["parking_count"] == 1
    assert row["storage_area_sqm"] == pytest.approx(4.0)
    assert row["balcony_direction"] == "East"  # translated from מזרח
    assert row["garden_area_sqm"] == pytest.approx(0.0)
    assert row["roof_area_sqm"] == pytest.approx(0.0)
    assert bool(row["is_top_floor"]) is False


def test_optional_columns_missing_does_not_crash_and_stays_null():
    # the original 7-column source (no enrichment columns at all) --
    # backward compatibility.
    raw = _raw([[1, 101, 3, 69.0, 12.0, "מזרח", None]])
    result = normalize_apartments(raw)

    row = result.iloc[0]
    assert len(result) == 1
    for field in (
        "parking_count",
        "storage_area_sqm",
        "balcony_direction",
        "garden_area_sqm",
        "roof_area_sqm",
        "is_top_floor",
    ):
        assert pd.isna(row[field])


def test_existing_required_schema_still_works_without_enrichment():
    # existing (pre-Feature #7.5) behavior must be unchanged for a plain
    # source file.
    raw = _raw([[1, 101, 3, 69.0, 12.0, "מזרח", None]])
    result = normalize_apartments(raw)
    assert list(result.columns) == NORMALIZED_APARTMENT_COLUMNS
    assert result.iloc[0]["interior_area_sqm"] == pytest.approx(69.0)


def test_parking_count_is_not_double_counted_across_duplex_rows():
    raw = _raw_enriched(
        [
            [9, 38, 7, 80.7, 19.6, "מערב", "דופלקס", 1, 8.0, "מזרח", 0.0, 51.0, False],
            [10, 38, 7, 89.4, 20.0, None, "דופלקס", 1, 8.0, "מזרח", 0.0, 51.0, False],
        ]
    )
    result = normalize_apartments(raw)

    row = result.iloc[0]
    assert row["num_levels"] == 2
    assert row["parking_count"] == 1  # NOT 2 -- repeated apartment-level value


def test_storage_area_is_not_double_counted_across_triplex_rows():
    raw = _raw_enriched(
        [
            [9, 36, 6, 73.4, 12.0, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, False],
            [10, 36, 6, 78.5, 20.0, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, False],
            [11, 36, 6, 103.3, 62.6, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, True],
        ]
    )
    result = normalize_apartments(raw)

    row = result.iloc[0]
    assert row["num_levels"] == 3
    assert row["storage_area_sqm"] == pytest.approx(8.0)  # NOT 24.0


def test_balcony_direction_is_preserved_and_translated():
    raw = _raw_enriched(
        [
            [1, 101, 3, 69.0, 12.0, "מזרח", None, 1, 4.0, "צפון-מערב", 0.0, 0.0, False],
        ]
    )
    result = normalize_apartments(raw)
    assert result.iloc[0]["balcony_direction"] == "North-West"


def test_garden_area_sqm_only_populated_for_garden_apartments():
    raw = _raw_enriched(
        [
            [
                "קרקע", 1, 3, 70.8, 100.0, "מזרח", "דירת גן",
                1, 4.0, "מזרח", 35.4, 0.0, False,
            ],
            [1, 4, 3, 69.0, 12.0, "מזרח", None, 1, 4.0, "מזרח", 0.0, 0.0, False],
        ]
    )
    result = normalize_apartments(raw)

    garden_apt = result[result["apartment_id"] == 1].iloc[0]
    regular_apt = result[result["apartment_id"] == 4].iloc[0]
    assert garden_apt["garden_area_sqm"] == pytest.approx(35.4)
    assert regular_apt["garden_area_sqm"] == pytest.approx(0.0)


def test_roof_area_sqm_uses_apartment_total_not_summed_across_rows():
    # roof_area_sqm is a single shared amenity for the whole multi-level
    # apartment -- the source repeats the SAME total on every row, and
    # normalization must take that value once (max), never sum it.
    raw = _raw_enriched(
        [
            [9, 36, 6, 73.4, 12.0, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, False],
            [10, 36, 6, 78.5, 20.0, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, False],
            [11, 36, 6, 103.3, 62.6, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, True],
        ]
    )
    result = normalize_apartments(raw)
    assert result.iloc[0]["roof_area_sqm"] == pytest.approx(76.6)  # NOT 229.8


def test_is_top_floor_is_true_if_any_level_reaches_the_top():
    raw = _raw_enriched(
        [
            [9, 36, 6, 73.4, 12.0, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, False],
            [10, 36, 6, 78.5, 20.0, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, False],
            [11, 36, 6, 103.3, 62.6, "מזרח", "טריפלקס", 2, 8.0, "מזרח", 0.0, 76.6, True],
        ]
    )
    result = normalize_apartments(raw)
    assert bool(result.iloc[0]["is_top_floor"]) is True


def test_is_top_floor_false_when_no_level_is_flagged():
    raw = _raw_enriched(
        [
            [9, 38, 7, 80.7, 19.6, "מערב", "דופלקס", 2, 8.0, "מזרח", 0.0, 51.0, False],
            [10, 38, 7, 89.4, 20.0, None, "דופלקס", 2, 8.0, "מזרח", 0.0, 51.0, False],
        ]
    )
    result = normalize_apartments(raw)
    assert bool(result.iloc[0]["is_top_floor"]) is False


def test_real_enriched_source_file_still_produces_39_apartments_with_correct_values():
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Apartment_example.xlsx"
    if not raw_path.exists():
        pytest.skip("raw source file not available")

    result = load_normalized_apartments(raw_path)

    assert len(result) == 39
    assert list(result.columns) == NORMALIZED_APARTMENT_COLUMNS

    # triplex apartments (36, 37) reach the building's top floor (11);
    # duplex apartments (38, 39) top out at floor 10 and do not.
    apt36 = result[result["apartment_id"] == 36].iloc[0]
    apt38 = result[result["apartment_id"] == 38].iloc[0]
    assert bool(apt36["is_top_floor"]) is True
    assert bool(apt38["is_top_floor"]) is False

    # parking/storage/roof are apartment-level, not summed across the
    # triplex's 3 source rows.
    assert apt36["parking_count"] == 2
    assert apt36["storage_area_sqm"] == pytest.approx(8.0)
    assert apt36["roof_area_sqm"] == pytest.approx(76.6)

    # garden apartments have a garden; regular apartments don't.
    garden_apts = result[result["apartment_id"].isin([1, 2, 3])]
    assert (garden_apts["garden_area_sqm"] > 0).all()
    regular_apt = result[result["apartment_id"] == 4].iloc[0]
    assert regular_apt["garden_area_sqm"] == pytest.approx(0.0)

    # directions (general air-direction) is unchanged, still raw Hebrew.
    assert apt36["directions"] == "מזרח"
