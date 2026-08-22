"""Tests for transaction_normalizer."""
import pandas as pd
import pytest

from src.data.transaction_normalizer import (
    build_address,
    normalize_floor,
    normalize_transaction,
    normalize_transactions,
)


def _raw_record(**overrides):
    record = {
        "objectid": 1627260,
        "settlementId": 5000,
        "settlementNameHeb": "תל אביב-יפו",
        "streetCode": 50000932,
        "streetNameHeb": "הלסינקי",
        "houseNum": 13,
        "floorNo": "שניה",
        "assetArea": 85,
        "dealAmount": 5254000,
        "dealId": 6308613950,
        "propertyTypeDescription": "דירה",
        "dealNatureDescription": "דירה בבית קומות",
        "assetRoomNum": 3,
        "neighborhood": "הצפון החדש סביבת כיכר המדינה",
        "dealDate": "2026-06-25T00:00:00.000Z",
        "gushNum": 6108,
        "parcelNum": 230,
        "subParcelNum": 19,
        "polygonId": "52315574",
        "shape": "MULTIPOLYGON(((0 0)))",
        "sourceorder": 2,
        "source": "GovMap",
        "source_url": "https://www.govmap.gov.il/api/real-estate/street-deals/6108-86",
        "data_retrieved_at": "2026-08-22T12:00:00+00:00",
    }
    record.update(overrides)
    return record


def test_address_construction():
    address = build_address("הלסינקי", 24, "תל אביב-יפו")
    assert address == "הלסינקי 24, תל אביב-יפו"


def test_address_construction_with_missing_parts():
    assert build_address(None, 24, "תל אביב-יפו") == "24, תל אביב-יפו"
    assert build_address("הלסינקי", None, "תל אביב-יפו") == "הלסינקי, תל אביב-יפו"
    assert build_address(None, None, None) is None


def test_rooms_and_area_and_price_are_carried_through():
    normalized = normalize_transaction(_raw_record())
    assert normalized["rooms"] == 3
    assert normalized["area_sqm"] == 85
    assert normalized["original_price"] == 5254000


def test_price_per_sqm_is_computed():
    normalized = normalize_transaction(_raw_record(dealAmount=5254000, assetArea=85))
    assert normalized["original_price_per_sqm"] == pytest.approx(5254000 / 85)


def test_price_per_sqm_is_none_when_area_is_zero_or_missing():
    normalized = normalize_transaction(_raw_record(assetArea=0))
    assert normalized["original_price_per_sqm"] is None

    normalized = normalize_transaction(_raw_record(assetArea=None))
    assert normalized["original_price_per_sqm"] is None


@pytest.mark.parametrize(
    "floor_raw,expected",
    [
        ("קרקע", 0),
        ("ראשונה", 1),
        ("שניה", 2),
        ("שנייה", 2),
        ("שלישית", 3),
        ("רביעית", 4),
        ("חמישית", 5),
        ("3", 3),
        (5, 5),
    ],
)
def test_hebrew_floor_conversion(floor_raw, expected):
    assert normalize_floor(floor_raw) == expected


def test_missing_or_unrecognized_floor_is_null():
    assert normalize_floor(None) is None
    assert normalize_floor("") is None
    assert normalize_floor(float("nan")) is None
    assert normalize_floor("לא ידוע") is None


def test_transaction_date_is_timezone_naive_for_cbs_lookup_compatibility():
    # GovMap's dealDate carries a "Z" (UTC) suffix; the CBS cache's monthly
    # `period` column (src/data/cbs_client.py) is timezone-naive, so the
    # normalized transaction_date must be too or CBS index lookups silently
    # fail to match.
    normalized = normalize_transaction(_raw_record(dealDate="2026-02-15T00:00:00.000Z"))
    assert normalized["transaction_date"].tzinfo is None
    assert normalized["transaction_date"] == pd.Timestamp(2026, 2, 15)


def test_normalize_transactions_returns_dataframe_with_expected_columns():
    df = normalize_transactions([_raw_record(), _raw_record(dealId=999)])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "deal_id" in df.columns
    assert set(df["deal_id"]) == {6308613950, 999}
