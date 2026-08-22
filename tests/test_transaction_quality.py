"""Tests for transaction_quality."""
import pandas as pd
import pytest

from src.data.transaction_quality import (
    REASON_MISSING_AREA,
    REASON_MISSING_PRICE,
    REASON_MISSING_ROOMS,
    REASON_NON_RESIDENTIAL,
    REASON_NON_RESIDENTIAL_DEAL_NATURE,
    evaluate_transaction,
    evaluate_transactions,
)


def _normal_transaction(**overrides):
    row = {
        "deal_id": 1,
        "address": "הלסינקי 13, תל אביב-יפו",
        "transaction_date": pd.Timestamp("2026-06-25"),
        "rooms": 3,
        "area_sqm": 85,
        "floor": 2,
        "original_price": 5254000,
        "original_price_per_sqm": 61811.76,
        "property_type": "דירה",
        "deal_nature": "דירה בבית קומות",
        "neighborhood": "הצפון החדש סביבת כיכר המדינה",
        "gush": 6108,
        "parcel": 230,
        "sub_parcel": 19,
        "source": "GovMap",
        "source_url": "https://www.govmap.gov.il/api/real-estate/street-deals/6108-86",
        "data_retrieved_at": pd.Timestamp("2026-08-22", tz="UTC"),
    }
    row.update(overrides)
    return row


def test_normal_residential_transaction_is_eligible():
    is_eligible, reasons = evaluate_transaction(_normal_transaction())
    assert is_eligible is True
    assert reasons == []


def test_missing_rooms_is_excluded():
    is_eligible, reasons = evaluate_transaction(_normal_transaction(rooms=None))
    assert is_eligible is False
    assert REASON_MISSING_ROOMS in reasons


def test_invalid_area_is_excluded():
    is_eligible, reasons = evaluate_transaction(_normal_transaction(area_sqm=0))
    assert is_eligible is False
    assert REASON_MISSING_AREA in reasons

    is_eligible, reasons = evaluate_transaction(_normal_transaction(area_sqm=None))
    assert is_eligible is False
    assert REASON_MISSING_AREA in reasons


def test_invalid_price_is_excluded():
    is_eligible, reasons = evaluate_transaction(_normal_transaction(original_price=0))
    assert is_eligible is False
    assert REASON_MISSING_PRICE in reasons

    is_eligible, reasons = evaluate_transaction(_normal_transaction(original_price=None))
    assert is_eligible is False
    assert REASON_MISSING_PRICE in reasons


def test_parking_storage_transaction_is_excluded():
    is_eligible, reasons = evaluate_transaction(_normal_transaction(property_type="חניה"))
    assert is_eligible is False
    assert REASON_NON_RESIDENTIAL in reasons

    is_eligible, reasons = evaluate_transaction(_normal_transaction(property_type="מחסן"))
    assert is_eligible is False
    assert REASON_NON_RESIDENTIAL in reasons


def test_office_deal_nature_is_excluded_even_with_generic_property_type():
    # real case: property_type="בנין" (building) doesn't itself look
    # non-residential, but deal_nature="משרד" (office) makes it clear
    # this isn't an apartment sale.
    is_eligible, reasons = evaluate_transaction(
        _normal_transaction(property_type="בנין", deal_nature="משרד")
    )
    assert is_eligible is False
    assert REASON_NON_RESIDENTIAL_DEAL_NATURE in reasons


def test_multiple_reasons_are_all_preserved():
    is_eligible, reasons = evaluate_transaction(
        _normal_transaction(rooms=None, original_price=None)
    )
    assert is_eligible is False
    assert REASON_MISSING_ROOMS in reasons
    assert REASON_MISSING_PRICE in reasons


def test_evaluate_transactions_never_drops_rows():
    df = pd.DataFrame(
        [_normal_transaction(deal_id=1), _normal_transaction(deal_id=2, rooms=None)]
    )
    result = evaluate_transactions(df)

    assert len(result) == 2
    assert list(result["is_eligible_comparable"]) == [True, False]
    assert result.iloc[1]["exclusion_reason"] == REASON_MISSING_ROOMS
    assert pd.isna(result.iloc[0]["exclusion_reason"])
