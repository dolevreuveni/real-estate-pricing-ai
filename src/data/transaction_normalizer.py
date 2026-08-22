"""Normalize raw GovMap transaction records into the project's internal
transaction schema.

This module never invents missing values: a floor that can't be
confidently parsed is left null, and so is any other field GovMap didn't
provide. It contains no data-quality judgement (see
transaction_quality.py) and no pricing math (see
src/pricing/index_adjustment.py).
"""
from __future__ import annotations

import pandas as pd

NORMALIZED_TRANSACTION_COLUMNS = [
    "deal_id",
    "address",
    "transaction_date",
    "rooms",
    "area_sqm",
    "floor",
    "original_price",
    "original_price_per_sqm",
    "property_type",
    "deal_nature",
    "neighborhood",
    "gush",
    "parcel",
    "sub_parcel",
    "source",
    "source_url",
    "data_retrieved_at",
]

# Hebrew ordinal floor names as they appear in GovMap's floorNo field.
# "קרקע" (ground floor) maps to 0. A value not in this map, and not a
# plain integer, is left null -- never guessed.
HEBREW_FLOOR_MAP = {
    "קרקע": 0,
    "ראשונה": 1,
    "שניה": 2,
    "שנייה": 2,
    "שלישית": 3,
    "רביעית": 4,
    "חמישית": 5,
    "שישית": 6,
    "שביעית": 7,
    "שמינית": 8,
    "תשיעית": 9,
    "עשירית": 10,
    "אחת עשרה": 11,
    "שתים עשרה": 12,
    "שתיים עשרה": 12,
    "שלוש עשרה": 13,
    "ארבע עשרה": 14,
    "חמש עשרה": 15,
    "שש עשרה": 16,
    "שבע עשרה": 17,
    "שמונה עשרה": 18,
    "תשע עשרה": 19,
    "עשרים": 20,
}


def normalize_floor(floor_raw) -> int | None:
    """Convert a GovMap floorNo value to an integer, or None if it can't
    be confidently interpreted."""
    if floor_raw is None:
        return None
    if isinstance(floor_raw, float) and pd.isna(floor_raw):
        return None
    if isinstance(floor_raw, (int, float)):
        return int(floor_raw)

    text = str(floor_raw).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return HEBREW_FLOOR_MAP.get(text)


def _clean_text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    as_float = _to_float(value)
    return int(as_float) if as_float is not None else None


def build_address(street, house_num, settlement) -> str | None:
    """Construct an address string from street, house number and settlement.

    Missing components are simply omitted -- never fabricated.
    """
    street = _clean_text(street)
    settlement = _clean_text(settlement)

    house_num_text = None
    house_num_value = _to_float(house_num)
    if house_num_value is not None:
        house_num_text = (
            str(int(house_num_value)) if house_num_value.is_integer() else str(house_num_value)
        )

    street_part = " ".join(p for p in [street, house_num_text] if p) or None
    parts = [p for p in [street_part, settlement] if p]
    return ", ".join(parts) if parts else None


def normalize_transaction(raw: dict) -> dict:
    """Normalize one raw GovMap deal record into the internal schema."""
    area_sqm = _to_float(raw.get("assetArea"))
    original_price = _to_float(raw.get("dealAmount"))
    original_price_per_sqm = (
        original_price / area_sqm
        if original_price is not None and area_sqm not in (None, 0)
        else None
    )

    transaction_date = None
    deal_date_raw = raw.get("dealDate")
    if deal_date_raw:
        try:
            transaction_date = pd.Timestamp(deal_date_raw)
            if transaction_date.tzinfo is not None:
                # GovMap's dealDate is UTC ("...Z"); drop the tz so this is
                # directly comparable to CBS's tz-naive monthly `period`
                # values in src/data/cbs_client.py.
                transaction_date = transaction_date.tz_localize(None)
            transaction_date = transaction_date.normalize()
        except (ValueError, TypeError):
            transaction_date = None

    return {
        "deal_id": raw.get("dealId"),
        "address": build_address(
            raw.get("streetNameHeb"), raw.get("houseNum"), raw.get("settlementNameHeb")
        ),
        "transaction_date": transaction_date,
        "rooms": _to_float(raw.get("assetRoomNum")),
        "area_sqm": area_sqm,
        "floor": normalize_floor(raw.get("floorNo")),
        "original_price": original_price,
        "original_price_per_sqm": original_price_per_sqm,
        "property_type": _clean_text(raw.get("propertyTypeDescription")),
        "deal_nature": _clean_text(raw.get("dealNatureDescription")),
        "neighborhood": _clean_text(raw.get("neighborhood")),
        "gush": _to_int(raw.get("gushNum")),
        "parcel": _to_int(raw.get("parcelNum")),
        "sub_parcel": _to_int(raw.get("subParcelNum")),
        "source": raw.get("source"),
        "source_url": raw.get("source_url"),
        "data_retrieved_at": raw.get("data_retrieved_at"),
    }


def normalize_transactions(raw_records) -> pd.DataFrame:
    """Normalize a list of raw GovMap records (or a DataFrame of them) into
    the internal schema."""
    if isinstance(raw_records, pd.DataFrame):
        raw_records = raw_records.to_dict(orient="records")
    normalized = [normalize_transaction(r) for r in raw_records]
    return pd.DataFrame(normalized, columns=NORMALIZED_TRANSACTION_COLUMNS)
