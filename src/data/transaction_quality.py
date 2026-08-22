"""Data-quality evaluation for normalized GovMap transactions.

We do not blindly treat every government record as a usable comparable.
Every transaction receives `is_eligible_comparable` and `exclusion_reason`
-- excluded transactions are never dropped, only flagged, so the pipeline
can explain exactly why a record wasn't used.

Rules here are deliberately conservative and explainable: missing core
fields, or a clearly non-residential property type. No statistical
outlier removal and no arbitrary price/price-per-sqm thresholds.

`is_eligible_comparable = True` means only that a transaction passed these
basic data-quality/residential checks and MAY be considered later by the
future Comparable Engine. It is NOT a statement that the transaction has
already been selected as a comparable for any specific apartment.
Geographic distance, apartment similarity, new-vs-second-hand matching,
and final comparable selection are separate, not-yet-built features.
"""
from __future__ import annotations

import pandas as pd

REASON_MISSING_PRICE = "missing_or_non_positive_price"
REASON_MISSING_AREA = "missing_or_non_positive_area"
REASON_MISSING_ROOMS = "missing_rooms"
REASON_NON_RESIDENTIAL = "non_residential_property_type"
REASON_NON_RESIDENTIAL_DEAL_NATURE = "non_residential_deal_nature"
REASON_INSUFFICIENT_DATA = "insufficient_core_data"

# GovMap/Tax-Authority property-type descriptions that are clearly not a
# residential apartment sale (parking, storage, commercial, land, etc).
# Preserved as the raw Hebrew terms used in propertyTypeDescription --
# not translated or fuzzy-matched.
NON_RESIDENTIAL_PROPERTY_TYPES = {
    "חניה",
    "חנייה",
    "מחסן",
    "מחסן/חניה",
    "משרד",
    "חנות",
    "מסחרי",
    "תעשיה",
    "תעשייה",
    "קרקע",
    "מבנה חקלאי",
}

# dealNatureDescription can indicate a non-residential deal even when
# propertyTypeDescription is a generic value like "בנין" (building) that
# isn't itself in NON_RESIDENTIAL_PROPERTY_TYPES -- e.g. a "בנין" record
# whose deal_nature is "משרד" (office) is an office sale, not an apartment.
# Kept to only the values actually observed in the live dataset so far;
# not a guessed/broad category list.
NON_RESIDENTIAL_DEAL_NATURES = {
    "משרד",
}


def evaluate_transaction(row: dict) -> tuple:
    """Return (is_eligible_comparable, exclusion_reasons) for one normalized row.

    Every rule is checked independently, so a row failing multiple rules
    gets all of the relevant reasons, not just the first one found.
    """
    reasons = []

    price = row.get("original_price")
    if price is None or pd.isna(price) or price <= 0:
        reasons.append(REASON_MISSING_PRICE)

    area = row.get("area_sqm")
    if area is None or pd.isna(area) or area <= 0:
        reasons.append(REASON_MISSING_AREA)

    rooms = row.get("rooms")
    if rooms is None or pd.isna(rooms):
        reasons.append(REASON_MISSING_ROOMS)

    property_type = row.get("property_type")
    if property_type in NON_RESIDENTIAL_PROPERTY_TYPES:
        reasons.append(REASON_NON_RESIDENTIAL)

    deal_nature = row.get("deal_nature")
    if deal_nature in NON_RESIDENTIAL_DEAL_NATURES:
        reasons.append(REASON_NON_RESIDENTIAL_DEAL_NATURE)

    address = row.get("address")
    if address is None or (isinstance(address, float) and pd.isna(address)):
        reasons.append(REASON_INSUFFICIENT_DATA)

    return (len(reasons) == 0, reasons)


def evaluate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add is_eligible_comparable and exclusion_reason columns.

    Never drops rows -- excluded transactions stay in the returned
    DataFrame with their reason(s) recorded.
    """
    result = transactions.copy()
    eligibility = []
    exclusion_reasons = []
    for _, row in result.iterrows():
        is_eligible, reasons = evaluate_transaction(row.to_dict())
        eligibility.append(is_eligible)
        exclusion_reasons.append(", ".join(reasons) if reasons else None)

    result["is_eligible_comparable"] = eligibility
    result["exclusion_reason"] = exclusion_reasons
    return result
