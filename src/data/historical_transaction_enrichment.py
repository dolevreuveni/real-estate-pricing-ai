"""Historical Regression training-eligibility enrichment.

This module answers one question per transaction: is this row allowed to
train the Historical Regression Model (src/pricing/regression_model.py)?
It does NOT touch general market-data quality (`is_eligible_comparable`,
`exclusion_reason` -- see src/data/transaction_quality.py, Feature #5)
and does NOT affect the Current Market model in any way.

Two problems motivated this module:

1. STRICT RESIDENTIAL WHITELIST -- `is_eligible_comparable` uses a
   blacklist of known-bad property types/deal natures (Feature #5), which
   is not strict enough for regression training: it lets through rows
   like property_type="בנין" with an untagged deal_nature, or
   deal_nature values ("קומבינציה", "חנות", "קרקע למגורים", ...) that are
   not ordinary apartment sales. Historical Regression training now
   requires an explicit POSITIVE whitelist:
       property_type == "דירה"
       AND deal_nature in {"דירה בבית קומות", "דירת גן"}

2. SOLD FRACTION ("חלק נמכר") -- the Israeli Tax Authority's own
   interface exposes a "sold fraction" field representing what portion of
   a property a recorded deal actually covers (e.g. 0.01 = 1% of the
   property, common for a small ownership-share transfer, not a full
   apartment sale). A ₪503,000 deal with sold_fraction=0.01 is really a
   ₪50,300,000-equivalent transaction, not a distressed ₪503K apartment.

   INVESTIGATION RESULT (see src/data/govmap_client.py and the raw HTTP
   response verified directly against
   https://www.govmap.gov.il/api/real-estate/street-deals/{polygon_id}):
   the public GovMap street-deals API does NOT expose any field
   corresponding to sold fraction. Every transaction record returned by
   that endpoint has exactly the same key set --
       objectid, settlementId, settlementNameHeb, settlementNameEng,
       streetCode, streetNameHeb, streetNameEng, houseNum, floorNo,
       assetArea, dealAmount, dealId, propertyTypeDescription,
       dealNatureDescription, assetRoomNum, neighborhood, dealDate,
       gushNum, parcelNum, subParcelNum, polygonId, shape, sourceorder
   -- with nothing resembling a fraction/share/part field, verified both
   through the Python client and a raw, unparsed curl request to the live
   endpoint. This was confirmed on the exact real transaction this
   investigation was triggered by: הלסינקי 8, תל אביב-יפו, 2024-04-09,
   dealId 6257204850, dealAmount 503000, assetArea 119 -- its raw record
   carries no sold-fraction-like field either.

   Because the field is genuinely unavailable from GovMap (not merely
   omitted for full-ownership deals -- there is no field to omit), this
   module does NOT assume missing sold_fraction == 1.0. `sold_fraction`
   and `full_ownership_price` are still added as columns (so a future
   data source that DOES expose the fraction needs no schema migration),
   but they are null for every row today. Instead, a conservative POC
   fallback heuristic flags likely-partial transactions by adjusted price
   per sqm (see SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS below) and excludes
   them from training -- flagged and preserved, never deleted, and never
   silently "corrected".
"""
from __future__ import annotations

import pandas as pd

from src.pricing.regression_features import FEATURE_COLUMNS, TARGET_COLUMN

RESIDENTIAL_PROPERTY_TYPE = "דירה"
RESIDENTIAL_DEAL_NATURE_WHITELIST = {"דירה בבית קומות", "דירת גן"}

# POC-only data-quality heuristic, NOT a general Israeli real-estate rule.
# Used only as a fallback because GovMap does not expose a sold-fraction
# field for us to validate transactions against directly. If sold_fraction
# ever becomes available and valid for a row, this heuristic must not gate
# that row (see evaluate_historical_training_eligibility below).
SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS = 40_000

REASON_NOT_ELIGIBLE_COMPARABLE = "not_eligible_comparable"
REASON_NON_RESIDENTIAL_PROPERTY_TYPE = "non_residential_property_type_strict"
REASON_NON_RESIDENTIAL_DEAL_NATURE = "non_residential_deal_nature_strict"
REASON_INVALID_SOLD_FRACTION = "invalid_sold_fraction"
REASON_SUSPICIOUS_PARTIAL_TRANSACTION = "suspicious_partial_transaction"
REASON_MISSING_MODEL_FIELDS = "missing_required_model_fields"


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_sold_fraction(value) -> tuple:
    """Validate a raw sold_fraction value.

    Returns (valid_fraction: float | None, is_invalid: bool):
      * missing/NaN            -> (None, False)  -- simply not applicable
      * malformed / <=0 / >1   -> (None, True)   -- an explicit validation
                                                     failure (0 < f <= 1 required)
      * valid                  -> (f, False)
    """
    if _is_missing(value):
        return None, False
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None, True
    if fraction <= 0 or fraction > 1:
        return None, True
    return fraction, False


def compute_full_ownership_price(reported_price, sold_fraction) -> float | None:
    """full_ownership_price = reported_price / sold_fraction.

    Never multiplies by the fraction. Assumes `sold_fraction` has already
    been validated (0 < sold_fraction <= 1) via validate_sold_fraction --
    pass the validated value, not a raw unchecked one. Returns None if
    either input is missing.
    """
    if sold_fraction is None or _is_missing(reported_price):
        return None
    return float(reported_price) / sold_fraction


def evaluate_historical_training_eligibility(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add sold-fraction/full-ownership audit columns and the final
    used_for_historical_model / historical_model_exclusion_reason flags.

    Never drops rows -- excluded transactions stay in the returned
    DataFrame with their reason(s) recorded, exactly like
    src/data/transaction_quality.py's is_eligible_comparable pattern.
    """
    result = transactions.copy()

    if "sold_fraction" not in result.columns:
        # GovMap does not currently expose this field (see module
        # docstring) -- added as an always-null column so the schema is
        # forward-compatible with a future source that does.
        result["sold_fraction"] = pd.NA

    full_ownership_prices = []
    full_ownership_prices_per_sqm = []
    invalid_sold_fraction_flags = []
    for _, row in result.iterrows():
        fraction, is_invalid = validate_sold_fraction(row.get("sold_fraction"))
        invalid_sold_fraction_flags.append(is_invalid)

        full_price = compute_full_ownership_price(row.get("original_price"), fraction)
        full_ownership_prices.append(full_price)

        area = row.get("area_sqm")
        if full_price is not None and not _is_missing(area) and area:
            full_ownership_prices_per_sqm.append(full_price / area)
        else:
            full_ownership_prices_per_sqm.append(None)

    result["full_ownership_price"] = full_ownership_prices
    result["full_ownership_price_per_sqm"] = full_ownership_prices_per_sqm

    # Fallback heuristic: only meaningful (and only ever applied) when
    # sold_fraction is unavailable for that row -- a row with a validated
    # sold_fraction is never gated by this heuristic, per the module spec.
    adjusted_price_per_sqm = result.get("adjusted_price_per_sqm")
    suspicious_flags = []
    for idx in result.index:
        fraction_missing = pd.isna(result.at[idx, "sold_fraction"])
        price_per_sqm = adjusted_price_per_sqm.at[idx] if adjusted_price_per_sqm is not None else None
        is_suspicious = (
            fraction_missing
            and price_per_sqm is not None
            and not pd.isna(price_per_sqm)
            and price_per_sqm < SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS
        )
        suspicious_flags.append(bool(is_suspicious))
    result["suspicious_partial_transaction"] = suspicious_flags

    required_model_columns = list(FEATURE_COLUMNS) + [TARGET_COLUMN]

    used_flags = []
    reasons = []
    for i, (_, row) in enumerate(result.iterrows()):
        row_reasons = []

        if not bool(row.get("is_eligible_comparable")):
            row_reasons.append(REASON_NOT_ELIGIBLE_COMPARABLE)

        if row.get("property_type") != RESIDENTIAL_PROPERTY_TYPE:
            row_reasons.append(REASON_NON_RESIDENTIAL_PROPERTY_TYPE)
        elif row.get("deal_nature") not in RESIDENTIAL_DEAL_NATURE_WHITELIST:
            row_reasons.append(REASON_NON_RESIDENTIAL_DEAL_NATURE)

        if invalid_sold_fraction_flags[i]:
            row_reasons.append(REASON_INVALID_SOLD_FRACTION)

        if suspicious_flags[i]:
            row_reasons.append(REASON_SUSPICIOUS_PARTIAL_TRANSACTION)

        if any(_is_missing(row.get(c)) for c in required_model_columns):
            row_reasons.append(REASON_MISSING_MODEL_FIELDS)

        used_flags.append(len(row_reasons) == 0)
        reasons.append(", ".join(row_reasons) if row_reasons else None)

    result["used_for_historical_model"] = used_flags
    result["historical_model_exclusion_reason"] = reasons

    return result
