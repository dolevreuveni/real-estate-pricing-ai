"""Tests for historical_transaction_enrichment -- the strict residential
whitelist and sold-fraction/full-ownership-price normalization used to
decide Historical Regression training eligibility.
"""
import pandas as pd
import pytest

from src.data.historical_transaction_enrichment import (
    REASON_INVALID_SOLD_FRACTION,
    REASON_MISSING_MODEL_FIELDS,
    REASON_NON_RESIDENTIAL_DEAL_NATURE,
    REASON_NON_RESIDENTIAL_PROPERTY_TYPE,
    REASON_NOT_ELIGIBLE_COMPARABLE,
    REASON_SUSPICIOUS_PARTIAL_TRANSACTION,
    SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS,
    compute_full_ownership_price,
    evaluate_historical_training_eligibility,
    validate_sold_fraction,
)


def _transaction_row(**overrides):
    row = {
        "deal_id": 1,
        "address": "הלסינקי 13, תל אביב-יפו",
        "rooms": 3.0,
        "area_sqm": 80.0,
        "floor": 2.0,
        "original_price": 4_000_000.0,
        "adjusted_price": 4_500_000.0,
        "adjusted_price_per_sqm": 56250.0,
        "property_type": "דירה",
        "deal_nature": "דירה בבית קומות",
        "is_eligible_comparable": True,
    }
    row.update(overrides)
    return row


def _evaluate_one(**overrides) -> dict:
    df = pd.DataFrame([_transaction_row(**overrides)])
    result = evaluate_historical_training_eligibility(df)
    return result.iloc[0].to_dict()


# --- 1/2: permitted type combinations -----------------------------------


def test_diraha_and_dira_bevet_komot_is_permitted():
    row = _evaluate_one(property_type="דירה", deal_nature="דירה בבית קומות")
    assert bool(row["used_for_historical_model"]) is True
    assert row["historical_model_exclusion_reason"] is None


def test_diraha_and_dirat_gan_is_permitted():
    row = _evaluate_one(property_type="דירה", deal_nature="דירת גן")
    assert bool(row["used_for_historical_model"]) is True
    assert row["historical_model_exclusion_reason"] is None


# --- 3/4: excluded deal_nature values (whitelist, not blacklist) --------


def test_deal_nature_chanut_is_excluded():
    row = _evaluate_one(property_type="דירה", deal_nature="חנות")
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_NON_RESIDENTIAL_DEAL_NATURE in row["historical_model_exclusion_reason"]


def test_deal_nature_kombinatzia_is_excluded():
    row = _evaluate_one(property_type="דירה", deal_nature="קומבינציה")
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_NON_RESIDENTIAL_DEAL_NATURE in row["historical_model_exclusion_reason"]


# --- 5: property_type outside whitelist ----------------------------------


def test_non_residential_property_type_is_excluded():
    row = _evaluate_one(property_type="בנין", deal_nature="דירה בבית קומות")
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_NON_RESIDENTIAL_PROPERTY_TYPE in row["historical_model_exclusion_reason"]
    # deal_nature check is not separately reported once property_type fails
    assert REASON_NON_RESIDENTIAL_DEAL_NATURE not in row["historical_model_exclusion_reason"]


# --- 6-9: sold-fraction normalization (division, never multiplication) --


def test_sold_fraction_1_0_leaves_price_unchanged():
    fraction, invalid = validate_sold_fraction(1.0)
    assert invalid is False
    price = compute_full_ownership_price(2_000_000, fraction)
    assert price == pytest.approx(2_000_000)


def test_sold_fraction_0_5_doubles_price():
    fraction, invalid = validate_sold_fraction(0.5)
    assert invalid is False
    price = compute_full_ownership_price(1_000_000, fraction)
    assert price == pytest.approx(2_000_000)


def test_sold_fraction_0_01_normalizes_by_100x():
    # the exact real-world example: Helsinki 8, 2024-04-09, reported 503,000
    fraction, invalid = validate_sold_fraction(0.01)
    assert invalid is False
    price = compute_full_ownership_price(503_000, fraction)
    assert price == pytest.approx(50_300_000)


def test_sold_fraction_0_001_normalizes_by_1000x():
    fraction, invalid = validate_sold_fraction(0.001)
    assert invalid is False
    price = compute_full_ownership_price(1_000, fraction)
    assert price == pytest.approx(1_000_000)


# --- 10/11: invalid sold_fraction ----------------------------------------


def test_sold_fraction_zero_or_negative_is_invalid():
    fraction, invalid = validate_sold_fraction(0)
    assert fraction is None
    assert invalid is True

    fraction, invalid = validate_sold_fraction(-0.5)
    assert fraction is None
    assert invalid is True


def test_sold_fraction_greater_than_one_is_invalid():
    fraction, invalid = validate_sold_fraction(1.5)
    assert fraction is None
    assert invalid is True


def test_row_with_invalid_sold_fraction_is_excluded_from_training():
    row = _evaluate_one(sold_fraction=1.5)
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_INVALID_SOLD_FRACTION in row["historical_model_exclusion_reason"]


# --- 12: original reported price is preserved -----------------------------


def test_original_reported_price_column_is_preserved():
    df = pd.DataFrame([_transaction_row(original_price=503_000.0)])
    result = evaluate_historical_training_eligibility(df)
    assert result.iloc[0]["original_price"] == pytest.approx(503_000.0)


# --- 13: training target uses the CBS-adjusted (full-ownership) price ----


def test_missing_adjusted_price_excludes_from_training():
    row = _evaluate_one(adjusted_price=None)
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_MISSING_MODEL_FIELDS in row["historical_model_exclusion_reason"]


# --- 14: non-training transactions remain in the dataset -----------------


def test_excluded_rows_are_never_dropped():
    df = pd.DataFrame(
        [
            _transaction_row(deal_id=1, property_type="דירה", deal_nature="דירה בבית קומות"),
            _transaction_row(deal_id=2, property_type="בנין", deal_nature="חנות"),
        ]
    )
    result = evaluate_historical_training_eligibility(df)
    assert len(result) == 2
    assert set(result["deal_id"]) == {1, 2}


# --- general data quality precondition ------------------------------------


def test_not_eligible_comparable_excludes_from_training_even_if_residential():
    row = _evaluate_one(
        property_type="דירה", deal_nature="דירה בבית קומות", is_eligible_comparable=False
    )
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_NOT_ELIGIBLE_COMPARABLE in row["historical_model_exclusion_reason"]


# --- suspicious partial transaction fallback heuristic --------------------


def test_suspicious_price_per_sqm_below_threshold_is_flagged_and_excluded():
    # mirrors the real Helsinki 8 case: sold_fraction unavailable, price/sqm far below threshold
    low_price_per_sqm = SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS - 1000
    row = _evaluate_one(adjusted_price_per_sqm=low_price_per_sqm)
    assert bool(row["suspicious_partial_transaction"]) is True
    assert bool(row["used_for_historical_model"]) is False
    assert REASON_SUSPICIOUS_PARTIAL_TRANSACTION in row["historical_model_exclusion_reason"]


def test_normal_price_per_sqm_above_threshold_is_not_flagged():
    normal_price_per_sqm = SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS + 10_000
    row = _evaluate_one(adjusted_price_per_sqm=normal_price_per_sqm)
    assert bool(row["suspicious_partial_transaction"]) is False


def test_heuristic_does_not_apply_when_sold_fraction_is_valid():
    # a valid sold_fraction must never be overridden by the fallback
    # price-per-sqm heuristic, even if the (correctly low, pre-fraction)
    # adjusted_price_per_sqm looks suspicious on its own.
    low_price_per_sqm = SUSPICIOUS_PRICE_PER_SQM_THRESHOLD_NIS - 1000
    row = _evaluate_one(adjusted_price_per_sqm=low_price_per_sqm, sold_fraction=1.0)
    assert bool(row["suspicious_partial_transaction"]) is False


# --- 15: full ownership price columns -------------------------------------


def test_full_ownership_price_and_per_sqm_computed_when_fraction_valid():
    df = pd.DataFrame(
        [_transaction_row(original_price=503_000.0, area_sqm=119.0, sold_fraction=0.01)]
    )
    result = evaluate_historical_training_eligibility(df)
    row = result.iloc[0]
    assert row["full_ownership_price"] == pytest.approx(50_300_000.0)
    assert row["full_ownership_price_per_sqm"] == pytest.approx(50_300_000.0 / 119.0)


def test_full_ownership_price_is_null_when_sold_fraction_unavailable():
    df = pd.DataFrame([_transaction_row(original_price=503_000.0, area_sqm=119.0)])
    result = evaluate_historical_training_eligibility(df)
    row = result.iloc[0]
    assert pd.isna(row["full_ownership_price"])
    assert pd.isna(row["full_ownership_price_per_sqm"])


def test_sold_fraction_column_is_added_when_absent_from_input():
    df = pd.DataFrame([_transaction_row()]).drop(columns=[])  # no sold_fraction column at all
    assert "sold_fraction" not in df.columns
    result = evaluate_historical_training_eligibility(df)
    assert "sold_fraction" in result.columns
    assert pd.isna(result.iloc[0]["sold_fraction"])


# --- regression guard: missing sold_fraction must not zero out training --
# GovMap does not expose a sold-fraction field for any transaction (see
# module docstring). A dataset entirely missing sold_fraction must still
# produce a non-zero, non-trivial used_for_historical_model population --
# missing sold_fraction on its own is "not applicable", never an
# automatic exclusion. Only the price-per-sqm fallback heuristic (a
# genuinely low price for the property) or the other independent checks
# (whitelist, required fields, is_eligible_comparable) may exclude a row.


def test_missing_sold_fraction_across_the_whole_dataset_does_not_zero_out_training():
    df = pd.DataFrame(
        [
            _transaction_row(deal_id=i, adjusted_price_per_sqm=50_000.0 + i)
            for i in range(10)
        ]
    )
    assert "sold_fraction" not in df.columns  # matches real GovMap data: field never present

    result = evaluate_historical_training_eligibility(df)

    assert result["sold_fraction"].isna().all()
    assert result["used_for_historical_model"].sum() == 10
    assert (result["used_for_historical_model"] == True).all()  # noqa: E712
