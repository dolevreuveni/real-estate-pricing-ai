"""Feature preparation for the baseline transaction-price regression.

Maps the historical-transactions schema and the target-apartment schema
onto one small, explicit set of canonical model features. Model fitting
and prediction live in regression_model.py -- this module only prepares
data (loading, filtering, and column mapping).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical predictor columns for the baseline model. Deliberately a
# short whitelist: area, rooms, and floor are the only fields that exist
# reliably in BOTH the historical transactions dataset and the target
# apartment dataset. Everything else -- deal_id, address, transaction_date,
# original_price, original_price_per_sqm, CBS index values,
# adjusted_price_per_sqm, source/source_url, exclusion_reason -- is
# deliberately excluded: it either leaks the target, identifies a record,
# or isn't an appropriate baseline predictor (see Feature #7 spec).
FEATURE_COLUMNS = ["area_sqm", "rooms", "floor"]
TARGET_COLUMN = "adjusted_price"

# Maps target-apartment columns (data/processed/apartments.csv) onto the
# canonical feature names above. For multi-level apartments (duplex/
# triplex), floor_min is used as the floor representation: it's the
# apartment's entry/ground floor and is directly comparable to a
# transaction's single-level `floor` value. floor_max and num_levels are
# not used as predictors in this baseline.
APARTMENT_FEATURE_MAP = {
    "interior_area_sqm": "area_sqm",
    "rooms": "rooms",
    "floor_min": "floor",
}


def load_transactions_csv(path: str | Path) -> pd.DataFrame:
    """Load data/external/transactions.csv for training-data preparation."""
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df["is_eligible_comparable"] = df["is_eligible_comparable"].astype(bool)
    if "used_for_historical_model" in df.columns:
        df["used_for_historical_model"] = df["used_for_historical_model"].astype(bool)
    return df


def select_training_transactions(transactions: pd.DataFrame) -> tuple:
    """Split transactions into (eligible, trainable) subsets.

    `eligible`  = is_eligible_comparable == True (Feature #5's general
                  market-data-quality flag). Returned unmodified -- nothing
                  is dropped or altered in the source view.
    `trainable` = used_for_historical_model == True. This column
                  (src/data/historical_transaction_enrichment.py) is the
                  single source of truth for Historical Regression
                  training eligibility: it already requires
                  is_eligible_comparable, the strict positive residential
                  whitelist (property_type=="דירה" AND deal_nature in
                  {"דירה בבית קומות","דירת גן"}), a valid sold-fraction
                  normalization (if applicable), and all required model
                  fields (FEATURE_COLUMNS + TARGET_COLUMN) present. A row
                  excluded from `trainable` is never removed from
                  `eligible` or from the source dataset on disk.

    Raises ValueError if `transactions` doesn't have the
    used_for_historical_model column -- run
    src.data.historical_transaction_enrichment.evaluate_historical_training_eligibility
    (already wired into scripts/fetch_transactions.py) first, rather than
    silently falling back to a weaker selection rule.
    """
    if "used_for_historical_model" not in transactions.columns:
        raise ValueError(
            "transactions is missing 'used_for_historical_model'. Run "
            "src.data.historical_transaction_enrichment.evaluate_historical_training_eligibility "
            "before selecting training transactions."
        )

    eligible = transactions[transactions["is_eligible_comparable"] == True].copy()  # noqa: E712
    trainable = transactions[transactions["used_for_historical_model"] == True].copy()  # noqa: E712

    return eligible, trainable


def transactions_to_training_frame(transactions: pd.DataFrame) -> tuple:
    """Return (X, y) ready for model fitting from a trainable transactions subset."""
    X = transactions[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
    y = transactions[TARGET_COLUMN].astype(float).reset_index(drop=True)
    return X, y


def apartments_to_feature_frame(apartments: pd.DataFrame) -> pd.DataFrame:
    """Map the target-apartment schema onto the canonical model features.

    Returns a DataFrame indexed like `apartments` with FEATURE_COLUMNS
    plus an `is_priceable` boolean flag for rows missing a required
    source field. Missing values are never filled/invented -- an
    apartment missing area, rooms, or floor_min is simply flagged
    unpriceable.
    """
    mapped = pd.DataFrame(index=apartments.index)
    for source_col, feature_col in APARTMENT_FEATURE_MAP.items():
        mapped[feature_col] = apartments[source_col]

    mapped["is_priceable"] = mapped[FEATURE_COLUMNS].notna().all(axis=1)
    return mapped
