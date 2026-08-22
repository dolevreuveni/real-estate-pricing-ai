"""Reusable logic for adjusting historical transaction prices to current
market conditions using a housing price index.

The adjustment never assumes a fixed appreciation rate: it always uses the
ratio between the current index and the index at the transaction date.

This module is pure math: it has no network/HTTP dependency. The CBS
integration lives in src/data/cbs_client.py; `adjust_transaction_price_using_cbs`
below is the only function here that talks to it, and it does so purely to
fetch index values before calling the same pure functions.
"""
from __future__ import annotations

import pandas as pd


def calculate_index_adjustment_factor(transaction_index: float, current_index: float) -> float:
    """Return current_index / transaction_index.

    Raises ValueError if either index is not a positive number.
    """
    _validate_positive_index("transaction_index", transaction_index)
    _validate_positive_index("current_index", current_index)
    return current_index / transaction_index


def adjust_transaction_price(
    original_price: float,
    transaction_index: float,
    current_index: float,
) -> float:
    """Adjust a historical transaction price to current market conditions.

    adjusted_price = original_price * (current_index / transaction_index)
    """
    if original_price < 0:
        raise ValueError(f"original_price must not be negative, got {original_price}.")
    factor = calculate_index_adjustment_factor(transaction_index, current_index)
    return original_price * factor


def _validate_positive_index(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}.")


def adjust_transaction_price_using_cbs(
    original_price: float,
    transaction_date,
    history: pd.DataFrame | None = None,
) -> dict:
    """Adjust a historical transaction price using the CBS Housing Price Index.

    Fetches price_index_at_transaction and the current stable index from
    src.data.cbs_client, then applies the same pure calculation functions
    used everywhere else in this module. Pass `history` (a DataFrame from
    cbs_client) to avoid hitting the local cache/network, e.g. in tests.

    Returns a dict with:
        price_index_at_transaction
        current_stable_price_index
        current_stable_index_period
        index_adjustment_factor
        adjusted_price
    """
    from src.data.cbs_client import get_index_for_date, get_latest_stable_index

    transaction_index = get_index_for_date(transaction_date, history=history)
    current_index, current_period = get_latest_stable_index(history=history)

    factor = calculate_index_adjustment_factor(transaction_index, current_index)
    adjusted_price = adjust_transaction_price(original_price, transaction_index, current_index)

    return {
        "price_index_at_transaction": transaction_index,
        "current_stable_price_index": current_index,
        "current_stable_index_period": current_period,
        "index_adjustment_factor": factor,
        "adjusted_price": adjusted_price,
    }


def enrich_transactions_with_cbs_index(
    transactions: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add CBS-index-based price adjustment columns to a transactions DataFrame.

    Adds: price_index_at_transaction, current_price_index,
    index_adjustment_factor, adjusted_price, adjusted_price_per_sqm.

    Enrichment is attempted for every row with a usable transaction_date
    and original_price, regardless of any comparable-eligibility flag the
    caller may have set elsewhere (e.g. transaction_quality.py) -- an
    excluded transaction may still be technically valid enough for CBS
    enrichment. Rows where enrichment isn't possible (no CBS index for
    that month, missing price/date) are left with null enrichment fields
    rather than a guessed value.
    """
    result = transactions.copy()
    for column in [
        "price_index_at_transaction",
        "current_price_index",
        "index_adjustment_factor",
        "adjusted_price",
        "adjusted_price_per_sqm",
    ]:
        result[column] = None

    for idx, row in result.iterrows():
        original_price = row.get("original_price")
        transaction_date = row.get("transaction_date")
        if original_price is None or pd.isna(original_price) or pd.isna(transaction_date):
            continue

        try:
            enrichment = adjust_transaction_price_using_cbs(
                original_price=float(original_price),
                transaction_date=transaction_date,
                history=history,
            )
        except ValueError:
            continue

        area_sqm = row.get("area_sqm")
        adjusted_price = enrichment["adjusted_price"]
        adjusted_price_per_sqm = (
            adjusted_price / area_sqm if area_sqm not in (None, 0) and not pd.isna(area_sqm) else None
        )

        result.at[idx, "price_index_at_transaction"] = enrichment["price_index_at_transaction"]
        result.at[idx, "current_price_index"] = enrichment["current_stable_price_index"]
        result.at[idx, "index_adjustment_factor"] = enrichment["index_adjustment_factor"]
        result.at[idx, "adjusted_price"] = adjusted_price
        result.at[idx, "adjusted_price_per_sqm"] = adjusted_price_per_sqm

    return result
