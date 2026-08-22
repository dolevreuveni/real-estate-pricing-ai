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
