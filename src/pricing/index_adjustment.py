"""Reusable logic for adjusting historical transaction prices to current
market conditions using a housing price index.

The adjustment never assumes a fixed appreciation rate: it always uses the
ratio between the current index and the index at the transaction date.
"""
from __future__ import annotations


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
