"""Combine the historical GovMap/CBS regression signal with the Current
Market signal into a recommended_marketing_price.

Pure math and validation only -- no model fitting, no data loading. The
weights are read from src/config/settings.py (HISTORICAL_MARKET_WEIGHT,
CURRENT_MARKET_WEIGHT) and validated eagerly at import time, so a
misconfiguration (e.g. weights that don't sum to 1.0) fails immediately
and loudly rather than silently producing a skewed price later.
"""
from __future__ import annotations

from src.config.settings import CURRENT_MARKET_WEIGHT, HISTORICAL_MARKET_WEIGHT

_WEIGHT_SUM_TOLERANCE = 1e-9


def validate_weights(historical_weight: float, current_market_weight: float) -> None:
    """Raise ValueError with a clear message if the weights are invalid."""
    if historical_weight < 0 or current_market_weight < 0:
        raise ValueError(
            f"Pricing weights must not be negative, got "
            f"historical={historical_weight}, current_market={current_market_weight}."
        )

    total = historical_weight + current_market_weight
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Pricing weights must sum to 1.0, got {historical_weight} + "
            f"{current_market_weight} = {total}."
        )


def combine_prices(
    historical_base_price,
    current_market_price,
    historical_weight: float = HISTORICAL_MARKET_WEIGHT,
    current_market_weight: float = CURRENT_MARKET_WEIGHT,
) -> float:
    """recommended_marketing_price = historical*w1 + current_market*w2.

    Raises ValueError if either price is missing/non-positive or the
    weights are invalid. Never silently substitutes one signal for the
    other, never invents a price, and never returns a non-positive result
    for valid inputs.
    """
    validate_weights(historical_weight, current_market_weight)

    if historical_base_price is None or historical_base_price != historical_base_price:  # NaN-safe
        raise ValueError("historical_base_price is missing; cannot combine prices.")
    if historical_base_price <= 0:
        raise ValueError(f"historical_base_price must be positive, got {historical_base_price}.")

    if current_market_price is None or current_market_price != current_market_price:  # NaN-safe
        raise ValueError("current_market_price is missing; cannot combine prices.")
    if current_market_price <= 0:
        raise ValueError(f"current_market_price must be positive, got {current_market_price}.")

    return historical_base_price * historical_weight + current_market_price * current_market_weight


# Fail fast: validate the configured weights as soon as this module is
# imported, not only when combine_prices() first gets called.
validate_weights(HISTORICAL_MARKET_WEIGHT, CURRENT_MARKET_WEIGHT)
