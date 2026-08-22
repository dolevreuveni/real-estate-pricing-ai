"""Tests for pricing_recommendation."""
import math

import pytest

from src.pricing.pricing_recommendation import combine_prices, validate_weights


def test_weights_that_sum_to_one_are_valid():
    validate_weights(0.7, 0.3)  # should not raise
    validate_weights(1.0, 0.0)
    validate_weights(0.0, 1.0)


def test_weights_that_do_not_sum_to_one_raise():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_weights(0.7, 0.4)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_weights(0.5, 0.4)


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="must not be negative"):
        validate_weights(-0.1, 1.1)


def test_weighted_price_calculation_is_correct():
    price = combine_prices(
        historical_base_price=1_000_000,
        current_market_price=2_000_000,
        historical_weight=0.7,
        current_market_weight=0.3,
    )
    assert price == pytest.approx(1_000_000 * 0.7 + 2_000_000 * 0.3)


def test_missing_historical_signal_is_not_silently_replaced():
    with pytest.raises(ValueError, match="historical_base_price"):
        combine_prices(historical_base_price=None, current_market_price=2_000_000)

    with pytest.raises(ValueError, match="historical_base_price"):
        combine_prices(historical_base_price=float("nan"), current_market_price=2_000_000)


def test_missing_current_market_signal_is_not_silently_replaced():
    with pytest.raises(ValueError, match="current_market_price"):
        combine_prices(historical_base_price=1_000_000, current_market_price=None)

    with pytest.raises(ValueError, match="current_market_price"):
        combine_prices(historical_base_price=1_000_000, current_market_price=float("nan"))


def test_zero_or_negative_price_is_rejected_not_silently_accepted():
    with pytest.raises(ValueError):
        combine_prices(historical_base_price=0, current_market_price=2_000_000)
    with pytest.raises(ValueError):
        combine_prices(historical_base_price=1_000_000, current_market_price=-500)


def test_invalid_weights_reject_combination_even_with_valid_prices():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        combine_prices(
            historical_base_price=1_000_000,
            current_market_price=2_000_000,
            historical_weight=0.5,
            current_market_weight=0.6,
        )


def test_valid_combination_never_produces_a_non_positive_result():
    price = combine_prices(1_000_000, 1_000_000)
    assert price > 0
    assert not math.isnan(price)
