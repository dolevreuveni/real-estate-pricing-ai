"""Tests for index_adjustment."""
import pytest

from src.pricing.index_adjustment import (
    adjust_transaction_price,
    calculate_index_adjustment_factor,
)


def test_no_index_change_factor_is_one():
    factor = calculate_index_adjustment_factor(transaction_index=500, current_index=500)
    assert factor == pytest.approx(1.0)

    adjusted = adjust_transaction_price(
        original_price=4_000_000, transaction_index=500, current_index=500
    )
    assert adjusted == pytest.approx(4_000_000)


def test_twenty_percent_index_increase():
    factor = calculate_index_adjustment_factor(transaction_index=500, current_index=600)
    assert factor == pytest.approx(1.2)

    adjusted = adjust_transaction_price(
        original_price=4_000_000, transaction_index=500, current_index=600
    )
    assert adjusted == pytest.approx(4_800_000)


def test_index_decrease_lowers_adjusted_price():
    factor = calculate_index_adjustment_factor(transaction_index=600, current_index=500)
    assert factor == pytest.approx(500 / 600)

    adjusted = adjust_transaction_price(
        original_price=4_800_000, transaction_index=600, current_index=500
    )
    assert adjusted == pytest.approx(4_000_000)


def test_invalid_zero_index_raises():
    with pytest.raises(ValueError):
        calculate_index_adjustment_factor(transaction_index=0, current_index=500)
    with pytest.raises(ValueError):
        calculate_index_adjustment_factor(transaction_index=500, current_index=0)


def test_invalid_negative_index_raises():
    with pytest.raises(ValueError):
        calculate_index_adjustment_factor(transaction_index=-500, current_index=500)
    with pytest.raises(ValueError):
        calculate_index_adjustment_factor(transaction_index=500, current_index=-500)


def test_negative_price_raises():
    with pytest.raises(ValueError):
        adjust_transaction_price(original_price=-1, transaction_index=500, current_index=600)
