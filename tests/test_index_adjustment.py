"""Tests for index_adjustment."""
import pandas as pd
import pytest

from src.data.cbs_client import INDEX_HISTORY_COLUMNS
from src.pricing.index_adjustment import (
    adjust_transaction_price,
    adjust_transaction_price_using_cbs,
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


def _fake_cbs_history() -> pd.DataFrame:
    rows = [
        {
            "period": pd.Timestamp(2026, 5, 1),
            "year": 2026,
            "month": 5,
            "index_value": 593.6,
            "is_provisional": True,
            "series_id": 40010,
            "series_name": "מחירי דירות",
            "source": "CBS (Israeli Central Bureau of Statistics)",
            "source_url": "https://api.cbs.gov.il/index/data/price",
            "data_retrieved_at": pd.Timestamp.now(tz="UTC"),
        },
        {
            "period": pd.Timestamp(2026, 2, 1),
            "year": 2026,
            "month": 2,
            "index_value": 600.0,
            "is_provisional": False,
            "series_id": 40010,
            "series_name": "מחירי דירות",
            "source": "CBS (Israeli Central Bureau of Statistics)",
            "source_url": "https://api.cbs.gov.il/index/data/price",
            "data_retrieved_at": pd.Timestamp.now(tz="UTC"),
        },
        {
            "period": pd.Timestamp(2024, 1, 1),
            "year": 2024,
            "month": 1,
            "index_value": 500.0,
            "is_provisional": False,
            "series_id": 40010,
            "series_name": "מחירי דירות",
            "source": "CBS (Israeli Central Bureau of Statistics)",
            "source_url": "https://api.cbs.gov.il/index/data/price",
            "data_retrieved_at": pd.Timestamp.now(tz="UTC"),
        },
    ]
    return pd.DataFrame(rows, columns=INDEX_HISTORY_COLUMNS)


def test_adjust_transaction_price_using_cbs():
    history = _fake_cbs_history()

    result = adjust_transaction_price_using_cbs(
        original_price=4_000_000,
        transaction_date="2024-01-01",
        history=history,
    )

    # transaction index = 500.0 (2024-01), current stable index = 600.0
    # (2026-02; 2026-05 is provisional and skipped)
    assert result["price_index_at_transaction"] == pytest.approx(500.0)
    assert result["current_stable_price_index"] == pytest.approx(600.0)
    assert result["current_stable_index_period"] == pd.Timestamp(2026, 2, 1)
    assert result["index_adjustment_factor"] == pytest.approx(1.2)
    assert result["adjusted_price"] == pytest.approx(4_800_000)
