"""End-to-end synthetic test of the Feature #9 strategy layer applied on
top of a (synthetic, in-memory) market recommendation, mirroring
scripts/generate_pricing_recommendations.py's strategy section without
touching real files or the network.
"""
import pandas as pd
import pytest

from src.pricing.strategy_adjustment import apply_strategy_adjustment, get_apartment_strategy_row


def _strategy_df(n=5):
    return pd.DataFrame(
        [
            {
                "apartment_id": i,
                "manual_adjustment_pct": 0.0,
                "manual_adjustment_amount": 0.0,
                "strategy_note": "",
            }
            for i in range(1, n + 1)
        ]
    )


def test_zero_default_strategy_means_final_equals_recommended():
    apartments = pd.DataFrame(
        [
            {"apartment_id": 1, "recommended_marketing_price": 4_270_000.0},
            {"apartment_id": 2, "recommended_marketing_price": 6_490_421.75},
            {"apartment_id": 3, "recommended_marketing_price": 3_501_004.23},
        ]
    )
    strategy_df = _strategy_df(3)

    final_prices = []
    for _, row in apartments.iterrows():
        strategy_row = get_apartment_strategy_row(strategy_df, row["apartment_id"])
        final_prices.append(
            apply_strategy_adjustment(
                row["recommended_marketing_price"],
                company_positioning_pct=0.0,
                sales_phase_pct=0.0,
                inventory_strategy_pct=0.0,
                manual_adjustment_pct=strategy_row["manual_adjustment_pct"],
                manual_adjustment_amount=strategy_row["manual_adjustment_amount"],
            )
        )

    apartments["final_strategy_price"] = final_prices

    # recommended_marketing_price is preserved, not overwritten
    assert "recommended_marketing_price" in apartments.columns
    # final_strategy_price is a separate, additional column
    assert "final_strategy_price" in apartments.columns
    # with all-zero strategy config, the two must be identical
    assert (
        apartments["final_strategy_price"] == apartments["recommended_marketing_price"]
    ).all()


def test_nonzero_project_strategy_shifts_all_apartments_consistently():
    apartments = pd.DataFrame(
        [
            {"apartment_id": 1, "recommended_marketing_price": 4_000_000.0},
            {"apartment_id": 2, "recommended_marketing_price": 6_000_000.0},
        ]
    )
    strategy_df = _strategy_df(2)

    final_prices = [
        apply_strategy_adjustment(
            row["recommended_marketing_price"],
            company_positioning_pct=0.02,
            sales_phase_pct=0.0,
            inventory_strategy_pct=0.0,
            manual_adjustment_pct=0.0,
            manual_adjustment_amount=0.0,
        )
        for _, row in apartments.iterrows()
    ]

    assert final_prices[0] == pytest.approx(4_000_000 * 1.02)
    assert final_prices[1] == pytest.approx(6_000_000 * 1.02)
