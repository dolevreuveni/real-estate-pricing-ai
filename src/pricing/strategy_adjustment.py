"""Company-strategy pricing layer.

Applies explicit, transparent percentage/amount adjustments on top of the
market-derived recommended_marketing_price (Feature #8) to produce
final_strategy_price.

This module contains NO property-feature premium logic (parking,
storage, balcony, floor, garden, roof, new-project positioning, top-floor
status, ...). Those characteristics are already reflected in the Current
Market regression signal that produced current_market_price (see
src/pricing/current_market_features.py) -- adding a separate premium for
any of them here would double-count them. This module only combines:

* three project-level strategy percentages (company positioning, sales
  phase, inventory strategy) from src/config/settings.py
* one apartment-specific manual percentage and one manual fixed amount
  from data/external/apartment_strategy_adjustments.csv
  (src/data/apartment_strategy_loader.py)

into a single, auditable final_strategy_price. With every adjustment at
its neutral default (0), final_strategy_price == recommended_marketing_price.
"""
from __future__ import annotations

import math

import pandas as pd


def _validate_number(name: str, value) -> float:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} must be a number, got NaN.")
    return float(value)


def calculate_project_strategy_factor(
    company_positioning_pct: float,
    sales_phase_pct: float,
    inventory_strategy_pct: float,
    manual_adjustment_pct: float,
) -> float:
    """project_strategy_factor = 1 + sum of all percentage adjustments."""
    total = (
        _validate_number("company_positioning_pct", company_positioning_pct)
        + _validate_number("sales_phase_pct", sales_phase_pct)
        + _validate_number("inventory_strategy_pct", inventory_strategy_pct)
        + _validate_number("manual_adjustment_pct", manual_adjustment_pct)
    )
    return 1.0 + total


def apply_strategy_adjustment(
    base_price: float,
    company_positioning_pct: float,
    sales_phase_pct: float,
    inventory_strategy_pct: float,
    manual_adjustment_pct: float,
    manual_adjustment_amount: float,
) -> float:
    """final_strategy_price = base_price * project_strategy_factor + manual_adjustment_amount.

    Raises ValueError for a non-numeric input, a non-positive
    base_price, or a non-positive result -- never silently substitutes a
    fallback value or clips to zero.
    """
    base_price = _validate_number("base_price", base_price)
    if base_price <= 0:
        raise ValueError(f"base_price must be positive, got {base_price}.")

    factor = calculate_project_strategy_factor(
        company_positioning_pct, sales_phase_pct, inventory_strategy_pct, manual_adjustment_pct
    )
    amount = _validate_number("manual_adjustment_amount", manual_adjustment_amount)

    result = base_price * factor + amount

    if result <= 0:
        raise ValueError(
            f"final_strategy_price must be positive, got {result} "
            f"(base_price={base_price}, project_strategy_factor={factor}, "
            f"manual_adjustment_amount={amount})."
        )
    return result


def get_apartment_strategy_row(strategy_df: pd.DataFrame, apartment_id) -> dict:
    """Return {manual_adjustment_pct, manual_adjustment_amount, strategy_note}
    for one apartment. Raises ValueError if the row is missing or
    duplicated -- never falls back to a neutral default silently."""
    matches = strategy_df[strategy_df["apartment_id"] == apartment_id]
    if matches.empty:
        raise ValueError(
            f"No strategy adjustment row found for apartment_id {apartment_id}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple strategy adjustment rows found for apartment_id {apartment_id}."
        )

    row = matches.iloc[0]
    return {
        "manual_adjustment_pct": _validate_number(
            "manual_adjustment_pct", float(row["manual_adjustment_pct"])
        ),
        "manual_adjustment_amount": _validate_number(
            "manual_adjustment_amount", float(row["manual_adjustment_amount"])
        ),
        "strategy_note": row["strategy_note"],
    }
