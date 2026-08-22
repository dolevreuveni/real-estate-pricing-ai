"""Tests for strategy_adjustment."""
import ast
import inspect

import pandas as pd
import pytest

from src.pricing import strategy_adjustment
from src.pricing.strategy_adjustment import (
    apply_strategy_adjustment,
    calculate_project_strategy_factor,
    get_apartment_strategy_row,
)


def test_zero_adjustments_return_original_price():
    price = apply_strategy_adjustment(
        base_price=4_270_000,
        company_positioning_pct=0.0,
        sales_phase_pct=0.0,
        inventory_strategy_pct=0.0,
        manual_adjustment_pct=0.0,
        manual_adjustment_amount=0.0,
    )
    assert price == pytest.approx(4_270_000)


def test_positive_percentage_adjustment():
    price = apply_strategy_adjustment(
        base_price=1_000_000,
        company_positioning_pct=0.05,
        sales_phase_pct=0.0,
        inventory_strategy_pct=0.0,
        manual_adjustment_pct=0.0,
        manual_adjustment_amount=0.0,
    )
    assert price == pytest.approx(1_050_000)


def test_negative_percentage_adjustment():
    price = apply_strategy_adjustment(
        base_price=1_000_000,
        company_positioning_pct=-0.03,
        sales_phase_pct=0.0,
        inventory_strategy_pct=0.0,
        manual_adjustment_pct=0.0,
        manual_adjustment_amount=0.0,
    )
    assert price == pytest.approx(970_000)


def test_manual_fixed_amount_adjustment():
    price = apply_strategy_adjustment(
        base_price=1_000_000,
        company_positioning_pct=0.0,
        sales_phase_pct=0.0,
        inventory_strategy_pct=0.0,
        manual_adjustment_pct=0.0,
        manual_adjustment_amount=25_000,
    )
    assert price == pytest.approx(1_025_000)


def test_combined_project_and_apartment_adjustment():
    # 2% company positioning + 1% sales phase - 0.5% inventory + 3% manual, plus a fixed amount
    price = apply_strategy_adjustment(
        base_price=4_000_000,
        company_positioning_pct=0.02,
        sales_phase_pct=0.01,
        inventory_strategy_pct=-0.005,
        manual_adjustment_pct=0.03,
        manual_adjustment_amount=10_000,
    )
    factor = 1 + 0.02 + 0.01 - 0.005 + 0.03
    assert price == pytest.approx(4_000_000 * factor + 10_000)


@pytest.mark.parametrize(
    "bad_value",
    ["5%", None, float("nan"), [0.1], {"pct": 0.1}, True],
)
def test_malformed_percentage_is_rejected(bad_value):
    with pytest.raises(ValueError):
        apply_strategy_adjustment(
            base_price=1_000_000,
            company_positioning_pct=bad_value,
            sales_phase_pct=0.0,
            inventory_strategy_pct=0.0,
            manual_adjustment_pct=0.0,
            manual_adjustment_amount=0.0,
        )


def test_malformed_manual_amount_is_rejected():
    with pytest.raises(ValueError):
        apply_strategy_adjustment(
            base_price=1_000_000,
            company_positioning_pct=0.0,
            sales_phase_pct=0.0,
            inventory_strategy_pct=0.0,
            manual_adjustment_pct=0.0,
            manual_adjustment_amount="10000",
        )


def test_invalid_non_positive_final_price_is_rejected():
    with pytest.raises(ValueError):
        apply_strategy_adjustment(
            base_price=1_000_000,
            company_positioning_pct=-1.5,  # factor becomes negative
            sales_phase_pct=0.0,
            inventory_strategy_pct=0.0,
            manual_adjustment_pct=0.0,
            manual_adjustment_amount=0.0,
        )


def test_invalid_base_price_is_rejected():
    with pytest.raises(ValueError):
        apply_strategy_adjustment(
            base_price=0,
            company_positioning_pct=0.0,
            sales_phase_pct=0.0,
            inventory_strategy_pct=0.0,
            manual_adjustment_pct=0.0,
            manual_adjustment_amount=0.0,
        )
    with pytest.raises(ValueError):
        apply_strategy_adjustment(
            base_price=-500,
            company_positioning_pct=0.0,
            sales_phase_pct=0.0,
            inventory_strategy_pct=0.0,
            manual_adjustment_pct=0.0,
            manual_adjustment_amount=0.0,
        )


def test_duplicate_apartment_strategy_row_is_rejected():
    df = pd.DataFrame(
        [
            {"apartment_id": 1, "manual_adjustment_pct": 0.0, "manual_adjustment_amount": 0.0, "strategy_note": ""},
            {"apartment_id": 1, "manual_adjustment_pct": 0.02, "manual_adjustment_amount": 0.0, "strategy_note": ""},
        ]
    )
    with pytest.raises(ValueError, match="Multiple"):
        get_apartment_strategy_row(df, 1)


def test_missing_apartment_strategy_row_is_rejected():
    df = pd.DataFrame(
        [{"apartment_id": 1, "manual_adjustment_pct": 0.0, "manual_adjustment_amount": 0.0, "strategy_note": ""}]
    )
    with pytest.raises(ValueError, match="No strategy adjustment row found"):
        get_apartment_strategy_row(df, 999)


def test_calculate_project_strategy_factor_is_one_plus_sum():
    factor = calculate_project_strategy_factor(0.02, -0.01, 0.005, 0.03)
    assert factor == pytest.approx(1 + 0.02 - 0.01 + 0.005 + 0.03)


def test_no_property_feature_premium_logic_exists_in_module():
    """Feature #9 must not introduce a separate premium for any
    property characteristic already used by the Current Market model --
    those are already reflected in current_market_price.

    Checks actual code identifiers (function/variable/parameter names)
    via the AST, deliberately ignoring the module's docstring/comments --
    the docstring legitimately *names* these features to explain why
    they're excluded, which is documentation, not logic.
    """
    source = inspect.getsource(strategy_adjustment)
    tree = ast.parse(source)

    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg.lower())
        elif isinstance(node, ast.FunctionDef):
            identifiers.add(node.name.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())

    forbidden_terms = [
        "parking",
        "storage",
        "balcony",
        "garden",
        "roof",
        "floor_premium",
        "new_project",
        "top_floor",
    ]
    for term in forbidden_terms:
        assert not any(term in ident for ident in identifiers), (
            f"strategy_adjustment.py must not reference '{term}' in a code identifier"
        )
