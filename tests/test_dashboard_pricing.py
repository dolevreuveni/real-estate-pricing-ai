"""Tests for dashboard/pricing.py -- the dashboard's pure (non-Streamlit)
logic layer: KPI calculation, apartment lookup, formatting, and loading
helpers. No Streamlit app/server is launched by these tests.
"""
from pathlib import Path

import pandas as pd
import pytest

from dashboard.pricing import (
    compute_project_kpis,
    derive_simulator_input_ranges,
    format_currency,
    format_percent,
    format_value,
    get_apartment_detail,
    get_valid_categories,
    load_csv_if_exists,
    load_json_if_exists,
    summarize_historical_transactions,
)
from src.config.settings import CURRENT_MARKET_DATA_TYPE, CURRENT_MARKET_INPUT_PATH


def _pricing_row(**overrides):
    row = {
        "apartment_id": 1,
        "interior_area_sqm": 70.0,
        "final_strategy_price": 4_000_000.0,
        "final_strategy_price_per_sqm": 57142.86,
        "pricing_status": "priced",
    }
    row.update(overrides)
    return row


def test_pricing_recommendations_csv_loads_correctly_via_pure_loader():
    from src.config.settings import PROCESSED_DATA_DIR

    path = PROCESSED_DATA_DIR / "apartment_pricing_recommendations.csv"
    if not path.exists():
        pytest.skip("apartment_pricing_recommendations.csv not available")

    df = load_csv_if_exists(path)
    assert df is not None
    assert len(df) == 39
    assert "final_strategy_price" in df.columns


def test_project_kpi_calculation():
    df = pd.DataFrame(
        [
            _pricing_row(apartment_id=1, final_strategy_price=4_000_000.0, final_strategy_price_per_sqm=50_000.0),
            _pricing_row(apartment_id=2, final_strategy_price=6_000_000.0, final_strategy_price_per_sqm=60_000.0),
        ]
    )
    kpis = compute_project_kpis(df)

    assert kpis["apartment_count"] == 2
    assert kpis["priced_count"] == 2
    assert kpis["average_final_price"] == pytest.approx(5_000_000.0)
    assert kpis["min_final_price"] == pytest.approx(4_000_000.0)
    assert kpis["max_final_price"] == pytest.approx(6_000_000.0)


def test_project_total_value():
    df = pd.DataFrame(
        [
            _pricing_row(apartment_id=1, final_strategy_price=4_000_000.0),
            _pricing_row(apartment_id=2, final_strategy_price=6_000_000.0),
            _pricing_row(apartment_id=3, final_strategy_price=5_000_000.0),
        ]
    )
    kpis = compute_project_kpis(df)
    assert kpis["total_project_value"] == pytest.approx(15_000_000.0)


def test_kpis_exclude_unpriced_apartments():
    df = pd.DataFrame(
        [
            _pricing_row(apartment_id=1, final_strategy_price=4_000_000.0, pricing_status="priced"),
            _pricing_row(apartment_id=2, final_strategy_price=None, pricing_status="missing_historical_signal"),
        ]
    )
    kpis = compute_project_kpis(df)
    assert kpis["apartment_count"] == 2
    assert kpis["priced_count"] == 1
    assert kpis["average_final_price"] == pytest.approx(4_000_000.0)


def test_kpis_handle_empty_or_none_dataframe():
    assert compute_project_kpis(None)["apartment_count"] == 0
    assert compute_project_kpis(pd.DataFrame())["apartment_count"] == 0


def test_apartment_selection_detail_extraction():
    df = pd.DataFrame([_pricing_row(apartment_id=1), _pricing_row(apartment_id=2, final_strategy_price=9_000_000.0)])
    detail = get_apartment_detail(df, 2)
    assert detail is not None
    assert detail["apartment_id"] == 2
    assert detail["final_strategy_price"] == pytest.approx(9_000_000.0)


def test_missing_apartment_is_handled_cleanly():
    df = pd.DataFrame([_pricing_row(apartment_id=1)])
    assert get_apartment_detail(df, 999) is None
    assert get_apartment_detail(None, 1) is None
    assert get_apartment_detail(pd.DataFrame(), 1) is None


def test_current_market_file_path_is_the_updated_file():
    assert CURRENT_MARKET_INPUT_PATH.name == "current_market_500_updated.xlsx"


def test_synthetic_poc_label_is_preserved():
    assert CURRENT_MARKET_DATA_TYPE == "synthetic_poc"


def test_load_csv_if_exists_handles_missing_file_cleanly(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.csv"
    assert load_csv_if_exists(missing_path) is None


def test_load_csv_if_exists_loads_a_real_file(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(path, index=False)
    df = load_csv_if_exists(path)
    assert df is not None
    assert list(df.columns) == ["a", "b"]


def test_load_json_if_exists_handles_missing_file_cleanly(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.json"
    assert load_json_if_exists(missing_path) is None


def test_load_json_if_exists_loads_a_real_file(tmp_path: Path):
    import json

    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"mae": 1.0}), encoding="utf-8")
    data = load_json_if_exists(path)
    assert data == {"mae": 1.0}


def test_get_valid_categories_derives_from_dataframe():
    df = pd.DataFrame({"property_type": ["Apartment", "Duplex", "Apartment", None]})
    categories = get_valid_categories(df, "property_type")
    assert categories == ["Apartment", "Duplex"]


def test_get_valid_categories_handles_missing_dataframe():
    assert get_valid_categories(None, "property_type") == []
    assert get_valid_categories(pd.DataFrame(), "property_type") == []


def test_derive_simulator_input_ranges_from_real_data():
    apartments = pd.DataFrame(
        [
            {"rooms": 3, "interior_area_sqm": 70.0, "floor_min": 1, "balcony_area_sqm": 12.0,
             "parking_count": 1, "storage_area_sqm": 4.0, "garden_area_sqm": 0.0, "roof_area_sqm": 0.0},
            {"rooms": 5, "interior_area_sqm": 111.0, "floor_min": 3, "balcony_area_sqm": 12.0,
             "parking_count": 2, "storage_area_sqm": 6.0, "garden_area_sqm": 0.0, "roof_area_sqm": 0.0},
        ]
    )
    market = pd.DataFrame(
        [
            {"rooms": 3, "area_sqm": 80.0, "floor": 2, "balcony_area_sqm": 10.0, "parking_count": 1,
             "storage_area_sqm": 3.0, "garden_area_sqm": 0.0, "roof_area_sqm": 0.0},
        ]
    )
    ranges = derive_simulator_input_ranges(apartments, market)

    assert ranges["rooms"]["min"] <= 3 <= ranges["rooms"]["max"]
    assert ranges["interior_area_sqm"]["min"] <= 80.0 <= ranges["interior_area_sqm"]["max"]


def test_derive_simulator_input_ranges_handles_missing_data():
    ranges = derive_simulator_input_ranges(None, None)
    assert ranges["rooms"]["default"] > 0
    assert ranges["interior_area_sqm"]["default"] > 0


def test_format_currency():
    assert format_currency(4_270_578) == "₪4,270,578"
    assert format_currency(None) == "-"
    assert format_currency(float("nan")) == "-"
    assert format_currency(4_270_578, compact=True) == "₪4.27M"
    assert format_currency(12_000, compact=True) == "₪12K"


def test_format_percent():
    assert format_percent(0.0) == "0.0%"
    assert format_percent(0.02) == "+2.0%"
    assert format_percent(-0.03) == "-3.0%"
    assert format_percent(None) == "-"


def test_format_value():
    assert format_value(None) == "-"
    assert format_value(float("nan")) == "-"
    assert format_value(True) == "Yes"
    assert format_value(False) == "No"
    assert format_value(3.0) == "3"
    assert format_value(12.0, " sqm") == "12 sqm"


def test_summarize_historical_transactions_exposes_new_audit_fields():
    df = pd.DataFrame(
        [
            {
                "is_eligible_comparable": True,
                "price_index_at_transaction": 500.0,
                "used_for_historical_model": True,
            },
            {
                "is_eligible_comparable": True,
                "price_index_at_transaction": 500.0,
                "used_for_historical_model": False,
            },
            {
                "is_eligible_comparable": False,
                "price_index_at_transaction": None,
                "used_for_historical_model": False,
            },
        ]
    )
    summary = summarize_historical_transactions(df)

    assert summary["total"] == 3
    assert summary["eligible"] == 2
    assert summary["excluded"] == 1
    assert summary["cbs_enriched"] == 2
    assert summary["cbs_missing"] == 1
    assert summary["used_for_historical_model"] == 1
    assert summary["not_used_for_historical_model"] == 2


def test_summarize_historical_transactions_handles_missing_data():
    empty_summary = summarize_historical_transactions(None)
    assert empty_summary["total"] == 0
    assert summarize_historical_transactions(pd.DataFrame())["total"] == 0


def test_summarize_historical_transactions_handles_dataframe_without_new_column():
    # a transactions.csv generated before this feature -- must not crash
    df = pd.DataFrame(
        [{"is_eligible_comparable": True, "price_index_at_transaction": 500.0}]
    )
    summary = summarize_historical_transactions(df)
    assert summary["used_for_historical_model"] == 0
    assert summary["not_used_for_historical_model"] == 1


def test_dashboard_used_for_model_count_matches_actual_regression_training_set():
    """Single source of truth guard: the dashboard's "Used for Model"
    count must always equal the number of rows the Historical Regression
    pipeline actually trains on -- both read used_for_historical_model
    from the same transactions.csv. This is the exact bug class behind
    a dashboard showing "Used for Model = 0" while the model itself
    trains on a non-empty population."""
    from src.data.market_data_loader import TRANSACTIONS_PATH
    from src.pricing.regression_features import (
        load_transactions_csv,
        select_training_transactions,
    )

    if not TRANSACTIONS_PATH.exists():
        pytest.skip("data/external/transactions.csv not available")

    transactions = load_transactions_csv(TRANSACTIONS_PATH)
    _, trainable = select_training_transactions(transactions)

    summary = summarize_historical_transactions(transactions)

    assert summary["used_for_historical_model"] == len(trainable)
    assert summary["used_for_historical_model"] > 0
