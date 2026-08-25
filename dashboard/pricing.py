"""Pure, non-Streamlit dashboard logic: model training orchestration,
project KPIs, apartment lookup, simulator input ranges/categories, and
display-value formatting.

Streamlit-specific code (widgets, caching decorators, layout) lives in
dashboard/data.py and dashboard/components.py -- everything here is
plain Python/pandas so it can be unit tested without launching Streamlit.
Business/pricing logic is never duplicated here: training and prediction
delegate entirely to the existing src/pricing modules.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.data.build_apartment_dataset import CSV_OUTPUT_PATH as APARTMENTS_CSV_PATH  # noqa: F401
from src.data.current_market_loader import load_current_market_listings
from src.data.market_data_loader import TRANSACTIONS_PATH
from src.pricing.current_market_features import (
    listings_to_training_frame as market_listings_to_training_frame,
)
from src.pricing.current_market_features import select_training_rows as select_market_training_rows
from src.pricing.current_market_model import train_and_evaluate as train_current_market
from src.pricing.custom_apartment_pricing import build_pricing_breakdown, predict_custom_apartment
from src.pricing.regression_features import (
    load_transactions_csv,
    select_training_transactions,
    transactions_to_training_frame,
)
from src.pricing.regression_model import train_and_evaluate as train_historical
from src.config.settings import CURRENT_MARKET_INPUT_PATH


def load_csv_if_exists(path: Path, **read_csv_kwargs) -> pd.DataFrame | None:
    """Return None (never raise) if `path` doesn't exist, so dashboard
    pages can render a graceful warning instead of crashing. Pure
    function -- no Streamlit caching involved, safe to unit test with a
    tmp_path fixture."""
    if not Path(path).exists():
        return None
    return pd.read_csv(path, **read_csv_kwargs)


def load_json_if_exists(path: Path) -> dict | None:
    """Return None (never raise) if `path` doesn't exist."""
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_historical_model() -> dict:
    """Train the historical regression model using the real
    data/external/transactions.csv training pipeline -- the exact same
    steps as scripts/train_baseline_pricing_model.py. Returns
    {"model", "report"}.
    """
    transactions = load_transactions_csv(TRANSACTIONS_PATH)
    _, trainable = select_training_transactions(transactions)
    X, y = transactions_to_training_frame(trainable)
    return train_historical(X, y)


def train_current_market_model() -> dict:
    """Train the Current Market model using ONLY
    data/external/current_market_500_updated.xlsx (CURRENT_MARKET_INPUT_PATH)
    -- the exact same steps as scripts/generate_pricing_recommendations.py.
    """
    listings = load_current_market_listings()
    trainable = select_market_training_rows(listings)
    X, y = market_listings_to_training_frame(trainable)
    return train_current_market(X, y, input_file=str(CURRENT_MARKET_INPUT_PATH))


def simulate_custom_apartment(
    apartment: dict,
    historical_model,
    current_market_model,
    company_positioning_pct: float = 0.0,
    sales_phase_pct: float = 0.0,
    inventory_strategy_pct: float = 0.0,
    manual_adjustment_pct: float = 0.0,
    manual_adjustment_amount: float = 0.0,
) -> dict:
    """Predict both signals for a custom apartment and build the full
    pricing breakdown -- thin wrapper over src/pricing/custom_apartment_pricing.py."""
    predictions = predict_custom_apartment(apartment, historical_model, current_market_model)
    return build_pricing_breakdown(
        predictions["historical_base_price"],
        predictions["current_market_price"],
        apartment["interior_area_sqm"],
        company_positioning_pct,
        sales_phase_pct,
        inventory_strategy_pct,
        manual_adjustment_pct,
        manual_adjustment_amount,
    )


def compute_project_kpis(pricing_df: pd.DataFrame | None) -> dict:
    """apartment_count, priced_count, average/min/max final price,
    average price/sqm, total project value. Uses only rows with
    pricing_status == 'priced'."""
    empty = {
        "apartment_count": 0,
        "priced_count": 0,
        "average_final_price": None,
        "average_final_price_per_sqm": None,
        "min_final_price": None,
        "max_final_price": None,
        "total_project_value": None,
    }
    if pricing_df is None or pricing_df.empty:
        return empty

    priced = pricing_df[pricing_df["pricing_status"] == "priced"]
    if priced.empty:
        return {**empty, "apartment_count": int(len(pricing_df))}

    return {
        "apartment_count": int(len(pricing_df)),
        "priced_count": int(len(priced)),
        "average_final_price": float(priced["final_strategy_price"].mean()),
        "average_final_price_per_sqm": float(priced["final_strategy_price_per_sqm"].mean()),
        "min_final_price": float(priced["final_strategy_price"].min()),
        "max_final_price": float(priced["final_strategy_price"].max()),
        "total_project_value": float(priced["final_strategy_price"].sum()),
    }


def get_apartment_detail(pricing_df: pd.DataFrame | None, apartment_id) -> dict | None:
    """Return the pricing_df row for `apartment_id` as a dict, or None if
    the pricing table is unavailable or the apartment doesn't exist."""
    if pricing_df is None or pricing_df.empty:
        return None
    matches = pricing_df[pricing_df["apartment_id"] == apartment_id]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def summarize_historical_transactions(transactions_df: pd.DataFrame | None) -> dict:
    """Summary counts for the dashboard's Historical Transactions tab,
    including the Historical-Regression-training-eligibility audit fields
    added by src/data/historical_transaction_enrichment.py (sold-fraction
    normalization + the strict residential whitelist)."""
    empty = {
        "total": 0,
        "eligible": 0,
        "excluded": 0,
        "cbs_enriched": 0,
        "cbs_missing": 0,
        "used_for_historical_model": 0,
        "not_used_for_historical_model": 0,
    }
    if transactions_df is None or transactions_df.empty:
        return empty

    total = len(transactions_df)
    eligible = int((transactions_df["is_eligible_comparable"] == True).sum())  # noqa: E712
    cbs_enriched = int(transactions_df["price_index_at_transaction"].notna().sum())
    used_for_model = (
        int((transactions_df["used_for_historical_model"] == True).sum())  # noqa: E712
        if "used_for_historical_model" in transactions_df.columns
        else 0
    )

    return {
        "total": total,
        "eligible": eligible,
        "excluded": total - eligible,
        "cbs_enriched": cbs_enriched,
        "cbs_missing": total - cbs_enriched,
        "used_for_historical_model": used_for_model,
        "not_used_for_historical_model": total - used_for_model,
    }


def get_valid_categories(market_df: pd.DataFrame | None, column: str) -> list:
    """Distinct non-null values for `column` in the Current Market
    dataset, sorted -- used to populate simulator dropdowns from real
    data rather than a hardcoded list."""
    if market_df is None or market_df.empty or column not in market_df.columns:
        return []
    return sorted(market_df[column].dropna().unique().tolist())


def derive_simulator_input_ranges(
    apartments_df: pd.DataFrame | None, market_df: pd.DataFrame | None
) -> dict:
    """Derive (min, max, default) for each numeric simulator input from
    the real apartments + Current Market datasets, so the simulator's
    valid ranges reflect the project's actual data rather than arbitrary
    hardcoded bounds."""

    def _range(series_list, fallback_min, fallback_max, fallback_default):
        usable = [s for s in series_list if s is not None and not s.empty]
        if not usable:
            return {"min": fallback_min, "max": fallback_max, "default": fallback_default}
        values = pd.concat(usable)
        if values.empty:
            return {"min": fallback_min, "max": fallback_max, "default": fallback_default}
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "default": float(values.median()),
        }

    apt_col = (lambda c: apartments_df[c] if apartments_df is not None and c in apartments_df else None)
    mkt_col = (lambda c: market_df[c] if market_df is not None and c in market_df else None)

    return {
        "rooms": _range([apt_col("rooms"), mkt_col("rooms")], 1, 8, 3),
        "interior_area_sqm": _range(
            [apt_col("interior_area_sqm"), mkt_col("area_sqm")], 25.0, 300.0, 70.0
        ),
        "floor": _range([apt_col("floor_min"), mkt_col("floor")], 0, 20, 2),
        "num_levels": {"min": 1, "max": 3, "default": 1},
        "balcony_area_sqm": _range(
            [apt_col("balcony_area_sqm"), mkt_col("balcony_area_sqm")], 0.0, 120.0, 12.0
        ),
        "parking_count": _range([apt_col("parking_count"), mkt_col("parking_count")], 0, 3, 1),
        "storage_area_sqm": _range(
            [apt_col("storage_area_sqm"), mkt_col("storage_area_sqm")], 0.0, 15.0, 4.0
        ),
        "garden_area_sqm": _range(
            [apt_col("garden_area_sqm"), mkt_col("garden_area_sqm")], 0.0, 60.0, 0.0
        ),
        "roof_area_sqm": _range(
            [apt_col("roof_area_sqm"), mkt_col("roof_area_sqm")], 0.0, 100.0, 0.0
        ),
    }


def format_currency(value, compact: bool = False) -> str:
    """₪-formatted currency, e.g. ₪4,270,578 or (compact) ₪4.27M / ₪12K."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if compact:
        abs_value = abs(value)
        if abs_value >= 1_000_000:
            return f"₪{value / 1_000_000:.2f}M"
        if abs_value >= 1_000:
            return f"₪{value / 1_000:.0f}K"
    return f"₪{value:,.0f}"


def format_percent(value) -> str:
    """e.g. 0.02 -> '+2.0%', -0.03 -> '-3.0%', 0.0 -> '0.0%'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.1f}%"


def format_value(value, suffix: str = "") -> str:
    """Generic display formatter: '-' for missing/NaN, 'Yes'/'No' for
    booleans, integers without a trailing '.0'."""
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return f"{value}{suffix}"
