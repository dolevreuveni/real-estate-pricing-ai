"""Orchestrates the existing pricing components to price ONE apartment --
either one of the 39 real project apartments or a custom apartment
entered in the dashboard's Price Simulator.

This module introduces NO new pricing methodology and duplicates NO
formulas: it only calls into the already-existing, already-tested
modules:

    src/pricing/regression_model.py        (historical prediction)
    src/pricing/current_market_model.py    (Current Market prediction)
    src/pricing/pricing_recommendation.py  (70/30 weighted blend)
    src/pricing/strategy_adjustment.py     (company strategy layer)

and reuses the same canonical FEATURE_COLUMNS whitelists already defined
in regression_features.py / current_market_features.py, so a custom
apartment is guaranteed to go through exactly the same features -- and
only those features -- as a real one. In particular this guarantees the
historical model never sees parking/storage/balcony/etc: those columns
simply aren't in HISTORICAL_FEATURE_COLUMNS.
"""
from __future__ import annotations

import pandas as pd

from src.config.settings import (
    CURRENT_MARKET_WEIGHT,
    HISTORICAL_MARKET_WEIGHT,
    TARGET_MARKET_SEGMENT,
)
from src.pricing.current_market_features import FEATURE_COLUMNS as MARKET_FEATURE_COLUMNS
from src.pricing.current_market_model import predict as predict_current_market
from src.pricing.pricing_recommendation import combine_prices
from src.pricing.pricing_utils import enforce_non_negative_predictions
from src.pricing.regression_features import FEATURE_COLUMNS as HISTORICAL_FEATURE_COLUMNS
from src.pricing.regression_model import predict as predict_historical
from src.pricing.strategy_adjustment import apply_strategy_adjustment

# Characteristics a dashboard user may enter for a custom (simulated)
# apartment. A superset of both models' feature whitelists plus a couple
# of descriptive-only fields (num_levels) that neither model uses as a
# predictor but that are shown in the apartment characteristics view.
CUSTOM_APARTMENT_FIELDS = [
    "rooms",
    "interior_area_sqm",
    "floor",
    "num_levels",
    "balcony_area_sqm",
    "balcony_direction",
    "parking_count",
    "storage_area_sqm",
    "garden_area_sqm",
    "roof_area_sqm",
    "is_top_floor",
    "property_type",
]


def custom_apartment_to_historical_features(apartment: dict) -> pd.DataFrame:
    """Map a custom apartment dict to the historical model's feature frame.

    Uses ONLY HISTORICAL_FEATURE_COLUMNS (area_sqm, rooms, floor) --
    parking/storage/balcony/etc are never passed to the historical model.
    This mirrors the model's real limitation: those fields don't exist in
    the GovMap/CBS training data (see src/pricing/regression_features.py).
    """
    row = {
        "area_sqm": apartment["interior_area_sqm"],
        "rooms": apartment["rooms"],
        "floor": apartment["floor"],
    }
    return pd.DataFrame([row])[HISTORICAL_FEATURE_COLUMNS]


def custom_apartment_to_market_features(apartment: dict) -> pd.DataFrame:
    """Map a custom apartment dict to the Current Market model's feature frame.

    market_segment is set from TARGET_MARKET_SEGMENT (project config),
    exactly as it is for the 39 real project apartments -- never inferred
    per custom apartment (see src/pricing/current_market_features.py).
    """
    row = {
        "area_sqm": apartment["interior_area_sqm"],
        "rooms": apartment["rooms"],
        "floor": apartment["floor"],
        "balcony_area_sqm": apartment["balcony_area_sqm"],
        "parking_count": apartment["parking_count"],
        "storage_area_sqm": apartment["storage_area_sqm"],
        "garden_area_sqm": apartment["garden_area_sqm"],
        "roof_area_sqm": apartment["roof_area_sqm"],
        "property_type": apartment["property_type"],
        "market_segment": TARGET_MARKET_SEGMENT,
        "balcony_direction": apartment["balcony_direction"],
        "is_top_floor": apartment["is_top_floor"],
    }
    return pd.DataFrame([row])[MARKET_FEATURE_COLUMNS]


def predict_custom_apartment(apartment: dict, historical_model, current_market_model) -> dict:
    """Predict historical_base_price and current_market_price for one
    custom apartment, using the SAME predict() functions used for the 39
    real project apartments.

    Raises ValueError if either model produces an invalid (negative)
    prediction -- never returns a fabricated fallback value.
    """
    historical_X = custom_apartment_to_historical_features(apartment)
    market_X = custom_apartment_to_market_features(apartment)

    historical_raw = predict_historical(historical_model, historical_X)
    market_raw = predict_current_market(current_market_model, market_X)

    historical_price = enforce_non_negative_predictions(historical_raw).iloc[0]
    current_market_price = enforce_non_negative_predictions(market_raw).iloc[0]

    if pd.isna(historical_price):
        raise ValueError("Historical model produced an invalid (negative) prediction.")
    if pd.isna(current_market_price):
        raise ValueError("Current Market model produced an invalid (negative) prediction.")

    return {
        "historical_base_price": float(historical_price),
        "current_market_price": float(current_market_price),
    }


def build_pricing_breakdown(
    historical_base_price: float,
    current_market_price: float,
    interior_area_sqm: float,
    company_positioning_pct: float = 0.0,
    sales_phase_pct: float = 0.0,
    inventory_strategy_pct: float = 0.0,
    manual_adjustment_pct: float = 0.0,
    manual_adjustment_amount: float = 0.0,
) -> dict:
    """Build the full, explainable pricing breakdown for one apartment
    from its two model prices.

    Reuses combine_prices() (Feature #8) and apply_strategy_adjustment()
    (Feature #9) exactly as scripts/generate_pricing_recommendations.py
    does -- so this produces identical numbers to the stored
    apartment_pricing_recommendations.csv for a real project apartment,
    and the same calculation for a fresh custom-apartment prediction.

    Returns a dict whose keys match apartment_pricing_recommendations.csv
    columns, so the same dashboard rendering component can display either
    a static CSV row or a freshly computed simulator result.
    """
    recommended_marketing_price = combine_prices(historical_base_price, current_market_price)
    historical_contribution = historical_base_price * HISTORICAL_MARKET_WEIGHT
    current_market_contribution = current_market_price * CURRENT_MARKET_WEIGHT

    final_strategy_price = apply_strategy_adjustment(
        recommended_marketing_price,
        company_positioning_pct,
        sales_phase_pct,
        inventory_strategy_pct,
        manual_adjustment_pct,
        manual_adjustment_amount,
    )

    final_strategy_price_per_sqm = (
        final_strategy_price / interior_area_sqm if interior_area_sqm else None
    )

    return {
        "historical_base_price": historical_base_price,
        "current_market_price": current_market_price,
        "historical_weight": HISTORICAL_MARKET_WEIGHT,
        "current_market_weight": CURRENT_MARKET_WEIGHT,
        "historical_contribution": historical_contribution,
        "current_market_contribution": current_market_contribution,
        "recommended_marketing_price": recommended_marketing_price,
        "company_positioning_adjustment_pct": company_positioning_pct,
        "sales_phase_adjustment_pct": sales_phase_pct,
        "inventory_strategy_adjustment_pct": inventory_strategy_pct,
        "manual_adjustment_pct": manual_adjustment_pct,
        "manual_adjustment_amount": manual_adjustment_amount,
        "final_strategy_price": final_strategy_price,
        "final_strategy_price_per_sqm": final_strategy_price_per_sqm,
    }
