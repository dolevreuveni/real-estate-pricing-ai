"""Streamlit-cached data loading for the dashboard.

Thin wrappers over existing src/data loaders and dashboard/pricing.py's
pure load_csv_if_exists/load_json_if_exists helpers and training
functions -- caching only, no new business/loading logic. Every loader
returns None (rather than raising) when its source file is missing, so
page code can render a graceful st.warning/st.error instead of crashing.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import pricing as pricing_logic
from src.config.settings import CURRENT_MARKET_INPUT_PATH, OUTPUT_DATA_DIR, PROCESSED_DATA_DIR
from src.data.build_apartment_dataset import CSV_OUTPUT_PATH as APARTMENTS_CSV_PATH
from src.data.current_market_loader import load_current_market_listings
from src.data.market_data_loader import TRANSACTIONS_PATH

RECOMMENDATIONS_CSV_PATH = PROCESSED_DATA_DIR / "apartment_pricing_recommendations.csv"
REGRESSION_MODEL_REPORT_PATH = OUTPUT_DATA_DIR / "regression_model_report.json"
CURRENT_MARKET_MODEL_REPORT_PATH = OUTPUT_DATA_DIR / "current_market_model_report.json"
STRATEGY_REPORT_PATH = OUTPUT_DATA_DIR / "strategy_pricing_report.json"


@st.cache_data
def load_pricing_recommendations() -> pd.DataFrame | None:
    return pricing_logic.load_csv_if_exists(RECOMMENDATIONS_CSV_PATH)


@st.cache_data
def load_apartments() -> pd.DataFrame | None:
    return pricing_logic.load_csv_if_exists(APARTMENTS_CSV_PATH)


@st.cache_data
def load_transactions() -> pd.DataFrame | None:
    return pricing_logic.load_csv_if_exists(TRANSACTIONS_PATH)


@st.cache_data
def load_current_market_raw() -> pd.DataFrame | None:
    if not CURRENT_MARKET_INPUT_PATH.exists():
        return None
    return load_current_market_listings()


@st.cache_data
def load_regression_model_report() -> dict | None:
    return pricing_logic.load_json_if_exists(REGRESSION_MODEL_REPORT_PATH)


@st.cache_data
def load_current_market_model_report() -> dict | None:
    return pricing_logic.load_json_if_exists(CURRENT_MARKET_MODEL_REPORT_PATH)


@st.cache_data
def load_strategy_report() -> dict | None:
    return pricing_logic.load_json_if_exists(STRATEGY_REPORT_PATH)


@st.cache_resource(show_spinner="Training historical model...")
def get_historical_model() -> dict:
    """Cached: trained once per Streamlit session, reused for every
    simulator prediction. Raises if data/external/transactions.csv is
    missing/invalid -- callers should catch and show st.error."""
    return pricing_logic.train_historical_model()


@st.cache_resource(show_spinner="Training Current Market model...")
def get_current_market_model() -> dict:
    """Cached: trained once per Streamlit session. Raises if
    data/external/current_market_500_updated.xlsx is missing/invalid --
    callers should catch and show st.error."""
    return pricing_logic.train_current_market_model()
