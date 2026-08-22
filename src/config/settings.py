"""Project-wide configuration for the target real-estate POC."""
from pathlib import Path

PROJECT_NAME = "Real Estate Pricing POC"
PROJECT_ADDRESS = "Helsinki 24"
PROJECT_CITY = "Tel Aviv"
PROJECT_NEIGHBORHOOD = "Kikar HaMedina"
SEARCH_RADIUS_KM = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "output"

# Israeli Central Bureau of Statistics (CBS) price-index API.
# Public API, no credentials required.
CBS_BASE_URL = "https://api.cbs.gov.il"
CBS_INDEX_ENDPOINT = "/index/data/price"
CBS_HOUSING_PRICE_INDEX_SERIES = "40010"
CBS_USER_AGENT = "real-estate-pricing-ai/1.0"
CBS_REQUEST_TIMEOUT_SECONDS = 15

# The Housing Price Index is bi-monthly and CBS revises recently published
# observations as more transaction data comes in. The CBS API does not
# expose an explicit provisional/final flag per observation, so the newest
# this-many observations are treated as provisional (see src/data/cbs_client.py).
CBS_PROVISIONAL_OBSERVATIONS_COUNT = 3

# Public GovMap real-estate transaction endpoints (see src/data/govmap_client.py).
# Verified during a read-only technical spike against the real, public API.
GOVMAP_BASE_URL = "https://www.govmap.gov.il"
GOVMAP_USER_AGENT = "real-estate-pricing-ai/1.0"
GOVMAP_REQUEST_TIMEOUT_SECONDS = 15

# The GovMap address-search endpoint tokenizes by the language of
# searchText: Hebrew text needs language="he", English text needs
# language="en" (verified live -- "he" against an English address string
# returns zero results). This must match the script PROJECT_ADDRESS /
# PROJECT_CITY are written in; both are currently English, hence "en".
GOVMAP_SEARCH_LANGUAGE = "en"
GOVMAP_SEARCH_FILTER_TYPE = "address"
GOVMAP_SEARCH_MAX_RESULTS = 10

# Radius (meters) for the deals-by-point lookup used to discover nearby
# polygon_ids. Deliberately a separate setting from SEARCH_RADIUS_KM: that
# value is the (future) comparable-search radius in km, while this is
# GovMap's own point-lookup radius in meters -- verified at 200m in the
# technical spike. Each extra 100m roughly doubles the number of GovMap
# API calls this pipeline makes (one street-deals call per nearby polygon).
GOVMAP_DEALS_RADIUS_M = 200

TRANSACTIONS_START_DATE = "2021-01"
TRANSACTIONS_PAGE_SIZE = 100
TRANSACTIONS_DEAL_TYPE = 2

# Baseline transaction-price regression (see src/pricing/regression_*.py).
REGRESSION_MODEL_VERSION = "baseline_linear_v1"
REGRESSION_RANDOM_STATE = 42
REGRESSION_TEST_SIZE = 0.2

# Current Market model (see src/data/current_market_loader.py and
# src/pricing/current_market_*.py). The input file is manually supplied
# SYNTHETIC POC data standing in for a future Yad2/Madlan/developer feed
# integration -- CURRENT_MARKET_DATA_TYPE is stamped onto every output
# this dataset produces so it can never be mistaken for real scraped
# market data.
# Feature #7.5: switched to the enriched current-market file, which adds
# parking_count, storage_area_sqm, balcony_direction, garden_area_sqm,
# roof_area_sqm, is_top_floor, and directions columns on top of the
# original current_market_500.xlsx schema. The old file is obsolete for
# this feature onward -- do not use it.
CURRENT_MARKET_INPUT_PATH = EXTERNAL_DATA_DIR / "current_market_500_updated.xlsx"
CURRENT_MARKET_SHEET_NAME = "Current_Market_Listings"
CURRENT_MARKET_DATA_TYPE = "synthetic_poc"
CURRENT_MARKET_MODEL_VERSION = "current_market_linear_v1"

# This project targets a NEW residential development (see PROJECT_NAME /
# PROJECT_ADDRESS above), not a second-hand resale. This is an explicit,
# documented strategic/config assumption -- never inferred or fabricated
# per apartment -- used only to select the "New Project" market_segment
# value when applying the Current Market model to the 39 target
# apartments, none of which carry a market_segment field of their own.
TARGET_MARKET_SEGMENT = "New Project"

# Weighted combination of the two independent pricing signals into
# recommended_marketing_price (see src/pricing/pricing_recommendation.py).
# Must sum to 1.0 -- validated at import time of that module and again
# every time prices are combined.
HISTORICAL_MARKET_WEIGHT = 0.70
CURRENT_MARKET_WEIGHT = 0.30
