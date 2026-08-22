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
