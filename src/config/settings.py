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
