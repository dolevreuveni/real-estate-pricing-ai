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
