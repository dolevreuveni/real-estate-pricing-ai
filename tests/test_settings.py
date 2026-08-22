"""Tests for the project configuration layer."""
from src.config import settings


def test_project_constants_are_defined():
    assert settings.PROJECT_NAME == "Real Estate Pricing POC"
    assert settings.PROJECT_ADDRESS == "Helsinki 24"
    assert settings.PROJECT_CITY == "Tel Aviv"
    assert settings.PROJECT_NEIGHBORHOOD == "Kikar HaMedina"
    assert settings.SEARCH_RADIUS_KM == 1.0


def test_data_dirs_are_under_project_root():
    assert settings.RAW_DATA_DIR == settings.PROJECT_ROOT / "data" / "raw"
    assert settings.PROCESSED_DATA_DIR == settings.PROJECT_ROOT / "data" / "processed"
    assert settings.EXTERNAL_DATA_DIR == settings.PROJECT_ROOT / "data" / "external"


def test_project_root_points_to_actual_repo_root():
    assert (settings.PROJECT_ROOT / "requirements.txt").exists()
    assert (settings.RAW_DATA_DIR / "Apartment_example.xlsx").exists()
