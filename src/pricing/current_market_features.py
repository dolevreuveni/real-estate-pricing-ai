"""Feature preparation for the Current Market pricing model.

Maps the Current Market Excel schema and the target-apartment schema onto
one small, HONEST set of canonical model features -- honest meaning every
feature used to predict a target apartment's price is either a real field
already present in data/processed/apartments.csv, or an explicit,
documented project-level config value (TARGET_MARKET_SEGMENT). Nothing
about a target apartment is invented or guessed.

Feature #7.5 enriched both data/processed/apartments.csv and
data/external/current_market_500_updated.xlsx with matching new fields
(parking_count, storage_area_sqm, balcony_direction, garden_area_sqm,
roof_area_sqm, is_top_floor), so this feature set grew from the original
5 honest features to include them all.

Deliberately still excluded, and why:

* building_age, condition, elevator, distance_from_project_km --
  data/processed/apartments.csv has no equivalent field, and the spec
  explicitly forbids inventing these for target apartments.
* `directions` (general apartment air-direction, e.g. "מזרח, מערב") --
  a different concept from `balcony_direction` (one balcony's compass
  facing). Since balcony_direction now exists explicitly and honestly on
  both sides, it is used directly; `directions` stays in the apartment
  output as valuable project information but is not forced into this
  model (see src/data/apartment_reader.py's module docstring for the
  full reasoning, and the original Feature #8 investigation that first
  established directions != balcony_direction).

Model fitting lives in current_market_model.py -- this module only
prepares data.
"""
from __future__ import annotations

import pandas as pd

from src.config.settings import TARGET_MARKET_SEGMENT

NUMERIC_FEATURES = [
    "area_sqm",
    "rooms",
    "floor",
    "balcony_area_sqm",
    "parking_count",
    "storage_area_sqm",
    "garden_area_sqm",
    "roof_area_sqm",
]
CATEGORICAL_FEATURES = ["property_type", "market_segment", "balcony_direction", "is_top_floor"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "asking_price"

# data/processed/apartments.csv column -> canonical feature name.
# floor_min is used as the floor representation for multi-level
# apartments, matching the convention already established for the
# historical regression model (src/pricing/regression_features.py).
# balcony_direction and is_top_floor map directly (no translation needed
# here): apartment_reader.py already normalizes balcony_direction into
# the same English compass vocabulary the Current Market dataset uses,
# and produces is_top_floor as the same True/False/None concept.
APARTMENT_FEATURE_MAP = {
    "interior_area_sqm": "area_sqm",
    "rooms": "rooms",
    "floor_min": "floor",
    "balcony_area_sqm": "balcony_area_sqm",
    "parking_count": "parking_count",
    "storage_area_sqm": "storage_area_sqm",
    "garden_area_sqm": "garden_area_sqm",
    "roof_area_sqm": "roof_area_sqm",
    "balcony_direction": "balcony_direction",
    "is_top_floor": "is_top_floor",
}

# apartments.csv property_type (Feature #1 vocabulary) -> current-market
# property_type (Excel vocabulary: Apartment / Duplex / Garden Apartment /
# Penthouse). "triplex" has no equivalent category in the current-market
# dataset and is deliberately left unmapped rather than guessed: the raw
# value passes through unmapped and becomes an unseen category to the
# model's OneHotEncoder(handle_unknown="ignore"), which contributes zero
# for that dimension instead of a fabricated match.
APARTMENT_TO_MARKET_PROPERTY_TYPE = {
    "regular": "Apartment",
    "garden": "Garden Apartment",
    "duplex": "Duplex",
}


def select_training_rows(listings: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of `listings` usable for training: rows with
    every required model field present. Nothing is dropped from the
    source dataset on disk -- this only affects what's used to fit the
    model."""
    required_columns = [TARGET_COLUMN] + FEATURE_COLUMNS
    complete_mask = listings[required_columns].notna().all(axis=1)
    return listings[complete_mask].copy()


def listings_to_training_frame(listings: pd.DataFrame) -> tuple:
    """Return (X, y) ready for model fitting from a trainable listings subset."""
    X = listings[FEATURE_COLUMNS].copy()
    y = listings[TARGET_COLUMN].astype(float).reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def apartments_to_market_feature_frame(apartments: pd.DataFrame) -> pd.DataFrame:
    """Map the target-apartment schema onto the Current Market model's
    canonical features.

    market_segment is not read from `apartments` (it has no such column);
    it is set uniformly from TARGET_MARKET_SEGMENT, an explicit
    project-level config assumption -- not fabricated per apartment.

    Returns a DataFrame indexed like `apartments` with FEATURE_COLUMNS
    plus an `is_priceable` boolean flag for rows missing a required
    source field. Missing values are never filled/invented.
    """
    mapped = pd.DataFrame(index=apartments.index)
    for source_col, feature_col in APARTMENT_FEATURE_MAP.items():
        mapped[feature_col] = apartments[source_col]

    mapped["property_type"] = apartments["property_type"].map(
        APARTMENT_TO_MARKET_PROPERTY_TYPE
    )
    # Unmapped categories (e.g. "triplex") pass through as their raw
    # value rather than becoming null -- they are still "priceable"
    # (see module docstring: unseen category -> OneHotEncoder ignores it).
    mapped["property_type"] = mapped["property_type"].fillna(apartments["property_type"])

    mapped["market_segment"] = TARGET_MARKET_SEGMENT

    # is_top_floor is boolean; OneHotEncoder handles bool categoricals
    # fine, but normalize None explicitly so a missing value is treated
    # as "unpriceable" rather than an accidental False.
    required_for_pricing = [c for c in FEATURE_COLUMNS if c not in ("market_segment",)]
    mapped["is_priceable"] = mapped[required_for_pricing].notna().all(axis=1)
    return mapped
