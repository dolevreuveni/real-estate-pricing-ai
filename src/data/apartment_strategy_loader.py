"""Loading and validation for the apartment-specific company-strategy
adjustments CSV.

data/external/apartment_strategy_adjustments.csv holds one row per target
apartment with optional manual pricing adjustments from the marketing
department (see src/pricing/strategy_adjustment.py for how they're
applied). All values start neutral (0 / "") -- populating them with a
real business decision is a deliberate future action, never invented by
this module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import APARTMENT_STRATEGY_ADJUSTMENTS_PATH

REQUIRED_COLUMNS = [
    "apartment_id",
    "manual_adjustment_pct",
    "manual_adjustment_amount",
    "strategy_note",
]

NUMERIC_COLUMNS = ["apartment_id", "manual_adjustment_pct", "manual_adjustment_amount"]


def load_apartment_strategy_adjustments(
    path: str | Path = APARTMENT_STRATEGY_ADJUSTMENTS_PATH,
) -> pd.DataFrame:
    """Load and validate the strategy-adjustments CSV.

    Raises a clear ValueError for a missing required column, a
    non-numeric value in a numeric column, or a duplicate apartment_id --
    never silently repairs malformed data.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"apartment_strategy_adjustments: missing required column(s) {missing} in {path}. "
            f"Expected columns: {REQUIRED_COLUMNS}."
        )

    df = df[REQUIRED_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"apartment_strategy_adjustments: column '{column}' in {path} contains a "
                f"non-numeric value."
            ) from exc

    duplicate_ids = df["apartment_id"][df["apartment_id"].duplicated()].unique().tolist()
    if duplicate_ids:
        raise ValueError(
            f"apartment_strategy_adjustments: duplicate apartment_id value(s) "
            f"{duplicate_ids} in {path}."
        )

    return df
