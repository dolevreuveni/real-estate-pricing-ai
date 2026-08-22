"""Small utilities shared by the historical regression and Current Market
pricing models.
"""
from __future__ import annotations

import pandas as pd


def enforce_non_negative_predictions(predictions: pd.Series) -> pd.Series:
    """Return a copy of `predictions` with negative values replaced by NaN.

    A negative predicted price is not a valid business value. It is never
    silently accepted or clipped to zero -- the row is treated as "cannot
    be priced" instead, while staying in the index so the caller can
    report exactly which row failed and why.
    """
    cleaned = predictions.astype(float).copy()
    cleaned[cleaned < 0] = float("nan")
    return cleaned
