"""Client for the Israeli Central Bureau of Statistics (CBS) price-index API.

This module only talks to the CBS API (over HTTP, with a timeout, never
swallowing network errors) and parses its response into a clean
pandas DataFrame. It does not contain pricing math -- see
src/pricing/index_adjustment.py for that.

Stable-index assumption:
CBS publishes the Housing Price Index (series 40010) monthly, and the
series compares recent two-month periods to prior ones. Per CBS
methodology, the most recently published observations are still subject
to revision as more transaction data is processed. The CBS API response
for this series (verified against the live endpoint) does not expose an
explicit provisional/final flag per observation, so this client falls
back to position-based logic: the newest
`CBS_PROVISIONAL_OBSERVATIONS_COUNT` observations (default 3, see
src/config/settings.py) are treated as provisional and are skipped by
`get_latest_stable_index()`. If CBS ever adds explicit status/finality
metadata to the response, that should be preferred over this fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config.settings import (
    CBS_BASE_URL,
    CBS_HOUSING_PRICE_INDEX_SERIES,
    CBS_INDEX_ENDPOINT,
    CBS_PROVISIONAL_OBSERVATIONS_COUNT,
    CBS_REQUEST_TIMEOUT_SECONDS,
    CBS_USER_AGENT,
    EXTERNAL_DATA_DIR,
)

CACHE_PATH = EXTERNAL_DATA_DIR / "cbs_housing_price_index.csv"
SOURCE_NAME = "CBS (Israeli Central Bureau of Statistics)"

INDEX_HISTORY_COLUMNS = [
    "period",
    "year",
    "month",
    "index_value",
    "is_provisional",
    "series_id",
    "series_name",
    "source",
    "source_url",
    "data_retrieved_at",
]


class CBSRequestError(RuntimeError):
    """Raised when the CBS API cannot be reached or returns an HTTP error."""


class CBSResponseError(ValueError):
    """Raised when the CBS response is malformed, missing data, or has an
    unsupported structure."""


def fetch_index_history(
    series_id: str = CBS_HOUSING_PRICE_INDEX_SERIES,
    timeout: float = CBS_REQUEST_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Call the CBS API and return the full observed history for `series_id`."""
    url = f"{CBS_BASE_URL}{CBS_INDEX_ENDPOINT}"
    params = {"id": series_id, "format": "json", "download": "false"}
    headers = {"User-Agent": CBS_USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise CBSRequestError(f"CBS API request to {url} failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise CBSResponseError(f"CBS API response from {url} is not valid JSON.") from exc

    return _parse_cbs_payload(payload, series_id=series_id, source_url=response.url)


def _parse_cbs_payload(payload: dict, series_id: str, source_url: str) -> pd.DataFrame:
    if not isinstance(payload, dict) or "month" not in payload:
        raise CBSResponseError(
            "Unsupported CBS response structure: expected a 'month' key with series data."
        )

    month_series = payload["month"]
    if not isinstance(month_series, list) or not month_series:
        raise CBSResponseError("CBS response contains no series data under 'month'.")

    series = month_series[0]
    observations = series.get("date")
    if not observations:
        raise CBSResponseError(
            f"CBS response for series {series_id} contains no index observations."
        )

    retrieved_at = datetime.now(timezone.utc)
    records = []
    for obs in observations:
        base = obs.get("currBase") or {}
        index_value = base.get("value")
        year = obs.get("year")
        month = obs.get("month")
        if index_value is None or year is None or month is None:
            raise CBSResponseError(
                f"CBS response for series {series_id} is missing an index value "
                f"for observation {obs}."
            )
        records.append(
            {
                "period": pd.Timestamp(year=int(year), month=int(month), day=1),
                "year": int(year),
                "month": int(month),
                "index_value": float(index_value),
                "series_id": series.get("code", series_id),
                "series_name": series.get("name"),
                "source": SOURCE_NAME,
                "source_url": source_url,
                "data_retrieved_at": retrieved_at,
            }
        )

    history = pd.DataFrame.from_records(records)
    history = history.sort_values("period", ascending=False).reset_index(drop=True)
    history["is_provisional"] = history.index < CBS_PROVISIONAL_OBSERVATIONS_COUNT
    return history[INDEX_HISTORY_COLUMNS]


def save_index_history_cache(history: pd.DataFrame, path: str | Path = CACHE_PATH) -> Path:
    """Write the index history to the local CSV cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history[INDEX_HISTORY_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_index_history_cache(path: str | Path = CACHE_PATH) -> pd.DataFrame:
    """Load a previously cached index history from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached CBS index history found at {path}. "
            f"Call refresh_index_history_cache() first."
        )
    history = pd.read_csv(path, parse_dates=["period", "data_retrieved_at"])

    missing = [c for c in INDEX_HISTORY_COLUMNS if c not in history.columns]
    if missing:
        raise CBSResponseError(
            f"Cached CBS index history at {path} is missing column(s) {missing}."
        )

    history["is_provisional"] = history["is_provisional"].astype(bool)
    return history[INDEX_HISTORY_COLUMNS]


def refresh_index_history_cache(
    series_id: str = CBS_HOUSING_PRICE_INDEX_SERIES,
    path: str | Path = CACHE_PATH,
) -> pd.DataFrame:
    """Fetch fresh data from the live CBS API and overwrite the local cache."""
    history = fetch_index_history(series_id=series_id)
    save_index_history_cache(history, path)
    return history


def get_index_for_date(transaction_date, history: pd.DataFrame | None = None) -> float:
    """Return the CBS index value for the month of `transaction_date`.

    Loads the cached history by default; pass `history` to use an
    already-loaded/fetched DataFrame instead (e.g. in tests).
    """
    if history is None:
        history = load_index_history_cache()

    target = pd.Timestamp(transaction_date).replace(day=1)
    match = history[history["period"] == target]
    if match.empty:
        raise CBSResponseError(
            f"No CBS index value found for {target.strftime('%Y-%m')}. "
            f"Available range: {history['period'].min()} to {history['period'].max()}."
        )
    return float(match.iloc[0]["index_value"])


def get_latest_stable_index(history: pd.DataFrame | None = None) -> tuple:
    """Return (index_value, period) for the most recent non-provisional observation.

    See the module docstring for why position-based logic is used.
    """
    if history is None:
        history = load_index_history_cache()

    stable = history[~history["is_provisional"]]
    if stable.empty:
        raise CBSResponseError("No stable (non-provisional) CBS index observation is available.")

    latest_stable = stable.sort_values("period", ascending=False).iloc[0]
    return float(latest_stable["index_value"]), latest_stable["period"]
