"""Client for the public GovMap real-estate transaction endpoints.

This module only talks to GovMap over HTTP (with a timeout, never
swallowing network errors) and returns clean Python data structures.
Normalization, quality evaluation and pricing math live elsewhere (see
transaction_normalizer.py, transaction_quality.py and
src/pricing/index_adjustment.py).

Endpoints (verified during a read-only technical spike against the real,
public API -- no CAPTCHA, auth, or rate-limit bypass involved):

    POST {GOVMAP_BASE_URL}/api/search-service/autocomplete
    GET  {GOVMAP_BASE_URL}/api/real-estate/deals/{x},{y}/{radius}
    GET  {GOVMAP_BASE_URL}/api/real-estate/street-deals/{polygon_id}

Coordinate system: the `shape` field returned by the autocomplete endpoint
("POINT(x y)") is passed straight through to the deals-by-point call
exactly as verified in the spike. These are Web Mercator (EPSG:3857)
values, not ITM -- the deals-by-point endpoint was verified to accept them
directly, so no coordinate conversion is performed here.

"Relevant" polygons: the deals-by-point endpoint returns every nearby
building/parcel with recorded deals, most of which are on other streets.
This client fetches street-deals for every polygon_id returned within the
configured radius and deduplicates the resulting records by dealId (the
same street's full deal history comes back from any of its polygon_ids,
as verified in the spike) -- this is simpler and more robust than trying
to match GovMap's Hebrew-only streetNameHeb against a search string that
may have been entered in a different language.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import requests

from src.config.settings import (
    GOVMAP_BASE_URL,
    GOVMAP_REQUEST_TIMEOUT_SECONDS,
    GOVMAP_SEARCH_FILTER_TYPE,
    GOVMAP_SEARCH_LANGUAGE,
    GOVMAP_SEARCH_MAX_RESULTS,
    GOVMAP_USER_AGENT,
)

SOURCE_NAME = "GovMap"
AUTOCOMPLETE_ENDPOINT = "/api/search-service/autocomplete"
DEALS_BY_POINT_ENDPOINT = "/api/real-estate/deals"
STREET_DEALS_ENDPOINT = "/api/real-estate/street-deals"

_SHAPE_POINT_RE = re.compile(r"POINT\s*\(\s*([\-0-9.]+)\s+([\-0-9.]+)\s*\)")


class GovMapRequestError(RuntimeError):
    """Raised when the GovMap API cannot be reached or returns an HTTP error."""


class GovMapResponseError(ValueError):
    """Raised when a GovMap response is malformed or has an unsupported structure."""


def _headers() -> dict:
    return {"User-Agent": GOVMAP_USER_AGENT, "Accept": "application/json"}


def _get(url: str, params: dict | None = None, timeout: float = GOVMAP_REQUEST_TIMEOUT_SECONDS):
    try:
        response = requests.get(url, params=params, headers=_headers(), timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise GovMapRequestError(f"GovMap API GET {url} failed: {exc}") from exc
    try:
        return response.json(), response.url
    except ValueError as exc:
        raise GovMapResponseError(f"GovMap API GET {url} did not return valid JSON.") from exc


def _post(url: str, json_body: dict, timeout: float = GOVMAP_REQUEST_TIMEOUT_SECONDS):
    try:
        response = requests.post(url, json=json_body, headers=_headers(), timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise GovMapRequestError(f"GovMap API POST {url} failed: {exc}") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise GovMapResponseError(f"GovMap API POST {url} did not return valid JSON.") from exc


def resolve_address(search_text: str) -> dict:
    """Resolve free text to GovMap's best-matching address record.

    `search_text` is sent through Python's normal JSON serialization
    (requests' `json=`), so Hebrew or any other text is encoded correctly
    without manual string manipulation.

    Returns the highest-scoring result of type "address" (including its
    raw GovMap `shape`, e.g. "POINT(x y)", for use with
    `get_deals_by_point`). Raises GovMapResponseError if no address match
    is found.
    """
    url = f"{GOVMAP_BASE_URL}{AUTOCOMPLETE_ENDPOINT}"
    payload = {
        "searchText": search_text,
        "language": GOVMAP_SEARCH_LANGUAGE,
        "filterType": GOVMAP_SEARCH_FILTER_TYPE,
        "isAccurate": False,
        "maxResults": GOVMAP_SEARCH_MAX_RESULTS,
    }
    data = _post(url, payload)

    if not isinstance(data, dict) or "results" not in data:
        raise GovMapResponseError(
            "Unsupported GovMap autocomplete response: expected a 'results' key."
        )

    address_results = [r for r in data["results"] if r.get("type") == "address"]
    if not address_results:
        raise GovMapResponseError(f"No GovMap address match found for {search_text!r}.")

    return max(address_results, key=lambda r: r.get("score", 0))


def parse_point_shape(shape: str) -> tuple:
    """Parse a GovMap "POINT(x y)" shape string into (x, y) floats."""
    match = _SHAPE_POINT_RE.match(shape.strip()) if shape else None
    if not match:
        raise GovMapResponseError(f"Could not parse GovMap point shape: {shape!r}")
    return float(match.group(1)), float(match.group(2))


def get_deals_by_point(x: float, y: float, radius: int) -> list:
    """Return the nearby building/polygon deal summaries around (x, y)."""
    url = f"{GOVMAP_BASE_URL}{DEALS_BY_POINT_ENDPOINT}/{x},{y}/{radius}"
    data, _ = _get(url)
    if not isinstance(data, list):
        raise GovMapResponseError(
            f"Unsupported GovMap deals-by-point response: expected a list, got {type(data)}."
        )
    return data


def get_street_deals(
    polygon_id: str,
    limit: int = 100,
    deal_type=None,
    start_date=None,
    end_date=None,
) -> list:
    """Return ALL street-deal transaction records for `polygon_id`.

    Paginates via limit/offset until GovMap's reported totalCount is
    exhausted. Each returned record is stamped with source, source_url
    (the exact request URL that returned it) and data_retrieved_at.
    """
    url = f"{GOVMAP_BASE_URL}{STREET_DEALS_ENDPOINT}/{polygon_id}"
    all_records = []
    offset = 0

    while True:
        params = {"limit": limit, "offset": offset}
        if deal_type is not None:
            params["dealType"] = deal_type
        if start_date is not None:
            params["startDate"] = start_date
        if end_date is not None:
            params["endDate"] = end_date

        page, request_url = _get(url, params=params)
        if not isinstance(page, dict) or "data" not in page:
            raise GovMapResponseError(
                f"Unsupported GovMap street-deals response for polygon {polygon_id}: "
                f"expected a 'data' key."
            )

        records = page["data"]
        if not isinstance(records, list):
            raise GovMapResponseError(
                f"Unsupported GovMap street-deals response for polygon {polygon_id}: "
                f"'data' is not a list."
            )

        retrieved_at = datetime.now(timezone.utc)
        for record in records:
            record["source"] = SOURCE_NAME
            record["source_url"] = request_url
            record["data_retrieved_at"] = retrieved_at
        all_records.extend(records)

        try:
            total_count = int(page.get("totalCount", len(all_records)))
        except (TypeError, ValueError):
            total_count = len(all_records)

        if not records or len(records) < limit or len(all_records) >= total_count:
            break
        offset += len(records)

    return all_records


def fetch_transactions_near_address(
    search_text: str,
    radius: int,
    deal_type=None,
    start_date=None,
    end_date=None,
    page_size: int = 100,
) -> tuple:
    """Full GovMap acquisition sequence for one project address.

    resolve_address -> get_deals_by_point -> get_street_deals for every
    nearby polygon_id -> deduplicate by dealId.

    Returns (resolved_address, transactions) where `transactions` is a
    list of raw GovMap deal dicts (see get_street_deals for the stamped
    source fields).
    """
    resolved = resolve_address(search_text)
    x, y = parse_point_shape(resolved["shape"])

    polygons = get_deals_by_point(x, y, radius)
    polygon_ids = [p["polygon_id"] for p in polygons if p.get("polygon_id")]

    seen_deal_ids = set()
    transactions = []
    for polygon_id in polygon_ids:
        records = get_street_deals(
            polygon_id,
            limit=page_size,
            deal_type=deal_type,
            start_date=start_date,
            end_date=end_date,
        )
        for record in records:
            deal_id = record.get("dealId")
            if deal_id is not None:
                if deal_id in seen_deal_ids:
                    continue
                seen_deal_ids.add(deal_id)
            transactions.append(record)

    return resolved, transactions
