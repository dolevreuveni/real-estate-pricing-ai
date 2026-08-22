"""Tests for cbs_client. All HTTP calls are mocked; no live network access."""
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.data.cbs_client import (
    CBSRequestError,
    CBSResponseError,
    INDEX_HISTORY_COLUMNS,
    fetch_index_history,
    get_index_for_date,
    get_latest_stable_index,
    load_index_history_cache,
    save_index_history_cache,
)


def _obs(year, month, value):
    return {
        "year": year,
        "percent": 0.0,
        "percentYear": 0.0,
        "currBase": {"baseDesc": "1993 average", "value": value},
        "prevBase": None,
        "month": month,
        "monthDesc": "",
    }


# Mirrors the real series-40010 shape (most recent observation first).
SAMPLE_OBSERVATIONS = [
    _obs(2026, 5, 593.6),
    _obs(2026, 4, 593.0),
    _obs(2026, 3, 599.6),
    _obs(2026, 2, 602.0),
    _obs(2026, 1, 599.6),
    _obs(2025, 12, 600.2),
]


def _fake_cbs_payload(observations):
    return {
        "month": [{"code": 40010, "name": "מחירי דירות", "date": observations}],
        "quarter": [],
        "paging": {},
    }


def _mock_response(payload, status_ok=True, url="https://api.cbs.gov.il/index/data/price"):
    response = Mock()
    response.url = url
    response.json.return_value = payload
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 error")
    return response


def test_valid_cbs_response_is_parsed_into_dataframe():
    payload = _fake_cbs_payload(SAMPLE_OBSERVATIONS)
    with patch("src.data.cbs_client.requests.get", return_value=_mock_response(payload)):
        history = fetch_index_history()

    assert isinstance(history, pd.DataFrame)
    assert list(history.columns) == INDEX_HISTORY_COLUMNS
    assert len(history) == 6
    assert history.iloc[0]["index_value"] == 593.6
    assert bool(history.iloc[0]["is_provisional"]) is True
    assert bool(history.iloc[3]["is_provisional"]) is False


def test_http_connection_failure_raises_cbs_request_error():
    with patch(
        "src.data.cbs_client.requests.get",
        side_effect=requests.exceptions.ConnectionError("network down"),
    ):
        with pytest.raises(CBSRequestError):
            fetch_index_history()


def test_http_error_status_raises_cbs_request_error():
    payload = _fake_cbs_payload(SAMPLE_OBSERVATIONS)
    with patch(
        "src.data.cbs_client.requests.get",
        return_value=_mock_response(payload, status_ok=False),
    ):
        with pytest.raises(CBSRequestError):
            fetch_index_history()


def test_malformed_json_response_raises_cbs_response_error():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("not json")
    with patch("src.data.cbs_client.requests.get", return_value=response):
        with pytest.raises(CBSResponseError):
            fetch_index_history()


def test_unsupported_response_structure_raises():
    with patch(
        "src.data.cbs_client.requests.get",
        return_value=_mock_response({"unexpected": "shape"}),
    ):
        with pytest.raises(CBSResponseError):
            fetch_index_history()


def test_missing_index_value_raises():
    bad_obs = _obs(2026, 5, 593.6)
    bad_obs["currBase"] = None
    payload = _fake_cbs_payload([bad_obs])
    with patch("src.data.cbs_client.requests.get", return_value=_mock_response(payload)):
        with pytest.raises(CBSResponseError):
            fetch_index_history()


def _sample_history() -> pd.DataFrame:
    payload = _fake_cbs_payload(SAMPLE_OBSERVATIONS)
    with patch("src.data.cbs_client.requests.get", return_value=_mock_response(payload)):
        return fetch_index_history()


def test_index_lookup_by_historical_date():
    history = _sample_history()
    value = get_index_for_date("2026-02-01", history=history)
    assert value == 602.0


def test_index_lookup_for_missing_date_raises():
    history = _sample_history()
    with pytest.raises(CBSResponseError):
        get_index_for_date("2020-01-01", history=history)


def test_latest_stable_index_skips_provisional_observations():
    history = _sample_history()
    value, period = get_latest_stable_index(history=history)

    # the 3 most recent observations (2026-05, 2026-04, 2026-03) are
    # provisional; the latest stable one is 2026-02 = 602.0
    assert value == 602.0
    assert period == pd.Timestamp(2026, 2, 1)


def test_local_cache_creation_and_loading(tmp_path):
    history = _sample_history()
    cache_path = tmp_path / "cbs_housing_price_index.csv"

    saved_path = save_index_history_cache(history, cache_path)
    assert saved_path == cache_path
    assert cache_path.exists()

    reloaded = load_index_history_cache(cache_path)
    assert list(reloaded.columns) == INDEX_HISTORY_COLUMNS
    assert len(reloaded) == len(history)
    assert reloaded.iloc[3]["index_value"] == 602.0
    assert bool(reloaded.iloc[0]["is_provisional"]) is True
    assert bool(reloaded.iloc[3]["is_provisional"]) is False


def test_loading_missing_cache_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_index_history_cache(missing_path)
