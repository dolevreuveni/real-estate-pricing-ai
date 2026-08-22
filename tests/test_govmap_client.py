"""Tests for govmap_client. All HTTP calls are mocked; no live network access."""
from unittest.mock import Mock, patch

import pytest
import requests

from src.data.govmap_client import (
    GovMapRequestError,
    GovMapResponseError,
    fetch_transactions_near_address,
    get_deals_by_point,
    get_street_deals,
    parse_point_shape,
    resolve_address,
)


def _autocomplete_payload():
    return {
        "resultsCount": 2,
        "results": [
            {
                "id": "address|ADDR|253837",
                "text": "Helsinki 24 Tel Aviv-Yafo",
                "type": "address",
                "score": 4014.8162,
                "shape": "POINT(3873103.0778553626 3774571.0556697305)",
                "data": {},
            },
            {
                "id": "address|ADDR|253208",
                "text": "Helsinki 1 Tel Aviv-Yafo",
                "type": "address",
                "score": 242.76727,
                "shape": "POINT(3872990.2169683594 3774270.370150718)",
                "data": {},
            },
        ],
        "aggregations": [{"key": "address", "count": 2}],
    }


def _deals_by_point_payload():
    return [
        {
            "dealscount": "6",
            "settlementNameHeb": "תל אביב-יפו",
            "streetNameHeb": "הלסינקי",
            "houseNum": 16,
            "polygon_id": "6108-86",
            "objectid": 62546,
        },
        {
            "dealscount": "1",
            "settlementNameHeb": "תל אביב -יפו",
            "streetNameHeb": None,
            "houseNum": None,
            "polygon_id": "6108-188",
            "objectid": 62380,
        },
    ]


def _street_deal_record(deal_id, house_num=13):
    return {
        "objectid": 1627260,
        "settlementId": 5000,
        "settlementNameHeb": "תל אביב-יפו",
        "settlementNameEng": "Tel Aviv-Yafo",
        "streetCode": 50000932,
        "streetNameHeb": "הלסינקי",
        "streetNameEng": "Helsinki",
        "houseNum": house_num,
        "floorNo": "שניה",
        "assetArea": 85,
        "dealAmount": 5254000,
        "dealId": deal_id,
        "propertyTypeDescription": "דירה",
        "dealNatureDescription": "דירה בבית קומות",
        "assetRoomNum": 3,
        "neighborhood": "הצפון החדש סביבת כיכר המדינה",
        "dealDate": "2026-06-25T00:00:00.000Z",
        "gushNum": 6108,
        "parcelNum": 230,
        "subParcelNum": 19,
        "polygonId": "52315574",
        "shape": "MULTIPOLYGON(((0 0,0 0,0 0)))",
        "sourceorder": 2,
    }


def _mock_post_response(payload, status_ok=True):
    response = Mock()
    response.json.return_value = payload
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 error")
    return response


def _mock_get_response(payload, url, status_ok=True):
    response = Mock()
    response.json.return_value = payload
    response.url = url
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 error")
    return response


def test_successful_address_resolution():
    with patch(
        "src.data.govmap_client.requests.post",
        return_value=_mock_post_response(_autocomplete_payload()),
    ):
        result = resolve_address("Helsinki 24 Tel Aviv")

    assert result["text"] == "Helsinki 24 Tel Aviv-Yafo"
    x, y = parse_point_shape(result["shape"])
    assert x == pytest.approx(3873103.0778553626)
    assert y == pytest.approx(3774571.0556697305)


def test_address_resolution_no_match_raises():
    empty_payload = {"resultsCount": 0, "results": [], "aggregations": []}
    with patch(
        "src.data.govmap_client.requests.post", return_value=_mock_post_response(empty_payload)
    ):
        with pytest.raises(GovMapResponseError):
            resolve_address("nonexistent address")


def test_transaction_retrieval_by_point():
    with patch(
        "src.data.govmap_client.requests.get",
        return_value=_mock_get_response(
            _deals_by_point_payload(), "https://www.govmap.gov.il/api/real-estate/deals/1,2/200"
        ),
    ):
        polygons = get_deals_by_point(1.0, 2.0, 200)

    assert len(polygons) == 2
    assert polygons[0]["polygon_id"] == "6108-86"


def test_street_deals_pagination_fetches_all_pages():
    page1 = {
        "totalCount": "3",
        "data": [_street_deal_record(1), _street_deal_record(2)],
        "limit": "2",
        "offset": 0,
    }
    page2 = {
        "totalCount": "3",
        "data": [_street_deal_record(3)],
        "limit": "2",
        "offset": 2,
    }
    responses = [
        _mock_get_response(page1, "https://www.govmap.gov.il/api/real-estate/street-deals/x?offset=0"),
        _mock_get_response(page2, "https://www.govmap.gov.il/api/real-estate/street-deals/x?offset=2"),
    ]
    with patch("src.data.govmap_client.requests.get", side_effect=responses) as mock_get:
        records = get_street_deals("6108-86", limit=2)

    assert mock_get.call_count == 2
    assert [r["dealId"] for r in records] == [1, 2, 3]
    # every record is stamped with source metadata
    assert all(r["source"] == "GovMap" for r in records)
    assert all(r["source_url"] for r in records)
    assert all(r["data_retrieved_at"] for r in records)


def test_street_deals_single_page_stops_without_extra_request():
    page = {"totalCount": "1", "data": [_street_deal_record(1)], "limit": "100", "offset": 0}
    with patch(
        "src.data.govmap_client.requests.get",
        return_value=_mock_get_response(page, "https://www.govmap.gov.il/api/real-estate/street-deals/x"),
    ) as mock_get:
        records = get_street_deals("6108-86", limit=100)

    assert mock_get.call_count == 1
    assert len(records) == 1


def test_http_error_raises_govmap_request_error():
    with patch(
        "src.data.govmap_client.requests.get",
        side_effect=requests.exceptions.ConnectionError("network down"),
    ):
        with pytest.raises(GovMapRequestError):
            get_deals_by_point(1.0, 2.0, 200)


def test_http_error_status_raises_govmap_request_error():
    with patch(
        "src.data.govmap_client.requests.post",
        return_value=_mock_post_response(_autocomplete_payload(), status_ok=False),
    ):
        with pytest.raises(GovMapRequestError):
            resolve_address("Helsinki 24 Tel Aviv")


def test_malformed_response_raises_govmap_response_error():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("not json")
    with patch("src.data.govmap_client.requests.post", return_value=response):
        with pytest.raises(GovMapResponseError):
            resolve_address("Helsinki 24 Tel Aviv")


def test_unsupported_deals_by_point_structure_raises():
    with patch(
        "src.data.govmap_client.requests.get",
        return_value=_mock_get_response({"unexpected": "shape"}, "https://www.govmap.gov.il/x"),
    ):
        with pytest.raises(GovMapResponseError):
            get_deals_by_point(1.0, 2.0, 200)


def test_unparseable_shape_raises():
    with pytest.raises(GovMapResponseError):
        parse_point_shape("not a point")


def test_fetch_transactions_near_address_deduplicates_across_polygons():
    autocomplete_response = _mock_post_response(_autocomplete_payload())

    street_deals_page = {
        "totalCount": "1",
        "data": [_street_deal_record(42)],
        "limit": "100",
        "offset": 0,
    }
    get_responses = [
        _mock_get_response(_deals_by_point_payload(), "https://www.govmap.gov.il/deals/1,2/200"),
        # both polygons happen to return the same underlying deal (42),
        # exactly like the real street-deals endpoint does for polygons
        # on the same street
        _mock_get_response(street_deals_page, "https://www.govmap.gov.il/street-deals/6108-86"),
        _mock_get_response(street_deals_page, "https://www.govmap.gov.il/street-deals/6108-188"),
    ]

    with patch("src.data.govmap_client.requests.post", return_value=autocomplete_response), patch(
        "src.data.govmap_client.requests.get", side_effect=get_responses
    ):
        resolved, transactions = fetch_transactions_near_address(
            "Helsinki 24 Tel Aviv", radius=200
        )

    assert resolved["text"] == "Helsinki 24 Tel Aviv-Yafo"
    assert len(transactions) == 1
    assert transactions[0]["dealId"] == 42
