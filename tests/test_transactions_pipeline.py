"""End-to-end test of the normalize -> evaluate -> CBS-enrich chain used by
scripts/fetch_transactions.py, using fake raw GovMap records and a fake CBS
history so it never touches the network or the local CBS cache.
"""
import pandas as pd
import pytest

from src.data.cbs_client import INDEX_HISTORY_COLUMNS
from src.data.transaction_normalizer import normalize_transactions
from src.data.transaction_quality import evaluate_transactions
from src.pricing.index_adjustment import enrich_transactions_with_cbs_index


def _fake_cbs_history() -> pd.DataFrame:
    rows = [
        {
            "period": pd.Timestamp(2026, 2, 1),
            "year": 2026,
            "month": 2,
            "index_value": 602.0,
            "is_provisional": False,
            "series_id": 40010,
            "series_name": "מחירי דירות",
            "source": "CBS (Israeli Central Bureau of Statistics)",
            "source_url": "https://api.cbs.gov.il/index/data/price",
            "data_retrieved_at": pd.Timestamp.now(tz="UTC"),
        },
        {
            "period": pd.Timestamp(2026, 6, 1),
            "year": 2026,
            "month": 6,
            "index_value": 593.6,
            "is_provisional": True,
            "series_id": 40010,
            "series_name": "מחירי דירות",
            "source": "CBS (Israeli Central Bureau of Statistics)",
            "source_url": "https://api.cbs.gov.il/index/data/price",
            "data_retrieved_at": pd.Timestamp.now(tz="UTC"),
        },
    ]
    return pd.DataFrame(rows, columns=INDEX_HISTORY_COLUMNS)


def _raw_record(**overrides):
    record = {
        "objectid": 1,
        "settlementId": 5000,
        "settlementNameHeb": "תל אביב-יפו",
        "streetCode": 50000932,
        "streetNameHeb": "הלסינקי",
        "houseNum": 13,
        "floorNo": "שניה",
        "assetArea": 85,
        "dealAmount": 5254000,
        "dealId": 1,
        "propertyTypeDescription": "דירה",
        "dealNatureDescription": "דירה בבית קומות",
        "assetRoomNum": 3,
        "neighborhood": "הצפון החדש סביבת כיכר המדינה",
        "dealDate": "2026-02-15T00:00:00.000Z",
        "gushNum": 6108,
        "parcelNum": 230,
        "subParcelNum": 19,
        "polygonId": "52315574",
        "shape": "MULTIPOLYGON(((0 0)))",
        "sourceorder": 2,
        "source": "GovMap",
        "source_url": "https://www.govmap.gov.il/api/real-estate/street-deals/6108-86",
        "data_retrieved_at": "2026-08-22T12:00:00+00:00",
    }
    record.update(overrides)
    return record


def test_pipeline_produces_eligible_and_excluded_rows_with_cbs_enrichment():
    raw_records = [
        _raw_record(dealId=1),  # normal, eligible
        _raw_record(dealId=2, assetRoomNum=None),  # missing rooms -> excluded
    ]

    normalized = normalize_transactions(raw_records)
    evaluated = evaluate_transactions(normalized)
    enriched = enrich_transactions_with_cbs_index(evaluated, history=_fake_cbs_history())

    assert len(enriched) == 2
    eligible_row = enriched[enriched["deal_id"] == 1].iloc[0]
    excluded_row = enriched[enriched["deal_id"] == 2].iloc[0]

    assert bool(eligible_row["is_eligible_comparable"]) is True
    assert pd.isna(eligible_row["exclusion_reason"])
    assert bool(excluded_row["is_eligible_comparable"]) is False
    assert excluded_row["exclusion_reason"] == "missing_rooms"

    # both rows get CBS enrichment regardless of eligibility -- the
    # excluded row's price/date are still technically valid
    assert eligible_row["price_index_at_transaction"] == pytest.approx(602.0)
    assert eligible_row["adjusted_price"] == pytest.approx(5254000)  # transaction IS the stable month
    assert excluded_row["price_index_at_transaction"] == pytest.approx(602.0)
    assert bool(excluded_row["is_eligible_comparable"]) is False
