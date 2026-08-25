"""Completed-transactions acquisition pipeline.

    configured project address
        -> GovMap (address resolution, nearby polygons, street deals)
        -> data/raw/govmap_transactions_raw.csv
        -> normalization
        -> data-quality evaluation
        -> CBS Housing Price Index enrichment
        -> data/external/transactions.csv

Run:
    python scripts/fetch_transactions.py
    python -m scripts.fetch_transactions
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import (
    EXTERNAL_DATA_DIR,
    GOVMAP_DEALS_RADIUS_M,
    PROJECT_ADDRESS,
    PROJECT_CITY,
    RAW_DATA_DIR,
    TRANSACTIONS_DEAL_TYPE,
    TRANSACTIONS_PAGE_SIZE,
    TRANSACTIONS_START_DATE,
)
from src.data.cbs_client import get_latest_stable_index, load_index_history_cache
from src.data.govmap_client import fetch_transactions_near_address
from src.data.historical_transaction_enrichment import evaluate_historical_training_eligibility
from src.data.transaction_normalizer import normalize_transactions
from src.data.transaction_quality import evaluate_transactions
from src.pricing.index_adjustment import enrich_transactions_with_cbs_index

import pandas as pd

RAW_OUTPUT_PATH = RAW_DATA_DIR / "govmap_transactions_raw.csv"
PROCESSED_OUTPUT_PATH = EXTERNAL_DATA_DIR / "transactions.csv"

RAW_COLUMNS = [
    "objectid",
    "settlementId",
    "settlementNameHeb",
    "streetCode",
    "streetNameHeb",
    "houseNum",
    "floorNo",
    "assetArea",
    "dealAmount",
    "dealId",
    "propertyTypeDescription",
    "dealNatureDescription",
    "assetRoomNum",
    "neighborhood",
    "dealDate",
    "gushNum",
    "parcelNum",
    "subParcelNum",
    "polygonId",
    "shape",
    "sourceorder",
    "source",
    "source_url",
    "data_retrieved_at",
]

FINAL_COLUMNS = [
    "deal_id",
    "address",
    "transaction_date",
    "rooms",
    "area_sqm",
    "floor",
    "original_price",
    "original_price_per_sqm",
    "price_index_at_transaction",
    "current_price_index",
    "index_adjustment_factor",
    "adjusted_price",
    "adjusted_price_per_sqm",
    # Not computed by this feature -- always null until a comparable-radius
    # distance calculation exists. See src/data/market_data_loader.py,
    # which requires this column as part of the canonical transactions
    # schema.
    "distance_from_project_km",
    "property_type",
    "deal_nature",
    "neighborhood",
    "gush",
    "parcel",
    "sub_parcel",
    # True only means "passed basic data-quality/residential checks, may
    # be considered later by the Comparable Engine" -- not "already
    # selected as a comparable" (see src/data/transaction_quality.py).
    "is_eligible_comparable",
    "exclusion_reason",
    # Sold-fraction / full-ownership-equivalent audit trail and Historical
    # Regression training eligibility -- see
    # src/data/historical_transaction_enrichment.py for the full
    # rationale (GovMap does not expose a sold-fraction field; these
    # columns exist for forward-compatibility and are currently null for
    # every row, with a POC price-per-sqm heuristic used as a fallback).
    "sold_fraction",
    "full_ownership_price",
    "full_ownership_price_per_sqm",
    "suspicious_partial_transaction",
    "used_for_historical_model",
    "historical_model_exclusion_reason",
    "source",
    "source_url",
    "data_retrieved_at",
]


def save_raw_transactions(raw_transactions: list, path: Path = RAW_OUTPUT_PATH) -> Path:
    """Save the untouched GovMap records as our audit trail. Nothing is
    filtered or removed here, however incomplete or suspicious."""
    raw_df = pd.DataFrame(raw_transactions)
    for column in RAW_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_df[RAW_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    search_text = f"{PROJECT_ADDRESS} {PROJECT_CITY}"
    retrieved_at = datetime.now(timezone.utc)

    resolved, raw_transactions = fetch_transactions_near_address(
        search_text=search_text,
        radius=GOVMAP_DEALS_RADIUS_M,
        deal_type=TRANSACTIONS_DEAL_TYPE,
        start_date=TRANSACTIONS_START_DATE,
        end_date=retrieved_at.strftime("%Y-%m"),
        page_size=TRANSACTIONS_PAGE_SIZE,
    )

    raw_path = save_raw_transactions(raw_transactions)

    normalized = normalize_transactions(raw_transactions)
    evaluated = evaluate_transactions(normalized)

    cbs_history = load_index_history_cache()
    cbs_enriched = enrich_transactions_with_cbs_index(evaluated, history=cbs_history)
    enriched = evaluate_historical_training_eligibility(cbs_enriched)

    for column in FINAL_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = None
    final = enriched[FINAL_COLUMNS].sort_values("transaction_date").reset_index(drop=True)

    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(PROCESSED_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    eligible = final[final["is_eligible_comparable"] == True]  # noqa: E712
    excluded = final[final["is_eligible_comparable"] == False]  # noqa: E712
    dates = final["transaction_date"].dropna()
    latest_stable_value, latest_stable_period = get_latest_stable_index(history=cbs_history)

    print(f"Resolved project address: {resolved.get('text')}")
    print(f"Transactions retrieved: {len(final)}")
    print(f"Eligible transactions: {len(eligible)}")
    print(f"Excluded transactions: {len(excluded)}")
    print(f"Earliest transaction: {dates.min() if not dates.empty else None}")
    print(f"Latest transaction: {dates.max() if not dates.empty else None}")
    print(
        f"Latest stable CBS index: {latest_stable_value} "
        f"(period: {latest_stable_period.strftime('%Y-%m')})"
    )
    print(f"Raw dataset: {raw_path}")
    print(f"Processed dataset: {PROCESSED_OUTPUT_PATH}")

    if not excluded.empty:
        print("\nExclusion reasons:")
        reason_counts: dict = {}
        for reasons_text in excluded["exclusion_reason"].fillna("(no reason recorded)"):
            for reason in reasons_text.split(", "):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")

    used_for_model = final[final["used_for_historical_model"] == True]  # noqa: E712
    not_used_for_model = final[final["used_for_historical_model"] == False]  # noqa: E712
    print("\nHistorical Regression training eligibility:")
    print(f"  Used for historical model: {len(used_for_model)}")
    print(f"  Not used for historical model: {len(not_used_for_model)}")
    if not not_used_for_model.empty:
        reason_counts = {}
        for reasons_text in not_used_for_model["historical_model_exclusion_reason"].fillna(
            "(no reason recorded)"
        ):
            for reason in reasons_text.split(", "):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
