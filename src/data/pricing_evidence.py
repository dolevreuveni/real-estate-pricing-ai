"""Pricing evidence / audit-trail foundation.

For every future pricing run, the system must be able to record exactly
which market records (transactions, market listings, competitor projects)
were used to determine each apartment's recommended price, and where each
record came from. This module only provides the schema and reusable
save/load utilities for that audit trail -- it does not select comparables
or compute prices.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.settings import OUTPUT_DATA_DIR

PRICING_EVIDENCE_COLUMNS = [
    "run_id",
    "target_apartment_id",
    "evidence_type",
    "address",
    "record_date",
    "rooms",
    "area_sqm",
    "floor",
    "original_price",
    "adjusted_price",
    "price_per_sqm",
    "distance_from_project_km",
    "similarity_score",
    "pricing_weight",
    "source",
    "source_url",
    "data_retrieved_at",
]

EVIDENCE_TYPE_TRANSACTION = "transaction"
EVIDENCE_TYPE_MARKET_LISTING = "market_listing"
EVIDENCE_TYPE_COMPETITOR_PROJECT = "competitor_project"
VALID_EVIDENCE_TYPES = {
    EVIDENCE_TYPE_TRANSACTION,
    EVIDENCE_TYPE_MARKET_LISTING,
    EVIDENCE_TYPE_COMPETITOR_PROJECT,
}

CSV_OUTPUT_PATH = OUTPUT_DATA_DIR / "pricing_evidence.csv"
XLSX_OUTPUT_PATH = OUTPUT_DATA_DIR / "pricing_evidence.xlsx"


def generate_run_id(now: datetime | None = None) -> str:
    """Generate a pricing-run identifier from a UTC timestamp.

    Example: pricing_run_20260822_153000
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    return f"pricing_run_{moment.strftime('%Y%m%d_%H%M%S')}"


def validate_evidence(evidence: pd.DataFrame) -> None:
    """Raise ValueError if `evidence` does not match the pricing-evidence schema."""
    missing = [c for c in PRICING_EVIDENCE_COLUMNS if c not in evidence.columns]
    if missing:
        raise ValueError(
            f"pricing_evidence: missing required column(s) {missing}. "
            f"Expected columns: {PRICING_EVIDENCE_COLUMNS}."
        )

    unknown_types = set(evidence["evidence_type"].dropna().unique()) - VALID_EVIDENCE_TYPES
    if unknown_types:
        raise ValueError(
            f"pricing_evidence: unknown evidence_type value(s) {sorted(unknown_types)}. "
            f"Expected one of {sorted(VALID_EVIDENCE_TYPES)}."
        )


def save_pricing_evidence_csv(
    evidence: pd.DataFrame, path: str | Path = CSV_OUTPUT_PATH
) -> Path:
    """Validate and save the evidence records to a single CSV file."""
    validate_evidence(evidence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence[PRICING_EVIDENCE_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_pricing_evidence_excel(
    evidence: pd.DataFrame, path: str | Path = XLSX_OUTPUT_PATH
) -> Path:
    """Validate and save the evidence records to an Excel workbook.

    Sheets:
        all_evidence               -- every evidence record
        transactions_used          -- evidence_type == "transaction"
        market_listings_used       -- evidence_type == "market_listing"
        competitor_projects_used   -- evidence_type == "competitor_project"
    """
    validate_evidence(evidence)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = evidence[PRICING_EVIDENCE_COLUMNS]
    sheets = {
        "all_evidence": ordered,
        "transactions_used": ordered[ordered["evidence_type"] == EVIDENCE_TYPE_TRANSACTION],
        "market_listings_used": ordered[
            ordered["evidence_type"] == EVIDENCE_TYPE_MARKET_LISTING
        ],
        "competitor_projects_used": ordered[
            ordered["evidence_type"] == EVIDENCE_TYPE_COMPETITOR_PROJECT
        ],
    }

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return path


def save_pricing_evidence(
    evidence: pd.DataFrame,
    csv_path: str | Path = CSV_OUTPUT_PATH,
    xlsx_path: str | Path = XLSX_OUTPUT_PATH,
) -> tuple[Path, Path]:
    """Validate and save the evidence records to both CSV and Excel."""
    validate_evidence(evidence)
    return (
        save_pricing_evidence_csv(evidence, csv_path),
        save_pricing_evidence_excel(evidence, xlsx_path),
    )
