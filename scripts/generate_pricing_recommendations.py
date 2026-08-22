"""Train the Current Market model, predict a current-market value for
every project apartment, combine it with the existing historical
GovMap/CBS regression price into a recommended_marketing_price, and then
apply the Feature #9 company-strategy layer on top to produce
final_strategy_price.

    data/external/current_market_500_updated.xlsx (SYNTHETIC POC data)
        -> Current Market model (LinearRegression + OneHotEncoder)
        -> current_market_price per apartment
    data/processed/apartment_base_prices.csv (Feature #7 output)
        -> historical_base_price per apartment
    combine_prices() with HISTORICAL_MARKET_WEIGHT / CURRENT_MARKET_WEIGHT
        -> recommended_marketing_price
    apply_strategy_adjustment() with COMPANY_POSITIONING_ADJUSTMENT_PCT /
    SALES_PHASE_ADJUSTMENT_PCT / INVENTORY_STRATEGY_ADJUSTMENT_PCT /
    data/external/apartment_strategy_adjustments.csv
        -> final_strategy_price
        -> data/processed/apartment_pricing_recommendations.{csv,xlsx}
        -> data/output/strategy_pricing_report.json

This script does not modify or retrain either regression model -- run
scripts/train_baseline_pricing_model.py first to (re)generate
data/processed/apartment_base_prices.csv.

Run:
    python scripts/generate_pricing_recommendations.py
    python -m scripts.generate_pricing_recommendations
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.train_baseline_pricing_model import APARTMENT_PRICES_CSV_PATH
from src.config.settings import (
    COMPANY_POSITIONING_ADJUSTMENT_PCT,
    CURRENT_MARKET_DATA_TYPE,
    CURRENT_MARKET_INPUT_PATH,
    CURRENT_MARKET_WEIGHT,
    HISTORICAL_MARKET_WEIGHT,
    INVENTORY_STRATEGY_ADJUSTMENT_PCT,
    OUTPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    SALES_PHASE_ADJUSTMENT_PCT,
)
from src.data.apartment_strategy_loader import load_apartment_strategy_adjustments
from src.data.build_apartment_dataset import CSV_OUTPUT_PATH as APARTMENTS_CSV_PATH
from src.data.current_market_loader import load_current_market_listings
from src.pricing.current_market_features import (
    apartments_to_market_feature_frame,
    listings_to_training_frame,
    select_training_rows,
)
from src.pricing.current_market_model import predict, train_and_evaluate
from src.pricing.pricing_recommendation import combine_prices, validate_weights
from src.pricing.pricing_utils import enforce_non_negative_predictions
from src.pricing.strategy_adjustment import apply_strategy_adjustment, get_apartment_strategy_row

MODEL_REPORT_PATH = OUTPUT_DATA_DIR / "current_market_model_report.json"
STRATEGY_REPORT_PATH = OUTPUT_DATA_DIR / "strategy_pricing_report.json"
RECOMMENDATIONS_CSV_PATH = PROCESSED_DATA_DIR / "apartment_pricing_recommendations.csv"
RECOMMENDATIONS_XLSX_PATH = PROCESSED_DATA_DIR / "apartment_pricing_recommendations.xlsx"

OUTPUT_COLUMNS = [
    "apartment_id",
    "rooms",
    "floor_min",
    "floor_max",
    "num_levels",
    "interior_area_sqm",
    "balcony_area_sqm",
    "balcony_direction",
    "directions",
    "parking_count",
    "storage_area_sqm",
    "garden_area_sqm",
    "roof_area_sqm",
    "is_top_floor",
    "property_type",
    "historical_base_price",
    "historical_base_price_per_sqm",
    "current_market_price",
    "current_market_price_per_sqm",
    "historical_weight",
    "current_market_weight",
    "historical_contribution",
    "current_market_contribution",
    "recommended_marketing_price",
    "recommended_marketing_price_per_sqm",
    "historical_model_version",
    "current_market_model_version",
    "current_market_data_type",
    "company_positioning_adjustment_pct",
    "sales_phase_adjustment_pct",
    "inventory_strategy_adjustment_pct",
    "manual_adjustment_pct",
    "manual_adjustment_amount",
    "strategy_note",
    "total_strategy_adjustment_pct",
    "final_strategy_price",
    "final_strategy_price_per_sqm",
    "pricing_status",
]


def main() -> None:
    validate_weights(HISTORICAL_MARKET_WEIGHT, CURRENT_MARKET_WEIGHT)

    print(f"Current Market input path: {CURRENT_MARKET_INPUT_PATH}")

    # --- Current Market model ---
    listings = load_current_market_listings()
    print(f"Current Market rows loaded: {len(listings)}")
    print(f"Current Market columns found: {list(listings.columns)}")

    trainable = select_training_rows(listings)
    X, y = listings_to_training_frame(trainable)
    fit = train_and_evaluate(X, y, input_file=str(CURRENT_MARKET_INPUT_PATH))
    model, report = fit["model"], fit["report"]
    report["pricing_weights"] = {
        "historical_market_weight": HISTORICAL_MARKET_WEIGHT,
        "current_market_weight": CURRENT_MARKET_WEIGHT,
    }

    MODEL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- Current Market predictions for the 39 apartments ---
    apartments = pd.read_csv(APARTMENTS_CSV_PATH)
    market_features = apartments_to_market_feature_frame(apartments)
    priceable_mask = market_features["is_priceable"]

    current_market_price = pd.Series(index=apartments.index, dtype="float64")
    if priceable_mask.any():
        raw = predict(model, market_features.loc[priceable_mask])
        current_market_price.update(enforce_non_negative_predictions(raw))

    # --- Historical predictions (Feature #7 output) ---
    historical = pd.read_csv(APARTMENT_PRICES_CSV_PATH)[
        ["apartment_id", "regression_base_price", "regression_base_price_per_sqm", "model_version"]
    ].rename(
        columns={
            "regression_base_price": "historical_base_price",
            "regression_base_price_per_sqm": "historical_base_price_per_sqm",
            "model_version": "historical_model_version",
        }
    )

    result = apartments.copy()
    result["current_market_price"] = current_market_price
    result["current_market_price_per_sqm"] = (
        result["current_market_price"] / result["interior_area_sqm"]
    )
    result = result.merge(historical, on="apartment_id", how="left")

    result["historical_weight"] = HISTORICAL_MARKET_WEIGHT
    result["current_market_weight"] = CURRENT_MARKET_WEIGHT
    result["current_market_model_version"] = report["model_version"]
    result["current_market_data_type"] = CURRENT_MARKET_DATA_TYPE

    recommended_prices = []
    contributions_hist = []
    contributions_market = []
    statuses = []
    for _, row in result.iterrows():
        hist_price = row["historical_base_price"]
        market_price = row["current_market_price"]
        hist_missing = pd.isna(hist_price)
        market_missing = pd.isna(market_price)

        if hist_missing and market_missing:
            statuses.append("missing_both_signals")
            recommended_prices.append(None)
            contributions_hist.append(None)
            contributions_market.append(None)
        elif hist_missing:
            statuses.append("missing_historical_signal")
            recommended_prices.append(None)
            contributions_hist.append(None)
            contributions_market.append(None)
        elif market_missing:
            statuses.append("missing_current_market_signal")
            recommended_prices.append(None)
            contributions_hist.append(None)
            contributions_market.append(None)
        else:
            try:
                price = combine_prices(hist_price, market_price)
                statuses.append("priced")
                recommended_prices.append(price)
                contributions_hist.append(hist_price * HISTORICAL_MARKET_WEIGHT)
                contributions_market.append(market_price * CURRENT_MARKET_WEIGHT)
            except ValueError:
                statuses.append("invalid_signal")
                recommended_prices.append(None)
                contributions_hist.append(None)
                contributions_market.append(None)

    result["recommended_marketing_price"] = recommended_prices
    result["historical_contribution"] = contributions_hist
    result["current_market_contribution"] = contributions_market
    result["recommended_marketing_price_per_sqm"] = (
        result["recommended_marketing_price"] / result["interior_area_sqm"]
    )
    result["pricing_status"] = statuses

    # --- Feature #9: company strategy pricing layer ---
    strategy_df = load_apartment_strategy_adjustments()

    manual_pcts = []
    manual_amounts = []
    strategy_notes = []
    final_strategy_prices = []
    final_statuses = []
    for _, row in result.iterrows():
        try:
            strategy_row = get_apartment_strategy_row(strategy_df, row["apartment_id"])
        except ValueError:
            manual_pcts.append(None)
            manual_amounts.append(None)
            strategy_notes.append(None)
            final_strategy_prices.append(None)
            final_statuses.append("missing_strategy_row")
            continue

        manual_pcts.append(strategy_row["manual_adjustment_pct"])
        manual_amounts.append(strategy_row["manual_adjustment_amount"])
        strategy_notes.append(strategy_row["strategy_note"])

        base_price = row["recommended_marketing_price"]
        if row["pricing_status"] != "priced" or pd.isna(base_price):
            final_strategy_prices.append(None)
            final_statuses.append(row["pricing_status"])
            continue

        try:
            final_strategy_prices.append(
                apply_strategy_adjustment(
                    base_price,
                    COMPANY_POSITIONING_ADJUSTMENT_PCT,
                    SALES_PHASE_ADJUSTMENT_PCT,
                    INVENTORY_STRATEGY_ADJUSTMENT_PCT,
                    strategy_row["manual_adjustment_pct"],
                    strategy_row["manual_adjustment_amount"],
                )
            )
            final_statuses.append("priced")
        except ValueError:
            final_strategy_prices.append(None)
            final_statuses.append("invalid_strategy_adjustment")

    result["company_positioning_adjustment_pct"] = COMPANY_POSITIONING_ADJUSTMENT_PCT
    result["sales_phase_adjustment_pct"] = SALES_PHASE_ADJUSTMENT_PCT
    result["inventory_strategy_adjustment_pct"] = INVENTORY_STRATEGY_ADJUSTMENT_PCT
    result["manual_adjustment_pct"] = manual_pcts
    result["manual_adjustment_amount"] = manual_amounts
    result["strategy_note"] = strategy_notes
    result["total_strategy_adjustment_pct"] = (
        COMPANY_POSITIONING_ADJUSTMENT_PCT
        + SALES_PHASE_ADJUSTMENT_PCT
        + INVENTORY_STRATEGY_ADJUSTMENT_PCT
        + result["manual_adjustment_pct"]
    )
    result["final_strategy_price"] = final_strategy_prices
    result["final_strategy_price_per_sqm"] = (
        result["final_strategy_price"] / result["interior_area_sqm"]
    )
    result["pricing_status"] = final_statuses

    final = result[OUTPUT_COLUMNS]

    RECOMMENDATIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(RECOMMENDATIONS_CSV_PATH, index=False, encoding="utf-8-sig")
    final.to_excel(RECOMMENDATIONS_XLSX_PATH, index=False)

    # --- report ---
    priced = final[final["pricing_status"] == "priced"]

    print("=== Synthetic Current Market model ===")
    print(f"Training rows: {report['training_rows']}  Test rows: {report['test_rows']}")
    print(f"Features: {report['features']}")
    print(f"MAE: {report['mae']:.2f}")
    print(f"RMSE: {report['rmse']:.2f}")
    print(f"R2: {report['r2']:.4f}")

    print("\n=== Final pricing ===")
    print(f"Apartments successfully priced: {len(priced)} / {len(final)}")
    if not priced.empty:
        print(f"Min recommended price: {priced['recommended_marketing_price'].min():.2f}")
        print(f"Max recommended price: {priced['recommended_marketing_price'].max():.2f}")
        print(f"Average recommended price: {priced['recommended_marketing_price'].mean():.2f}")
        print(f"Min final strategy price: {priced['final_strategy_price'].min():.2f}")
        print(f"Max final strategy price: {priced['final_strategy_price'].max():.2f}")
        print(f"Average final strategy price: {priced['final_strategy_price'].mean():.2f}")
    not_priced = final[final["pricing_status"] != "priced"]
    if not not_priced.empty:
        print("Apartments not priced:")
        for _, row in not_priced.iterrows():
            print(f"  apartment_id {row['apartment_id']}: {row['pricing_status']}")

    # --- Feature #9: strategy summary report ---
    apartments_with_manual_adjustments = int(
        (
            (final["manual_adjustment_pct"].fillna(0) != 0)
            | (final["manual_adjustment_amount"].fillna(0) != 0)
        ).sum()
    )
    strategy_report = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "company_positioning_adjustment_pct": COMPANY_POSITIONING_ADJUSTMENT_PCT,
        "sales_phase_adjustment_pct": SALES_PHASE_ADJUSTMENT_PCT,
        "inventory_strategy_adjustment_pct": INVENTORY_STRATEGY_ADJUSTMENT_PCT,
        "apartment_count": int(len(final)),
        "apartments_with_manual_adjustments": apartments_with_manual_adjustments,
        "average_market_recommended_price": (
            float(priced["recommended_marketing_price"].mean()) if not priced.empty else None
        ),
        "average_final_strategy_price": (
            float(priced["final_strategy_price"].mean()) if not priced.empty else None
        ),
        "total_market_recommended_value": (
            float(priced["recommended_marketing_price"].sum()) if not priced.empty else None
        ),
        "total_final_strategy_value": (
            float(priced["final_strategy_price"].sum()) if not priced.empty else None
        ),
    }
    STRATEGY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STRATEGY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(strategy_report, f, ensure_ascii=False, indent=2)

    print(f"\nCurrent Market model report: {MODEL_REPORT_PATH}")
    print(f"Strategy pricing report: {STRATEGY_REPORT_PATH}")
    print(f"Recommendations CSV: {RECOMMENDATIONS_CSV_PATH}")
    print(f"Recommendations XLSX: {RECOMMENDATIONS_XLSX_PATH}")


if __name__ == "__main__":
    main()
