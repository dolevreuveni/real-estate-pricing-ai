"""Train the baseline transaction-price regression model and generate a
base marketing price for every project apartment.

    data/external/transactions.csv (GovMap + CBS adjusted)
        -> select eligible, trainable transactions
        -> fit LinearRegression(area_sqm, rooms, floor -> adjusted_price)
        -> evaluate on a held-out test split
        -> save model report (data/output/regression_model_report.json)
        -> map data/processed/apartments.csv onto the same features
        -> predict regression_base_price for each apartment
        -> save data/processed/apartment_base_prices.{csv,xlsx}

`regression_base_price` is the transaction-based base market value from
this historical GovMap + CBS regression -- it is NOT yet the final
recommended marketing price. No balcony/direction/new-project premiums are
applied here; those are future adjustment layers.

Run:
    python scripts/train_baseline_pricing_model.py
    python -m scripts.train_baseline_pricing_model
"""
import json
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config.settings import OUTPUT_DATA_DIR, PROCESSED_DATA_DIR
from src.data.build_apartment_dataset import CSV_OUTPUT_PATH as APARTMENTS_CSV_PATH
from src.data.market_data_loader import TRANSACTIONS_PATH
from src.pricing.regression_features import (
    apartments_to_feature_frame,
    load_transactions_csv,
    select_training_transactions,
    transactions_to_training_frame,
)
from src.pricing.regression_model import (
    enforce_non_negative_predictions,
    predict,
    train_and_evaluate,
)

MODEL_REPORT_PATH = OUTPUT_DATA_DIR / "regression_model_report.json"
APARTMENT_PRICES_CSV_PATH = PROCESSED_DATA_DIR / "apartment_base_prices.csv"
APARTMENT_PRICES_XLSX_PATH = PROCESSED_DATA_DIR / "apartment_base_prices.xlsx"

OUTPUT_COLUMNS = [
    "apartment_id",
    "rooms",
    "floor_min",
    "floor_max",
    "num_levels",
    "interior_area_sqm",
    "balcony_area_sqm",
    "directions",
    "property_type",
    "regression_base_price",
    "regression_base_price_per_sqm",
    "model_version",
]


def main() -> None:
    transactions = load_transactions_csv(TRANSACTIONS_PATH)
    eligible, trainable = select_training_transactions(transactions)
    # `trainable` now requires the strict residential whitelist and
    # sold-fraction/full-ownership validation, not just field completeness
    # (see src/data/historical_transaction_enrichment.py) -- this count
    # therefore spans all historical_model_exclusion_reason values, not
    # only missing fields.
    excluded_from_training = len(eligible) - len(trainable)

    X, y = transactions_to_training_frame(trainable)
    fit = train_and_evaluate(X, y)
    model, report = fit["model"], fit["report"]

    MODEL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    apartments = pd.read_csv(APARTMENTS_CSV_PATH)
    features = apartments_to_feature_frame(apartments)
    priceable_mask = features["is_priceable"]

    predictions = pd.Series(index=apartments.index, dtype="float64")
    if priceable_mask.any():
        raw = predict(model, features.loc[priceable_mask])
        predictions.update(enforce_non_negative_predictions(raw))

    result_df = apartments.copy()
    result_df["regression_base_price"] = predictions
    result_df["regression_base_price_per_sqm"] = (
        result_df["regression_base_price"] / result_df["interior_area_sqm"]
    )
    result_df["model_version"] = report["model_version"]

    final = result_df[OUTPUT_COLUMNS]

    APARTMENT_PRICES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(APARTMENT_PRICES_CSV_PATH, index=False, encoding="utf-8-sig")
    final.to_excel(APARTMENT_PRICES_XLSX_PATH, index=False)

    priced = final[final["regression_base_price"].notna()]
    unpriced = final[final["regression_base_price"].isna()]

    print("=== Training ===")
    print(f"Eligible transactions available: {len(eligible)}")
    print(f"Transactions used for training: {len(trainable)}")
    print(f"Excluded from training (residential whitelist / sold-fraction / missing fields): {excluded_from_training}")
    print(f"Features: {report['training_features']}")

    print("\n=== Evaluation ===")
    print(f"Train rows: {report['train_row_count']}  Test rows: {report['test_row_count']}")
    print(f"MAE: {report['mae']:.2f}")
    print(f"RMSE: {report['rmse']:.2f}")
    print(f"R2: {report['r2']:.4f}")
    print(f"Intercept: {report['intercept']:.2f}")
    for name, coef in report["coefficients"].items():
        print(f"Coefficient ({name}): {coef:.2f}")

    print("\n=== Project pricing ===")
    print(f"Apartments successfully priced: {len(priced)} / {len(final)}")
    if not unpriced.empty:
        print(f"Apartments NOT priced (missing required feature or invalid prediction): "
              f"{list(unpriced['apartment_id'])}")
    if not priced.empty:
        print(f"Min predicted price: {priced['regression_base_price'].min():.2f}")
        print(f"Max predicted price: {priced['regression_base_price'].max():.2f}")
        print(f"Average predicted price: {priced['regression_base_price'].mean():.2f}")

    print(f"\nModel report: {MODEL_REPORT_PATH}")
    print(f"Apartment prices CSV: {APARTMENT_PRICES_CSV_PATH}")
    print(f"Apartment prices XLSX: {APARTMENT_PRICES_XLSX_PATH}")


if __name__ == "__main__":
    main()
