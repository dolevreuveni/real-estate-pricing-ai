"""End-to-end synthetic test of the baseline regression pricing pipeline
(select training data -> train -> map apartments -> predict), mirroring
scripts/train_baseline_pricing_model.py without touching real files or
the network.
"""
import numpy as np
import pandas as pd

from src.pricing.regression_features import (
    apartments_to_feature_frame,
    select_training_transactions,
    transactions_to_training_frame,
)
from src.pricing.regression_model import (
    enforce_non_negative_predictions,
    predict,
    train_and_evaluate,
)


def _synthetic_transactions(n=80, seed=1):
    rng = np.random.default_rng(seed)
    area = rng.uniform(40, 150, n)
    rooms = rng.integers(2, 6, n).astype(float)
    floor = rng.integers(0, 10, n).astype(float)
    price = 18_000 * area + 70_000 * rooms + 12_000 * floor + 900_000
    return pd.DataFrame(
        {
            "deal_id": range(n),
            "area_sqm": area,
            "rooms": rooms,
            "floor": floor,
            "adjusted_price": price,
            "is_eligible_comparable": True,
            # this pipeline test is about training/prediction plumbing, not
            # the residential whitelist itself (see
            # test_historical_transaction_enrichment.py for that) -- mark
            # every synthetic row trainable directly.
            "used_for_historical_model": True,
        }
    )


def _synthetic_apartments():
    return pd.DataFrame(
        [
            {"apartment_id": 1, "interior_area_sqm": 70.0, "rooms": 3, "floor_min": 1},
            {"apartment_id": 2, "interior_area_sqm": 110.0, "rooms": 5, "floor_min": 3},
            {"apartment_id": 3, "interior_area_sqm": 255.2, "rooms": 6, "floor_min": 9},
        ]
    )


def test_full_pipeline_prices_all_valid_apartments_with_expected_schema():
    transactions = _synthetic_transactions()
    _, trainable = select_training_transactions(transactions)
    X, y = transactions_to_training_frame(trainable)
    result = train_and_evaluate(X, y)

    apartments = _synthetic_apartments()
    features = apartments_to_feature_frame(apartments)
    assert features["is_priceable"].all()

    raw_predictions = predict(result["model"], features)
    predictions = enforce_non_negative_predictions(raw_predictions)

    final = apartments.copy()
    final["regression_base_price"] = predictions
    final["regression_base_price_per_sqm"] = (
        final["regression_base_price"] / final["interior_area_sqm"]
    )
    final["model_version"] = result["report"]["model_version"]

    expected_columns = {
        "apartment_id",
        "rooms",
        "floor_min",
        "interior_area_sqm",
        "regression_base_price",
        "regression_base_price_per_sqm",
        "model_version",
    }
    assert expected_columns.issubset(set(final.columns))
    assert final["regression_base_price"].notna().all()
    assert (final["regression_base_price"] > 0).all()


def test_apartment_missing_required_feature_is_not_priced_not_invented():
    transactions = _synthetic_transactions()
    _, trainable = select_training_transactions(transactions)
    X, y = transactions_to_training_frame(trainable)
    result = train_and_evaluate(X, y)

    apartments = pd.DataFrame(
        [
            {"apartment_id": 1, "interior_area_sqm": 70.0, "rooms": 3, "floor_min": 1},
            # missing floor_min entirely -- must not be priced or guessed
            {"apartment_id": 2, "interior_area_sqm": 90.0, "rooms": 4, "floor_min": None},
        ]
    )
    features = apartments_to_feature_frame(apartments)
    priceable_mask = features["is_priceable"]
    assert list(priceable_mask) == [True, False]

    predictions = pd.Series(index=apartments.index, dtype="float64")
    predictions.update(
        enforce_non_negative_predictions(predict(result["model"], features.loc[priceable_mask]))
    )

    assert predictions.loc[0] > 0
    assert pd.isna(predictions.loc[1])
