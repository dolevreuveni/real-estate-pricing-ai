"""Manual/optional check of the real CBS API connection.

Not part of the pytest suite (pytest only runs mocked tests -- see
tests/test_cbs_client.py). Run this directly to verify live connectivity
and refresh the local cache with real data:

    python scripts/test_cbs_connection.py
    python -m scripts.test_cbs_connection
"""
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import CBS_HOUSING_PRICE_INDEX_SERIES
from src.data.cbs_client import get_latest_stable_index, refresh_index_history_cache


def main() -> None:
    print(f"CBS Housing Price Index series: {CBS_HOUSING_PRICE_INDEX_SERIES}")

    history = refresh_index_history_cache()
    print(f"Fetched {len(history)} observations. Cache refreshed.")

    print("\nLatest observations:")
    print(history.head(5).to_string(index=False))

    value, period = get_latest_stable_index(history=history)
    print(f"\nLatest stable index: {value} (period: {period.strftime('%Y-%m')})")


if __name__ == "__main__":
    main()
