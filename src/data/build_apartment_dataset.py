"""Build the normalized apartment dataset from the raw source Excel file.

Run as a module from the project root:
    python -m src.data.build_apartment_dataset
"""
from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.apartment_reader import load_normalized_apartments

RAW_PATH = RAW_DATA_DIR / "Apartment_example.xlsx"
CSV_OUTPUT_PATH = PROCESSED_DATA_DIR / "apartments.csv"
XLSX_OUTPUT_PATH = PROCESSED_DATA_DIR / "apartments.xlsx"


def main() -> None:
    apartments = load_normalized_apartments(RAW_PATH)

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    apartments.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    apartments.to_excel(XLSX_OUTPUT_PATH, index=False)

    print(f"Normalized apartments: {len(apartments)}")
    print(f"Columns: {list(apartments.columns)}")


if __name__ == "__main__":
    main()
