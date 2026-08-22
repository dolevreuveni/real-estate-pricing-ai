"""Reading and normalization of the raw apartment-mix Excel file.

The source spreadsheet lists one row per apartment *level*: a regular or
garden apartment has exactly one row, while a duplex/triplex has one row
per floor it occupies (and sometimes, but not always, an extra "total"
row summarizing it). This module collapses that into one row per
apartment, using the apartment number as the identifier.

Feature #7.5 adds six OPTIONAL enrichment columns (parking_count,
storage_area_sqm, balcony_direction, garden_area_sqm, roof_area_sqm,
is_top_floor) on top of the original required schema. They are optional
for backward compatibility: a source file without them still normalizes
successfully, with these fields left null rather than invented -- only
the seven original fields (floor, apartment number, rooms, area, balcony
area, directions, notes) remain required.

Apartment-level vs. per-floor aggregation for the new fields:
parking_count, storage_area_sqm, garden_area_sqm, and roof_area_sqm are
whole-apartment attributes (e.g. one shared roof, one storage unit) that
the source repeats identically on every level-row of a multi-level
apartment -- they are aggregated with max(), never sum(), so a duplex
with parking_count=1 on both of its rows still normalizes to
parking_count=1, not 2. is_top_floor is aggregated with a boolean OR
(True if ANY level reaches the building's top floor). balcony_direction
is aggregated the same union-of-distinct-values way as the existing
`directions` field, then translated from the source's Hebrew compass
terms into English compass words so it lines up with the Current Market
dataset's vocabulary (see HEBREW_TO_ENGLISH_COMPASS below) -- `directions`
itself is left untouched in Hebrew, since general apartment air-direction
and balcony direction are different concepts and existing behavior for
`directions` must not change.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

# Actual column headers in the source file are in Hebrew and are not
# assumed to be in any fixed order or exact wording. Each internal field
# name is mapped to the header strings that may represent it.
HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "floor": ("מס' קומה", "מספר קומה", "קומה"),
    "apartment_number": ("מספר דירה", "מס' דירה", "דירה"),
    "rooms": ("מס' חדרים", "מספר חדרים", "חדרים"),
    "interior_area_sqm": ('שטח דירה (מ"ר)', "שטח דירה", 'שטח (מ"ר)'),
    "balcony_area_sqm": ("שטח מרפסת", 'שטח מרפסת (מ"ר)'),
    "directions": ("כיווני אוויר", "כיוונים", "כיווני אויר"),
    "notes": ("הערות",),
}

# Optional enrichment columns (Feature #7.5). Missing from the source
# file -> the corresponding output field is left null, never invented.
OPTIONAL_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "parking_count": ("מספר חניות", "מס' חניות", "חניות"),
    "storage_area_sqm": ("שטח מחסן", 'שטח מחסן (מ"ר)'),
    "balcony_direction": ("כיוון מרפסת",),
    "garden_area_sqm": ("שטח גינה", 'שטח גינה (מ"ר)'),
    "roof_area_sqm": ("שטח גג מוצמד", "שטח גג", 'שטח גג מוצמד (מ"ר)'),
    "is_top_floor": ("קומה אחרונה",),
}

# Apartment-level (not per-floor) optional numeric fields -- aggregated
# with max() across an apartment's source rows, never summed.
OPTIONAL_NUMERIC_MAX_FIELDS = [
    "parking_count",
    "storage_area_sqm",
    "garden_area_sqm",
    "roof_area_sqm",
]

# Final normalized column order.
NORMALIZED_APARTMENT_COLUMNS = [
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
    "notes",
]

# Marker values that appear as *cell contents* (not headers).
GROUND_FLOOR_LABEL = "קרקע"
TOTAL_ROW_LABEL = 'סה"כ'
GARDEN_LABEL = "דירת גן"
DUPLEX_LABEL = "דופלקס"
TRIPLEX_LABEL = "טריפלקס"

# balcony_direction translation: source Hebrew compass term -> English,
# to match the Current Market dataset's vocabulary (see
# src/pricing/current_market_features.py). Applied only to
# balcony_direction, never to the general `directions` field.
HEBREW_TO_ENGLISH_COMPASS = {
    "מזרח": "East",
    "מערב": "West",
    "צפון": "North",
    "דרום": "South",
    "צפון-מזרח": "North-East",
    "צפון-מערב": "North-West",
    "דרום-מזרח": "South-East",
    "דרום-מערב": "South-West",
}


def _normalize_header(value: object) -> str:
    return str(value).strip()


def _build_column_map(columns: list) -> dict[str, object]:
    """Map internal field names to the actual column found in the sheet.

    Raises if a REQUIRED field's column can't be found. Use
    `_build_optional_column_map` for the optional enrichment fields.
    """
    normalized = {_normalize_header(c): c for c in columns}
    column_map: dict[str, object] = {}
    for field, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                column_map[field] = normalized[alias]
                break
        else:
            raise ValueError(
                f"Could not find a column for '{field}'. "
                f"Looked for any of {aliases} among columns {list(columns)}."
            )
    return column_map


def _build_optional_column_map(columns: list) -> dict[str, object]:
    """Map optional enrichment field names to their column, where present.

    Never raises: a source file missing some or all of these columns is
    still valid -- the corresponding fields are simply absent.
    """
    normalized = {_normalize_header(c): c for c in columns}
    column_map: dict[str, object] = {}
    for field, aliases in OPTIONAL_HEADER_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                column_map[field] = normalized[alias]
                break
    return column_map


def read_raw_apartment_sheet(path: str | Path) -> pd.DataFrame:
    """Read the source Excel file as-is. Never writes back to `path`."""
    return pd.read_excel(path, sheet_name=0)


def _to_floor_number(value: object) -> Optional[int]:
    if isinstance(value, str) and value.strip() == GROUND_FLOOR_LABEL:
        return 0
    if pd.isna(value):
        return None
    return int(value)


def _is_total_row(floor_value: object) -> bool:
    return isinstance(floor_value, str) and floor_value.strip() == TOTAL_ROW_LABEL


def _clean_text(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _derive_property_type(notes: list, num_levels: int) -> str:
    joined = " ".join(notes)
    if TRIPLEX_LABEL in joined:
        return "triplex"
    if DUPLEX_LABEL in joined:
        return "duplex"
    if GARDEN_LABEL in joined:
        return "garden"
    # Fall back to level count when notes don't state the type explicitly,
    # since not every multi-level apartment is annotated in the notes.
    if num_levels >= 3:
        return "triplex"
    if num_levels == 2:
        return "duplex"
    return "regular"


def _aggregate_optional_numeric_max(series: pd.Series) -> Optional[float]:
    """Apartment-level optional numeric field: max() across the
    apartment's source rows (never sum -- see module docstring). Returns
    None (not 0) when the source column has no value for this apartment."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def _aggregate_is_top_floor(series: pd.Series) -> Optional[bool]:
    """True if ANY of the apartment's levels is flagged as the building's
    top floor. None (not False) when the source has no value at all."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return bool(numeric.max() > 0)


def _aggregate_balcony_direction(series: pd.Series) -> Optional[str]:
    """Union of distinct balcony_direction values across an apartment's
    levels (same pattern as `directions`/`notes`), translated to English
    compass terms. An untranslatable raw value is kept as-is rather than
    dropped."""
    values = sorted(
        {
            HEBREW_TO_ENGLISH_COMPASS.get(v, v)
            for v in (_clean_text(x) for x in series)
            if v
        }
    )
    return ", ".join(values) if values else None


def normalize_apartments(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-level source rows into one row per apartment."""
    columns = _build_column_map(list(raw.columns))
    optional_columns = _build_optional_column_map(list(raw.columns))
    all_columns = {**columns, **optional_columns}

    df = raw.rename(columns={source: field for field, source in all_columns.items()})
    df = df[list(all_columns.keys())].copy()

    # "Total" rows repeat information already present in the per-level
    # rows and are not used for aggregation: some multi-level apartments
    # have no total row at all, so correctness cannot depend on them.
    is_total = df["floor"].apply(_is_total_row)
    summary_rows = df[is_total].copy()
    level_rows = df[~is_total].copy()

    level_rows["floor"] = level_rows["floor"].apply(_to_floor_number)
    level_rows["apartment_number"] = level_rows["apartment_number"].astype(int)

    # Exact duplicate level rows (e.g. accidental re-entry) must not be
    # double-counted when aggregating areas.
    level_rows = level_rows.drop_duplicates()

    records = []
    for apartment_number, group in level_rows.groupby("apartment_number", sort=True):
        floors = group["floor"].dropna().tolist()
        notes = sorted({n for n in (_clean_text(v) for v in group["notes"]) if n})
        directions = sorted({d for d in (_clean_text(v) for v in group["directions"]) if d})
        rooms_values = sorted(set(group["rooms"].dropna().tolist()))
        if len(rooms_values) > 1:
            warnings.warn(
                f"Apartment {apartment_number}: inconsistent room counts "
                f"across levels: {rooms_values}. Using the maximum."
            )

        num_levels = len(group)
        record = {
            "apartment_id": int(apartment_number),
            "rooms": rooms_values[-1] if rooms_values else None,
            "floor_min": min(floors) if floors else None,
            "floor_max": max(floors) if floors else None,
            "num_levels": num_levels,
            "interior_area_sqm": round(float(group["interior_area_sqm"].sum()), 2),
            "balcony_area_sqm": round(float(group["balcony_area_sqm"].fillna(0).sum()), 2),
            "directions": ", ".join(directions) if directions else None,
            "notes": ", ".join(notes) if notes else None,
        }

        for field in OPTIONAL_NUMERIC_MAX_FIELDS:
            value = _aggregate_optional_numeric_max(group[field]) if field in group.columns else None
            # parking_count is a count, not an area -- keep it int-like
            # for readability once it's known to be present.
            record[field] = int(value) if field == "parking_count" and value is not None else value
        record["is_top_floor"] = (
            _aggregate_is_top_floor(group["is_top_floor"])
            if "is_top_floor" in group.columns
            else None
        )
        record["balcony_direction"] = (
            _aggregate_balcony_direction(group["balcony_direction"])
            if "balcony_direction" in group.columns
            else None
        )

        record["property_type"] = _derive_property_type(notes, num_levels)
        records.append(record)

    result = pd.DataFrame.from_records(records)
    result = result.sort_values("apartment_id").reset_index(drop=True)
    result = result.reindex(columns=NORMALIZED_APARTMENT_COLUMNS)
    _validate_against_summary_rows(result, summary_rows)
    return result


def _validate_against_summary_rows(normalized: pd.DataFrame, summary_rows: pd.DataFrame) -> None:
    """Best-effort cross-check against any "total" rows present.

    Summary rows are never used to compute values (some multi-level
    apartments don't have one), but when present they're a useful sanity
    check on the interior area aggregated from the individual levels.
    """
    if summary_rows.empty or normalized.empty:
        return
    computed_by_id = normalized.set_index("apartment_id")["interior_area_sqm"]
    for _, row in summary_rows.iterrows():
        apartment_number = int(row["apartment_number"])
        summary_area = row["interior_area_sqm"]
        if pd.isna(summary_area) or apartment_number not in computed_by_id.index:
            continue
        computed_area = computed_by_id.loc[apartment_number]
        if abs(float(summary_area) - float(computed_area)) > 0.5:
            warnings.warn(
                f"Apartment {apartment_number}: aggregated interior area "
                f"({computed_area}) does not match its summary row "
                f"({summary_area})."
            )


def load_normalized_apartments(path: str | Path) -> pd.DataFrame:
    """Read the raw source file and return one normalized row per apartment."""
    raw = read_raw_apartment_sheet(path)
    return normalize_apartments(raw)
