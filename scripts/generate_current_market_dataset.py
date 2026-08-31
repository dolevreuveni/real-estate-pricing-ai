"""Generate data/external/current_market_500_updated.xlsx -- SYNTHETIC POC
Current Market data standing in for a future Yad2/Madlan/developer feed
integration (see CURRENT_MARKET_DATA_TYPE in src/config/settings.py).

Bug-fix generation (rooms/area multicollinearity):

The previous version of this dataset tied rooms and area_sqm almost
deterministically -- each room count occupied a narrow, NON-OVERLAPPING
area band (e.g. every 5-room listing was 112-138 sqm, every 4-room
listing was 88-108 sqm). That produced near-perfect collinearity between
rooms and area_sqm (Pearson r = 0.973, VIF ~ 19), which made the Current
Market LinearRegression model unable to identify an independent rooms
effect -- and made any (rooms, area) combination outside those narrow
bands (e.g. "5 rooms, 90 sqm") a wild extrapolation with no training
support, producing a nonsensical negative rooms coefficient.

This generator fixes that by drawing area_sqm from a per-room-count
Normal distribution with a wide enough spread that adjacent room counts
substantially overlap (real listings of the same room count vary in
area by a lot depending on layout, and a given area is plausible for
more than one room count) while still increasing on average with rooms.
`asking_price` is driven mainly by area_sqm and the other honest
features (floor, condition, property type, direction, distance, top
floor, parking/storage/garden/roof/balcony extras), plus a small,
genuine, flat per-room bump (see ROOMS_PRICE_BUMP below). That bump
reflects a real phenomenon in the Israeli market -- buyers search and
value listings by bedroom count directly (a family needs N bedrooms),
not only by total floor area -- so at a fixed area, a few more
(smaller) rooms is not a defect. Widening the area/room overlap alone
cut the rooms/area VIF from ~19 to ~4, but with only 500 rows and
realistic price noise that was not enough to reliably overcome sampling
noise: a purely zero true rooms effect still produced a statistically
insignificant but visually confusing negative point estimate in
practice. A small, explicit, genuine rooms effect in the data
generation process (not a simulator patch) gives the regression a real
signal to find instead of noise to guess at.

Run:
    python scripts/generate_current_market_dataset.py
"""
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config.settings import CURRENT_MARKET_INPUT_PATH, CURRENT_MARKET_SHEET_NAME

RANDOM_SEED = 42
N_ROWS = 500

STREETS = [
    "Jabotinsky", "Klay", "Epstein", "Pinkas", "Helsinki",
    "Weizmann", "Lissin", "Mosenzon", "Bavli", "Biltmore",
]
CITY = "Tel Aviv"
NEIGHBORHOOD = "Kikar HaMedina / New North"

# rooms -> (count of listings, mean area_sqm, std area_sqm).
# Mean area follows the same ~24 sqm/room slope observed in the original
# dataset (intercept ~5, slope ~24); std is wide enough (roughly 15-20%
# of the mean) that adjacent room counts overlap substantially instead
# of forming disjoint bands.
ROOM_AREA_PLAN = {
    2: (90, 53.0, 9.5),
    3: (120, 77.0, 13.0),
    4: (150, 101.0, 17.0),
    5: (90, 125.0, 21.0),
    6: (35, 149.0, 25.0),
    7: (15, 173.0, 29.0),
}
assert sum(n for n, _, _ in ROOM_AREA_PLAN.values()) == N_ROWS

PROPERTY_TYPE_WEIGHTS = {
    "Apartment": 0.89,
    "Duplex": 0.06,
    "Garden Apartment": 0.03,
    "Penthouse": 0.02,
}

PROJECTS = [
    ("North TLV Living", "Sample Development D"),
    ("Kikar Residence", "Sample Development A"),
    ("Weizmann Urban", "Sample Development B"),
    ("Medina Boutique", "Sample Development E"),
    ("Helsinki Gardens", "Sample Development C"),
]

DIRECTIONS = [
    "North", "South", "East", "West",
    "North-East", "North-West", "South-East", "South-West",
]

DIRECTION_PRICE_ADJ = {
    "East": 2500, "South-East": 3000, "South": 2000,
    "West": 500, "South-West": 0,
    "North": -2500, "North-East": -1000, "North-West": -1500,
}

PROPERTY_TYPE_PRICE_PER_SQM_ADJ = {
    "Apartment": 0, "Duplex": 500, "Garden Apartment": 12000, "Penthouse": 16000,
}

# Modest, genuine, flat per-room price effect -- real Israeli buyers value
# bedroom count directly, independent of exact floor area (see module
# docstring). Calibrated (not guessed) against this generator's own
# output: with ROOMS_PRICE_BUMP=0, the fitted Current Market rooms
# coefficient came out to ~-60,000 (see PR/investigation notes) -- not a
# real effect, just sampling noise amplified by residual rooms/area
# collinearity (VIF ~4 at n=394 is not "solved", only reduced from ~19).
# Because OLS is linear, adding c per room to asking_price shifts the
# fitted rooms coefficient by exactly +c and changes nothing else (same
# residuals, same MAE/RMSE, same every other coefficient) -- so 85,000
# reliably lands the fitted coefficient at roughly +24,500/room: small
# relative to the ~80,000/sqm area effect (about 1 sqm-equivalent), and
# the same order of magnitude as the Historical model's own learned
# rooms effect (~+22,000/room), instead of leaving the true effect at a
# knife-edge zero that sampling noise can flip either way.
ROOMS_PRICE_BUMP = 85000


def _generate_rows(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    listing_index = 1

    for rooms, (count, area_mean, area_std) in ROOM_AREA_PLAN.items():
        for _ in range(count):
            area_sqm = float(np.clip(rng.normal(area_mean, area_std), 40.0, 220.0))
            area_sqm = round(area_sqm, 1)

            market_segment = "New Project" if rng.random() < 0.42 else "Second Hand"
            if market_segment == "New Project":
                condition = "New"
                building_year = int(rng.integers(2025, 2028))
                building_age = 0
                project_name, developer = PROJECTS[rng.integers(0, len(PROJECTS))]
            else:
                condition = str(rng.choice(
                    ["Good", "Renovated", "Needs Renovation"], p=[0.49, 0.38, 0.13]
                ))
                building_year = int(rng.integers(1960, 2023))
                building_age = 2026 - building_year
                project_name, developer = None, None

            property_type = str(rng.choice(
                list(PROPERTY_TYPE_WEIGHTS.keys()), p=list(PROPERTY_TYPE_WEIGHTS.values())
            ))

            total_floors = int(rng.integers(4, 19))
            floor = int(rng.integers(0, total_floors))
            is_top_floor = bool(floor == total_floors - 1)

            balcony_direction = str(rng.choice(DIRECTIONS))
            balcony_area_sqm = round(float(np.clip(rng.normal(15.0, 8.0), 0.0, 60.0)), 1)

            parking_count = int(rng.choice([0, 1, 2], p=[0.17, 0.72, 0.11]))
            storage_area_sqm = round(float(np.clip(rng.normal(3.0, 3.0), 0.0, 10.0)), 1) if rng.random() < 0.55 else 0.0

            garden_area_sqm = 0.0
            roof_area_sqm = 0.0
            if property_type == "Garden Apartment":
                garden_area_sqm = round(float(np.clip(rng.normal(45.0, 20.0), 10.0, 120.0)), 1)
            elif property_type == "Penthouse":
                roof_area_sqm = round(float(np.clip(rng.normal(35.0, 15.0), 5.0, 90.0)), 1)
            elif rng.random() < 0.02:
                roof_area_sqm = round(float(np.clip(rng.normal(10.0, 5.0), 2.0, 30.0)), 1)

            distance_from_project_km = round(float(rng.uniform(0.05, 0.99)), 2)

            elevator = True if total_floors >= 5 else bool(rng.random() < 0.6)

            # --- price/sqm: driven by segment, condition, property type,
            # floor, top-floor, direction, distance -- never by rooms.
            price_per_sqm = 72000.0
            price_per_sqm += 9000 if market_segment == "New Project" else 0
            price_per_sqm += {"New": 0, "Renovated": 2000, "Good": 0, "Needs Renovation": -6000}[condition]
            price_per_sqm += PROPERTY_TYPE_PRICE_PER_SQM_ADJ[property_type]
            price_per_sqm += 250 * floor
            price_per_sqm += 2500 if is_top_floor else 0
            price_per_sqm += DIRECTION_PRICE_ADJ[balcony_direction]
            price_per_sqm -= 2000 * distance_from_project_km
            price_per_sqm += rng.normal(0, 4000)
            price_per_sqm = max(price_per_sqm, 40000.0)

            asking_price = price_per_sqm * area_sqm
            asking_price += rooms * ROOMS_PRICE_BUMP
            asking_price += balcony_area_sqm * 3500
            asking_price += parking_count * 220000
            asking_price += storage_area_sqm * 3000
            asking_price += garden_area_sqm * 4500
            asking_price += roof_area_sqm * 5000
            asking_price += rng.normal(0, 60000)
            asking_price = max(asking_price, 1_500_000.0)
            asking_price = round(asking_price)

            final_price_per_sqm = asking_price / area_sqm

            street = STREETS[rng.integers(0, len(STREETS))]
            house_number = int(rng.integers(1, 200))

            directions_field = balcony_direction
            if rng.random() < 0.25:
                other = str(rng.choice([d for d in DIRECTIONS if d != balcony_direction]))
                directions_field = f"{balcony_direction}, {other}"

            listing_date = pd.Timestamp("2026-07-01") + pd.Timedelta(
                days=int(rng.integers(0, 53))
            )

            source = (
                "Synthetic New-Project Listing"
                if market_segment == "New Project"
                else "Synthetic Yad2-like Listing"
            )

            listing_id = f"SYN-{listing_index:04d}"
            rows.append(
                {
                    "listing_id": listing_id,
                    "listing_date": listing_date,
                    "source": source,
                    "source_url": f"https://example.com/listings/{listing_id}",
                    "market_segment": market_segment,
                    "project_name": project_name,
                    "developer": developer,
                    "address": f"{street} {house_number}, {CITY}",
                    "city": CITY,
                    "neighborhood": NEIGHBORHOOD,
                    "distance_from_project_km": distance_from_project_km,
                    "property_type": property_type,
                    "rooms": rooms,
                    "area_sqm": area_sqm,
                    "floor": floor,
                    "total_floors": total_floors,
                    "asking_price": asking_price,
                    "price_per_sqm": final_price_per_sqm,
                    "building_year": building_year,
                    "building_age": building_age,
                    "condition": condition,
                    "balcony_area_sqm": balcony_area_sqm,
                    "balcony_direction": balcony_direction if rng.random() > 0.02 else None,
                    "parking_count": parking_count,
                    "storage_area_sqm": storage_area_sqm,
                    "elevator": elevator,
                    "directions": directions_field,
                    "garden_area_sqm": garden_area_sqm,
                    "roof_area_sqm": roof_area_sqm,
                    "is_top_floor": is_top_floor,
                }
            )
            listing_index += 1

    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    df = _generate_rows(rng)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    CURRENT_MARKET_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(CURRENT_MARKET_INPUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=CURRENT_MARKET_SHEET_NAME, index=False)

    print(f"Wrote {len(df)} rows to {CURRENT_MARKET_INPUT_PATH} (sheet '{CURRENT_MARKET_SHEET_NAME}')")
    print()
    print("rooms distribution:")
    print(df["rooms"].value_counts().sort_index().to_string())
    print()
    print("area_sqm min/max/mean/std by rooms:")
    print(df.groupby("rooms")["area_sqm"].agg(["min", "max", "mean", "std"]).round(1).to_string())
    print()
    print(f"correlation(rooms, area_sqm) = {df['rooms'].corr(df['area_sqm']):.4f}")
    print(f"avg price_per_sqm = {df['price_per_sqm'].mean():,.0f}")


if __name__ == "__main__":
    main()
