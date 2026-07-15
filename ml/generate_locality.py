"""
generate_locality.py
---------------------
Builds a small, fictional locality of houses laid out on a street grid for
the interactive map UI. Each house gets map coordinates (for SVG placement)
plus a realistic feature set (used to compute its live prediction when the
user clicks it in the UI).

Run:
    python ml/generate_locality.py
"""

import json
from pathlib import Path

import numpy as np

RANDOM_SEED = 7
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "locality" / "locality_houses.json"

STREET_NAMES = ["Maple", "Birch", "Cedar", "Willow", "Elm", "Aspen"]
MAP_WIDTH = 720
MAP_HEIGHT = 480

# 3 streets (rows) x up to 8 houses (columns), forming a small grid locality.
ROWS = 3
COLS = 8


def generate_locality(seed: int = RANDOM_SEED) -> list:
    rng = np.random.default_rng(seed)
    houses = []
    house_id = 1

    margin_x, margin_y = 70, 90
    row_gap = (MAP_HEIGHT - 2 * margin_y) / (ROWS - 1)
    col_gap = (MAP_WIDTH - 2 * margin_x) / (COLS - 1)

    for row in range(ROWS):
        street = STREET_NAMES[row % len(STREET_NAMES)]
        for col in range(COLS):
            x = round(margin_x + col * col_gap + rng.uniform(-10, 10), 1)
            y = round(margin_y + row * row_gap + rng.uniform(-10, 10), 1)

            area_sqft = int(rng.normal(1350, 450))
            area_sqft = max(350, min(area_sqft, 4200))
            bedrooms = int(rng.integers(1, 6))
            bathrooms = max(1, bedrooms - int(rng.integers(0, 2)))
            age_years = int(rng.gamma(2.0, 7))
            distance_km = round(float(np.clip(rng.gamma(2.0, 3.5), 0.3, 25)), 1)
            floor_number = int(rng.integers(0, 4))  # small locality: low-rise homes
            has_parking = int(rng.binomial(1, 0.65))
            has_garden = int(rng.binomial(1, 0.4))
            locality_score = round(float(np.clip(rng.normal(6.5, 1.5), 1, 10)), 1)

            houses.append({
                "id": house_id,
                "address": f"{10 + col * 2} {street} Street",
                "x": x,
                "y": y,
                "features": {
                    "area_sqft": area_sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "age_years": age_years,
                    "distance_km": distance_km,
                    "floor_number": floor_number,
                    "has_parking": has_parking,
                    "has_garden": has_garden,
                    "locality_score": locality_score,
                },
            })
            house_id += 1

    return houses


if __name__ == "__main__":
    houses = generate_locality()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "map_width": MAP_WIDTH,
            "map_height": MAP_HEIGHT,
            "streets": STREET_NAMES[:ROWS],
            "houses": houses,
        }, f, indent=2)
    print(f"Saved {len(houses)} houses to {OUTPUT_PATH}")
