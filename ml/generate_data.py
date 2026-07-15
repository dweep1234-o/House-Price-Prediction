"""
generate_data.py
-----------------
Generates a synthetic housing training dataset. Price is built from a
realistic, human-interpretable linear formula (bigger area = more,
older = less, closer to the city center = more, parking/garden add value,
a locality amenity score adds value) plus noise — so a trained linear
model recovers coefficients that translate directly into an itemized
price calculation for the UI.

NOTE ON DATA: This is SYNTHETIC data, not real listings. Swap in a real,
licensed housing dataset for production use (see data/README.md).

Run:
    python ml/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 2500

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "housing_training.csv"

# Ground-truth coefficients (also mirrored in data/README.md for reference).
BASE_PRICE = 45_000
COEFS = {
    "area_sqft": 118,
    "bedrooms": 9_000,
    "bathrooms": 6_500,
    "age_years": -950,
    "distance_km": -2_100,
    "floor_number": 650,
    "has_parking": 11_000,
    "has_garden": 8_500,
    "locality_score": 14_000,
}


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    area_sqft = rng.normal(1400, 550, n_samples).clip(250, 5500).round(0)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = (bedrooms - rng.integers(0, 2, n_samples)).clip(1, 5)
    age_years = rng.gamma(2.0, 8, n_samples).clip(0, 75).round(0)
    distance_km = rng.gamma(2.0, 4, n_samples).clip(0.2, 35).round(1)
    floor_number = rng.integers(0, 20, n_samples)
    has_parking = rng.binomial(1, 0.6, n_samples)
    has_garden = rng.binomial(1, 0.35, n_samples)
    locality_score = rng.normal(6, 2, n_samples).clip(0, 10).round(1)

    price = (
        BASE_PRICE
        + COEFS["area_sqft"] * area_sqft
        + COEFS["bedrooms"] * bedrooms
        + COEFS["bathrooms"] * bathrooms
        + COEFS["age_years"] * age_years
        + COEFS["distance_km"] * distance_km
        + COEFS["floor_number"] * floor_number
        + COEFS["has_parking"] * has_parking
        + COEFS["has_garden"] * has_garden
        + COEFS["locality_score"] * locality_score
    )
    noise = rng.normal(0, 14_000, n_samples)
    price = (price + noise).clip(35_000, None).round(0)

    df = pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "distance_km": distance_km,
        "floor_number": floor_number,
        "has_parking": has_parking,
        "has_garden": has_garden,
        "locality_score": locality_score,
        "price": price,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["price"].describe())
