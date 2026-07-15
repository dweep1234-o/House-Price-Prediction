"""
preprocessing.py
-----------------
Shared feature list, ranges, and the ground-truth pricing formula used by
both the synthetic data generator and the training/inference code, so
everything stays consistent.
"""

from typing import List

FEATURE_COLUMNS: List[str] = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_km",
    "floor_number",
    "has_parking",
    "has_garden",
    "locality_score",
]

TARGET_COLUMN = "price"

FEATURE_LABELS = {
    "area_sqft": "Area",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "age_years": "Age",
    "distance_km": "Distance to city center",
    "floor_number": "Floor",
    "has_parking": "Parking",
    "has_garden": "Garden",
    "locality_score": "Locality score",
}

FEATURE_UNITS = {
    "area_sqft": "sq ft",
    "bedrooms": "",
    "bathrooms": "",
    "age_years": "yrs",
    "distance_km": "km",
    "floor_number": "",
    "has_parking": "",
    "has_garden": "",
    "locality_score": "/10",
}

# Reasonable bounds used for basic input validation.
FEATURE_RANGES = {
    "area_sqft": (200, 6000),
    "bedrooms": (0, 8),
    "bathrooms": (0, 6),
    "age_years": (0, 80),
    "distance_km": (0.1, 40),
    "floor_number": (0, 40),
    "has_parking": (0, 1),
    "has_garden": (0, 1),
    "locality_score": (0, 10),
}


def validate_ranges(values: dict) -> list:
    """Returns a list of human-readable error strings, empty if all valid."""
    errors = []
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in values:
            errors.append(f"Missing field: {col}")
            continue
        v = values[col]
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number.")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{col} should be between {lo} and {hi}.")
    return errors
