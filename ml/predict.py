"""
predict.py
----------
Loads the trained linear model and produces both a price prediction and an
itemized "calculation" breakdown (base price + each feature's dollar
contribution), which is what the map UI displays when a house is clicked.
"""

from pathlib import Path

import joblib

from preprocessing import FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_UNITS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python ml/train.py` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_price(house: dict) -> dict:
    """
    house: dict with keys matching FEATURE_COLUMNS
    returns: predicted price + an itemized breakdown of how it was calculated
    """
    bundle = _load()
    intercept = bundle["intercept"]
    coefficients = bundle["coefficients"]

    breakdown = [{
        "feature": "base_price",
        "label": "Base price",
        "value": None,
        "unit": "",
        "coefficient": None,
        "contribution": round(intercept, 2),
    }]

    running_total = intercept
    for col in FEATURE_COLUMNS:
        value = float(house[col])
        coef = coefficients[col]
        contribution = round(coef * value, 2)
        running_total += contribution
        breakdown.append({
            "feature": col,
            "label": FEATURE_LABELS[col],
            "value": value,
            "unit": FEATURE_UNITS[col],
            "coefficient": coef,
            "contribution": contribution,
        })

    predicted_price = round(running_total, 2)

    return {
        "predicted_price": predicted_price,
        "breakdown": breakdown,
        "model_name": bundle.get("model_name", "unknown"),
    }


if __name__ == "__main__":
    sample = {
        "area_sqft": 1450,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 8,
        "distance_km": 4.5,
        "floor_number": 3,
        "has_parking": 1,
        "has_garden": 0,
        "locality_score": 7.2,
    }
    result = predict_price(sample)
    print("Predicted price:", result["predicted_price"])
    for row in result["breakdown"]:
        print(row)
