"""
train.py
--------
Trains the house price model. Linear Regression is trained on raw
(unscaled) features so its coefficients are directly interpretable in
dollars — this is what powers the itemized "calculation" breakdown in the
UI. A Random Forest is also trained purely as an accuracy comparison and
is recorded in metrics.json, but Linear Regression is what gets served.

Run:
    python ml/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "housing_training.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python ml/generate_data.py` first.")
    return pd.read_csv(DATA_PATH)


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, preds)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, preds)), 2),
        "r2": round(float(r2_score(y_test, preds)), 4),
    }


def main():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Served model: Linear Regression (interpretable coefficients) ---
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    linreg_metrics = evaluate(linreg, X_test, y_test)

    coefficients = {col: round(float(c), 2) for col, c in zip(FEATURE_COLUMNS, linreg.coef_)}
    intercept = round(float(linreg.intercept_), 2)

    print("=== Linear Regression (served model) ===")
    print(json.dumps(linreg_metrics, indent=2))
    print("Intercept (base price):", intercept)
    print("Coefficients:", json.dumps(coefficients, indent=2))

    # --- Comparison-only model: Random Forest ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test)

    print("\n=== Random Forest (comparison only, not served) ===")
    print(json.dumps(rf_metrics, indent=2))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": linreg,
        "model_name": "linear_regression",
        "features": FEATURE_COLUMNS,
        "intercept": intercept,
        "coefficients": coefficients,
    }, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "served_model": "linear_regression",
            "results": {
                "linear_regression": linreg_metrics,
                "random_forest_comparison": rf_metrics,
            },
            "intercept": intercept,
            "coefficients": coefficients,
        }, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
