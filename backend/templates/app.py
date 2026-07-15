"""
app.py
------
Flask application that serves the interactive locality map UI and a JSON
prediction API backed by the trained linear regression model.

Run (from project root):
    python backend/app.py

Then open http://127.0.0.1:5003 in your browser.
"""

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from predict import predict_price  # noqa: E402
from preprocessing import validate_ranges  # noqa: E402

LOCALITY_PATH = ROOT / "data" / "locality" / "locality_houses.json"

app = Flask(__name__)

_locality_cache = None


def load_locality() -> dict:
    global _locality_cache
    if _locality_cache is None:
        if not LOCALITY_PATH.exists():
            raise FileNotFoundError(
                f"{LOCALITY_PATH} not found. Run `python ml/generate_locality.py` first."
            )
        with open(LOCALITY_PATH) as f:
            _locality_cache = json.load(f)
    return _locality_cache


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/locality")
def api_locality():
    return jsonify(load_locality())


@app.route("/api/predict/<int:house_id>")
def api_predict(house_id):
    locality = load_locality()
    house = next((h for h in locality["houses"] if h["id"] == house_id), None)

    if house is None:
        return jsonify({"error": f"No house with id {house_id}."}), 404

    errors = validate_ranges(house["features"])
    if errors:
        return jsonify({"error": "Invalid feature values", "details": errors}), 400

    try:
        result = predict_price(house["features"])
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    result["house_id"] = house_id
    result["address"] = house["address"]
    result["features"] = house["features"]
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
