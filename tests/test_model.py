"""
test_model.py
--------------
Tests for the model, the prediction/breakdown logic, the locality data, and
the Flask API.

Run (from project root):
    pytest -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

from predict import predict_price  # noqa: E402
from preprocessing import FEATURE_COLUMNS, validate_ranges  # noqa: E402

SAMPLE_HOUSE = {
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

SMALL_OLD_FAR_HOUSE = {
    "area_sqft": 500,
    "bedrooms": 1,
    "bathrooms": 1,
    "age_years": 60,
    "distance_km": 25,
    "floor_number": 0,
    "has_parking": 0,
    "has_garden": 0,
    "locality_score": 2,
}


def test_model_artifact_exists():
    assert (ROOT / "models" / "model.pkl").exists(), (
        "models/model.pkl is missing. Run `python ml/train.py` first."
    )


def test_locality_data_exists():
    assert (ROOT / "data" / "locality" / "locality_houses.json").exists(), (
        "locality_houses.json is missing. Run `python ml/generate_locality.py` first."
    )


def test_predict_returns_expected_shape():
    result = predict_price(SAMPLE_HOUSE)
    assert "predicted_price" in result
    assert "breakdown" in result
    assert len(result["breakdown"]) == len(FEATURE_COLUMNS) + 1  # + base price row


def test_predict_price_is_positive():
    result = predict_price(SAMPLE_HOUSE)
    assert result["predicted_price"] > 0


def test_breakdown_sums_to_predicted_price():
    result = predict_price(SAMPLE_HOUSE)
    total = sum(row["contribution"] for row in result["breakdown"])
    assert abs(total - result["predicted_price"]) < 0.5


def test_predict_directionality():
    """A bigger, newer, closer-in house should be worth more than a small, old, far one."""
    good = predict_price(SAMPLE_HOUSE)
    bad = predict_price(SMALL_OLD_FAR_HOUSE)
    assert good["predicted_price"] > bad["predicted_price"]


def test_validate_ranges_flags_out_of_bounds():
    bad_input = dict(SAMPLE_HOUSE)
    bad_input["bedrooms"] = 50  # out of the allowed 0-8 range
    errors = validate_ranges(bad_input)
    assert any("bedrooms" in e for e in errors)


def test_validate_ranges_flags_missing_field():
    incomplete = {k: v for k, v in SAMPLE_HOUSE.items() if k != "area_sqft"}
    errors = validate_ranges(incomplete)
    assert any("area_sqft" in e for e in errors)


@pytest.fixture
def client():
    from app import app as flask_app

    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Locality House Price Explorer" in resp.data


def test_locality_endpoint(client):
    resp = client.get("/api/locality")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["houses"]) > 0
    assert "x" in data["houses"][0] and "y" in data["houses"][0]


def test_predict_endpoint_success(client):
    locality = client.get("/api/locality").get_json()
    first_id = locality["houses"][0]["id"]
    resp = client.get(f"/api/predict/{first_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predicted_price" in data
    assert "breakdown" in data


def test_predict_endpoint_unknown_house(client):
    resp = client.get("/api/predict/999999")
    assert resp.status_code == 404
