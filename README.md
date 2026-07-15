# 🏘️ Locality House Price Explorer

A full-stack machine learning project that predicts house prices for a
small, interactive neighborhood map. Hover a house for a quick look, click
it, and watch the prediction come up with a full itemized calculation —
base price plus each feature's dollar contribution, straight from the
trained model's coefficients.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ✨ Features

- **Interactive locality map** — an SVG neighborhood of 24 houses across 3 streets; hover for a tooltip, click for the full breakdown
- **Live, itemized price calculation** — every prediction shows *base price + area + bedrooms + ... = total*, not just a black-box number
- **Interpretable ML model** — Linear Regression trained on raw (unscaled) features so its coefficients translate directly into dollar contributions
- **Accuracy comparison** — a Random Forest is also trained and benchmarked in `metrics.json`, even though the linear model is what's served (for interpretability)
- **REST API** — `GET /api/locality` and `GET /api/predict/<house_id>` for programmatic access
- **Tests** — pytest suite covering the model, the calculation breakdown, and the API
- **CI** — GitHub Actions workflow that regenerates data, retrains, and runs tests on every push
- **Reproducible data** — synthetic dataset + synthetic locality generator so the project runs out of the box with no external map/API keys

## 📁 Project structure

```
house-price-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions: install → generate data → train → test
├── data/
│   ├── raw/
│   │   └── housing_training.csv  # synthetic training dataset (generated)
│   ├── locality/
│   │   └── locality_houses.json  # the 24-house map locality (generated)
│   └── README.md                 # data dictionary + ground-truth pricing formula
├── ml/
│   ├── generate_data.py          # builds the synthetic training dataset
│   ├── generate_locality.py      # builds the map locality (coordinates + features)
│   ├── preprocessing.py          # shared feature list, labels, units, validation
│   ├── train.py                   # trains + evaluates + saves the model
│   └── predict.py                 # inference + itemized calculation breakdown
├── models/
│   ├── model.pkl                  # trained linear regression (+ coefficients)
│   └── metrics.json                # evaluation metrics from the last training run
├── backend/
│   ├── app.py                     # Flask app: map UI + /api/locality + /api/predict
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js           # draws the map, handles hover/click, renders results
├── tests/
│   └── test_model.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone / open the project folder, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the training dataset and the map locality (both already included, but this regenerates them)
python ml/generate_data.py
python ml/generate_locality.py

# 4. Train the model
python ml/train.py

# 5. Run the app
python backend/app.py
```

Then open **http://127.0.0.1:5003** in your browser. Hover over any house
on the map for a quick preview, then click it to see the predicted price
and the full calculation breakdown on the right.

### Run the tests

```bash
pytest -v
```

## 🧠 How it works

1. **`ml/generate_data.py`** builds a synthetic training dataset where price
   is a linear combination of 9 features (area, bedrooms, bathrooms, age,
   distance to city center, floor, parking, garden, and a locality amenity
   score) plus noise — see [`data/README.md`](data/README.md) for the exact
   formula.
2. **`ml/train.py`** trains a `LinearRegression` on the **raw, unscaled**
   features, so each coefficient is directly interpretable in dollars (e.g.
   "+$118 per sq ft"). A `RandomForestRegressor` is also trained purely for
   an accuracy comparison, recorded in `models/metrics.json`.
3. **`ml/generate_locality.py`** creates a small fictional locality —
   "Maple Grove" — placing 24 houses across 3 streets with map coordinates
   and their own feature values.
4. **`ml/predict.py`** takes a house's features and returns both the
   predicted price and an itemized breakdown: `base price + (coefficient ×
   value)` for every feature, which sums exactly to the predicted price.
5. **`backend/app.py`** exposes:
   - `GET /` — the map UI
   - `GET /api/locality` — the full locality (houses + coordinates)
   - `GET /api/predict/<house_id>` — predicted price + breakdown for one house
   - `GET /api/health` — health check
6. **`backend/static/js/script.js`** draws the SVG map from `/api/locality`,
   wires up hover tooltips and click selection, and renders the calculation
   table when a house is clicked.

### Latest evaluation (from `models/metrics.json`)

| Model                      | MAE       | RMSE      | R²      |
|------------------------------|-----------|-----------|---------|
| Linear Regression ✅ (served) | $11,581   | $14,742   | **0.959** |
| Random Forest (comparison)   | $16,059   | $19,722   | 0.926   |

*(Regenerate by re-running `python ml/train.py`; numbers will vary slightly with the random seed.)*

## 🔌 API example

```bash
curl http://127.0.0.1:5003/api/predict/1
```

```json
{
  "predicted_price": 355286.25,
  "address": "10 Maple Street",
  "breakdown": [
    { "feature": "base_price", "label": "Base price", "contribution": 48061.6 },
    { "feature": "area_sqft", "label": "Area", "value": 1226, "unit": "sq ft", "coefficient": 117.75, "contribution": 144361.5 },
    { "feature": "locality_score", "label": "Locality score", "value": 7.0, "unit": "/10", "coefficient": 13776.48, "contribution": 96435.36 }
  ]
}
```

## 🔁 Using a real locality or real housing data

- To change the map's neighborhood, edit the constants in `ml/generate_locality.py` (street names, grid size) and re-run it.
- To train on real housing data, replace `data/raw/housing_training.csv` with a real dataset sharing the same 9 feature columns + `price`, then re-run `python ml/train.py`. See [`data/README.md`](data/README.md) for details, including a note on the linear-model assumption behind the calculation breakdown.

## ⚠️ Disclaimer

This tool is an **educational demo**. The locality, houses, and prices are
entirely fictional/synthetic and do not represent any real property or
market. It is not a real estate valuation tool.

## 📄 License

MIT — see [LICENSE](LICENSE).




# 👩🏻 INTERN ID - CITS5351
