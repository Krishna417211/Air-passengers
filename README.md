# ✈️ Air Passengers Forecast API

Forecast monthly international **airline passengers** with classic time-series models
(SARIMAX and Holt-Winters), served over a clean **FastAPI**.

Originally the `AIR_Passengers.ipynb` notebook — refactored into a reproducible project
with the fitted model **committed so it runs out of the box**.

---

## ✨ Highlights

- 🚀 **FastAPI** with interactive Swagger docs at `/docs`.
- 📦 **Ready to run** — fitted parameters committed; no training needed to forecast.
- 🔁 **Reproducible** — dataset auto-downloads, one command retrains.
- 📅 **Two models** — SARIMAX (with confidence intervals) and Holt-Winters.
- ✅ **Validated** query params (bad input → `422`).

---

## 🚀 Quickstart

```bash
git clone https://github.com/Krishna417211/Air-passengers.git
cd Air-passengers
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs**.

---

## 📡 API reference

| Method | Path                                   | Description                              |
|--------|----------------------------------------|------------------------------------------|
| `GET`  | `/health`                              | Liveness + model-loaded flag             |
| `GET`  | `/model/info`                          | Version, order, holdout metrics          |
| `GET`  | `/forecast?steps=N&model=sarima`       | Forecast next `N` months                 |

`model` is either `sarima` (default, includes confidence intervals) or `holtwinters`.

### Example — `curl`

```bash
curl "http://127.0.0.1:8000/forecast?steps=3&model=sarima"
```

```json
{
  "model_version": "1.0.0",
  "model": "sarima",
  "last_observed_date": "1960-12-01",
  "steps": 3,
  "forecast": [
    {"date": "1961-01-01", "passengers": 447.2, "lower": 424.0, "upper": 470.4},
    {"date": "1961-02-01", "passengers": 422.7, "lower": 394.8, "upper": 450.7}
  ]
}
```

### Example — Python

```python
import requests
r = requests.get("http://127.0.0.1:8000/forecast",
                 params={"steps": 12, "model": "holtwinters"})
for p in r.json()["forecast"]:
    print(p["date"], p["passengers"])
```

---

## 🧠 The models

- **Dataset:** the classic [international airline passengers](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)
  monthly series (1949–1960, 144 months). Downloads automatically on first run.
- **SARIMAX(1,1,1)×(1,1,1,12)** on the passenger levels — includes confidence intervals.
  Only its fitted parameters are stored (`models/air_meta.json`, a few hundred bytes)
  and the model is rebuilt on load via `smooth()` — instant, no re-optimization.
- **Holt-Winters** (additive trend, multiplicative seasonality) — fit on demand.

**Holdout (last 12 months, 1960):** SARIMAX RMSE ≈ 21.6 · Holt-Winters RMSE ≈ 15.8.

Passenger counts are in thousands, matching the source series. Forecasts start in 1961
(the series ends Dec 1960); the models capture the trend + yearly seasonality.

---

## 🔁 Reproduce / retrain

```bash
PYTHONPATH=src python -m air_passengers.train
```

---

## 🛠️ Tech stack

FastAPI · Uvicorn · statsmodels (SARIMAX, Holt-Winters) · pandas · NumPy · scikit-learn
