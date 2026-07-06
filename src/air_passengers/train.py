"""Train the Air Passengers forecasters and save a reusable artifact.

The notebook explored ARIMA, SARIMAX, and Holt-Winters Exponential Smoothing. For a
clean, interpretable API we forecast the *actual passenger counts* (levels) with:
  * SARIMAX(1,1,1)x(1,1,1,12)  - primary, with confidence intervals
  * Holt-Winters (additive trend, multiplicative seasonality) - alternative

Both are evaluated on the last 12 months (1960) as a holdout. Only the fitted SARIMAX
parameters are persisted (tiny); the model is rebuilt on load via smooth().

Run:  python -m air_passengers.train
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import load_series
from .model import META_PATH

VERSION = "1.0.0"
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)
SPLIT_DATE = "1960-01-01"


def _fit_sarimax(series):
    return SARIMAX(series, order=ORDER, seasonal_order=SEASONAL_ORDER,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def train() -> dict:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    s = load_series()
    train_s, test_s = s[s.index < SPLIT_DATE], s[s.index >= SPLIT_DATE]

    # SARIMAX holdout.
    sar_eval = _fit_sarimax(train_s)
    sar_pred = sar_eval.get_forecast(steps=len(test_s)).predicted_mean
    sar_metrics = {"mae": round(float(mean_absolute_error(test_s, sar_pred)), 4),
                   "rmse": round(_rmse(test_s, sar_pred), 4)}

    # Holt-Winters holdout.
    hw_eval = ExponentialSmoothing(train_s, trend="add", seasonal="mul",
                                   seasonal_periods=12).fit()
    hw_pred = hw_eval.forecast(len(test_s))
    hw_metrics = {"mae": round(float(mean_absolute_error(test_s, hw_pred)), 4),
                  "rmse": round(_rmse(test_s, hw_pred), 4)}

    print("SARIMAX holdout:", sar_metrics)
    print("Holt-Winters holdout:", hw_metrics)

    # Serving SARIMAX on the full series -> persist params only.
    res = _fit_sarimax(s)
    meta = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_months": int(len(s)),
        "last_observed_date": str(s.index[-1].date()),
        "order": list(ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "params": [float(p) for p in res.params],
        "metrics": {"sarimax": sar_metrics, "holtwinters": hw_metrics},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved params + meta -> {META_PATH}")
    return meta


if __name__ == "__main__":
    train()
