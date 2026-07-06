"""Model artifact: rebuild the forecasters and produce passenger forecasts.

Only the fitted SARIMAX parameters are persisted (tiny). At load the model is rebuilt
on the series and parameters applied via ``smooth()`` (instant). Holt-Winters is cheap
(144 points) so it is fit lazily on first use.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import load_series

META_PATH = Path(__file__).resolve().parents[2] / "models" / "air_meta.json"


@dataclass
class AirModel:
    sarimax_results: object
    meta: dict
    series: object
    _hw: object = field(default=None)

    @classmethod
    def load(cls) -> "AirModel":
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {META_PATH}. "
                "Train it first with: python -m air_passengers.train"
            )
        warnings.filterwarnings("ignore")
        meta = json.loads(META_PATH.read_text())
        s = load_series()
        model = SARIMAX(s, order=tuple(meta["order"]),
                        seasonal_order=tuple(meta["seasonal_order"]),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.smooth(np.array(meta["params"]))
        return cls(sarimax_results=results, meta=meta, series=s)

    @property
    def version(self) -> str:
        return self.meta.get("version", "1.0.0")

    def forecast(self, steps: int, model: str = "sarima") -> list[dict]:
        if model == "holtwinters":
            return self._forecast_hw(steps)
        return self._forecast_sarima(steps)

    def _forecast_sarima(self, steps: int) -> list[dict]:
        fc = self.sarimax_results.get_forecast(steps=steps)
        mean, ci = fc.predicted_mean, fc.conf_int()
        return [
            {
                "date": mean.index[i].strftime("%Y-%m-%d"),
                "passengers": round(float(mean.iloc[i]), 1),
                "lower": round(float(ci.iloc[i, 0]), 1),
                "upper": round(float(ci.iloc[i, 1]), 1),
            }
            for i in range(steps)
        ]

    def _forecast_hw(self, steps: int) -> list[dict]:
        if self._hw is None:
            self._hw = ExponentialSmoothing(
                self.series, trend="add", seasonal="mul", seasonal_periods=12
            ).fit()
        fc = self._hw.forecast(steps)
        return [
            {"date": fc.index[i].strftime("%Y-%m-%d"),
             "passengers": round(float(fc.iloc[i]), 1),
             "lower": None, "upper": None}
            for i in range(steps)
        ]
