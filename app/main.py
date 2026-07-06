"""FastAPI service for the Air Passengers forecaster.

Run:  uvicorn app.main:app --reload   ->   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from air_passengers.model import AirModel  # noqa: E402

from app.schemas import (  # noqa: E402
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
)

state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = AirModel.load()
    except FileNotFoundError as exc:
        print(f"[warning] {exc}")
        state["model"] = None
    yield
    state.clear()


app = FastAPI(
    title="Air Passengers Forecast API",
    description=(
        "Forecast monthly international airline passengers with SARIMAX or "
        "Holt-Winters. Built from the AIR_Passengers notebook."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _model() -> AirModel:
    m = state.get("model")
    if m is None:
        raise HTTPException(503, "Model not loaded. Run `python -m air_passengers.train`.")
    return m


@app.get("/", tags=["meta"])
def root():
    return {"name": "Air Passengers Forecast API", "docs": "/docs",
            "forecast": "GET /forecast?steps=12&model=sarima"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", model_loaded=state.get("model") is not None)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info():
    m = _model()
    meta = m.meta
    return ModelInfoResponse(
        version=m.version, trained_at=meta.get("trained_at", ""),
        n_months=meta.get("n_months", 0),
        last_observed_date=meta.get("last_observed_date", ""),
        order=meta.get("order", []), seasonal_order=meta.get("seasonal_order", []),
        metrics=meta.get("metrics", {}),
    )


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(
    steps: int = Query(12, ge=1, le=240, description="Number of future months."),
    model: str = Query("sarima", pattern="^(sarima|holtwinters)$",
                       description="Forecasting model: 'sarima' or 'holtwinters'."),
):
    m = _model()
    return ForecastResponse(
        model_version=m.version, model=model,
        last_observed_date=m.meta.get("last_observed_date", ""),
        steps=steps, forecast=m.forecast(steps, model=model),
    )
