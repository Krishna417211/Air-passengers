"""Pydantic response models for the Air Passengers forecast API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    date: str = Field(..., description="Forecast month (YYYY-MM-01).")
    passengers: float = Field(..., description="Predicted passenger count (thousands).")
    lower: float | None = Field(None, description="Lower CI bound (SARIMAX only).")
    upper: float | None = Field(None, description="Upper CI bound (SARIMAX only).")


class ForecastResponse(BaseModel):
    model_version: str
    model: str
    last_observed_date: str
    steps: int
    forecast: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    version: str
    trained_at: str
    n_months: int
    last_observed_date: str
    order: list[int]
    seasonal_order: list[int]
    metrics: dict
