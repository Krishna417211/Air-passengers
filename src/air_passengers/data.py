"""Dataset loading for the Air Passengers forecaster.

The notebook read a local ``/content/AirPassengers.csv`` (the classic monthly
international airline passengers series, 1949-1960). It downloads automatically from
a public, stable mirror so the project is reproducible for anyone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "airline-passengers.csv"
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOCAL_CSV = DATA_DIR / "AirPassengers.csv"


def download_raw(dest: Path | None = None) -> Path:
    dest = dest or LOCAL_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"Downloading dataset from {DATA_URL} ...")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    dest.write_text(resp.text)
    print(f"Saved dataset to {dest}")
    return dest


def load_series(path: Path | None = None) -> pd.Series:
    """Return the monthly passenger count as a date-indexed series (freq='MS')."""
    if path is None:
        path = download_raw()
    df = pd.read_csv(path)
    # Mirror header is Month,Passengers; the notebook used '#Passengers'.
    df.columns = ["Month", "passengers"]
    df["Month"] = pd.to_datetime(df["Month"])
    s = df.set_index("Month")["passengers"].astype(float)
    s.index.freq = "MS"
    return s
