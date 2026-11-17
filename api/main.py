"""Small API that serves the latest forecast.

Run locally:  uvicorn api.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
FORECASTS = Path(os.environ.get("FORECASTS_FILE", ROOT / "artifacts" / "forecasts.csv"))
MONITORING = ROOT / "results" / "monitoring.csv"
BACKTEST = ROOT / "results" / "backtest_summary.csv"

app = FastAPI(title="Saudi POS spend forecast", version="0.1.0",
              description="Weekly point-of-sale spending forecast by sector and city, based on SAMA data.")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(503, f"{path.name} not found. Run the forecast job first.")
    return pd.read_csv(path)


@app.get("/health")
def health():
    return {"status": "ok", "forecast_file": FORECASTS.exists()}


@app.get("/series")
def series():
    df = _read(FORECASTS)
    return {"sectors": sorted(df.sector.unique()), "cities": sorted(df.city.unique())}


@app.get("/forecast")
def forecast(sector: str = Query(..., examples=["Restaurants and Cafes"]),
             city: str = Query(..., examples=["Riyadh"]),
             horizon: int = Query(4, ge=1, le=4, description="weeks ahead (1-4)")):
    df = _read(FORECASTS)
    sel = df[(df.sector.str.lower() == sector.lower()) & (df.city.str.lower() == city.lower())
             & (df.horizon <= horizon)]
    if sel.empty:
        raise HTTPException(404, "No forecast for this sector/city. See /series for valid values.")
    return {
        "sector": sel.sector.iloc[0], "city": sel.city.iloc[0],
        "last_actual_week": sel.origin_week.iloc[0], "issued_at": sel.issued_at.iloc[0],
        "unit": "same unit as the SAMA report (value of transactions)",
        "forecast": sel.sort_values("horizon")[["horizon", "target_week", "forecast"]].to_dict("records"),
    }


@app.get("/accuracy")
def accuracy():
    out = {}
    if BACKTEST.exists():
        out["backtest"] = pd.read_csv(BACKTEST).to_dict("records")
    if MONITORING.exists():
        out["live"] = pd.read_csv(MONITORING).tail(16).to_dict("records")
    return out


from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")