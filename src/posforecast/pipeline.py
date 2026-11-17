"""Weekly jobs: issue a forecast, and check old forecasts against new actuals."""
from __future__ import annotations
import pandas as pd

from . import config
from .metrics import mape
from .model import Forecaster


def forecast(df: pd.DataFrame) -> pd.DataFrame:
    model = Forecaster().fit(df)
    model.save()
    fc = model.predict_from(df)
    fc["forecast"] = fc["forecast"].round(2)
    fc["issued_at"] = pd.Timestamp.now(tz="Asia/Riyadh").strftime("%Y-%m-%d")
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fc.to_csv(config.FORECASTS_FILE, index=False)

    # keep every forecast we ever issued, so we can grade it later
    if config.FORECAST_LOG.exists():
        log = pd.read_csv(config.FORECAST_LOG, parse_dates=["origin_week", "target_week"])
        log = pd.concat([log, fc]).drop_duplicates(["sector", "city", "origin_week", "horizon"], keep="first")
    else:
        log = fc
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    log.to_csv(config.FORECAST_LOG, index=False)
    return fc


def monitor(df: pd.DataFrame) -> pd.DataFrame:
    """Match logged forecasts with actuals that have arrived since, and record the error per week."""
    if not config.FORECAST_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(config.FORECAST_LOG, parse_dates=["origin_week", "target_week"])
    actual = df.dropna(subset=["value"]).rename(columns={"week_start": "target_week", "value": "actual"})
    m = log.merge(actual[["sector", "city", "target_week", "actual"]], on=["sector", "city", "target_week"])
    if m.empty:
        return pd.DataFrame()
    rows = (m.groupby(["target_week", "horizon"])
             .apply(lambda g: pd.Series({"mape": mape(g.actual, g.forecast), "series": len(g)}), include_groups=False)
             .reset_index())
    rows["mape"] = rows["mape"].round(2)
    rows.to_csv(config.MONITORING_FILE, index=False)
    return rows
