"""Rolling-origin backtest.

Pick an origin week, train only on data up to it, forecast the next 4 weeks, compare with
what really happened. Move the origin forward and repeat. This simulates using the model
live, week after week. A random train/test split would leak future weeks into training.
"""
from __future__ import annotations
import pandas as pd

from . import config
from .baseline import seasonal_naive
from .calendar_sa import calendar_table
from .metrics import mape, wape
from .model import Forecaster


def origins(df: pd.DataFrame, n: int = config.BACKTEST_ORIGINS, step: int = config.BACKTEST_STEP):
    last = df["week_start"].max()
    first = last - pd.Timedelta(weeks=max(config.HORIZONS))
    return [first - pd.Timedelta(weeks=step * i) for i in range(n)][::-1]


def run(df: pd.DataFrame, n: int = config.BACKTEST_ORIGINS, verbose: bool = True):
    actual = df.dropna(subset=["value"])[["sector", "city", "week_start", "value"]].rename(
        columns={"week_start": "target_week", "value": "actual"})
    detail = []
    for i, o in enumerate(origins(df, n)):
        model = Forecaster().fit(df, cutoff=o)
        pred = model.predict_from(df, origin=o)
        base = seasonal_naive(df, o)
        res = pred.merge(base, on=["sector", "city", "origin_week", "horizon", "target_week"], how="left")
        detail.append(res.merge(actual, on=["sector", "city", "target_week"], how="inner"))
        if verbose:
            print(f"origin {i + 1}/{n}: {o.date()}")
    detail = pd.concat(detail, ignore_index=True).dropna(subset=["baseline"])

    def summarize(g):
        return pd.Series({"model_mape": mape(g.actual, g.forecast), "baseline_mape": mape(g.actual, g.baseline),
                          "model_wape": wape(g.actual, g.forecast), "baseline_wape": wape(g.actual, g.baseline),
                          "n": len(g)})

    overall = summarize(detail).to_frame().T.assign(sector="ALL")
    by_sector = detail.groupby("sector").apply(summarize, include_groups=False).reset_index()
    summary = pd.concat([overall, by_sector], ignore_index=True)
    summary["improvement_pts"] = summary["baseline_mape"] - summary["model_mape"]
    summary = summary[["sector", "model_mape", "baseline_mape", "improvement_pts", "model_wape", "baseline_wape", "n"]]

    by_week = detail.groupby("target_week").apply(summarize, include_groups=False).reset_index()
    by_week = by_week.merge(calendar_table(by_week["target_week"]).rename(columns={"week_start": "target_week"}),
                            on="target_week", how="left")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    by_week.round(2).to_csv(config.BACKTEST_BY_WEEK, index=False)
    detail.to_csv(config.BACKTEST_DETAIL, index=False)
    summary.round(2).to_csv(config.BACKTEST_FILE, index=False)
    return summary, detail
