"""Seasonal-naive baseline: the forecast for a week is the value of the same week last year.
Any model we build has to beat this, otherwise it is not worth using."""
import pandas as pd

from . import config


def seasonal_naive(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    hist = df[df["week_start"] <= origin].dropna(subset=["value"])
    lookup = hist.set_index(["sector", "city", "week_start"])["value"]
    rows = []
    for (s, c), _ in hist.groupby(["sector", "city"]):
        for h in config.HORIZONS:
            target = origin + pd.Timedelta(weeks=h)
            ref = target - pd.Timedelta(weeks=config.SEASON)
            rows.append({"sector": s, "city": c, "origin_week": origin, "horizon": h,
                         "target_week": target, "baseline": lookup.get((s, c, ref))})
    return pd.DataFrame(rows)
