"""Turn the weekly table into a supervised learning table.

Each row = (series, origin week t, horizon h). Features only use data up to week t.
Target = log(value at t+h) - log(value at t), i.e. the growth from the last known week.
Predicting growth instead of the raw level lets one model learn from big and small series together.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .calendar_sa import calendar_table

CAL_COLS = ["woy", "month", "ramadan_days", "ramadan_last10_days", "pre_eid_days",
            "eid_fitr_days", "eid_adha_days", "salary_week", "post_salary_week",
            "national_day", "founding_day"]


def _series_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week_start").copy()
    y = np.log1p(g["value"])
    g["y_log"] = y
    for lag in config.LAGS:
        # difference from the current week, so the feature does not depend on series size
        g[f"lag_{lag}"] = y.shift(lag) - y
    for w in config.ROLL_WINDOWS:
        g[f"roll_mean_{w}"] = y.rolling(w, min_periods=w // 2).mean() - y
        g[f"roll_std_{w}"] = y.rolling(w, min_periods=w // 2).std()
    g["level"] = y
    return g


def build_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features at every origin week (no target yet)."""
    df = df.dropna(subset=["value"])
    parts = [_series_features(g) for _, g in df.groupby(["sector", "city"])]
    feats = pd.concat(parts, ignore_index=True)
    feats["series_len"] = feats.groupby(["sector", "city"]).cumcount() + 1
    return feats


def expand_horizons(feats: pd.DataFrame, df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Add one copy of each origin row per horizon, plus calendar features of the *target* week."""
    rows = []
    actual = df.dropna(subset=["value"]).set_index(["sector", "city", "week_start"])["value"]
    for h in config.HORIZONS:
        f = feats.copy()
        f["horizon"] = h
        f["target_week"] = f["week_start"] + pd.Timedelta(weeks=h)
        # seasonal reference: same week last year, relative to current level
        ref_week = f["target_week"] - pd.Timedelta(weeks=config.SEASON)
        ref = actual.reindex(list(zip(f["sector"], f["city"], ref_week))).to_numpy()
        f["last_year_same_week"] = np.log1p(ref) - f["y_log"]
        if with_target:
            tgt = actual.reindex(list(zip(f["sector"], f["city"], f["target_week"]))).to_numpy()
            f["target"] = np.log1p(tgt) - f["y_log"]
            f["actual"] = tgt
        rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    cal = calendar_table(out["target_week"]).rename(columns={"week_start": "target_week"})
    out = out.merge(cal, on="target_week", how="left")
    # calendar of the origin week too: growth depends on where we start (e.g. starting inside Ramadan)
    cal0 = calendar_table(out["week_start"])[["week_start", "ramadan_days", "eid_fitr_days", "eid_adha_days"]]
    cal0.columns = ["week_start", "o_ramadan_days", "o_eid_fitr_days", "o_eid_adha_days"]
    return out.merge(cal0, on="week_start", how="left")


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = [f"lag_{l}" for l in config.LAGS]
    base += [f"roll_mean_{w}" for w in config.ROLL_WINDOWS] + [f"roll_std_{w}" for w in config.ROLL_WINDOWS]
    base += ["level", "horizon", "last_year_same_week", "o_ramadan_days", "o_eid_fitr_days", "o_eid_adha_days"]
    base += CAL_COLS + ["sector_code", "city_code"]
    return [c for c in base if c in df.columns]


def add_codes(df: pd.DataFrame, sectors: list[str], cities: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["sector_code"] = df["sector"].map({s: i for i, s in enumerate(sectors)}).fillna(-1).astype(int)
    df["city_code"] = df["city"].map({c: i for i, c in enumerate(cities)}).fillna(-1).astype(int)
    return df
