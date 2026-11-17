import numpy as np
import pandas as pd

from posforecast import data
from posforecast.baseline import seasonal_naive
from posforecast.calendar_sa import week_flags
from posforecast.features import build_origin_features, expand_horizons
from posforecast.metrics import mape
from posforecast.model import Forecaster


def make_df(weeks=120):
    w = pd.date_range("2023-01-01", periods=weeks, freq="7D")
    rows = []
    for s in ["A", "B"]:
        for i, d in enumerate(w):
            rows.append({"week_start": d, "sector": s, "city": "Riyadh", "value": 100 + i + 10 * np.sin(i / 4)})
    return pd.DataFrame(rows)


def test_sector_aliases_are_merged():
    raw = pd.DataFrame({"starting_date": ["2024-01-07"] * 2, "sectors": ["Other", "Others"],
                        "city": ["riyadh"] * 2, "value": [1.0, 2.0]})
    out = data.clean(raw)
    assert len(out) == 1 and out.value.iloc[0] == 3.0 and out.city.iloc[0] == "Riyadh"


def test_measure_column_keeps_value_rows():
    raw = pd.DataFrame({"starting_date": ["2024-01-07"] * 2, "sectors": ["X"] * 2, "city": ["Jeddah"] * 2,
                        "unit": ["Value of Transactions", "Number of Transactions"], "value": [500.0, 5.0]})
    assert data.clean(raw).value.tolist() == [500.0]


def test_eid_week_flag():
    assert week_flags(pd.Timestamp("2025-03-30"))["eid_fitr_days"] > 0
    assert week_flags(pd.Timestamp("2025-01-05"))["ramadan_days"] == 0


def test_features_do_not_look_ahead():
    df = make_df()
    feats = build_origin_features(df)
    t = expand_horizons(feats, df)
    # the target is in the future, the lag features are in the past
    row = t[(t.sector == "A") & (t.horizon == 1)].iloc[60]
    assert row.target_week > row.week_start


def test_baseline_uses_last_year():
    df = make_df()
    origin = df.week_start.iloc[80]
    b = seasonal_naive(df, origin)
    target = origin + pd.Timedelta(weeks=1)
    expected = df[(df.sector == "A") & (df.week_start == target - pd.Timedelta(weeks=52))].value.iloc[0]
    assert b[(b.sector == "A") & (b.horizon == 1)].baseline.iloc[0] == expected


def test_model_trains_and_predicts():
    df = make_df()
    origin = df.week_start.iloc[-5]
    fc = Forecaster().fit(df, cutoff=origin).predict_from(df, origin=origin)
    assert len(fc) == 2 * 4 and fc.forecast.gt(0).all()
    actual = df[df.week_start > origin].merge(fc, left_on=["sector", "week_start"], right_on=["sector", "target_week"])
    assert mape(actual.value, actual.forecast) < 15
