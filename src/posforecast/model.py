"""Model training and prediction. LightGBM if installed, otherwise scikit-learn's HistGradientBoosting."""
from __future__ import annotations
import pickle

import numpy as np
import pandas as pd

from . import config
from .features import add_codes, build_origin_features, expand_horizons, feature_columns

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def make_regressor():
    if lgb is not None:
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                 min_child_samples=20, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, objective="l1",
                                 random_state=config.RANDOM_STATE, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.03,
                                         max_leaf_nodes=31, min_samples_leaf=20,
                                         random_state=config.RANDOM_STATE)


class Forecaster:
    def __init__(self):
        self.model = None
        self.sectors: list[str] = []
        self.cities: list[str] = []
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> "Forecaster":
        """Train on everything whose *target week* is <= cutoff. This is what stops future leakage."""
        if cutoff is not None:
            df = df[df["week_start"] <= cutoff]
        self.sectors = sorted(df["sector"].unique())
        self.cities = sorted(df["city"].unique())
        table = expand_horizons(build_origin_features(df), df, with_target=True)
        table = table[(table["series_len"] >= config.SEASON) & table["target"].notna()]
        table = add_codes(table, self.sectors, self.cities)
        self.features = feature_columns(table)
        # recent weeks matter more than 2020 lockdown weeks
        age = (table["target_week"].max() - table["target_week"]).dt.days / 365
        weights = np.exp(-0.5 * age)
        self.model = make_regressor()
        self.model.fit(table[self.features], table["target"], sample_weight=weights)
        return self

    def predict_from(self, df: pd.DataFrame, origin: pd.Timestamp | None = None) -> pd.DataFrame:
        """Forecast h=1..4 from the origin week (default: last week in df) for every series."""
        if origin is not None:
            df = df[df["week_start"] <= origin]
        feats = build_origin_features(df)
        last = feats.groupby(["sector", "city"])["week_start"].transform("max")
        feats = feats[(feats["week_start"] == last) & (feats["series_len"] >= config.MIN_HISTORY)]
        if origin is not None:
            feats = feats[feats["week_start"] == origin]
        table = add_codes(expand_horizons(feats, df, with_target=False), self.sectors, self.cities)
        growth = self.model.predict(table[self.features])
        table["forecast"] = np.expm1(table["y_log"] + growth)
        return table[["sector", "city", "week_start", "horizon", "target_week", "forecast"]].rename(
            columns={"week_start": "origin_week"})

    def save(self, path=None):
        path = path or config.ARTIFACTS / "model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=None) -> "Forecaster":
        with open(path or config.ARTIFACTS / "model.pkl", "rb") as f:
            return pickle.load(f)
