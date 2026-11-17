"""Central settings: paths and modelling constants. Change values here, not inside the code."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("POS_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "pos_weekly.csv"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

FORECASTS_FILE = ARTIFACTS / "forecasts.csv"          # latest 4-week forecast (served by the API)
FORECAST_LOG = RESULTS / "forecast_log.csv"           # every forecast ever issued (append-only)
MONITORING_FILE = RESULTS / "monitoring.csv"          # forecast vs actual, filled week by week
BACKTEST_FILE = RESULTS / "backtest_summary.csv"
BACKTEST_DETAIL = RESULTS / "backtest_detail.csv"
BACKTEST_BY_WEEK = RESULTS / "backtest_by_week.csv"     # error per week: shows where the model fails

KAPSARC_DATASET = "point-of-sale-transactions-by-sector-and-city"
KAPSARC_CSV_URL = (
    "https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/"
    f"{KAPSARC_DATASET}/exports/csv?delimiter=%2C"
)

HORIZONS = [1, 2, 3, 4]        # weeks ahead
SEASON = 52                    # weeks in a year (seasonal-naive baseline)
LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINDOWS = [4, 8, 13]
MIN_HISTORY = 60               # a series needs this many weeks before we model it
BACKTEST_ORIGINS = 26          # 26 origins x 2 weeks = one year, so Ramadan and both Eids are included
BACKTEST_STEP = 2              # weeks between origins
RANDOM_STATE = 42
