"""Command line entry point.

    python -m posforecast.cli prepare      # raw file -> clean weekly table
    python -m posforecast.cli backtest     # rolling backtest vs seasonal-naive
    python -m posforecast.cli forecast     # train on all data, forecast next 4 weeks
    python -m posforecast.cli weekly       # prepare + monitor + forecast (used by GitHub Actions)
"""
import argparse
from pathlib import Path

import pandas as pd

from . import data, backtest, pipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="posforecast")
    p.add_argument("command", choices=["prepare", "backtest", "forecast", "monitor", "weekly"])
    p.add_argument("--raw", type=Path, help="path to raw CSV/JSON (default: newest file in data/raw)")
    p.add_argument("--origins", type=int, default=None, help="number of backtest origins")
    a = p.parse_args(argv)
    pd.set_option("display.width", 140)

    if a.command in ("prepare", "weekly"):
        df = data.prepare(a.raw)
        print(f"{len(df):,} rows | {df.groupby(['sector','city']).ngroups} series | "
              f"{df.week_start.min().date()} -> {df.week_start.max().date()}")
    else:
        df = data.load_processed()

    if a.command == "backtest":
        summary, _ = backtest.run(df, n=a.origins or backtest.config.BACKTEST_ORIGINS)
        print(summary.round(2).to_string(index=False))
    if a.command in ("monitor", "weekly"):
        m = pipeline.monitor(df)
        print(m.tail(8).to_string(index=False) if not m.empty else "no forecasts to grade yet")
    if a.command in ("forecast", "weekly"):
        fc = pipeline.forecast(df)
        print(f"forecast written: {len(fc)} rows, target weeks {fc.target_week.min().date()} -> {fc.target_week.max().date()}")


if __name__ == "__main__":
    main()
