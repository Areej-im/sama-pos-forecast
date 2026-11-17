"""Streamlit dashboard: forecast vs actual, and how the error moves week by week.

Run: streamlit run dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="POS spend forecast", layout="wide")


@st.cache_data(ttl=3600)
def load(name, dates=()):
    p = ROOT / name
    return pd.read_csv(p, parse_dates=list(dates)) if p.exists() else pd.DataFrame()


hist = load("data/processed/pos_weekly.csv", ["week_start"])
fc = load("artifacts/forecasts.csv", ["origin_week", "target_week"])
log = load("results/forecast_log.csv", ["origin_week", "target_week"])
mon = load("results/monitoring.csv", ["target_week"])
bt = load("results/backtest_summary.csv")

st.title("Weekly POS spending forecast")
st.caption("Source: SAMA weekly point-of-sale reports (via KAPSARC). Forecast 1-4 weeks ahead by sector and city.")

if hist.empty or fc.empty:
    st.warning("No data yet. Run `python -m posforecast.cli weekly` first.")
    st.stop()

c1, c2, c3 = st.columns(3)
sector = c1.selectbox("Sector", sorted(fc.sector.unique()))
city = c2.selectbox("City", sorted(fc[fc.sector == sector].city.unique()))
weeks = c3.slider("Weeks of history", 12, 156, 52)

h = hist[(hist.sector == sector) & (hist.city == city)].tail(weeks)
f = fc[(fc.sector == sector) & (fc.city == city)]
past = log[(log.sector == sector) & (log.city == city) & (log.horizon == 1)]

chart = pd.concat([
    h.set_index("week_start")["value"].rename("actual"),
    past.set_index("target_week")["forecast"].rename("past forecast (1 week ahead)"),
    f.set_index("target_week")["forecast"].rename("next 4 weeks"),
], axis=1)
st.line_chart(chart)

st.subheader("Next 4 weeks")
st.dataframe(f[["horizon", "target_week", "forecast"]].assign(target_week=f.target_week.dt.date),
             hide_index=True, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Backtest (MAPE %)")
    if not bt.empty:
        st.dataframe(bt[["sector", "model_mape", "baseline_mape", "improvement_pts"]], hide_index=True)
with right:
    st.subheader("Live error since launch")
    if mon.empty:
        st.info("Live error appears after the first week of new SAMA data.")
    else:
        st.line_chart(mon.pivot_table(index="target_week", columns="horizon", values="mape"))
