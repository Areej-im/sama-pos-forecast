# Saudi POS spend forecast

Every week SAMA publishes how much was spent through point-of-sale terminals in Saudi Arabia, by sector and by city. This project forecasts the next four weeks for each series, serves the forecast through an API and a dashboard, and includes a weekly job that grades past forecasts against new data.

**Dashboard:** `<add link>` · **API docs:** `<add link>/docs`

## Data

SAMA weekly POS reports, from the KAPSARC data portal. May 2020 to July 2025 (the portal dataset has not been updated since July 2025).

29 weekly series: 17 sectors at national level, and 12 cities with all sectors combined. Sector-by-city detail is not in this dataset. Values are in thousand SAR.

## The problem

Spending in Saudi Arabia follows two calendars. Salaries arrive on the 27th of each Gregorian month, while Ramadan and the two Eids follow the Hijri calendar and move about 11 days earlier every year. A model that only knows "week of the year" misses the biggest swings.

Example: the highest food & beverage week in the last year is the week of 23 Feb 2025, the week before Ramadan started.

## Results

Rolling-origin backtest over the last year of data (July 2024 to July 2025): 26 forecast origins, 1 to 4 weeks ahead, 3,016 forecasts. This period includes Ramadan, Eid al-Fitr and Eid al-Adha.

| | MAPE | WAPE |
|---|---|---|
| Seasonal naive (same week last year) | 16.8% | 12.0% |
| LightGBM | **9.2%** | **6.4%** |

Selected series (MAPE):

| Series | Model | Baseline |
|---|---|---|
| All cities | 6.3% | 11.1% |
| Electronics | 8.7% | 30.5% |
| Clothing and footwear | 12.2% | 24.1% |
| Gas stations | 4.1% | 12.1% |
| Restaurants and cafés | 6.9% | 8.8% |

Full table: `results/backtest_summary.csv`. Error per week: `results/backtest_by_week.csv`.

### Where it still struggles

- **Education** (53.6% vs 45.1% baseline). Spending is concentrated in a few back-to-school weeks whose dates change every year. The model has no school-calendar feature yet.
- **Hotels** (12.6% vs 8.8%). The same-week-last-year pattern is stronger than what the model learns here. For now the baseline is the better choice for these two sectors.
- **Public utilities**: the baseline error is 85% because the series changed level between years. The model (11.5%) handles this because it predicts growth from recent weeks, not the raw level.

## How it works

1. **Data.** `scripts/download_data.py` downloads the dataset. `posforecast/data.py` keeps the transaction value rows (not counts or % change), merges sector names written two ways, and puts every series on a regular weekly grid.
2. **Baseline.** Same week last year. The model has to beat this.
3. **Model.** One LightGBM model for all 29 series. It predicts growth from the last known week (log difference), so large and small series can share one model. Features: recent weeks, rolling averages, same week last year, Ramadan / pre-Eid / Eid days, salary week, national days, series id.
4. **Evaluation.** Rolling origin: train up to week t, forecast t+1 to t+4, compare, move t forward. A random split would leak future weeks into training.
5. **Serving.** FastAPI serves the latest forecast. Streamlit shows actual vs forecast and the backtest.
6. **Monitoring.** A GitHub Actions job runs every Tuesday: downloads the data, grades past forecasts, issues a new one and commits the results.

**Limitation:** since the portal dataset stopped in July 2025, the weekly job currently finds no new weeks. Next step is to read SAMA's weekly PDF reports directly.

## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python scripts/download_data.py
python -m posforecast.cli prepare
python -m posforecast.cli backtest
python -m posforecast.cli forecast

python -m uvicorn api.main:app --reload      # http://127.0.0.1:8000/docs
python -m streamlit run dashboard/app.py
```

API:

```
GET /forecast?sector=Total&city=Riyadh&horizon=4
GET /series
GET /accuracy
```

## Project layout

```
src/posforecast/   cleaning, calendar, features, model, backtest, weekly jobs
api/               FastAPI
dashboard/         Streamlit
scripts/           data download
tests/             pytest
.github/workflows/ weekly job + tests
```

Hijri event dates are in `src/posforecast/calendar_sa.py` and need a yearly check.