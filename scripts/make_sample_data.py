"""Create a fake dataset with the same columns as the KAPSARC file.
Only for testing the pipeline before you download the real data. Do not report results from it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posforecast.calendar_sa import week_flags  # noqa: E402

rng = np.random.default_rng(0)
sectors = {"Restaurants and Cafes": 900, "Clothing and Footwear": 500, "Food and Beverages": 1200,
           "Gas Stations": 400, "Other": 300, "Others": 80}
cities = {"Riyadh": 1.0, "Jeddah": 0.7, "Dammam": 0.4, "Makkah": 0.3}
weeks = pd.date_range("2021-01-03", "2026-10-25", freq="7D")  # Sundays

rows = []
for s, base in sectors.items():
    for c, mult in cities.items():
        level = base * mult
        for i, w in enumerate(weeks):
            f = week_flags(w)
            trend = 1 + 0.0015 * i
            effect = 1 + 0.04 * f["salary_week"] + 0.03 * f["ramadan_days"] / 7 \
                + 0.25 * f["pre_eid_days"] / 7 * (s == "Clothing and Footwear") \
                - 0.15 * f["eid_fitr_days"] / 4 + 0.05 * f["national_day"]
            for kind, scale in (("Value of Transactions", 1.0), ("Number of Transactions", 0.01)):
                v = level * trend * effect * scale * rng.lognormal(0, 0.04)
                rows.append({"starting_date": w.date(), "sectors": s, "city": c,
                             "unit": kind, "value": round(v, 2)})

out = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_pos.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out} ({len(rows)} rows)")
