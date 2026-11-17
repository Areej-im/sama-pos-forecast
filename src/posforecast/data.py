"""Load the raw SAMA/KAPSARC file and turn it into one clean weekly table:
week_start | sector | city | value
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd

from . import config

# Some sectors appear with two spellings in the source. Add to this map when you find more.
SECTOR_ALIASES = {
    "miscellaneous goods & services": "Miscellaneous Goods and Services",
    "restaurants & cafes": "Restaurants and Cafes",
    "restaurants and café": "Restaurants and Cafes",
    "restaurants and cafes": "Restaurants and Cafes",
    "beverages and food": "Food and Beverages",
    "food & beverages": "Food and Beverages",
    "food and beverages": "Food and Beverages",
    "clothing & footwear": "Clothing and Footwear",
    "clothing and footwear": "Clothing and Footwear",
    "gas stations": "Gas Stations",
    "fuel stations": "Gas Stations",
}

COLUMN_HINTS = {
    "week_start": ["starting_date", "start_date", "week_start", "date", "week"],
    "sector": ["sectors", "sector", "category", "name"],
    "city": ["city", "region"],
    "value": ["value", "amount", "sales"],
}
MEASURE_HINTS = ["unit", "measure", "indicator", "type", "transactions_sales"]


def read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):  # API style: {"results": [...]} or {"records": [{"fields": {...}}]}
            data = data.get("results") or [r.get("fields", r) for r in data.get("records", [])]
        return pd.DataFrame(data)
    # KAPSARC exports use ';' by default, our download script uses ','. Let pandas sniff it.
    return pd.read_csv(path, sep=None, engine="python")


def _clean_name(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def detect_columns(df: pd.DataFrame) -> dict:
    cols = {_clean_name(c): c for c in df.columns}
    found = {}
    for target, hints in COLUMN_HINTS.items():
        for h in hints:
            match = next((orig for clean, orig in cols.items()
                          if (clean == h or clean.startswith(h)) and orig not in found.values()), None)
            if match is not None:
                found[target] = match
                break
    missing = set(COLUMN_HINTS) - set(found)
    if missing:
        raise ValueError(f"Could not find columns {missing}. Columns in file: {list(df.columns)}. "
                         "Rename them or add a hint in COLUMN_HINTS.")
    for clean, orig in cols.items():
        if any(h in clean for h in MEASURE_HINTS) and orig not in found.values():
            found["measure"] = orig
            break
    return found


def normalize_sector(name: str) -> str:
    key = re.sub(r"\s+", " ", str(name).strip()).lower()
    return SECTOR_ALIASES.get(key, re.sub(r"\s+", " ", str(name).strip()).title())


def clean(df: pd.DataFrame, measure_keyword: str = "value") -> pd.DataFrame:
    cols = detect_columns(df)
    if "measure" in cols:
        # The file mixes "number of transactions" and "value of transactions". Keep the value rows.
        m = df[cols["measure"]].astype(str).str.lower()
        keep = m.str.contains(measure_keyword) & ~m.str.contains("change|%")
        if keep.any():
            df = df[keep]

    out = pd.DataFrame({
        "week_start": pd.to_datetime(df[cols["week_start"]], errors="coerce"),
        "sector": df[cols["sector"]].map(normalize_sector),
        "city": df[cols["city"]].astype(str).str.strip().str.title(),
        "value": pd.to_numeric(df[cols["value"]], errors="coerce"),
    }).dropna()
    out["week_start"] = out["week_start"].dt.normalize()

    # Merged sector names can create duplicates for the same week; add them up.
    out = out.groupby(["week_start", "sector", "city"], as_index=False)["value"].sum()
    out = out[out["value"] > 0]
    return out.sort_values(["sector", "city", "week_start"]).reset_index(drop=True)


def fill_weekly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Make every series have one row per week. Short gaps (<=2 weeks) are interpolated, longer stay empty."""
    parts = []
    for (s, c), g in df.groupby(["sector", "city"]):
        g = g.set_index("week_start").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="7D")
        g = g.reindex(idx)
        g["value"] = g["value"].interpolate(limit=2, limit_area="inside")
        g["sector"], g["city"] = s, c
        parts.append(g.rename_axis("week_start").reset_index())
    return pd.concat(parts, ignore_index=True)


def prepare(raw_path: Path | None = None) -> pd.DataFrame:
    if raw_path is None:
        files = sorted(config.RAW_DIR.glob("*.csv")) + sorted(config.RAW_DIR.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No data in {config.RAW_DIR}. Run scripts/download_data.py first.")
        raw_path = files[-1]
    df = fill_weekly_grid(clean(read_raw(raw_path)))
    config.PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED, index=False)
    return df


def load_processed() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED, parse_dates=["week_start"])
