"""Download the weekly POS dataset from the KAPSARC data portal into data/raw/.

Usage: python scripts/download_data.py
If the URL changes, open the dataset page on data.kapsarc.org, click Export -> CSV,
and save the file into data/raw/ yourself.
"""
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posforecast import config  # noqa: E402


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RAW_DIR / f"pos_{date.today():%Y%m%d}.csv"
    r = requests.get(config.KAPSARC_CSV_URL, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"saved {out} ({len(r.content) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
