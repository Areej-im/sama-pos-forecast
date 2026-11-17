"""Saudi calendar features: Hijri events (Ramadan, Eids), salary week, national days.

Hijri dates move ~11 days earlier every Gregorian year, so a model that only knows
"week of year" cannot learn them. We hard-code the Gregorian dates of each event.
Dates are the commonly announced Umm al-Qura dates; the actual start can shift by one
day after moon sighting. Verify and extend this table every year (see GUIDE).
"""
from __future__ import annotations
import pandas as pd

# (Ramadan first day, Eid al-Fitr day, Eid al-Adha day)
HIJRI_EVENTS = {
    2019: ("2019-05-06", "2019-06-04", "2019-08-11"),
    2020: ("2020-04-24", "2020-05-24", "2020-07-31"),
    2021: ("2021-04-13", "2021-05-13", "2021-07-20"),
    2022: ("2022-04-02", "2022-05-02", "2022-07-09"),
    2023: ("2023-03-23", "2023-04-21", "2023-06-28"),
    2024: ("2024-03-11", "2024-04-10", "2024-06-16"),
    2025: ("2025-03-01", "2025-03-30", "2025-06-06"),
    2026: ("2026-02-18", "2026-03-20", "2026-05-27"),
    2027: ("2027-02-08", "2027-03-10", "2027-05-16"),  # estimated - confirm when announced
}
SALARY_DAY = 27          # government salaries are paid on the 27th of each Gregorian month
EID_WINDOW_DAYS = 4      # holiday spending window around each Eid


def _days(start: str, n: int) -> set[pd.Timestamp]:
    s = pd.Timestamp(start)
    return {s + pd.Timedelta(days=i) for i in range(n)}


def _build_day_sets():
    ramadan, last10, fitr, adha, pre_eid = set(), set(), set(), set(), set()
    for r_start, fitr_day, adha_day in HIJRI_EVENTS.values():
        r = pd.Timestamp(r_start)
        f = pd.Timestamp(fitr_day)
        n = (f - r).days
        ramadan |= _days(r_start, n)
        last10 |= {f - pd.Timedelta(days=i) for i in range(1, 11)}
        fitr |= _days(fitr_day, EID_WINDOW_DAYS)
        adha |= _days(adha_day, EID_WINDOW_DAYS)
        pre_eid |= {f - pd.Timedelta(days=i) for i in range(1, 8)}
        pre_eid |= {pd.Timestamp(adha_day) - pd.Timedelta(days=i) for i in range(1, 8)}
    return ramadan, last10, fitr, adha, pre_eid


RAMADAN, RAMADAN_LAST10, EID_FITR, EID_ADHA, PRE_EID = _build_day_sets()


def week_flags(week_start: pd.Timestamp) -> dict:
    """Calendar features for the 7-day week that starts on `week_start` (SAMA weeks run Sun-Sat)."""
    days = [week_start + pd.Timedelta(days=i) for i in range(7)]
    count = lambda s: sum(d in s for d in days)
    return {
        "woy": int(week_start.isocalendar().week),
        "month": week_start.month,
        "ramadan_days": count(RAMADAN),
        "ramadan_last10_days": count(RAMADAN_LAST10),
        "pre_eid_days": count(PRE_EID),
        "eid_fitr_days": count(EID_FITR),
        "eid_adha_days": count(EID_ADHA),
        "salary_week": int(any(d.day == SALARY_DAY for d in days)),
        "post_salary_week": int(any(d.day in (28, 29, 30, 31, 1, 2, 3) for d in days)),
        "national_day": int(any(d.month == 9 and d.day == 23 for d in days)),
        "founding_day": int(any(d.month == 2 and d.day == 22 and d.year >= 2022 for d in days)),
    }


def calendar_table(week_starts) -> pd.DataFrame:
    ws = pd.to_datetime(pd.Series(sorted(set(week_starts))))
    rows = [{"week_start": w, **week_flags(w)} for w in ws]
    return pd.DataFrame(rows)
