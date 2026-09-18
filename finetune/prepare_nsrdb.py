# -*- coding: utf-8 -*-
"""Turn a raw NSRDB download into the CSV the environment reads.

The NSRDB export gives irradiance, weather and solar ZENITH, but no solar
azimuth and no look-ahead columns. This script adds both, producing the file the
simulation and the planner expect (data/2020.csv), and optionally the 56-day
evaluation subset.

Raw NSRDB layout: two metadata header rows, then the column header, then
10-minute records for the whole year.

Derived columns
---------------
    Datetime                    from Year/Month/Day/Hour/Minute
    Solar Azimuth Angle         pvlib solar position at the site, local time
    next_10min_solar_azimuth    one step ahead
    next_10min_solar_zenith     one step ahead
    next_10min_DNI              one step ahead
    next_10min_DHI              one step ahead
    next_30min_average_DNI      mean of the next three steps

The look-ahead columns are taken from the realised series, i.e. an idealised
nowcast. run_forecast_sensitivity.py and run_planner_forecast_sensitivity.py
quantify how much the results depend on that idealisation.

finetune/preprocess_csv.py is a LATER, separate step: it reads the file produced
here and adds noisy 20-60 minute horizons for planner fine-tuning, writing
data/2020_enriched.csv.

Usage
-----
    python finetune/prepare_nsrdb.py "5771095_35.20_126.85_2020.csv"
    python finetune/prepare_nsrdb.py raw.csv --out data/2020.csv --subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Site of the NSRDB grid cell used in the paper (Jeollanam-do, near Gwangju).
LATITUDE = 35.20
LONGITUDE = 126.85
TIMEZONE = "Asia/Seoul"

# Evaluation subset: days 9-22 of four months, one per season.
SUBSET_MONTHS = (1, 4, 7, 10)
SUBSET_DAYS = range(9, 23)

LOOKAHEAD = [
    "next_10min_solar_azimuth",
    "next_10min_solar_zenith",
    "next_10min_DNI",
    "next_10min_DHI",
    "next_30min_average_DNI",
]


def prepare(raw_csv: Path) -> pd.DataFrame:
    import pvlib

    # Two NSRDB metadata rows sit above the real column header.
    df = pd.read_csv(raw_csv, skiprows=2)
    df.columns = [c.strip() for c in df.columns]

    df["Datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.sort_values("Datetime").reset_index(drop=True)

    pos = pvlib.solarposition.get_solarposition(
        time=df["Datetime"].dt.tz_localize(TIMEZONE),
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )
    df["Solar Azimuth Angle"] = pos["azimuth"].values

    df["next_10min_solar_azimuth"] = df["Solar Azimuth Angle"].shift(-1)
    df["next_10min_solar_zenith"] = df["Solar Zenith Angle"].shift(-1)
    df["next_10min_DNI"] = df["DNI"].shift(-1)
    df["next_10min_DHI"] = df["DHI"].shift(-1)
    df["next_30min_average_DNI"] = (
        df["DNI"].shift(-1) + df["DNI"].shift(-2) + df["DNI"].shift(-3)
    ) / 3.0
    # The last rows have no successor; hold the final observation.
    df[LOOKAHEAD] = df[LOOKAHEAD].ffill()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", help="raw NSRDB export, e.g. 5771095_35.20_126.85_2020.csv")
    ap.add_argument("--out", default=str(ROOT / "data" / "2020.csv"))
    ap.add_argument("--subset", action="store_true",
                    help="also write the 56-day evaluation subset")
    ap.add_argument("--subset-out", default=str(ROOT / "data" / "2020_4months_2weeks.csv"))
    args = ap.parse_args()

    df = prepare(Path(args.raw_csv))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("wrote %s  (%d rows, %d columns)" % (out, len(df), len(df.columns)))

    if args.subset:
        sub = df[df["Month"].isin(SUBSET_MONTHS) & df["Day"].isin(SUBSET_DAYS)]
        sub = sub.reset_index(drop=True)
        sub.to_csv(args.subset_out, index=False)
        print("wrote %s  (%d rows = %d days)"
              % (args.subset_out, len(sub), len(sub) // 144))


if __name__ == "__main__":
    main()
