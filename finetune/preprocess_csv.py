"""
Enrich data/2020.csv with forecast columns for fine-tuning.

Adds shifted "next N-min" features that mirror what goal_wrapper._telemetry()
provides at inference time:

  Solar angles (no noise — astronomically exact):
    next_20min / 30min / 40min / 50min / 60min solar_azimuth & solar_zenith

  DNI / DHI (±2 % uniform noise — simulates real forecast uncertainty):
    next_20min / 30min / 40min / 50min / 60min DNI & DHI

Saves to data/2020_enriched.csv (leaves original untouched).

Usage:
    python preprocess_csv.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
CSV_IN   = ROOT / "data" / "2020.csv"
CSV_OUT  = ROOT / "data" / "2020_enriched.csv"

NOISE_PCT = 0.02   # ±2 % uniform noise on DNI/DHI forecasts
SEED      = 42
STEP_LAGS = [(1, "10min"), (2, "20min"), (3, "30min"),
             (4, "40min"), (5, "50min"), (6, "60min")]


def _noisy(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Multiply by (1 + ε), ε ~ Uniform(-NOISE_PCT, NOISE_PCT), clip ≥ 0."""
    eps = rng.uniform(-NOISE_PCT, NOISE_PCT, size=len(series))
    return (series * (1.0 + eps)).clip(lower=0.0).round(1)


def main() -> None:
    rng = np.random.default_rng(SEED)

    print(f"Loading {CSV_IN} ...", end=" ", flush=True)
    df = pd.read_csv(CSV_IN)
    print(f"{len(df)} rows, {len(df.columns)} existing columns")

    # ── solar angles: exact shift, fill tail with last known value ────────
    for steps, label in STEP_LAGS:
        col_az  = f"next_{label}_solar_azimuth"
        col_zen = f"next_{label}_solar_zenith"
        if col_az not in df.columns:
            df[col_az]  = df["Solar Azimuth Angle"].shift(-steps).ffill()
        if col_zen not in df.columns:
            df[col_zen] = df["Solar Zenith Angle"].shift(-steps).ffill()

    # ── DNI / DHI: shift + noise, fill tail with 0 (nighttime default) ───
    for steps, label in STEP_LAGS:
        col_dni = f"next_{label}_DNI"
        col_dhi = f"next_{label}_DHI"
        if col_dni not in df.columns:
            df[col_dni] = _noisy(df["DNI"].shift(-steps).fillna(0.0), rng)
        if col_dhi not in df.columns:
            df[col_dhi] = _noisy(df["DHI"].shift(-steps).fillna(0.0), rng)

    # ── recompute next_30min_average_DNI from noisy forecast ─────────────
    df["next_30min_average_DNI"] = (
        df[["next_10min_DNI", "next_20min_DNI", "next_30min_DNI"]].mean(axis=1).round(1)
    )

    print(f"\nNew columns added:")
    new_cols = [c for c in df.columns if c not in pd.read_csv(CSV_IN, nrows=0).columns]
    for c in new_cols:
        print(f"  {c}")

    print(f"\nWriting {CSV_OUT} ...", end=" ", flush=True)
    df.to_csv(CSV_OUT, index=False)
    print("done.")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
