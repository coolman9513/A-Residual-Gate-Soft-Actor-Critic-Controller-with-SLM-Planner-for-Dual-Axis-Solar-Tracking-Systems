"""Pre-compute static per-step values for SolarTrackerEnv and save to .npz.

All quantities computed here are functions only of the dataset row (solar
geometry + weather) — they do NOT depend on the agent's panel orientation.
Loading the .npz at training time lets the environment skip ~half the
per-step math (two POA calls, one ideal-orientation solve, time-feature
trig) while remaining fully correct.

Usage
-----
    python finetune/precompute_env_cache.py data/2020_4months_2weeks.csv
    python finetune/precompute_env_cache.py data/2020_4months.csv

The output file is written next to the input CSV with suffix _precomputed.npz.
SolarTrackerEnv auto-detects and loads it via _try_load_step_cache().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent


def _load_panel_and_env_params(schema_path: Path) -> dict:
    """Load panel model and env parameters from schema.json."""
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    pa = schema.get("panel", {}).get("attributes", {})
    bounds = schema.get("panel_angle_bounds", {
        "azimuth": {"low": 70.0, "high": 290.0},
        "tilt": {"low": 10.0, "high": 85.0},
    })
    gen = schema.get("generation", {})
    tw = schema.get("tracking_window", {})
    return {
        "area_m2": float(pa.get("area_m2", 10.0)),
        "efficiency": float(pa.get("efficiency", 0.20)),
        "performance_ratio": float(pa.get("performance_ratio", 0.90)),
        "temperature_coefficient": float(pa.get("temperature_coefficient", 0.004)),
        "reference_temperature_c": float(pa.get("reference_temperature_c", 25.0)),
        "max_power_kw": float(pa.get("max_power_kw", 0.0)),
        "azimuth_low": float(bounds["azimuth"]["low"]),
        "azimuth_high": float(bounds["azimuth"]["high"]),
        "tilt_low": float(bounds["tilt"]["low"]),
        "tilt_high": float(bounds["tilt"]["high"]),
        "dni_min_threshold": float(gen.get("dni_min_threshold", 0.0)),
        "seconds_per_time_step": float(schema.get("seconds_per_time_step", 600.0)),
        "tracking_start_hour": float(tw.get("start_hour", 7.0)),
        "tracking_end_hour": float(tw.get("end_hour", 18.0)),
    }


def precompute(csv_path: Path, params: dict, output_path: Path | None = None) -> Path:
    """Compute static per-step arrays and save to .npz next to the CSV."""

    df = pd.read_csv(csv_path)
    N = len(df)
    print(f"Pre-computing {N} rows from {csv_path.name} ...", flush=True)

    az_low  = params["azimuth_low"]
    az_high = params["azimuth_high"]
    tl_low  = params["tilt_low"]
    tl_high = params["tilt_high"]
    dni_thresh = params["dni_min_threshold"]
    spt = params["seconds_per_time_step"]

    solar_az  = df["Solar Azimuth Angle"].values % 360.0
    solar_zen = df["Solar Zenith Angle"].values
    dni       = df["DNI"].values.astype(float)
    dhi       = df["DHI"].values.astype(float)
    temp      = df["Temperature"].values.astype(float)
    hour      = df["Hour"].values.astype(float)
    minute    = df["Minute"].values.astype(float)

    # ---- Ideal azimuth: closest allowed azimuth to the sun ----
    in_range = (solar_az >= az_low) & (solar_az <= az_high)
    def _circ(a, b):
        return np.abs((a - b + 180.0) % 360.0 - 180.0)
    closer_low = _circ(solar_az, az_low) <= _circ(solar_az, az_high)
    ideal_az = np.where(in_range, solar_az,
                        np.where(closer_low, az_low, az_high)).astype(np.float64)

    # ---- Ideal tilt: POA-maximizing tilt under mechanical bounds ----
    nighttime = (solar_zen >= 90.0) | (dni < dni_thresh)
    theta_z   = np.radians(np.clip(solar_zen, 0.0, 180.0))
    delta_az  = np.radians((solar_az - ideal_az + 180.0) % 360.0 - 180.0)
    a_coef    = dni * np.cos(theta_z) + dhi / 2.0
    b_coef    = dni * np.sin(theta_z) * np.cos(delta_az)
    raw_tilt  = np.degrees(np.arctan2(b_coef, a_coef))
    ideal_tilt = np.clip(raw_tilt, tl_low, tl_high)
    ideal_tilt = np.where(nighttime, tl_low, ideal_tilt)

    # ---- Ideal POA irradiance (vectorised) ----
    ideal_tilt_rad = np.radians(ideal_tilt)
    ideal_az_rad   = np.radians(ideal_az)
    solar_az_rad   = np.radians(solar_az)
    cos_i = (np.cos(theta_z) * np.cos(ideal_tilt_rad)
             + np.sin(theta_z) * np.sin(ideal_tilt_rad) * np.cos(solar_az_rad - ideal_az_rad))
    cos_i = np.maximum(cos_i, 0.0)
    poa_direct  = np.maximum(dni, 0.0) * cos_i
    poa_diffuse = np.maximum(dhi, 0.0) * (1.0 + np.cos(ideal_tilt_rad)) / 2.0
    ideal_poa   = np.maximum(poa_direct + poa_diffuse, 0.0)
    ideal_poa   = np.where(nighttime, 0.0, ideal_poa)
    poa_direct  = np.where(nighttime, 0.0, poa_direct)
    poa_diffuse = np.where(nighttime, 0.0, poa_diffuse)

    # ---- Ideal power and energy ----
    base_kw = (ideal_poa * params["area_m2"] * params["efficiency"]
               * params["performance_ratio"] / 1000.0)
    temp_factor = np.maximum(
        0.0,
        1.0 - params["temperature_coefficient"] * np.maximum(0.0, temp - params["reference_temperature_c"])
    )
    ideal_power_kw = base_kw * temp_factor
    if params["max_power_kw"] > 0.0:
        ideal_power_kw = np.minimum(ideal_power_kw, params["max_power_kw"])
    ideal_power_kw  = np.maximum(ideal_power_kw, 0.0)
    ideal_energy_kwh = ideal_power_kw * spt / 3600.0

    # ---- Cyclic time features ----
    tod_angle = 2.0 * np.pi * (hour * 60.0 + minute) / (24.0 * 60.0)
    if "Datetime" in df.columns:
        doy = pd.to_datetime(df["Datetime"]).dt.dayofyear.values.astype(float)
    else:
        doy = np.ones(N, dtype=float)
    doy_angle = 2.0 * np.pi * (doy - 1.0) / 366.0

    # ---- Tracking window active ----
    time_dec = hour + minute / 60.0
    tracking_window_active = (
        (time_dec >= params["tracking_start_hour"]) &
        (time_dec <  params["tracking_end_hour"])
    )

    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_precomputed.npz")

    np.savez_compressed(
        str(output_path),
        ideal_azimuth           = ideal_az.astype(np.float32),
        ideal_tilt              = ideal_tilt.astype(np.float32),
        ideal_poa_w_per_m2      = ideal_poa.astype(np.float32),
        ideal_poa_direct_w_per_m2  = poa_direct.astype(np.float32),
        ideal_poa_diffuse_w_per_m2 = poa_diffuse.astype(np.float32),
        ideal_power_kw          = ideal_power_kw.astype(np.float32),
        ideal_energy_kwh        = ideal_energy_kwh.astype(np.float32),
        sin_time_of_day         = np.sin(tod_angle).astype(np.float32),
        cos_time_of_day         = np.cos(tod_angle).astype(np.float32),
        sin_day_of_year         = np.sin(doy_angle).astype(np.float32),
        cos_day_of_year         = np.cos(doy_angle).astype(np.float32),
        tracking_window_active  = tracking_window_active,
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved: {output_path.name}  ({size_kb:.0f} KB, {N} rows)", flush=True)
    print(f"  ideal_power_kw  max={ideal_power_kw.max():.4f}  mean={ideal_power_kw[ideal_power_kw>0].mean():.4f}")
    print(f"  tracking_window active steps: {tracking_window_active.sum()} / {N}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute static env cache for SolarTrackerEnv.")
    parser.add_argument("csv_path", help="Path to dataset CSV (e.g. data/2020_4months_2weeks.csv)")
    parser.add_argument("--schema", default=None, help="Path to schema.json (default: st/data/schema.json)")
    parser.add_argument("--output", default=None, help="Output .npz path (default: <csv_stem>_precomputed.npz)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = _ROOT / csv_path
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    schema_path = Path(args.schema) if args.schema else _ROOT / "st" / "data" / "schema.json"
    if not schema_path.exists():
        print(f"Error: schema not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    params = _load_panel_and_env_params(schema_path)
    output_path = Path(args.output) if args.output else None
    precompute(csv_path, params, output_path)


if __name__ == "__main__":
    main()
