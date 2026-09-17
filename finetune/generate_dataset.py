"""
Generate fine-tuning dataset for Qwen2.5-0.5B solar tracker planner.

Uses data/2020_enriched.csv (run preprocess_csv.py first) which contains
full 10-60 min forecast columns that exactly mirror goal_wrapper._telemetry().

Usage:
    python generate_dataset.py
Outputs:
    finetune/data/train.jsonl
    finetune/data/val.jsonl
    finetune/data/stats.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "2020_enriched.csv"
OUT_DIR  = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ── solar tracker constants ─────────────────────────────────────────────────
AZ_MIN, AZ_MAX     = 70.0, 290.0
TILT_MIN, TILT_MAX = 10.0, 85.0
TRACK_START_H, TRACK_END_H = 7, 18

BUDGET_TIERS = [0.0, 3.0, 6.0, 12.0, 20.0, 30.0]


# ── helpers ───────────────────────────────────────────────────────────────────

def _dni_bucket(max_dni: float) -> str:
    if max_dni >= 700:   return "CLEAR_STRONG"
    if max_dni >= 500:   return "CLEAR_MODERATE"
    if max_dni >= 300:   return "PARTIAL_CLOUD"
    if max_dni >= 100:   return "OVERCAST_DIM"
    return "NIGHT_OR_HEAVY_OVERCAST"


def _physics_budget(max_dni: float) -> float:
    raw = float(np.clip(max_dni * 0.043, 0.0, 30.0))
    return min(BUDGET_TIERS, key=lambda t: abs(t - raw))


def _ideal_az_tilt(solar_az: float, solar_zenith: float) -> tuple[float, float]:
    az   = float(np.clip(solar_az,    AZ_MIN,   AZ_MAX))
    tilt = float(np.clip(90.0 - solar_zenith, TILT_MIN, TILT_MAX))
    return round(az, 1), round(tilt, 1)


def _f(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    return float(v) if pd.notna(v) else default


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    renames = {
        "Solar Azimuth Angle":   "solar_azimuth",
        "Solar Zenith Angle":    "solar_zenith",
        "DNI":                   "dni",
        "DHI":                   "dhi",
        "GHI":                   "ghi",
        "Temperature":           "temperature",
        "Wind Speed":            "wind_speed",
        "Cloud Type":            "cloud_type",
        "Hour":                  "hour",
        "Minute":                "minute",
        "Month":                 "month",
        "Day":                   "day",
        # existing 10-min forecasts
        "next_10min_solar_azimuth": "az10",
        "next_10min_solar_zenith":  "zen10",
        "next_10min_DNI":           "dni10",
        "next_10min_DHI":           "dhi10",
        # enriched forecasts
        "next_20min_solar_azimuth": "az20",
        "next_20min_solar_zenith":  "zen20",
        "next_30min_solar_azimuth": "az30",
        "next_30min_solar_zenith":  "zen30",
        "next_40min_solar_azimuth": "az40",
        "next_40min_solar_zenith":  "zen40",
        "next_50min_solar_azimuth": "az50",
        "next_50min_solar_zenith":  "zen50",
        "next_60min_solar_azimuth": "az60",
        "next_60min_solar_zenith":  "zen60",
        "next_20min_DNI":           "dni20",
        "next_20min_DHI":           "dhi20",
        "next_30min_DNI":           "dni30",
        "next_30min_DHI":           "dhi30",
        "next_40min_DNI":           "dni40",
        "next_40min_DHI":           "dhi40",
        "next_50min_DNI":           "dni50",
        "next_50min_DHI":           "dhi50",
        "next_60min_DNI":           "dni60",
        "next_60min_DHI":           "dhi60",
        "next_30min_average_DNI":   "avg30_dni",
    }
    df = df.rename(columns=renames)

    # numeric coercion
    for col in ["hour", "minute", "dni", "dhi", "solar_zenith", "solar_azimuth",
                "cloud_type", "wind_speed", "temperature",
                "az10", "zen10", "az20", "zen20", "az30", "zen30",
                "az40", "zen40", "az50", "zen50", "az60", "zen60",
                "dni10", "dhi10", "dni20", "dhi20", "dni30", "dhi30",
                "dni40", "dhi40", "dni50", "dhi50", "dni60", "dhi60",
                "avg30_dni"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["dni"]          = df["dni"].clip(lower=0)
    df["dhi"]          = df["dhi"].clip(lower=0)
    df["solar_zenith"] = df["solar_zenith"].replace(0.0, 90.0)
    return df


# ── sample builder ────────────────────────────────────────────────────────────

def _build_sample(row: pd.Series) -> dict | None:
    hour   = int(_f(row, "hour"))
    minute = int(_f(row, "minute"))
    if not (TRACK_START_H <= hour < TRACK_END_H):
        return None

    # current conditions
    dni         = _f(row, "dni")
    dhi         = _f(row, "dhi")
    solar_az    = _f(row, "solar_azimuth", 180.0)
    solar_zen   = _f(row, "solar_zenith",   90.0)
    cloud_type  = int(_f(row, "cloud_type"))
    wind_speed  = _f(row, "wind_speed")
    temperature = _f(row, "temperature", 20.0)

    # 6-step DNI forecast (10–60 min ahead, with noise already baked in)
    dni_forecast = [
        max(0.0, _f(row, "dni10")),
        max(0.0, _f(row, "dni20")),
        max(0.0, _f(row, "dni30")),
        max(0.0, _f(row, "dni40")),
        max(0.0, _f(row, "dni50")),
        max(0.0, _f(row, "dni60")),
    ]
    avg30_dni    = float(np.mean(dni_forecast[:3]))   # first 3 steps = 30 min
    max_fore_dni = max(dni, avg30_dni)

    bucket = _dni_bucket(max_fore_dni)
    budget = _physics_budget(max_fore_dni)

    tracking_window = TRACK_START_H <= hour < TRACK_END_H
    hold = (
        solar_zen >= 90.0
        or not tracking_window
        or (dni < 80.0 and avg30_dni < 80.0 and solar_zen > 75.0)
    )
    if hold:
        budget = min(budget, 3.0)

    # solar-position targets — mirrors goal_wrapper._ideal_orientation_at()
    current_target  = _ideal_az_tilt(solar_az, solar_zen)
    az30, zen30     = _f(row, "az30", solar_az), _f(row, "zen30", solar_zen)
    az60, zen60     = _f(row, "az60", solar_az), _f(row, "zen60", solar_zen)
    future_30_target = _ideal_az_tilt(az30, zen30)
    future_60_target = _ideal_az_tilt(az60, zen60)

    # pre_position uses the target at the end of the planning horizon
    horizon_min = 60 if hour < 16 else 30
    pre_target  = future_60_target if horizon_min == 60 else future_30_target
    pre_az, pre_tilt = pre_target

    target_az, target_tilt = current_target
    confidence = round(float(np.clip(max_fore_dni / 900.0, 0.2, 1.0)), 2)
    time_until_end = max(0.0, (TRACK_END_H * 60.0) - (hour * 60.0 + minute))

    # ── telemetry — identical key layout to goal_wrapper._telemetry() ────
    telemetry = {
        "hour":                              hour,
        "minute":                            minute,
        "solar_azimuth_deg":                 round(solar_az, 2),
        "solar_zenith_deg":                  round(solar_zen, 2),
        "dni":                               round(dni, 1),
        "dhi":                               round(dhi, 1),
        "dni_forecast_next_60min_10min_steps": [round(v, 1) for v in dni_forecast],
        "next_10min_dni":                    round(dni_forecast[0], 1),
        "next_30min_average_dni":            round(avg30_dni, 1),
        "cloud_type":                        cloud_type,
        "wind_speed":                        round(wind_speed, 2),
        "temperature":                       round(temperature, 1),
        "tracking_window_active":            tracking_window,
        "dni_bucket":                        bucket,
        "max_forecast_dni":                  round(max_fore_dni, 1),
        "time_until_tracking_end_min":       round(time_until_end, 1),
        "current_target":       {"azimuth": target_az,  "tilt": target_tilt},
        "future_30min_target":  {"azimuth": future_30_target[0], "tilt": future_30_target[1]},
        "future_target":        {"azimuth": future_60_target[0], "tilt": future_60_target[1]},
    }

    goal = {
        "target_tilt":       target_tilt  if not hold else round(float(np.clip(45.0, TILT_MIN, TILT_MAX)), 1),
        "target_az":         target_az    if not hold else 180.0,
        "motion_budget_deg": budget,
        "pre_position_tilt": pre_tilt     if not hold else target_tilt,
        "pre_position_az":   pre_az       if not hold else target_az,
        "hold":              hold,
        "horizon_min":       horizon_min,
        "confidence":        confidence,
    }

    return {"telemetry": telemetry, "goal": goal}


# ── prompt builder ─────────────────────────────────────────────────────────────

def _to_chat_messages(sample: dict) -> list[dict]:
    telemetry_str = json.dumps(sample["telemetry"], separators=(", ", ": "))
    goal_str      = json.dumps(sample["goal"])
    bucket        = sample["telemetry"]["dni_bucket"]
    max_dni       = sample["telemetry"]["max_forecast_dni"]

    reasoning = (
        "Hold: low sun or night, budget=0-3."
        if sample["goal"]["hold"]
        else f"Bucket {bucket}, max_DNI={max_dni:.0f}, budget={sample['goal']['motion_budget_deg']:.0f}."
    )

    system = (
        "You are a compact solar-tracker planner. "
        "Use the dni_bucket label and sun position to choose a strategy and motion budget. "
        "Return one short Reasoning line (max 20 words), then exactly one valid JSON object. "
        "Do not write step-by-step analysis."
    )

    user = f"""Plan the next 30-60 simulated minutes for a dual-axis solar tracker.

Decision steps — apply in order, stop at first match:
1. solar_zenith_deg >= 90 OR tracking_window_active is false  -> hold=true, motion_budget_deg=0
2. NIGHT_OR_HEAVY_OVERCAST, OR (DNI<80 AND next_30min_average_dni<80 AND zenith>75) -> hold=true, motion_budget_deg=3
3. CLEAR_STRONG  (max_forecast_dni>=700) -> hold=false, motion_budget_deg=30
4. CLEAR_MODERATE(max_forecast_dni>=500) -> hold=false, motion_budget_deg=20
5. PARTIAL_CLOUD (max_forecast_dni>=300) -> hold=false, motion_budget_deg=12
6. OVERCAST_DIM  (max_forecast_dni>=100) -> hold=false, motion_budget_deg=6
7. tracking_window_active=true (fallback)-> hold=false, motion_budget_deg=3

motion_budget_deg must be one of: 0, 3, 6, 12, 20, 30.
pre_position_tilt/az = future_30min_target when horizon_min=30, future_target when horizon_min=60.
Telemetry: {telemetry_str}

One Reasoning line (max 20 words), then JSON:
{{"target_tilt":<deg>,"target_az":<deg>,"motion_budget_deg":<value>,"pre_position_tilt":<deg>,"pre_position_az":<deg>,"hold":<true|false>,"horizon_min":<30|60>,"confidence":<0-1>}}"""

    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": f"{reasoning}\n{goal_str}"},
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def generate(val_fraction: float = 0.1, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Run preprocess_csv.py first. Expected: {CSV_PATH}"
        )

    print(f"Loading {CSV_PATH}...", end=" ", flush=True)
    df = _load_csv(CSV_PATH)
    print(f"{len(df)} rows")

    # Oversample clear-sky buckets to correct the 86% night/cloud imbalance.
    # CLEAR_STRONG (DNI>=700) appears in only 7.9% of raw samples; these are
    # the most important for aggressive tracking and Qwen's hold=False decisions.
    OVERSAMPLE = {"CLEAR_STRONG": 3, "CLEAR_MODERATE": 2}

    all_samples: list[dict] = []
    for i in range(len(df)):
        row = df.iloc[i]
        sample = _build_sample(row)
        if sample is not None:
            n = OVERSAMPLE.get(sample["telemetry"]["dni_bucket"], 1)
            all_samples.extend([sample] * n)

    print(f"\nTotal samples after oversampling: {len(all_samples)}")
    budgets = [s["goal"]["motion_budget_deg"] for s in all_samples]
    for b in BUDGET_TIERS:
        count = budgets.count(b)
        print(f"  budget={b:4.0f}: {count:5d}  ({count / len(budgets) * 100:.1f}%)")

    random.shuffle(all_samples)
    n_val         = int(len(all_samples) * val_fraction)
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        out_path = OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({"messages": _to_chat_messages(s)}) + "\n")
        print(f"Wrote {len(samples)} samples -> {out_path}")

    stats = {
        "total":              len(all_samples),
        "train":              len(train_samples),
        "val":                len(val_samples),
        "source":             str(CSV_PATH),
        "budget_distribution": {str(b): budgets.count(b) for b in BUDGET_TIERS},
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved -> {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    generate()
