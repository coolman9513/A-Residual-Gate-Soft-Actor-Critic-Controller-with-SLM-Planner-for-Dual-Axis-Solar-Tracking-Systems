"""Evaluation metrics in a CityLearn-like tabular format."""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return a safe scalar division result."""

    denominator = float(denominator)
    if abs(denominator) < 1e-12:
        return default

    return float(numerator) / denominator


def _day_index(history: pd.DataFrame, seconds_per_time_step: float) -> pd.Series:
    """Return deterministic episode-local day index."""

    steps_per_day = max(int(round(24 * 3600 / float(seconds_per_time_step))), 1)
    first_step = float(history["time_step"].min()) if "time_step" in history else 0.0
    return np.floor((history["time_step"].astype(float) - first_step) / steps_per_day).astype(int)


def _weather_condition(cloud_type: float) -> str:
    """Map NREL-style numeric cloud type to coarse weather categories."""

    if pd.isna(cloud_type):
        return "unknown"
    if cloud_type <= 1:
        return "clear"
    if cloud_type <= 4:
        return "partly_cloudy"
    return "overcast"


def summarize_tracker_metrics(
    history: pd.DataFrame,
    seconds_per_time_step: float,
    rated_power_kw: float,
    fixed_energy_kwh: Optional[float] = None,
    rule_energy_kwh: Optional[float] = None,
    movement_threshold_deg: float = 0.1,
) -> Dict[str, float]:
    """Return key tracker metrics from a simulation history.

    Implemented metric groups:
    - Energy: daily yield, tracking efficiency, tracking gain, capacity use,
      cosine loss, and POA irradiance.
    - Operation: jerk-based smoothness, adjustment count, movement efficiency.
    - Robustness: coarse weather-conditioned tracking efficiency and monthly
      seasonal consistency.

    Motor-energy, safety-compliance, and recovery metrics are intentionally not
    implemented yet because they require extra physical assumptions.
    """

    if history.empty:
        return {
            "total_harvested_energy_kwh": 0.0,
            "average_energy_per_day_kwh": 0.0,
            "average_daily_generated_energy_kwh": 0.0,
            "daily_energy_yield_mean_kwh": 0.0,
            "tracking_efficiency": 0.0,
            "tracking_gain_over_fixed_panel_pct": 0.0,
            "capacity_factor": 0.0,
            "capacity_utilization_ratio": 0.0,
            "cosine_loss": 0.0,
            "mean_poa_irradiance_w_per_m2": 0.0,
            "movement_magnitude_deg": 0.0,
            "action_smoothness_deg": 0.0,
            "tracking_smoothness_jerk_deg2": 0.0,
            "number_of_adjustments": 0.0,
            "movement_efficiency_kwh_per_degree_vs_fixed": 0.0,
            "mean_tracking_error_deg": 0.0,
            "weather_te_clear": 0.0,
            "weather_te_partly_cloudy": 0.0,
            "weather_te_overcast": 0.0,
            "seasonal_consistency_monthly_te_std": 0.0,
            "improvement_over_fixed_panel": 0.0,
            "improvement_over_rule_based": 0.0,
        }

    total_energy = float(history["generated_energy_kwh"].sum())
    total_hours = len(history) * float(seconds_per_time_step) / 3600.0
    total_days = max(total_hours / 24.0, 1e-9)
    avg_day_energy = total_energy / total_days

    day_index = _day_index(history, seconds_per_time_step)
    daily_energy = history.groupby(day_index)["generated_energy_kwh"].sum()
    daily_energy_mean = float(daily_energy.mean()) if len(daily_energy) else 0.0

    ideal_energy = float(history.get("ideal_generated_energy_kwh", pd.Series(dtype=float)).sum())
    tracking_efficiency = _safe_divide(total_energy, ideal_energy)

    movement_series = np.abs(history["panel_azimuth_delta_deg"]) + np.abs(history["panel_tilt_delta_deg"])
    movement = float(movement_series.sum())
    action_smoothness = float(movement_series.mean())

    action_values = history[["panel_azimuth_deg", "panel_tilt_deg"]].astype(float).to_numpy()
    if len(action_values) >= 4:
        jerk = np.diff(action_values, n=3, axis=0)
        tracking_smoothness_jerk = float(np.mean(np.sum(jerk**2, axis=1)))
    else:
        tracking_smoothness_jerk = 0.0

    number_of_adjustments = float((movement_series > float(movement_threshold_deg)).sum())
    mean_tracking_error = float(history["tracking_error_deg"].mean())
    capacity_factor = total_energy / max(rated_power_kw * total_hours, 1e-9)
    sun_hours = float((history["ideal_poa_w_per_m2"] > 0.0).sum()) * float(seconds_per_time_step) / 3600.0
    capacity_utilization_ratio = total_energy / max(rated_power_kw * sun_hours, 1e-9)

    ideal_direct = float(history.get("ideal_poa_direct_w_per_m2", pd.Series(dtype=float)).sum())
    actual_direct = float(history.get("poa_direct_w_per_m2", pd.Series(dtype=float)).sum())
    cosine_loss = max(0.0, 1.0 - _safe_divide(actual_direct, ideal_direct, default=1.0))
    mean_poa = float(history["poa_w_per_m2"].mean())

    improvement_fixed = 0.0 if fixed_energy_kwh is None else (total_energy - fixed_energy_kwh) / max(fixed_energy_kwh, 1e-9)
    improvement_rule = 0.0 if rule_energy_kwh is None else (total_energy - rule_energy_kwh) / max(rule_energy_kwh, 1e-9)
    tracking_gain_pct = improvement_fixed * 100.0
    extra_energy_vs_fixed = 0.0 if fixed_energy_kwh is None else total_energy - fixed_energy_kwh
    movement_efficiency = _safe_divide(extra_energy_vs_fixed, movement)

    metrics = {
        "total_harvested_energy_kwh": total_energy,
        "average_energy_per_day_kwh": avg_day_energy,
        "average_daily_generated_energy_kwh": daily_energy_mean,
        "daily_energy_yield_mean_kwh": daily_energy_mean,
        "daily_energy_yield_min_kwh": float(daily_energy.min()) if len(daily_energy) else 0.0,
        "daily_energy_yield_max_kwh": float(daily_energy.max()) if len(daily_energy) else 0.0,
        "tracking_efficiency": tracking_efficiency,
        "tracking_gain_over_fixed_panel_pct": tracking_gain_pct,
        "capacity_factor": capacity_factor,
        "capacity_utilization_ratio": capacity_utilization_ratio,
        "cosine_loss": cosine_loss,
        "mean_poa_irradiance_w_per_m2": mean_poa,
        "movement_magnitude_deg": movement,
        "action_smoothness_deg": action_smoothness,
        "tracking_smoothness_jerk_deg2": tracking_smoothness_jerk,
        "number_of_adjustments": number_of_adjustments,
        "movement_efficiency_kwh_per_degree_vs_fixed": movement_efficiency,
        "mean_tracking_error_deg": mean_tracking_error,
        "improvement_over_fixed_panel": improvement_fixed,
        "improvement_over_rule_based": improvement_rule,
    }

    if "cloud_type" in history.columns:
        weather = history.copy()
        weather["weather_condition"] = weather["cloud_type"].map(_weather_condition)
        for condition in ["clear", "partly_cloudy", "overcast"]:
            subset = weather[weather["weather_condition"] == condition]
            metrics[f"weather_te_{condition}"] = _safe_divide(
                float(subset["generated_energy_kwh"].sum()),
                float(subset["ideal_generated_energy_kwh"].sum()),
            )

    if "month" in history.columns:
        monthly = history.groupby(history["month"].astype(int)).agg(
            actual=("generated_energy_kwh", "sum"),
            ideal=("ideal_generated_energy_kwh", "sum"),
        )
        monthly_te = monthly.apply(lambda row: _safe_divide(row["actual"], row["ideal"]), axis=1)
        metrics["seasonal_consistency_monthly_te_std"] = float(monthly_te.std(ddof=0)) if len(monthly_te) else 0.0

    return metrics


def metrics_to_dataframe(metrics: Dict[str, float], name: str = "Tracker") -> pd.DataFrame:
    """Convert metric dictionary to CityLearn-like evaluation dataframe."""

    rows = [{"cost_function": key, "value": value, "name": name, "level": "tracker"} for key, value in metrics.items()]
    return pd.DataFrame(rows)
