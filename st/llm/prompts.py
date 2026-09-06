"""Prompt builders for Mode-B context and Mode-A reward guidance."""

from __future__ import annotations

from typing import Mapping

REGIME_LIST_TEXT = "clear_stable, clear_variable, partly_cloudy, overcast_stable, overcast_variable, rain_storm, extreme_event"
STRATEGY_LIST_TEXT = "peak_track, track, low_motion_track, stow"


def build_mode_b_prompt(telemetry: Mapping[str, float]) -> list[dict[str, str]]:
    """Build a strict JSON prompt for hourly Mode-A+B guidance."""

    system = (
        "/no_think\n"
        "You are a compact solar-tracker weather-regime and reward-weight classifier. "
        "Do not reason out loud. Do not use markdown. "
        "Return exactly one valid JSON object and nothing else."
    )
    user = f"""
/no_think
Classify the current solar-tracker operating regime and high-level tracking strategy.
Also choose safe reward weights for the next simulated hour.

Allowed regime values: {REGIME_LIST_TEXT}
Allowed strategy values: {STRATEGY_LIST_TEXT}

Use these meanings:
- track: useful irradiance is available and normal tracking is beneficial.
- peak_track: strong direct irradiance is available during a high-value solar period; accurate azimuth and especially tilt alignment are worth the movement.
- low_motion_track: irradiance is weak, DNI is below 150 W/m2, and next_30min_average_DNI is below 150 W/m2; avoid jitter and aggressive chasing, but continue slow repositioning toward expected sun position because tracker actions are bounded delta-angle commands.
- stow: nighttime or severe weather/wind risk.

Also return two scalar guidance values:
- tilt_priority: number from 0.0 to 1.0. Use high values when tilt alignment is important during peak energy periods.
- movement_budget: number from 0.0 to 1.0. Use high values when movement is worth the energy gain, and low values for cloudy/night/low-energy periods.

Return reward_weights for exactly three components:
- alignment: rewards energy-aware sun-panel alignment. This is more important during high DNI and low solar zenith.
- movement: penalizes total azimuth/tilt movement.
- smoothness: penalizes jittery changes in movement.

The three reward weights must be decimal numbers, nonnegative, and sum to 1.0.
If DNI or next_30min_average_DNI is at least 150 W/m2, prefer track rather than low_motion_track.
For useful daylight, alignment should normally be at least 0.50.
For low_motion_track, movement and smoothness should be important but should not eliminate alignment.
For peak_track, alignment should be dominant, about 0.70, while movement and smoothness should not be zero.
Do not include a direct energy weight because energy is already represented through irradiance-weighted alignment.

Telemetry:
- hour: {telemetry.get('hour')}
- minute: {telemetry.get('minute')}
- solar_zenith_deg: {telemetry.get('solar_zenith_deg')}
- DNI_W_m2: {telemetry.get('dni')}
- DHI_W_m2: {telemetry.get('dhi')}
- next_10min_DNI_W_m2: {telemetry.get('next_10min_dni')}
- next_10min_DHI_W_m2: {telemetry.get('next_10min_dhi')}
- next_30min_average_DNI_W_m2: {telemetry.get('next_30min_average_dni')}
- cloud_type: {telemetry.get('cloud_type')}  # categorical code from 0 to 9, not a percentage or cloud-cover fraction
- wind_speed_mps: {telemetry.get('wind_speed')}
- temperature_C: {telemetry.get('temperature')}
- dni_clear_sky_ratio: {telemetry.get('dni_clear_sky_ratio')}

Return exactly this JSON schema:
{{"regime":"one_allowed_regime", "strategy":"one_allowed_strategy", "tilt_priority":0.0, "movement_budget":0.0, "reward_weights":{{"alignment":0.0,"movement":0.0,"smoothness":0.0}}}}

Valid example:
{{"regime":"clear_stable","strategy":"peak_track","tilt_priority":0.95,"movement_budget":0.85,"reward_weights":{{"alignment":0.70,"movement":0.15,"smoothness":0.15}}}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
