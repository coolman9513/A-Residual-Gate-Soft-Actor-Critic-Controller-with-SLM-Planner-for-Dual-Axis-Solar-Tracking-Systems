"""Prompt builders for goal-conditioned SLM solar-tracker planning."""

from __future__ import annotations

from typing import Mapping


def build_goal_prompt(telemetry: Mapping[str, object]) -> list[dict[str, str]]:
    """Build a strict JSON prompt for hourly goal-conditioned guidance."""

    system = (
        "You are a compact solar-tracker planner. Use the dni_bucket label and sun position "
        "to choose a strategy and motion budget. Return one short Reasoning line, maximum 20 words, "
        "followed immediately by exactly one valid JSON object. Do not write step-by-step analysis."
    )
    user = f"""
Plan the next 30 to 60 simulated minutes for a dual-axis solar tracker.

Decision steps: apply in order and stop at the first matching rule.
1. If solar_zenith_deg >= 90 OR tracking_window_active is false:
   -> hold=true, motion_budget_deg=0.
2. If dni_bucket is NIGHT_OR_HEAVY_OVERCAST OR (DNI < 80 AND next_30min_average_DNI < 80 AND solar_zenith_deg > 75):
   -> hold=true, motion_budget_deg=3.
3. If dni_bucket is CLEAR_STRONG (max_forecast_dni >= 700):
   -> hold=false, motion_budget_deg=30.
4. If dni_bucket is CLEAR_MODERATE (max_forecast_dni >= 500):
   -> hold=false, motion_budget_deg=20.
5. If dni_bucket is PARTIAL_CLOUD (max_forecast_dni >= 300):
   -> hold=false, motion_budget_deg=12.
6. If dni_bucket is OVERCAST_DIM (max_forecast_dni >= 100):
   -> hold=false, motion_budget_deg=6.
7. Otherwise, if tracking_window_active is true:
   -> hold=false, motion_budget_deg=3.

Important rules:
- hold=true is ONLY for night, outside tracking window, or deep low-sun overcast.
- During usable daylight, keep hold=false so the tracker can at least reposition slowly.
- Do not output angles outside the mechanical limits.
- motion_budget_deg is the maximum useful total azimuth+tilt movement this hour.
- Choose motion_budget_deg only from: 0, 3, 6, 12, 20, 30.
- Continuous sanity-check: budget_deg ≈ clip(max_forecast_dni × 0.043, 0, 30). If your rule choice differs by more than 5 from this, reconsider.
- cloud_type ranges 0-9: 0-2 = clear/thin (higher budget), 3-6 = moderate cloud, 7-9 = heavy/deep (lower budget or hold).
- Reference: continuous sun tracking needs about 15 deg/hour azimuth plus about 3 deg/hour tilt.

Telemetry JSON:
{telemetry}

First write one short Reasoning line, maximum 20 words. Do not write step-by-step analysis.
Then output exactly one JSON object with these keys and real numeric values:
{{"target_tilt":<deg_10_to_85>,"target_az":<deg_70_to_290>,"motion_budget_deg":<one_of_0_3_6_12_20_30>,"pre_position_tilt":<deg_10_to_85>,"pre_position_az":<deg_70_to_290>,"hold":<true_or_false>,"horizon_min":<30_or_60>,"confidence":<0_to_1>}}

Examples:
{{"target_tilt":30.0,"target_az":175.0,"motion_budget_deg":30.0,"pre_position_tilt":28.0,"pre_position_az":195.0,"hold":false,"horizon_min":60,"confidence":0.95}}
{{"target_tilt":40.0,"target_az":155.0,"motion_budget_deg":20.0,"pre_position_tilt":38.0,"pre_position_az":170.0,"hold":false,"horizon_min":60,"confidence":0.88}}
{{"target_tilt":50.0,"target_az":160.0,"motion_budget_deg":12.0,"pre_position_tilt":48.0,"pre_position_az":175.0,"hold":false,"horizon_min":60,"confidence":0.65}}
{{"target_tilt":42.0,"target_az":185.0,"motion_budget_deg":6.0,"pre_position_tilt":40.0,"pre_position_az":195.0,"hold":false,"horizon_min":30,"confidence":0.50}}
{{"target_tilt":55.0,"target_az":145.0,"motion_budget_deg":3.0,"pre_position_tilt":53.0,"pre_position_az":155.0,"hold":false,"horizon_min":30,"confidence":0.40}}
{{"target_tilt":45.0,"target_az":180.0,"motion_budget_deg":0.0,"pre_position_tilt":45.0,"pre_position_az":180.0,"hold":true,"horizon_min":60,"confidence":0.90}}

Field meanings:
- target_tilt, target_az: desired panel orientation right now. If hold=true, use the current panel orientation from telemetry.
- pre_position_tilt, pre_position_az: desired orientation at horizon_min minutes.
- motion_budget_deg: allowed total azimuth+tilt movement this hour.
- hold: true means reward stillness; use it only for night/outside-window/deep low-sun overcast.
- horizon_min: use 30 or 60 minutes.
- confidence: number from 0 to 1.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



def build_goal_retry_prompt(telemetry: Mapping[str, object]) -> list[dict[str, str]]:
    """Build a strict JSON-only retry prompt after a malformed response."""

    system = (
        "Return only one valid JSON object. No reasoning, no markdown, no explanation. "
        "Use double quotes for keys and booleans true/false."
    )
    user = f"""
Your previous answer was invalid or too verbose. Return ONLY valid JSON.

Decision rules (use dni_bucket from telemetry):
- If solar_zenith_deg >= 90 or tracking_window_active is false: hold=true, motion_budget_deg=0.
- If NIGHT_OR_HEAVY_OVERCAST or (DNI < 80 and next_30min_average_DNI < 80 and solar_zenith_deg > 75): hold=true, motion_budget_deg=3.
- If CLEAR_STRONG (max_forecast_dni >= 700): hold=false, motion_budget_deg=30.
- If CLEAR_MODERATE (max_forecast_dni >= 500): hold=false, motion_budget_deg=20.
- If PARTIAL_CLOUD (max_forecast_dni >= 300): hold=false, motion_budget_deg=12.
- If OVERCAST_DIM (max_forecast_dni >= 100): hold=false, motion_budget_deg=6.
- Otherwise during active daylight: hold=false, motion_budget_deg=3.

Telemetry JSON:
{telemetry}

Required JSON schema:
{{"target_tilt":45.0,"target_az":180.0,"motion_budget_deg":12.0,"pre_position_tilt":45.0,"pre_position_az":180.0,"hold":false,"horizon_min":60,"confidence":0.80}}

Return only the JSON object now.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ── v2: autonomous / oracle-trained planner ────────────────────────────────────
# The v1 prompt above hard-codes a DNI-threshold decision ladder ("Decision steps
# 1-7"). Fine-tuning on it is precisely why the SLM reproduces an FSM
# (PROJECT_HANDOFF.md §7). v2 states the ACTION SPACE and the field semantics but
# gives no policy: the model must infer the mapping from telemetry to plan from
# the hindsight-optimal targets it was trained on.
#
# The same builder is used for dataset generation and at inference so the
# training and serving prompt distributions match exactly.

AUTHORITY_BUDGETS = {"hold": 3.0, "cloud": 12.0, "clear": 30.0}


def build_goal_prompt_v2(telemetry: Mapping[str, object],
                        instruction: str = "") -> list[dict[str, str]]:
    """Ladder-free prompt for the oracle-trained autonomous planner.

    `instruction` is an optional operator directive in plain language, e.g.
    "the actuators are aging, prefer less movement". It is the programmability
    channel the paper's title claims: it re-tasks the planner with no retraining,
    which a fixed feature-vector classifier cannot do at all. Empty by default so
    the prompt is byte-identical to the one the adapter was fine-tuned on.
    """
    directive = (chr(10) + "Operator instruction: " + instruction.strip() + chr(10)) if instruction else ""

    system = (
        "You are a solar-tracker planner. Given telemetry, choose this hour's plan: "
        "where to point now, where to be at the end of the horizon, how much corrective "
        "movement to authorise, and whether to hold. Reply with exactly one valid JSON "
        "object and nothing else. No reasoning, no markdown, no explanation."
    )
    user = f"""
Plan the next 30 to 60 simulated minutes for a dual-axis solar tracker.

The tracker always follows an astronomical base schedule. Your motion_budget_deg
sets how much CORRECTIVE movement, on top of that base, is authorised this hour:
  3   minimal authority  - stay essentially on the base schedule
  12  moderate authority - correct pointing when it is worth the motor wear
  30  full authority     - track the sun as closely as the mechanism allows
Choose exactly one of 3, 12 or 30. Higher authority earns more energy but costs
motor start/stop cycles, which wear the drive. Spend it only where it pays.

Mechanical limits: azimuth 70-290 deg, tilt 10-85 deg.
{directive}
Field meanings:
- target_tilt, target_az: where the panel should point right now.
- pre_position_tilt, pre_position_az: where it should be horizon_min minutes from now.
- motion_budget_deg: corrective authority for this hour (3, 12 or 30).
- hold: true only when the sun is down, outside the tracking window, or the light
  is too weak for pointing to matter. Use motion_budget_deg=3 when hold is true.
- horizon_min: 30 or 60.
- confidence: 0 to 1 - how clearly this plan beats the alternatives.

Telemetry JSON:
{telemetry}

Return exactly one JSON object, nothing else:
{{"target_tilt":<deg>,"target_az":<deg>,"motion_budget_deg":<3|12|30>,"pre_position_tilt":<deg>,"pre_position_az":<deg>,"hold":<true|false>,"horizon_min":<30|60>,"confidence":<0-1>}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# Wear directives used by BOTH the instruction-conditioned training set and the
# programmability evaluation, so the two can never drift apart. Each maps to the
# per-activation wear cost whose hindsight-optimal policy it describes.
WEAR_DIRECTIVES = {
    "max_energy": (
        0.0,
        "Motor wear is not a concern at this site. Maximise energy capture, and "
        "authorise generous corrective movement whenever it helps."),
    "balanced": (
        3.79e-4,
        "Balance energy capture against motor wear using standard operating policy."),
    "min_movement": (
        3.0e-3,
        "The actuators at this site are aging and near end of life. Strongly prefer "
        "keeping the tracker still; authorise corrective movement only when the "
        "energy gain is large."),
}
