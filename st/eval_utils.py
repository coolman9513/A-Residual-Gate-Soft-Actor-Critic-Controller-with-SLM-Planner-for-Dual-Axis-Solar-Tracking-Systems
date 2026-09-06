"""Evaluation helpers for SLM-SAC-Auto experiments.

Complements st/utilities.py (which already has evaluate_tracker_policy and
metric_from_eval). This module adds RBC/fixed baseline rollouts, LLM-guided
evaluation, and energy aggregation helpers.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── energy helpers ─────────────────────────────────────────────────────────────

def total_energy_from_history(history) -> float:
    """Sum generated_energy_kwh across all steps in history."""
    if history is None:
        return float("nan")
    h = pd.DataFrame(history)
    if h.empty or "generated_energy_kwh" not in h.columns:
        return float("nan")
    return float(pd.to_numeric(h["generated_energy_kwh"], errors="coerce").sum())


def daily_energy_series(history, seconds_per_time_step: float = 600.0) -> pd.Series:
    """Return a Series of daily energy sums indexed by day number."""
    h = pd.DataFrame(history)
    steps_per_day = int(round(24 * 3600 / seconds_per_time_step))
    first_step = float(h["time_step"].min()) if len(h) else 0.0
    day_idx = np.floor((h["time_step"].astype(float) - first_step) / steps_per_day).astype(int)
    return h.groupby(day_idx)["generated_energy_kwh"].sum()


def daily_summary(history, label: str) -> pd.DataFrame:
    """Aggregate per-step history to daily statistics."""
    h = pd.DataFrame(history).copy()
    if "date" not in h.columns:
        return h
    return h.groupby("date", as_index=False).agg(
        controller=("date", lambda _: label),
        generated_energy_kwh=("generated_energy_kwh", "sum"),
        ideal_generated_energy_kwh=("ideal_generated_energy_kwh", "sum"),
        movement_deg=("panel_azimuth_delta_deg", lambda x: np.abs(x).sum()),
        tilt_movement_deg=("panel_tilt_delta_deg", lambda x: np.abs(x).sum()),
        tracking_error_deg=("tracking_error_deg", "mean"),
    )


# ── baseline rollouts ──────────────────────────────────────────────────────────

def _run_env_episode(env, action_fn, label: str = ""):
    """Step an environment using action_fn(observations) until done."""
    observations = env.reset()
    done = False
    step_count = 0
    total_steps = getattr(env, "time_steps", 0)
    t0 = time.time()
    while not done:
        action = action_fn(observations)
        observations, _, done, _ = env.step(action)
        step_count += 1
        if label and step_count % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {label}: {step_count}/{total_steps} steps ({elapsed:.0f}s)", flush=True)
    if label:
        print(f"  {label}: done — {step_count} steps in {time.time() - t0:.1f}s", flush=True)
    return env


def rollout_rbc_for_schema(schema_template: dict):
    """Run the Rule-Based Controller on schema_template. Returns (history, evaluation)."""
    from environment import SolarTrackerEnv
    from rbc import SolarTrackerRBC, average_tracker_schedule

    env = SolarTrackerEnv(schema=copy.deepcopy(schema_template))
    schedule = average_tracker_schedule(env, use_episode_window=False)
    agent = SolarTrackerRBC(env=env, schedule=schedule, outside_tracking_policy="hold_last", debug=False)

    print("RBC rollout started …", flush=True)
    _run_env_episode(env, lambda obs: agent.predict(obs, deterministic=True), label="RBC")
    return env.history.copy(), env.evaluate(include_baselines=True)


def rollout_fixed_for_schema(schema_template: dict, fixed_az: float = 180.0, fixed_tilt: float = 30.0):
    """Run the fixed-panel baseline. Returns (history, evaluation)."""
    from environment import SolarTrackerEnv

    schema = copy.deepcopy(schema_template)
    schema["initial_orientation"] = {"azimuth_deg": fixed_az, "tilt_deg": fixed_tilt}
    schema["auto_reset_to_initial_orientation"] = False
    env = SolarTrackerEnv(schema=schema)

    print("Fixed-panel rollout started …", flush=True)
    action = env.action_from_target_angles(fixed_az, fixed_tilt).tolist()
    _run_env_episode(env, lambda _obs: [action], label="Fixed")
    return env.history.copy(), env.evaluate(include_baselines=True)


def rollout_sac_for_schema(agent, schema_template: dict):
    """Run a trained SAC-Auto (or Meta-SAC) agent. Returns (history, evaluation)."""
    from utilities import SolarTrackerMetaSACWrapper

    eval_env = SolarTrackerMetaSACWrapper(schema_template, reward_scale=50.0)
    state = eval_env.reset()
    done = False
    step_count = 0
    t0 = time.time()
    print("SAC-Auto rollout started …", flush=True)
    while not done:
        action, _ = agent.select_action(state, eval=True, mode="running")
        state, _, done, _ = eval_env.step(action)
        step_count += 1
        if step_count % 500 == 0:
            print(f"  SAC: {step_count} steps ({time.time() - t0:.0f}s)", flush=True)
    print(f"  SAC: done — {step_count} steps in {time.time() - t0:.1f}s", flush=True)
    return eval_env.history.copy(), eval_env.evaluate(include_baselines=True)


# ── residual controller evaluation ─────────────────────────────────────────────

def movement_stats(history) -> dict:
    """Return total and per-step movement (azimuth + tilt) from a history df."""
    h = pd.DataFrame(history)
    if h.empty:
        return {"movement_total_deg": float("nan"), "movement_per_step_deg": float("nan")}
    total = float(
        pd.to_numeric(h.get("panel_azimuth_delta_deg"), errors="coerce").abs().sum()
        + pd.to_numeric(h.get("panel_tilt_delta_deg"), errors="coerce").abs().sum()
    )
    return {
        "movement_total_deg": total,
        "movement_per_step_deg": total / max(len(h), 1),
    }


def movement_profile_stats(history, move_eps: float = 0.5) -> dict:
    """Motor-wear metrics: how OFTEN and how SMOOTHLY the panel moves.

    For a good tracker the total angular distance is largely fixed by the sun's
    path, so what actually wears the hardware is the number of start/stop cycles
    (activations) and the jerk (sudden direction changes). These are highly
    reducible at near-zero energy cost by moving less often in bigger steps.
    """
    h = pd.DataFrame(history)
    if h.empty:
        return {"activations": 0, "total_move_deg": 0.0, "mean_move_per_activation": 0.0,
                "jerk_deg2": 0.0, "az_reversals": 0}
    az = pd.to_numeric(h.get("panel_azimuth_delta_deg"), errors="coerce").fillna(0.0).to_numpy()
    ti = pd.to_numeric(h.get("panel_tilt_delta_deg"), errors="coerce").fillna(0.0).to_numpy()
    step_move = np.abs(az) + np.abs(ti)
    activations = int((step_move > move_eps).sum())
    total = float(step_move.sum())
    jerk = float(np.mean(np.diff(az) ** 2 + np.diff(ti) ** 2)) if len(az) > 1 else 0.0
    nz = az[np.abs(az) > move_eps]
    reversals = int((np.diff(np.sign(nz)) != 0).sum()) if len(nz) > 1 else 0
    return {
        "activations": activations,
        "total_move_deg": total,
        "mean_move_per_activation": total / max(activations, 1),
        "jerk_deg2": jerk,
        "az_reversals": reversals,
    }


def wind_safety_stats(
    env,
    threshold: float = 15.0,
    stow_tilt: float = 10.0,
    margin: float = 5.0,
) -> dict:
    """Wind-safety metric from a wrapper's per-step (wind, tilt) log.

    A step is 'unsafe' if the effective wind is at/above threshold while the
    panel is NOT laid flat (tilt above stow_tilt + margin). This can be computed
    for ANY controller run through the wind wrapper — energy-optimal ones simply
    never stow, so they accumulate large exposure.
    """
    log = list(getattr(env, "wind_metric_log", []) or [])
    if not log:
        return {"storm_steps": 0, "unsafe_steps": 0, "protected_fraction": float("nan"),
                "exposure_integral_ms_steps": 0.0, "peak_wind_ms": 0.0}
    winds = np.array([w for w, _ in log], dtype=float)
    tilts = np.array([t for _, t in log], dtype=float)
    storm = winds >= threshold
    stowed = tilts <= (stow_tilt + margin)
    unsafe = storm & (~stowed)
    exposure = float(np.maximum(winds - threshold, 0.0)[unsafe].sum())
    return {
        "storm_steps": int(storm.sum()),
        "unsafe_steps": int(unsafe.sum()),
        "protected_fraction": float(1.0 - unsafe.sum() / max(int(storm.sum()), 1)),
        "exposure_integral_ms_steps": exposure,
        "peak_wind_ms": float(winds.max()),
    }


def rollout_residual_for_schema(
    agent,
    schema_template: dict,
    authority_mode: str = "goal",
    guidance=None,
    use_llm: bool = False,
    fixed_authority: tuple = (6.0, 4.0, float("inf")),
    authority_table=None,
    slew_deg=None,
    seed: int = 0,
    label: str = "",
    hold_base: bool = True,
    action_mode: str = "residual",
    rule_dni_threshold: float = 100.0,
    storm_wind=None,
    wind_stow: bool = False,
    wind_stow_threshold: float = 15.0,
    wind_stow_tilt: float = 10.0,
    wind_stow_lookahead: int = 0,
    hold_tilt_deg: float = 25.0,
    cloud_anticipate_lookahead: int = 0,
    cloud_recovery_dni: float = 150.0,
    move_deadband_deg: float = 0.0,
    disable_guidance: bool = False,
):
    """Roll out a trained residual SAC-Auto agent. Returns (history, evaluation, env).

    Pass agent=None to run the floor policy (base only) — with hold_base=False
    this must reproduce the RBC baseline exactly.
    Pass agent="oracle" to follow the physics correction fully — the upper
    bound of what the RL policy could learn.
    Pass agent="rule" for the hand-written gate baseline: follow the physics
    correction only when current DNI >= rule_dni_threshold, else stay on the
    base. A learned gate must beat this simple rule to justify RL.
    """
    from train_utils import make_residual_env

    env = make_residual_env(
        copy.deepcopy(schema_template),
        authority_mode=authority_mode,
        use_llm=use_llm,
        guidance=guidance,
        fixed_authority=fixed_authority,
        authority_table=authority_table,
        slew_deg=slew_deg,
        seed=seed,
        hold_base=hold_base,
        action_mode=action_mode,
        storm_wind=storm_wind,
        wind_stow=wind_stow,
        wind_stow_threshold=wind_stow_threshold,
        wind_stow_tilt=wind_stow_tilt,
        wind_stow_lookahead=wind_stow_lookahead,
        hold_tilt_deg=hold_tilt_deg,
        cloud_anticipate_lookahead=cloud_anticipate_lookahead,
        cloud_recovery_dni=cloud_recovery_dni,
        move_deadband_deg=move_deadband_deg,
        disable_guidance=disable_guidance,
    )
    label = label or f"Residual[{authority_mode}]"
    state = env.reset()
    done = False
    step_count = 0
    t0 = time.time()
    print(f"{label} rollout started …", flush=True)
    while not done:
        if agent is None:
            action = env.floor_action()
        elif isinstance(agent, str) and agent == "oracle":
            action = env.oracle_action()
        elif isinstance(agent, str) and agent == "rule":
            row = env.env._row_at_time_step(env.env.time_step)
            dni = float(row[env.env.OBSERVATION_COLUMN_MAPPING["dni"]])
            action = env.oracle_action() if dni >= rule_dni_threshold else env.floor_action()
        else:
            action, _ = agent.select_action(state, eval=True, mode="running")
        state, _, done, _ = env.step(action)
        step_count += 1
        if step_count % 500 == 0:
            print(f"  {label}: {step_count} steps ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {label}: done — {step_count} steps in {time.time() - t0:.1f}s", flush=True)
    return env.history.copy(), env.evaluate(include_baselines=True), env


def residual_ablation(
    agent,
    schema_template: dict,
    llm_guidance=None,
    modes: tuple = ("fixed", "random", "goal", "oracle"),
    fixed_authority: tuple = (6.0, 4.0, float("inf")),
    seed: int = 0,
    hold_base: bool = True,
    action_mode: str = "residual",
) -> pd.DataFrame:
    """Run the authority-mode ablation and return a comparison table.

    modes: any of 'fixed' (constant authority), 'random' (random per-hour
    regime), 'goal' (SLM-driven when llm_guidance is provided, otherwise the
    rule-based fallback), 'oracle' (physics oracle actions under goal-mode
    authority — the learnability upper bound, ignores `agent`). The gap
    between 'goal' and fixed/random is the causal contribution of the goal
    signal; the gap between 'goal' and 'oracle' is what learning left behind.
    """
    rows = []
    for mode in modes:
        guidance = llm_guidance if mode in ("goal", "oracle") else None
        history, evaluation, _ = rollout_residual_for_schema(
            "oracle" if mode == "oracle" else agent,
            schema_template,
            authority_mode="goal" if mode == "oracle" else mode,
            guidance=guidance,
            use_llm=False,
            fixed_authority=fixed_authority,
            seed=seed,
            label=f"Ablation[{mode}]",
            hold_base=hold_base,
            action_mode=action_mode,
        )
        metrics = dict(zip(evaluation["cost_function"], evaluation["value"]))
        moves = movement_stats(history)
        rows.append(
            {
                "authority_mode": mode,
                "energy_kwh": total_energy_from_history(history),
                "tracking_efficiency": metrics.get("tracking_efficiency", float("nan")),
                "movement_total_deg": moves["movement_total_deg"],
                "movement_per_step_deg": moves["movement_per_step_deg"],
                "jerk_deg2": metrics.get("tracking_smoothness_jerk_deg2", float("nan")),
            }
        )
    return pd.DataFrame(rows)


# ── LLM-guided evaluation ──────────────────────────────────────────────────────

def evaluate_llm_tracker_policy(
    agent,
    schema_template: dict,
    guidance=None,
    regime_router=None,
    include_baselines: bool = True,
):
    """Evaluate the SLM-SAC-Auto controller.

    Parameters
    ----------
    agent:
        Trained SAC-Auto (or Meta-SAC) agent. Used only when regime_router is None.
    schema_template:
        Environment schema (passed to LLMGoalConditionedMetaSACWrapper).
    guidance:
        Pre-built LLMGoalGuidance instance. If None, a new one is created with
        use_llm=True and the local inference server.
    regime_router:
        Optional RegimeRouter. When provided it overrides the single agent for
        action selection based on the current SLM goal.
    include_baselines:
        Whether to include fixed/RBC results in the evaluation dict.

    Returns (history_df, evaluation_df, eval_env)
    """
    from llm.goal_guidance import LLMGoalGuidance
    from llm.goal_wrapper import LLMGoalConditionedMetaSACWrapper

    if guidance is None:
        guidance = LLMGoalGuidance(enabled=True, use_llm=True, verbose=False)

    eval_env = LLMGoalConditionedMetaSACWrapper(
        schema_template=copy.deepcopy(schema_template),
        reward_scale=50.0,
        guidance=guidance,
        regime_router=regime_router,
    )

    state = eval_env.reset()
    done = False
    step_count = 0
    t0 = time.time()
    # Regime agents were trained on the plain base observation (no LLM context).
    # Slice off only those dims when routing so network shapes match.
    base_obs_dim = len(eval_env.base_obs_low) if hasattr(eval_env, "base_obs_low") else len(state)
    print("SLM evaluation rollout started …", flush=True)

    while not done:
        if regime_router is not None:
            action, _ = regime_router.select_action(state[:base_obs_dim], eval_env.last_goal, eval=True)
        else:
            action, _ = agent.select_action(state, eval=True, mode="running")
        state, _, done, _ = eval_env.step(action)
        step_count += 1
        if step_count % 500 == 0:
            print(f"  SLM eval: {step_count} steps ({time.time() - t0:.0f}s)", flush=True)

    print(f"  SLM eval: done — {step_count} steps in {time.time() - t0:.1f}s", flush=True)
    evaluation = eval_env.evaluate(include_baselines=include_baselines)
    return eval_env.history.copy(), evaluation, eval_env
