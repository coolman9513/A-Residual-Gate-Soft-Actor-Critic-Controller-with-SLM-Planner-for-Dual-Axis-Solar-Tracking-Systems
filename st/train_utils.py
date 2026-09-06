"""Training orchestration helpers for SLM-SAC-Auto experiments.

These complement st/utilities.py (which has load_flat_config, evaluate_tracker_policy,
SolarTrackerMetaSACWrapper). This module adds dataset filtering, schema construction,
RBC replay seeding, and season-curriculum helpers.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ── schema helpers ─────────────────────────────────────────────────────────────

def build_schema(
    base_schema: dict,
    data_csv: str,
    start_step: int,
    end_step: int,
    episode_steps: Optional[int] = None,
    rolling_split: bool = False,
    random_split: bool = False,
    fixed_az: float = 180.0,
    fixed_tilt: float = 30.0,
) -> dict:
    """Return a ready-to-use schema_run from a base schema and dataset parameters."""
    schema_run = copy.deepcopy(base_schema)
    schema_run["root_directory"] = str(_DATA_DIR)
    schema_run["data"] = {"filename": data_csv, "datetime_column": "Datetime"}
    schema_run["simulation_start_time_step"] = int(start_step)
    schema_run["simulation_end_time_step"] = int(end_step)
    schema_run["episode_time_steps"] = int(episode_steps or (end_step - start_step + 1))
    schema_run["rolling_episode_split"] = bool(rolling_split)
    schema_run["random_episode_split"] = bool(random_split)
    schema_run["seconds_per_time_step"] = float(base_schema.get("seconds_per_time_step", 600))
    schema_run["initial_orientation"] = {"azimuth_deg": 100.0, "tilt_deg": 85.0}
    schema_run["auto_reset_to_initial_orientation"] = True
    schema_run.setdefault("baselines", {}).setdefault("fixed", {})
    schema_run["baselines"]["fixed"]["azimuth_deg"] = fixed_az
    schema_run["baselines"]["fixed"]["tilt_deg"] = fixed_tilt
    return schema_run


def make_season_schemas(base_schema: dict, data_dir: Optional[Path] = None) -> dict:
    """Return a dict with pre-built schemas for each curriculum phase.

    Keys: 'summer', 'spring_autumn', 'full'
    Files assumed at data_dir: 2020_summer.csv, 2020_spring_autumn.csv,
                                2020_4months_2weeks.csv
    """
    d = Path(data_dir or _DATA_DIR)

    def _row_count(csv_name: str) -> int:
        path = d / csv_name
        if not path.exists():
            return 0
        return sum(1 for _ in path.open(encoding="utf-8")) - 1  # exclude header

    def _schema(csv_name: str) -> dict:
        n = _row_count(csv_name)
        return build_schema(base_schema, csv_name, 0, max(n - 1, 0))

    return {
        "summer":        _schema("2020_summer.csv"),
        "spring_autumn": _schema("2020_spring_autumn.csv"),
        "full":          _schema("2020_4months_2weeks.csv"),
    }


# ── dataset filtering ──────────────────────────────────────────────────────────

def filter_season_csv(src_csv: str, months: list[int], dst_csv: str) -> int:
    """Write rows matching any of months (1-12) from src_csv to dst_csv.

    Returns the number of rows written (excluding header).
    """
    src = Path(src_csv) if Path(src_csv).is_absolute() else _DATA_DIR / src_csv
    dst = Path(dst_csv) if Path(dst_csv).is_absolute() else _DATA_DIR / dst_csv
    df = pd.read_csv(src)
    month_col = next((c for c in df.columns if c.strip().lower() == "month"), None)
    if month_col is None:
        raise ValueError(f"No 'Month' column found in {src}. Columns: {list(df.columns)}")
    filtered = df[df[month_col].isin(months)].reset_index(drop=True)
    filtered.to_csv(dst, index=False)
    print(f"filter_season_csv: {len(filtered)} rows (months={months}) → {dst}")
    return len(filtered)


def filter_regime_csv(src_csv: str, regime: str, dst_csv: str, dni_col: str = "DNI") -> int:
    """Write rows by dominant daily DNI regime to dst_csv.

    Uses the mean of daytime (DNI > 0) rows per day to avoid nights dragging
    the average to zero — daily median of all rows would classify even clear
    days as 'hold' because 12 h of night set the median to 0.

    regime: 'clear'  (daytime-mean DNI >= 400 W/m²)
             'cloud'  (100 <= daytime-mean DNI < 400 W/m²)
             'hold'   (daytime-mean DNI < 100 W/m², or all-night day)
    Returns the number of rows written.
    """
    src = Path(src_csv) if Path(src_csv).is_absolute() else _DATA_DIR / src_csv
    dst = Path(dst_csv) if Path(dst_csv).is_absolute() else _DATA_DIR / dst_csv
    df = pd.read_csv(src)

    month_col  = next((c for c in df.columns if c.strip().lower() == "month"), "Month")
    day_col    = next((c for c in df.columns if c.strip().lower() == "day"),   "Day")
    actual_dni = next((c for c in df.columns if c.strip().lower() == dni_col.lower()), dni_col)

    # Mean over daylight steps only; days with no sun get NaN → filled with 0.
    daytime = df[df[actual_dni] > 0]
    day_mean = (
        daytime.groupby([month_col, day_col])[actual_dni]
        .mean()
        .reset_index()
        .rename(columns={actual_dni: "day_mean_dni"})
    )
    # Days that appear in df but not in daytime (all-night) get mean = 0.
    all_days = df[[month_col, day_col]].drop_duplicates()
    day_mean = all_days.merge(day_mean, on=[month_col, day_col], how="left").fillna(0)

    if regime == "clear":
        mask = day_mean["day_mean_dni"] >= 400
    elif regime == "cloud":
        mask = (day_mean["day_mean_dni"] >= 100) & (day_mean["day_mean_dni"] < 400)
    elif regime == "hold":
        mask = day_mean["day_mean_dni"] < 100
    else:
        raise ValueError(f"regime must be 'clear', 'cloud', or 'hold', got {regime!r}")

    regime_days = day_mean[mask][[month_col, day_col]]
    filtered = df.merge(regime_days, on=[month_col, day_col]).reset_index(drop=True)
    filtered.to_csv(dst, index=False)
    print(f"filter_regime_csv: {len(filtered)} rows (regime={regime!r}, threshold=400/100) → {dst}")
    return len(filtered)


# ── residual RL over RBC (SLM-gated) ──────────────────────────────────────────

# Per-regime deadband multiplier: coast more (bigger deadband) when the hour is
# less productive, so the motor activates less often where accuracy matters least.
_REGIME_DEADBAND_FACTOR = {"clear": 1.0, "cloud": 1.5, "hold": 3.0}

RESIDUAL_CONTEXT_NAMES = [
    "residual_authority_azimuth",
    "residual_authority_tilt",
    "residual_budget_used",
    "residual_base_action_azimuth",
    "residual_base_action_tilt",
    # Physics feature: the residual that would point the panel at the per-step
    # ideal orientation (solar-position physics, standard in real trackers).
    # Without it SAC must reverse-engineer solar geometry from a ~1e-4 kWh/step
    # reward signal — the oracle test showed it never does. With it, RL's job
    # becomes deciding WHEN following physics is worth the movement.
    "residual_needed_azimuth",
    "residual_needed_tilt",
]


def _lazy_residual_imports():
    """Import wrapper dependencies lazily so plain helpers stay import-light."""
    from llm.goal_wrapper import LLMGoalConditionedMetaSACWrapper
    from llm.goal_guidance import LLMGoalGuidance
    from llm.regime_router import Authority, DEFAULT_AUTHORITY_TABLE, authority_for
    from rbc import average_tracker_schedule, schedule_lookup
    return (
        LLMGoalConditionedMetaSACWrapper,
        LLMGoalGuidance,
        Authority,
        DEFAULT_AUTHORITY_TABLE,
        authority_for,
        average_tracker_schedule,
        schedule_lookup,
    )


# The main schema still carries the ~250x squared movement penalties from the
# earlier soft-penalty experiment. For the residual task movement is already
# hard-bounded by the authority clamp + hour budget, so those penalties only
# suppress the residual: a 6 deg correction would cost more reward than the
# maximum possible step energy. Residual envs revert them to the reward.py
# defaults; the small linear terms stay so useless base motion is still pruned.
RESIDUAL_REWARD_OVERRIDES = {
    "azimuth_squared_movement_penalty": 2e-6,
    "tilt_squared_movement_penalty": 3e-6,
    "smoothness_penalty": 2e-6,
}


def make_storm_schedule(
    n_steps: int,
    seed: int = 0,
    n_storms: int = 8,
    peak_wind: float = 18.0,
    ramp: int = 2,
    steps_per_day: int = 144,
):
    """Return (wind_array, storm_list): synthetic high-wind events over an episode.

    The NSRDB Gwangju data never exceeds 8 m/s, so a wind-safety experiment needs
    injected storms. This is a deterministic stress-test scenario, NOT a claim
    about real Gwangju wind. Storms are placed at varied times of day — some at
    midday, where stowing genuinely costs tracking energy, so the safety-vs-energy
    trade-off is real. Each storm ramps up to ~peak_wind (m/s) and back down.
    """
    rng = np.random.RandomState(int(seed))
    wind = np.zeros(int(n_steps), dtype=np.float32)
    n_days = max(int(n_steps) // int(steps_per_day), 3)
    choose = min(int(n_storms), n_days - 2)
    days = sorted(rng.choice(range(1, n_days - 1), size=choose, replace=False))
    storms = []
    for k, d in enumerate(days):
        if k % 3 == 0:
            start_in_day = rng.randint(42, 54)    # ~07:00-09:00 morning
        elif k % 3 == 1:
            start_in_day = rng.randint(66, 84)    # ~11:00-14:00 midday (sun conflict)
        else:
            start_in_day = rng.randint(96, 108)   # ~16:00-18:00 evening
        duration = int(rng.randint(6, 18))        # 1-3 hours
        peak = float(peak_wind + rng.uniform(-2.0, 3.0))
        s0 = d * steps_per_day + start_in_day
        for j in range(duration):
            i = s0 + j
            if i >= n_steps:
                break
            if j < ramp:
                frac = (j + 1) / (ramp + 1)
            elif j >= duration - ramp:
                frac = (duration - j) / (ramp + 1)
            else:
                frac = 1.0
            wind[i] = max(float(wind[i]), peak * frac)
        storms.append({"day": int(d), "start_step": int(s0),
                       "duration": duration, "peak_wind": round(peak, 1)})
    return wind, storms


def make_residual_env(
    schema_template: dict,
    authority_mode: str = "goal",
    use_llm: bool = False,
    guidance=None,
    reward_scale: float = 50.0,
    fixed_authority: tuple = (6.0, 4.0, float("inf")),
    authority_table=None,
    slew_deg: Optional[tuple] = None,
    query_interval_steps: int = 6,
    seed: int = 0,
    verbose: bool = False,
    reward_overrides: Optional[dict] = RESIDUAL_REWARD_OVERRIDES,
    hold_base: bool = True,
    action_mode: str = "residual",
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
    """Build a ResidualTrackerWrapper.

    authority_mode:
        'goal'   — authority from the SLM goal regime (hold/clear/cloud).
        'fixed'  — constant authority (Stage-1 / ablation baseline).
        'random' — per-hour random regime authority (ablation baseline).
    use_llm=False uses the deterministic rule-based fallback goals (no
    inference subprocess) — the right setting for training; switch to a
    use_llm=True guidance at evaluation time for the real SLM.
    reward_overrides (default RESIDUAL_REWARD_OVERRIDES) patches the schema's
    reward attributes; pass None to keep the schema reward untouched. Reward
    changes do not affect the energy/movement evaluation metrics.
    hold_base=True (lever 2): the SLM 'hold' regime freezes the base at the
    current pose instead of sweeping the RBC schedule through dead hours.
    Pass hold_base=False for the exact-RBC-parity floor check.
    """
    _, LLMGoalGuidance, _, _, _, _, _ = _lazy_residual_imports()
    if guidance is None:
        guidance = LLMGoalGuidance(enabled=True, use_llm=use_llm, verbose=verbose)
    schema_template = copy.deepcopy(schema_template)
    if reward_overrides:
        schema_template.setdefault("reward_function", {}).setdefault(
            "attributes", {}
        ).update(reward_overrides)
    return ResidualTrackerWrapper(
        schema_template=schema_template,
        reward_scale=reward_scale,
        guidance=guidance,
        authority_mode=authority_mode,
        fixed_authority=fixed_authority,
        authority_table=authority_table,
        slew_deg=slew_deg,
        query_interval_steps=query_interval_steps,
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


def _build_residual_wrapper_class():
    (
        LLMGoalConditionedMetaSACWrapper,
        _LLMGoalGuidance,
        Authority,
        DEFAULT_AUTHORITY_TABLE,
        authority_for,
        average_tracker_schedule,
        schedule_lookup,
    ) = _lazy_residual_imports()

    class ResidualTrackerWrapper(LLMGoalConditionedMetaSACWrapper):
        """Residual RL over the RBC prior with SLM-gated hard authority.

        Action semantics: the agent outputs a normalized residual a ∈ [-1, 1]².
        Every step the wrapper computes the RBC base delta (from the fixed
        daily schedule; hold-last outside the tracking window) and applies

            command = clip(base + a * authority, ±slew)

        with the cumulative residual motion per goal-hour projected onto the
        remaining hour budget. a = 0 reproduces RBC exactly, so RBC-level
        energy is a structural floor. Movement and jerk are bounded by
        construction (authority clamp + slew limit + smooth base), not by
        reward shaping.

        Reward: the plain environment reward (energy + tracking efficiency +
        mild movement terms) — stationary, unlike the goal-shaped reward of
        the parent class. The SLM influences behaviour only through the
        observation context and the hard authority limits, both of which are
        active at evaluation time.
        """

        # `last_history` was reassigned a full DataFrame copy on every step and is
        # never read anywhere in the codebase. Resolving it lazily keeps the read
        # API identical while removing that per-step cost.
        @property
        def last_history(self):
            return self.env.history

        @last_history.setter
        def last_history(self, value):
            return


        def __init__(
            self,
            schema_template,
            reward_scale: float = 50.0,
            guidance=None,
            authority_mode: str = "goal",
            fixed_authority: tuple = (6.0, 4.0, float("inf")),
            authority_table=None,
            slew_deg: Optional[tuple] = None,
            query_interval_steps: int = 6,
            seed: int = 0,
            hold_base: bool = True,
            hold_tilt_deg: float = 25.0,
            action_mode: str = "residual",
            storm_wind=None,
            wind_stow: bool = False,
            wind_stow_threshold: float = 15.0,
            wind_stow_tilt: float = 10.0,
            wind_stow_lookahead: int = 0,
            cloud_anticipate_lookahead: int = 0,
            cloud_recovery_dni: float = 150.0,
            move_deadband_deg: float = 0.0,
            disable_guidance: bool = False,
        ):
            assert authority_mode in {"goal", "fixed", "random"}
            assert action_mode in {"residual", "gate"}
            self.authority_mode = str(authority_mode)
            # Ablation: fully remove the SLM. The goal context vector is zeroed
            # in the observation, so the agent sees only physical sensors (solar
            # geometry, DNI, panel angles) and the physics correction — no
            # weather-regime label, no hold/anticipation/deadband. Pair with
            # authority_mode='fixed', hold_base=False, cloud_anticipate=0,
            # move_deadband=0 for a clean "gate over RBC, no SLM" baseline.
            self.disable_guidance = bool(disable_guidance)
            # Wind-stow (safety experiment): storm_wind is a per-dataset-step
            # array of injected high-wind (m/s); effective wind = max(data, storm).
            # When wind_stow is True and effective wind >= threshold, the base is
            # forced to the safe stow pose (azimuth held, tilt = wind_stow_tilt,
            # near-flat) and residual authority is zeroed, regardless of what RL
            # or the schedule want. This is the SLM's safety decision expressed
            # as a hard override. wind_stow=False keeps the energy-optimal
            # behaviour (never stows) so the two can be compared on identical
            # storms. The per-step (effective_wind, resulting_tilt) log feeds the
            # safety metric for ANY controller run through this wrapper.
            self._storm_wind = None if storm_wind is None else np.asarray(storm_wind, dtype=np.float32)
            self.wind_stow = bool(wind_stow)
            self.wind_stow_threshold = float(wind_stow_threshold)
            self.wind_stow_tilt = float(wind_stow_tilt)
            # Anticipatory stow: trigger if wind reaches threshold within the next
            # `wind_stow_lookahead` steps, so the panel is already flat when the
            # storm peaks. This is the "SLM reads the wind forecast and
            # pre-positions" behaviour; lookahead=0 is purely reactive.
            self.wind_stow_lookahead = int(wind_stow_lookahead)
            # Cloud anticipation: during a hold (heavy overcast), look ahead up to
            # cloud_anticipate_lookahead steps for the sun's return (DNI recovering
            # to cloud_recovery_dni). If found, the base pre-positions toward the
            # ideal orientation at that recovery step instead of freezing, so the
            # slew-limited panel is already aligned when the sun comes back — the
            # "SLM reads the forecast and pre-rotates" behaviour. lookahead=0
            # keeps the plain freeze. The hold pose itself uses hold_tilt_deg
            # (set it to the minimum tilt to lay near-flat = max diffuse capture).
            self.cloud_anticipate_lookahead = int(cloud_anticipate_lookahead)
            self.cloud_recovery_dni = float(cloud_recovery_dni)
            # Motion deadband: skip a move whose per-axis step is below the
            # deadband (scaled per regime). The panel then coasts and only
            # activates the motor when the drift grows past the deadband — far
            # fewer start/stop cycles and smoother motion, at near-zero energy
            # cost (cosine law). 0 disables it (fine tracking, every step).
            self.move_deadband_deg = float(move_deadband_deg)
            self.wind_metric_log = []
            # action_mode:
            #   'residual' — the agent outputs the correction itself (a*authority).
            #   'gate'     — the agent outputs how much of the PHYSICS correction
            #                to apply: action a in [-1,1] maps to gate (a+1)/2 in
            #                [0,1], applied to the authority-clipped oracle
            #                residual. Gate 1 everywhere reproduces the oracle
            #                (27.87 kWh reachable by construction); gate 0 is the
            #                base. RL only decides WHEN the correction is worth
            #                the movement — a dense, same-step reward signal.
            self.action_mode = str(action_mode)
            # Lever 2: when the goal regime is 'hold' (overcast/dim hours), the
            # BASE stops following the RBC schedule — the ~230 deg/day sweep on
            # dead days is wasted motion. Inside the tracking window the base
            # keeps the current azimuth but targets hold_tilt_deg: a flat-ish
            # tilt collects diffuse light almost optimally ((1+cos25°)/2≈0.95),
            # whereas freezing at the near-vertical morning pose loses ~40% of
            # it. Outside the window the pose freezes fully (the overnight
            # auto-reset is not counted as movement). With hold_base=False the
            # base is always the RBC schedule and the zero-residual rollout
            # reproduces RBC exactly.
            self.hold_base = bool(hold_base)
            self.hold_tilt_deg = float(hold_tilt_deg)
            self.fixed_authority = Authority(
                "fixed",
                float(fixed_authority[0]),
                float(fixed_authority[1]),
                float(fixed_authority[2]),
            )
            self.authority_table = dict(authority_table or DEFAULT_AUTHORITY_TABLE)
            self._slew_setting = slew_deg
            self._seed = int(seed)
            self._rng = np.random.RandomState(self._seed)
            self.hour_residual_used_deg = 0.0
            self._active_authority: Optional[Authority] = None
            self._random_authority: Optional[Authority] = None
            super().__init__(
                schema_template=schema_template,
                reward_scale=reward_scale,
                guidance=guidance,
                query_interval_steps=query_interval_steps,
            )

        # ── construction ──────────────────────────────────────────────────
        def _build_env(self):
            self.context_names = list(self.context_names) + list(RESIDUAL_CONTEXT_NAMES)
            super()._build_env()
            env_box = self.env.action_space[0]
            self._env_action_high = self._np.asarray(env_box.high, dtype=self._np.float32)
            self._slew = self._np.asarray(
                self._slew_setting if self._slew_setting is not None else env_box.high,
                dtype=self._np.float32,
            )
            self._slew = self._np.minimum(self._np.abs(self._slew), self._env_action_high)
            # Agent acts in normalized residual space.
            self.action_space = self._spaces.Box(
                low=-self._np.ones(2, dtype=self._np.float32),
                high=self._np.ones(2, dtype=self._np.float32),
                dtype=self._np.float32,
            )
            schedule = average_tracker_schedule(self.env, use_episode_window=False)
            self._rbc_lookup = schedule_lookup(schedule)

        def clone_for_eval(self):
            """Fresh wrapper with identical settings (guidance is shared)."""
            return ResidualTrackerWrapper(
                schema_template=self.schema_template,
                reward_scale=self.reward_scale,
                guidance=self.guidance,
                authority_mode=self.authority_mode,
                fixed_authority=(
                    self.fixed_authority.azimuth_deg,
                    self.fixed_authority.tilt_deg,
                    self.fixed_authority.hour_budget_deg,
                ),
                authority_table=self.authority_table,
                slew_deg=self._slew_setting,
                query_interval_steps=self.query_interval_steps,
                seed=self._seed,
                hold_base=self.hold_base,
                hold_tilt_deg=self.hold_tilt_deg,
                action_mode=self.action_mode,
                storm_wind=self._storm_wind,
                wind_stow=self.wind_stow,
                wind_stow_threshold=self.wind_stow_threshold,
                wind_stow_tilt=self.wind_stow_tilt,
                wind_stow_lookahead=self.wind_stow_lookahead,
                cloud_anticipate_lookahead=self.cloud_anticipate_lookahead,
                cloud_recovery_dni=self.cloud_recovery_dni,
                move_deadband_deg=self.move_deadband_deg,
                disable_guidance=self.disable_guidance,
            )

        # ── wind ──────────────────────────────────────────────────────────
        def _abs_step(self) -> int:
            return int(self.env.episode_tracker.episode_start_time_step) + int(self.env.time_step)

        def _effective_wind(self) -> float:
            data_wind = self._row_value(
                self.env._row_at_time_step(self.env.time_step), "wind_speed"
            )
            storm = 0.0
            if self._storm_wind is not None:
                i = self._abs_step()
                if 0 <= i < len(self._storm_wind):
                    storm = float(self._storm_wind[i])
            return max(float(data_wind), storm)

        def _peak_wind_ahead(self) -> float:
            peak = self._effective_wind()
            k = self.wind_stow_lookahead
            if self._storm_wind is not None and k > 0:
                i = self._abs_step()
                j0, j1 = i + 1, min(i + 1 + k, len(self._storm_wind))
                if j1 > j0:
                    peak = max(peak, float(self._storm_wind[j0:j1].max()))
            return peak

        def _wind_stow_now(self) -> bool:
            return self.wind_stow and self._peak_wind_ahead() >= self.wind_stow_threshold

        def _cloud_recovery_step(self):
            """First future step within lookahead where DNI recovers, else None."""
            k = self.cloud_anticipate_lookahead
            if k <= 0:
                return None
            ts = int(self.env.time_step)
            end = int(self.env.episode_tracker.episode_time_steps) - 1
            for off in range(1, k + 1):
                t = min(ts + off, end)
                if self._row_value(self.env._row_at_time_step(t), "dni") >= self.cloud_recovery_dni:
                    return t
            return None

        # ── base (RBC) action ─────────────────────────────────────────────
        def _base_target_angles(self) -> tuple:
            row = self.env._row_at_time_step(self.env.time_step)
            # Safety override: during a storm, lay the panel flat (stow) and hold
            # azimuth — beats every other base rule.
            if self._wind_stow_now():
                return float(self.env.panel_azimuth_deg), self.wind_stow_tilt
            # Lever 2: SLM 'hold' regime stops the schedule sweep. In-window:
            # azimuth stays, tilt goes flat-ish to keep collecting diffuse.
            # Out-of-window: freeze fully (overnight reset is free).
            if (
                self.hold_base
                and self._active_authority is not None
                and self._active_authority.regime == "hold"
            ):
                # Anticipation: if the sun returns within the lookahead window,
                # pre-position toward where it will be so the slew-limited panel
                # arrives aligned instead of catching up afterwards.
                t_rec = self._cloud_recovery_step()
                if t_rec is not None:
                    ideal = self._ideal_orientation_at(t_rec)
                    return ideal["azimuth"], ideal["tilt"]
                azimuth = float(self.env.panel_azimuth_deg)
                if self.env._is_tracking_window_active(row):
                    return azimuth, self.hold_tilt_deg
                return azimuth, float(self.env.panel_tilt_deg)
            mapping = self.env.OBSERVATION_COLUMN_MAPPING
            key = (int(row[mapping["hour"]]), int(row[mapping["minute"]]))
            if key in self._rbc_lookup:
                return self._rbc_lookup[key]
            # hold_last outside the tracking window
            return float(self.env.panel_azimuth_deg), float(self.env.panel_tilt_deg)

        def _base_action_delta(self):
            azimuth, tilt = self._base_target_angles()
            return self._np.asarray(
                self.env.action_from_target_angles(azimuth, tilt), dtype=self._np.float32
            )

        # ── authority ─────────────────────────────────────────────────────
        def _refresh_authority(self, bucket_changed: bool):
            if self.authority_mode == "fixed":
                self._active_authority = self.fixed_authority
            elif self.authority_mode == "random":
                if bucket_changed or self._random_authority is None:
                    regime = self._rng.choice(sorted(self.authority_table.keys()))
                    self._random_authority = self.authority_table[str(regime)]
                self._active_authority = self._random_authority
            else:  # 'goal'
                self._active_authority = authority_for(self.last_goal, self.authority_table)

        def _needed_residual(self, base_delta):
            """Residual (deg) that would point the panel at the ideal orientation."""
            ideal = self._ideal_orientation_at(int(self.env.time_step))
            desired = self._np.asarray(
                self.env.action_from_target_angles(ideal["azimuth"], ideal["tilt"]),
                dtype=self._np.float32,
            )
            return desired - base_delta

        def floor_action(self):
            """Action that reproduces the base exactly (the guaranteed floor)."""
            if self.action_mode == "gate":
                return -self._np.ones(2, dtype=self._np.float32)
            return self._np.zeros(2, dtype=self._np.float32)

        def oracle_action(self):
            """Action that follows the physics correction fully (upper bound)."""
            if self.action_mode == "gate":
                return self._np.ones(2, dtype=self._np.float32)
            authority = self._active_authority or self.authority_table["hold"]
            needed = self._needed_residual(self._base_action_delta())
            action = self._np.zeros(2, dtype=self._np.float32)
            if authority.azimuth_deg > 1e-9:
                action[0] = self._np.clip(needed[0] / authority.azimuth_deg, -1.0, 1.0)
            if authority.tilt_deg > 1e-9:
                action[1] = self._np.clip(needed[1] / authority.tilt_deg, -1.0, 1.0)
            return action

        def _context_vector(self):
            bucket = int(self.env.time_step) // max(1, self.query_interval_steps)
            bucket_changed = bucket != self.previous_goal_bucket
            base_context = super()._context_vector()  # refreshes last_goal
            if self.disable_guidance:
                # No SLM: strip the goal channels; keep physics features below.
                base_context = self._np.zeros_like(base_context)
            if bucket_changed:
                self.hour_residual_used_deg = 0.0
            self._refresh_authority(bucket_changed)
            authority = self._active_authority
            base_delta = self._base_action_delta()
            needed = self._needed_residual(base_delta)
            high = self._np.maximum(self._env_action_high, 1e-6)
            if np.isfinite(authority.hour_budget_deg) and authority.hour_budget_deg > 0:
                used_fraction = min(self.hour_residual_used_deg / authority.hour_budget_deg, 1.0)
            else:
                used_fraction = 0.0
            extra = self._np.array(
                [
                    float(self._np.clip(authority.azimuth_deg / high[0], 0.0, 1.0)),
                    float(self._np.clip(authority.tilt_deg / high[1], 0.0, 1.0)),
                    float(used_fraction),
                    float(self._np.clip((base_delta[0] / high[0] + 1.0) / 2.0, 0.0, 1.0)),
                    float(self._np.clip((base_delta[1] / high[1] + 1.0) / 2.0, 0.0, 1.0)),
                    float(self._np.clip((needed[0] / high[0] + 1.0) / 2.0, 0.0, 1.0)),
                    float(self._np.clip((needed[1] / high[1] + 1.0) / 2.0, 0.0, 1.0)),
                ],
                dtype=self._np.float32,
            )
            return self._np.concatenate([base_context, extra]).astype(self._np.float32)

        # ── episode flow ──────────────────────────────────────────────────
        def reset(self):
            self.hour_residual_used_deg = 0.0
            self._active_authority = None
            self._random_authority = None
            self._rng = np.random.RandomState(self._seed)
            self.wind_metric_log = []
            return super().reset()

        def step(self, action):
            a = self._np.clip(
                self._np.asarray(action, dtype=self._np.float32).reshape(-1)[:2], -1.0, 1.0
            )
            eff_wind = self._effective_wind()
            stow_now = self._wind_stow_now()
            authority = self._active_authority or self.authority_table["hold"]
            base = self._base_action_delta()
            auth_vec = self._np.array(
                [authority.azimuth_deg, authority.tilt_deg], dtype=self._np.float32
            )
            if stow_now:
                # Safety: no residual — the panel goes to (and stays at) the stow
                # pose defined by the base override. RL cannot pull it off.
                residual = self._np.zeros(2, dtype=self._np.float32)
            elif self.action_mode == "gate":
                gate = (a + 1.0) / 2.0
                oracle_residual = self._np.clip(
                    self._needed_residual(base), -auth_vec, auth_vec
                )
                residual = (gate * oracle_residual).astype(self._np.float32)
            else:
                residual = (a * auth_vec).astype(self._np.float32)
            # Hard hour-budget projection: residual motion (not RBC motion) is
            # capped, so the guaranteed base behaviour is never blocked.
            if np.isfinite(authority.hour_budget_deg):
                remaining = max(authority.hour_budget_deg - self.hour_residual_used_deg, 0.0)
                needed = float(self._np.abs(residual).sum())
                if needed > remaining:
                    residual *= (remaining / needed) if needed > 0 else 0.0
            command = self._np.clip(base + residual, -self._slew, self._slew)
            # Deadband: below the (regime-scaled) threshold, don't move that axis.
            # Skipped, the drift accumulates into the next step's base delta until
            # it exceeds the deadband — coarse tracking with fewer activations.
            # Never applied during a safety stow.
            if self.move_deadband_deg > 0.0 and not stow_now:
                db = self.move_deadband_deg * _REGIME_DEADBAND_FACTOR.get(authority.regime, 1.0)
                if abs(float(command[0])) < db:
                    command[0] = 0.0
                if abs(float(command[1])) < 0.6 * db:
                    command[1] = 0.0
            applied_residual = command - base

            obs, reward, done, info = self.env.step([command.astype(self._np.float32)])
            self.hour_residual_used_deg += float(self._np.abs(applied_residual).sum())
            # Read the raw append-only row dict instead of rebuilding the whole
            # episode DataFrame; .get() behaves identically on a dict.
            _rows = self.env._history_rows
            if len(_rows) > 0:
                row = _rows[-1]
                self.hour_motion_so_far_deg += abs(
                    float(row.get("panel_azimuth_delta_deg", 0.0))
                ) + abs(float(row.get("panel_tilt_delta_deg", 0.0)))

            self.wind_metric_log.append((float(eff_wind), float(self.env.panel_tilt_deg)))

            enriched = self._enrich_normalized_obs(obs[0])
            info = dict(info or {})
            info.update(
                {
                    "residual_regime": authority.regime,
                    "residual_authority_azimuth_deg": float(authority.azimuth_deg),
                    "residual_authority_tilt_deg": float(authority.tilt_deg),
                    "residual_base_azimuth_delta_deg": float(base[0]),
                    "residual_base_tilt_delta_deg": float(base[1]),
                    "residual_applied_azimuth_deg": float(applied_residual[0]),
                    "residual_applied_tilt_deg": float(applied_residual[1]),
                    "residual_hour_used_deg": float(self.hour_residual_used_deg),
                    "effective_wind_ms": float(eff_wind),
                    "wind_stow_active": bool(stow_now),
                }
            )
            return enriched, float(reward[0]) * self.reward_scale, bool(done), info

    return ResidualTrackerWrapper


class _ResidualLazy:
    """Defer class construction until llm/rbc imports are available."""

    _cls = None

    def __call__(self, *args, **kwargs):
        if _ResidualLazy._cls is None:
            _ResidualLazy._cls = _build_residual_wrapper_class()
        return _ResidualLazy._cls(*args, **kwargs)


ResidualTrackerWrapper = _ResidualLazy()


def train_residual_sac(
    agent,
    train_env,
    memory,
    kl_memory,
    cfg: dict,
    start_episode: int = 1,
    label: str = "Residual-SAC",
    eval_env=None,
):
    """Train SAC-Auto on a ResidualTrackerWrapper.

    Mirrors the notebook train_sac_auto loop, but runs the deterministic
    best-policy evaluation through a residual eval env (the plain
    evaluate_tracker_policy would rebuild a non-residual env with mismatched
    observation/action spaces).
    """
    from utilities import metric_from_eval

    if eval_env is None:
        eval_env = train_env.clone_for_eval()

    def _deterministic_eval():
        state = eval_env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state, eval=True, mode="running")
            state, _, done, _ = eval_env.step(action)
        return eval_env.evaluate()

    best_metric = cfg.get("best_metric", "total_harvested_energy_kwh")
    best_value = -np.inf
    best_episode = None
    best_policy = None
    best_log_alpha = None
    total_steps = 0
    updates = 0
    history = []

    for episode in range(int(start_episode), int(cfg["num_episodes"]) + 1):
        train_env.set_reward_episode(episode)
        state = train_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        indicators = np.zeros(6, dtype=np.float32)

        while not done:
            if total_steps < int(cfg.get("start_steps", 0)):
                action = train_env.action_space.sample()
                log_prob = np.zeros(1, dtype=np.float32)
            else:
                action, log_prob = agent.select_action(state)

            if len(memory) > int(cfg["batch_size"]):
                for _ in range(int(cfg["updates_per_step"])):
                    indicators += agent.update_parameters(
                        memory, kl_memory, int(cfg["batch_size"]), updates
                    )
                    updates += 1
            else:
                indicators[0] += agent.log_alpha.item()

            next_state, reward, done, _ = train_env.step(action)
            mask = 0.0 if done else 1.0
            memory.push(state, action, log_prob, reward, next_state, mask)
            kl_memory.push(state, action, log_prob, reward, next_state, mask)
            state = next_state
            ep_steps += 1
            total_steps += 1
            ep_reward += reward

        indicators /= max(ep_steps, 1)
        ep_eval = train_env.evaluate()
        metric_vals = dict(zip(ep_eval["cost_function"], ep_eval["value"]))

        row = {
            "episode": episode,
            "total_numsteps": total_steps,
            "episode_steps": ep_steps,
            "scaled_reward": ep_reward,
            "total_harvested_energy_kwh": metric_vals.get("total_harvested_energy_kwh", np.nan),
            "tracking_efficiency": metric_vals.get("tracking_efficiency", np.nan),
            "movement_magnitude_deg": metric_vals.get("movement_magnitude_deg", np.nan),
            "tracking_smoothness_jerk_deg2": metric_vals.get("tracking_smoothness_jerk_deg2", np.nan),
            "mean_tracking_error_deg": metric_vals.get("mean_tracking_error_deg", np.nan),
            "log_alpha": float(indicators[0]),
            "critic_1_loss": float(indicators[1]),
            "critic_2_loss": float(indicators[2]),
            "policy_loss": float(indicators[3]),
            "alpha_loss": float(indicators[4]),
            "entropy": float(indicators[5]),
        }

        if episode == 1 or episode % int(cfg.get("eval_interval", 10)) == 0:
            det_eval = _deterministic_eval()
            det_vals = dict(zip(det_eval["cost_function"], det_eval["value"]))
            det_score = metric_from_eval(det_eval, best_metric)
            row[f"eval_{best_metric}"] = det_score
            row["eval_tracking_efficiency"] = det_vals.get("tracking_efficiency", np.nan)
            row["eval_movement_magnitude_deg"] = det_vals.get("movement_magnitude_deg", np.nan)
            if det_score > best_value:
                best_value = det_score
                best_episode = episode
                best_policy = copy.deepcopy(agent.policy.state_dict())
                best_log_alpha = agent.log_alpha.detach().clone()

        history.append(row)
        print(
            f"{label} ep {episode:3d}/{cfg['num_episodes']} | "
            f"energy={row['total_harvested_energy_kwh']:.3f} kWh | "
            f"movement={row['movement_magnitude_deg']:.0f} deg | "
            f"log_alpha={row['log_alpha']:.3f}",
            flush=True,
        )

    if best_policy is not None:
        agent.policy.load_state_dict(best_policy)
        agent.log_alpha.data.copy_(best_log_alpha)
        print(f"Loaded best policy from episode {best_episode} ({best_metric}={best_value:.3f})")

    return {
        "history": history,
        "best_metric_value": best_value,
        "best_metric_name": best_metric,
        "best_episode": best_episode,
        "best_policy_state": best_policy,
        "best_log_alpha": best_log_alpha,
        "total_numsteps": total_steps,
        "updates": updates,
    }


# ── replay seeding ─────────────────────────────────────────────────────────────

def seed_replay_with_rbc(memory, kl_memory, schema_template: dict, episodes: int = 1) -> int:
    """Pre-populate replay buffers with RBC rollouts before RL training.

    Fills both memory and kl_memory with high-quality RBC demonstrations so
    the agent learns from good behaviour from the start rather than random noise.
    Returns the total number of seeded transitions.
    """
    from utilities import SolarTrackerMetaSACWrapper, _ensure_training_imports
    from rbc import SolarTrackerRBC, average_tracker_schedule

    _ensure_training_imports()

    seeded_steps = 0
    for ep in range(int(episodes)):
        seed_env = SolarTrackerMetaSACWrapper(schema_template, reward_scale=50.0)
        schedule = average_tracker_schedule(seed_env.env, use_episode_window=False)
        rbc = SolarTrackerRBC(seed_env.env, schedule=schedule, outside_tracking_policy="hold_last")
        state = seed_env.reset()
        done = False
        while not done:
            action = np.asarray(
                rbc.predict(seed_env.env.observations, deterministic=True)[0], dtype=np.float32
            )
            next_state, reward, done, _ = seed_env.step(action)
            mask = 0.0 if done else 1.0
            log_prob = np.zeros(1, dtype=np.float32)
            memory.push(state, action, log_prob, reward, next_state, mask)
            kl_memory.push(state, action, log_prob, reward, next_state, mask)
            state = next_state
            seeded_steps += 1
        print(f"RBC seed episode {ep + 1}/{episodes}: {seeded_steps} total transitions")
    return seeded_steps
