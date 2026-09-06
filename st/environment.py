"""CityLearn-style environment for single dual-axis solar tracker control."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from gym import Env, spaces
except Exception:  # pragma: no cover
    try:
        from gymnasium import Env, spaces
    except Exception:  # pragma: no cover
        class Env:  # type: ignore
            """Fallback minimal Env base when gym/gymnasium are unavailable."""

        class _Box:
            def __init__(self, low, high, dtype=np.float32):
                self.low = np.array(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype)
                self.dtype = dtype
                self.shape = self.low.shape

            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)

        class _Spaces:
            Box = _Box

        spaces = _Spaces()

from citylearn.base import Environment, EpisodeTracker
from st.metrics import metrics_to_dataframe, summarize_tracker_metrics
from st.reward import SolarTrackerReward
from st.schema import load_schema
from st.tracker import (
    PanelModel,
    compute_plane_of_array_irradiance,
    compute_power_and_energy,
    incidence_cosine,
    wrap_azimuth_degrees,
)


class SolarTrackerEnv(Environment, Env):
    """Simulation environment for direct control of panel azimuth and tilt.

    Modeling assumptions:
    - Control actions are absolute panel setpoints or bounded deltas, depending
      on schema["action_mode"].
    - Panel azimuth convention is [0, 360) and constrained by schema bounds.
    - PV irradiance uses DNI + DHI POA model and yields zero for zenith >= 90 deg.
    """

    OBSERVATION_COLUMN_MAPPING = {
        "month": "Month",
        "day": "Day",
        "hour": "Hour",
        "minute": "Minute",
        "solar_azimuth_angle": "Solar Azimuth Angle",
        "solar_zenith_angle": "Solar Zenith Angle",
        "next_10min_solar_azimuth": "next_10min_solar_azimuth",
        "next_10min_solar_zenith": "next_10min_solar_zenith",
        "temperature": "Temperature",
        "dhi": "DHI",
        "dni": "DNI",
        "next_10min_DNI": "next_10min_DNI",
        "next_10min_DHI": "next_10min_DHI",
        "next_30min_average_DNI": "next_30min_average_DNI",
        "ghi": "GHI",
        "wind_speed": "Wind Speed",
        "cloud_type": "Cloud Type",
        "clearsky_dhi": "Clearsky DHI",
        "clearsky_dni": "Clearsky DNI",
        "clearsky_ghi": "Clearsky GHI",
        "relative_humidity": "Relative Humidity",
        "surface_albedo": "Surface Albedo",
    }

    def __init__(
        self,
        schema: Union[str, Path, Mapping[str, Any]],
        root_directory: Union[str, Path] = None,
        simulation_start_time_step: int = None,
        simulation_end_time_step: int = None,
        episode_time_steps: Union[int, List[Tuple[int, int]]] = None,
        rolling_episode_split: bool = None,
        random_episode_split: bool = None,
        seconds_per_time_step: float = None,
        reward_function: SolarTrackerReward = None,
        central_agent: bool = None,
        random_seed: int = None,
        **kwargs: Any,
    ):
        self.schema = load_schema(schema)
        self._history_rows: List[Dict[str, float]] = []
        # Cached DataFrame for the `history` property. Rebuilding it on every
        # access made each rollout O(n^2) in episode length.
        self._history_df: Optional[pd.DataFrame] = None
        self._history_df_len: int = -1
        self.__rewards: List[List[float]] = []
        self.__episode_rewards: List[Mapping[str, List[float]]] = []

        (
            self.root_directory,
            self.data,
            self.episode_time_steps,
            self.rolling_episode_split,
            self.random_episode_split,
            parsed_seconds_per_time_step,
            self.central_agent,
            self.observation_config,
            self.action_config,
            self.panel_model,
            self.initial_panel_orientation,
            self.baseline_config,
            self.reward_function,
            self.episode_tracker,
        ) = self._load(
            root_directory=root_directory,
            simulation_start_time_step=simulation_start_time_step,
            simulation_end_time_step=simulation_end_time_step,
            episode_time_steps=episode_time_steps,
            rolling_episode_split=rolling_episode_split,
            random_episode_split=random_episode_split,
            seconds_per_time_step=seconds_per_time_step,
            reward_function=reward_function,
            central_agent=central_agent,
            random_seed=random_seed,
        )

        super().__init__(
            seconds_per_time_step=parsed_seconds_per_time_step,
            random_seed=random_seed,
            episode_tracker=self.episode_tracker,
        )

        self.active_observations = [name for name, metadata in self.observation_config.items() if metadata["active"]]
        self.active_actions = [name for name, metadata in self.action_config.items() if metadata["active"]]
        self._action_low = np.array([self.action_config[name]["low"] for name in self.active_actions], dtype=np.float32)
        self._action_high = np.array([self.action_config[name]["high"] for name in self.active_actions], dtype=np.float32)
        self._panel_low = np.array([self.panel_angle_bounds["azimuth"]["low"], self.panel_angle_bounds["tilt"]["low"]], dtype=np.float32)
        self._panel_high = np.array([self.panel_angle_bounds["azimuth"]["high"], self.panel_angle_bounds["tilt"]["high"]], dtype=np.float32)
        self._observation_low, self._observation_high = self._estimate_observation_bounds()
        self.reward_function.env_metadata = self.get_metadata()
        # Load pre-computed static per-step cache (ideal orientation, POA, power, time features).
        # Generated by finetune/precompute_env_cache.py; auto-detected from data filename.
        self._step_cache: Optional[dict] = self._try_load_step_cache()
        self.reset()
        self.episode_tracker.reset_episode_index()

    @property
    def time_steps(self) -> int:
        return self.episode_tracker.episode_time_steps

    @property
    def done(self) -> bool:
        return self.time_step == self.time_steps - 1

    @property
    def observation_space(self) -> List[spaces.Box]:
        return [spaces.Box(low=self._observation_low, high=self._observation_high, dtype=np.float32)]

    @property
    def action_space(self) -> List[spaces.Box]:
        return [spaces.Box(low=self._action_low, high=self._action_high, dtype=np.float32)]

    @property
    def observation_names(self) -> List[List[str]]:
        return [self.active_observations]

    @property
    def action_names(self) -> List[List[str]]:
        return [self.active_actions]

    @property
    def panel_azimuth_deg(self) -> float:
        return self._panel_azimuth_deg

    @property
    def panel_tilt_deg(self) -> float:
        return self._panel_tilt_deg

    @property
    def panel_azimuth_bounds(self) -> Tuple[float, float]:
        return float(self._panel_low[0]), float(self._panel_high[0])

    @property
    def panel_tilt_bounds(self) -> Tuple[float, float]:
        return float(self._panel_low[1]), float(self._panel_high[1])

    @property
    def observations(self) -> List[List[float]]:
        return [list(self._current_observation().values())]

    @property
    def rewards(self) -> List[List[float]]:
        return self.__rewards

    @property
    def episode_rewards(self) -> List[Mapping[str, List[float]]]:
        return self.__episode_rewards

    @property
    def history(self) -> pd.DataFrame:
        # Rows are append-only between resets, so the row count is a sufficient
        # cache key; reset() forces invalidation by setting _history_df_len=-1.
        n = len(self._history_rows)
        if self._history_df_len != n:
            self._history_df = pd.DataFrame(self._history_rows)
            self._history_df_len = n
        return self._history_df

    def wrap_azimuth(self, angle_deg: float) -> float:
        return wrap_azimuth_degrees(angle_deg)

    def clip_action(self, action: Union[List[float], np.ndarray]) -> np.ndarray:
        return np.clip(np.array(action, dtype=np.float32), self._action_low, self._action_high)

    def apply_action_deadband(self, action: np.ndarray) -> np.ndarray:
        """Suppress tiny delta commands that cause tracker hunting/jitter.

        Deadband is applied only in delta-action mode. It represents actuator
        resolution/control tolerance: if the commanded movement is smaller than
        the configured threshold, the tracker holds its current angle.
        """

        if self.action_mode != "delta":
            return action

        deadband = np.array(
            [self.action_deadband_azimuth_deg, self.action_deadband_tilt_deg],
            dtype=np.float32,
        )
        return np.where(np.abs(action) < deadband, 0.0, action).astype(np.float32)

    def clip_panel_angles(self, angles: Union[List[float], np.ndarray]) -> np.ndarray:
        """Clip physical panel angles to safe mechanical limits."""

        return np.clip(np.array(angles, dtype=np.float32), self._panel_low, self._panel_high)

    def action_from_target_angles(self, azimuth_deg: float, tilt_deg: float) -> np.ndarray:
        """Convert desired absolute panel angles into this environment's action format."""

        target = self.clip_panel_angles([azimuth_deg, tilt_deg])

        if self.action_mode == "delta":
            delta = target - np.array([self.panel_azimuth_deg, self.panel_tilt_deg], dtype=np.float32)
            return self.clip_action(delta)

        return self.clip_action(target)

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            "reward_function": self.reward_function.__class__.__name__,
            "central_agent": self.central_agent,
            "buildings": [{"name": "Tracker_1"}],
            "panel": {
                "area_m2": self.panel_model.area_m2,
                "efficiency": self.panel_model.efficiency,
                "performance_ratio": self.panel_model.performance_ratio,
                "rated_power_kw": self.panel_model.effective_max_power_kw(),
                "environment_file": __file__,
            },
        }

    def get_info(self) -> Mapping[str, Any]:
        if len(self._history_rows) == 0:
            return {}

        last = self._history_rows[-1]
        return {
            "generated_energy_kwh": last["generated_energy_kwh"],
            "generated_power_kw": last["generated_power_kw"],
            "poa_w_per_m2": last["poa_w_per_m2"],
            "tracking_error_deg": last["tracking_error_deg"],
        }

    def next_time_step(self):
        super().next_time_step()

    def reset(self) -> List[List[float]]:
        super().reset()
        self.episode_tracker.next_episode(
            self.episode_time_steps,
            self.rolling_episode_split,
            self.random_episode_split,
            self.random_seed,
        )
        initial_action = self.clip_panel_angles(
            [float(self.initial_panel_orientation["azimuth"]), float(self.initial_panel_orientation["tilt"])]
        )
        self._panel_azimuth_deg = float(initial_action[0])
        self._panel_tilt_deg = float(initial_action[1])
        self._previous_panel_azimuth_deg = self._panel_azimuth_deg
        self._previous_panel_tilt_deg = self._panel_tilt_deg
        self._previous_generated_energy_kwh = 0.0
        self._history_rows = []
        self._history_df_len = -1
        self.__rewards = [[]]
        self.reward_function.reset()
        self._append_current_time_step_row(azimuth_delta_deg=0.0, tilt_delta_deg=0.0)
        return self.observations

    def step(self, actions: List[List[float]]) -> Tuple[List[List[float]], List[float], bool, Dict[str, Any]]:
        self.next_time_step()
        parsed_action = self._parse_actions(actions)
        clipped_action = self.apply_action_deadband(self.clip_action(parsed_action))

        if self.action_mode == "delta":
            target_angles = np.array([self._panel_azimuth_deg, self._panel_tilt_deg], dtype=np.float32) + clipped_action
            new_angles = self.clip_panel_angles(target_angles)
        else:
            new_angles = self.clip_panel_angles(clipped_action)

        automatic_orientation_reset = False

        if self.auto_reset_to_initial_orientation and not self._is_tracking_window_active(self._row_at_time_step(self.time_step)):
            initial_angles = self.clip_panel_angles(
                [self.initial_panel_orientation["azimuth"], self.initial_panel_orientation["tilt"]]
            )
            new_angles = initial_angles
            automatic_orientation_reset = True

        new_azimuth = float(new_angles[0])
        new_tilt = float(new_angles[1])
        actual_azimuth_delta = new_azimuth - self._panel_azimuth_deg
        actual_tilt_delta = new_tilt - self._panel_tilt_deg
        azimuth_delta = 0.0 if automatic_orientation_reset else actual_azimuth_delta
        tilt_delta = 0.0 if automatic_orientation_reset else actual_tilt_delta
        self._previous_panel_azimuth_deg = self._panel_azimuth_deg
        self._previous_panel_tilt_deg = self._panel_tilt_deg
        self._panel_azimuth_deg = new_azimuth
        self._panel_tilt_deg = new_tilt
        self._append_current_time_step_row(
            azimuth_delta_deg=azimuth_delta,
            tilt_delta_deg=tilt_delta,
            automatic_orientation_reset=automatic_orientation_reset,
            actual_azimuth_delta_deg=actual_azimuth_delta,
            actual_tilt_delta_deg=actual_tilt_delta,
        )
        reward_observation = [
            {
                "generated_energy_kwh": self._history_rows[-1]["generated_energy_kwh"],
                "generated_power_kw": self._history_rows[-1]["generated_power_kw"],
                "ideal_generated_energy_kwh": self._history_rows[-1]["ideal_generated_energy_kwh"],
                "ideal_generated_power_kw": self._history_rows[-1]["ideal_generated_power_kw"],
                "panel_azimuth_delta_deg": azimuth_delta,
                "panel_tilt_delta_deg": tilt_delta,
            }
        ]
        reward = self.reward_function.calculate(reward_observation)
        self.__rewards.append(reward)

        if self.done:
            rewards = np.array(self.__rewards[1:], dtype=np.float32)
            self.__episode_rewards.append(
                {
                    "min": rewards.min(axis=0).tolist(),
                    "max": rewards.max(axis=0).tolist(),
                    "sum": rewards.sum(axis=0).tolist(),
                    "mean": rewards.mean(axis=0).tolist(),
                }
            )

        return self.observations, reward, self.done, dict(self.get_info())

    def evaluate(self, include_baselines: bool = True) -> pd.DataFrame:
        """Return CityLearn-like KPI table for the current simulation run."""

        baseline_metrics = self._compute_baseline_metrics() if include_baselines else {}
        metrics = summarize_tracker_metrics(
            history=self.history,
            seconds_per_time_step=self.seconds_per_time_step,
            rated_power_kw=self.panel_model.effective_max_power_kw(),
            fixed_energy_kwh=baseline_metrics.get("fixed_energy_kwh"),
            rule_energy_kwh=baseline_metrics.get("rule_energy_kwh"),
        )
        return metrics_to_dataframe(metrics)

    def _compute_baseline_metrics(self) -> Dict[str, float]:
        fixed_config = self.baseline_config.get("fixed", {})
        fixed_env = self._new_env_for_baseline(
            {
                "initial_orientation": {
                    "azimuth_deg": fixed_config.get("azimuth_deg", 180.0),
                    "tilt_deg": fixed_config.get("tilt_deg", 30.0),
                },
                "auto_reset_to_initial_orientation": False,
            }
        )
        rule_env = self._new_env_for_baseline()
        fixed_agent = self._instantiate_agent(
            self.schema["baselines"]["fixed"]["agent"]["type"],
            fixed_env,
            {
                **self.schema["baselines"]["fixed"]["agent"].get("attributes", {}),
                "fixed_azimuth_deg": fixed_config.get("azimuth_deg", 180.0),
                "fixed_tilt_deg": fixed_config.get("tilt_deg", 30.0),
            },
        )
        rule_agent = self._instantiate_agent(
            self.schema["baselines"]["rule_based"]["agent"]["type"],
            rule_env,
            self.schema["baselines"]["rule_based"]["agent"].get("attributes", {}),
        )
        fixed_metrics = self._rollout_agent(fixed_env, fixed_agent)
        rule_metrics = self._rollout_agent(rule_env, rule_agent)
        return {"fixed_energy_kwh": fixed_metrics["total_harvested_energy_kwh"], "rule_energy_kwh": rule_metrics["total_harvested_energy_kwh"]}

    def _new_env_for_baseline(self, schema_updates: Mapping[str, Any] = None) -> "SolarTrackerEnv":
        schema = deepcopy(self.schema)

        if schema_updates:
            schema.update(schema_updates)

        return SolarTrackerEnv(
            schema=schema,
            simulation_start_time_step=self.episode_tracker.episode_start_time_step,
            simulation_end_time_step=self.episode_tracker.episode_end_time_step,
            episode_time_steps=self.time_steps,
            rolling_episode_split=False,
            random_episode_split=False,
            random_seed=self.random_seed,
        )

    def _rollout_agent(self, env: "SolarTrackerEnv", agent) -> Dict[str, float]:
        observations = env.reset()
        done = False

        while not done:
            action = agent.predict(observations, deterministic=True)
            observations, _, done, _ = env.step(action)

        return summarize_tracker_metrics(
            history=env.history,
            seconds_per_time_step=env.seconds_per_time_step,
            rated_power_kw=env.panel_model.effective_max_power_kw(),
        )

    def _instantiate_agent(self, type_path: str, env: "SolarTrackerEnv", attributes: Mapping[str, Any]):
        module_name = ".".join(type_path.split(".")[:-1])
        class_name = type_path.split(".")[-1]
        constructor = getattr(importlib.import_module(module_name), class_name)
        return constructor(env=env, **attributes)

    def _parse_actions(self, actions: List[List[float]]) -> np.ndarray:
        if self.central_agent:
            parsed = np.array(actions[0], dtype=np.float32)
        else:
            if isinstance(actions[0], (float, int, np.floating)):
                parsed = np.array(actions, dtype=np.float32)
            else:
                parsed = np.array(actions[0], dtype=np.float32)

        assert parsed.shape[0] == len(self.active_actions), (
            f"Expected {len(self.active_actions)} actions, got {parsed.shape[0]}."
        )
        return parsed

    def _try_load_step_cache(self) -> Optional[dict]:
        """Auto-detect and load pre-computed static per-step cache for this dataset.

        Looks for <data_stem>_precomputed.npz in root_directory.
        Generate with: python finetune/precompute_env_cache.py <csv_path>
        """
        data_filename = self.schema.get("data", {}).get("filename", "")
        stem = Path(data_filename).stem
        cache_path = Path(self.root_directory) / f"{stem}_precomputed.npz"
        if cache_path.exists():
            try:
                cache = dict(np.load(str(cache_path)))
                print(f"[env] Step cache loaded: {cache_path.name} ({len(next(iter(cache.values())))} rows)",
                      file=sys.stderr, flush=True)
                return cache
            except Exception as exc:
                print(f"[env] Warning: step cache load failed ({cache_path.name}): {exc}",
                      file=sys.stderr, flush=True)
        return None

    def _row_at_time_step(self, time_step: int) -> pd.Series:
        index = self.episode_tracker.episode_start_time_step + time_step
        return self.data.iloc[index]

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Return a stable irradiance ratio without exploding at night."""

        denominator = float(denominator)
        if abs(denominator) < 1e-9:
            return 0.0

        return float(numerator) / denominator

    def _day_of_year(self, row: pd.Series) -> float:
        """Infer day of year from the configured datetime column or month/day."""

        datetime_column = self.schema.get("data", {}).get("datetime_column")

        if datetime_column in row.index:
            try:
                return float(pd.to_datetime(row[datetime_column]).dayofyear)
            except Exception:
                pass

        try:
            month = int(row[self.OBSERVATION_COLUMN_MAPPING["month"]])
            day = int(row[self.OBSERVATION_COLUMN_MAPPING["day"]])
            return float(pd.Timestamp(year=2020, month=month, day=day).dayofyear)
        except Exception:
            return 1.0

    def _time_features(self, row: pd.Series) -> Dict[str, float]:
        """Cyclic time encodings that avoid discontinuities at midnight/year-end."""

        hour = float(row[self.OBSERVATION_COLUMN_MAPPING["hour"]])
        minute = float(row[self.OBSERVATION_COLUMN_MAPPING["minute"]])
        time_of_day_angle = 2.0 * np.pi * ((hour * 60.0) + minute) / (24.0 * 60.0)
        day_of_year_angle = 2.0 * np.pi * (self._day_of_year(row) - 1.0) / 366.0

        return {
            "sin_time_of_day": float(np.sin(time_of_day_angle)),
            "cos_time_of_day": float(np.cos(time_of_day_angle)),
            "sin_day_of_year": float(np.sin(day_of_year_angle)),
            "cos_day_of_year": float(np.cos(day_of_year_angle)),
        }

    def _is_tracking_window_active(self, row: pd.Series) -> bool:
        """Return whether agent actions should control the tracker at this timestamp."""

        hour = float(row[self.OBSERVATION_COLUMN_MAPPING["hour"]])
        minute = float(row[self.OBSERVATION_COLUMN_MAPPING["minute"]])
        time_decimal = hour + minute / 60.0
        return self.tracking_start_hour <= time_decimal < self.tracking_end_hour

    def _closest_allowed_azimuth(self, solar_azimuth_deg: float) -> float:
        """Return allowed azimuth closest to the sun in circular degrees."""

        solar_azimuth = self.wrap_azimuth(solar_azimuth_deg)
        low = float(self._panel_low[0])
        high = float(self._panel_high[0])

        if low <= solar_azimuth <= high:
            return solar_azimuth

        def circular_distance(a: float, b: float) -> float:
            return abs((a - b + 180.0) % 360.0 - 180.0)

        return low if circular_distance(solar_azimuth, low) <= circular_distance(solar_azimuth, high) else high

    def _ideal_panel_orientation(self, dni: float, dhi: float, solar_zenith_deg: float, solar_azimuth_deg: float) -> Tuple[float, float]:
        """Return POA-maximizing panel orientation under mechanical limits.

        For the DNI + isotropic DHI model, azimuth should minimize incidence
        angle to the sun, while tilt maximizes:

            DHI/2 + (DNI*cos(theta_z) + DHI/2)*cos(beta)
                  + DNI*sin(theta_z)*cos(delta_azimuth)*sin(beta)

        This is the true bounded optimum for the same irradiance model used by
        the environment, so tracking-efficiency denominators remain meaningful.
        """

        ideal_azimuth = self._closest_allowed_azimuth(solar_azimuth_deg)

        if float(solar_zenith_deg) >= 90.0 or float(dni) < float(self.dni_min_threshold):
            return ideal_azimuth, float(self._panel_low[1])

        theta_z = np.radians(np.clip(float(solar_zenith_deg), 0.0, 180.0))
        delta_azimuth = np.radians(
            (self.wrap_azimuth(solar_azimuth_deg) - self.wrap_azimuth(ideal_azimuth) + 180.0) % 360.0 - 180.0
        )
        a = float(dni) * np.cos(theta_z) + float(dhi) / 2.0
        b = float(dni) * np.sin(theta_z) * np.cos(delta_azimuth)
        unconstrained_tilt = np.degrees(np.arctan2(b, a))
        ideal_tilt = float(np.clip(unconstrained_tilt, self._panel_low[1], self._panel_high[1]))
        return ideal_azimuth, ideal_tilt

    def _current_observation(self) -> Dict[str, float]:
        row = self._row_at_time_step(self.time_step)
        _cache = self._step_cache
        if _cache is not None:
            _abs = self.episode_tracker.episode_start_time_step + self.time_step
            time_features = {
                "sin_time_of_day": float(_cache["sin_time_of_day"][_abs]),
                "cos_time_of_day": float(_cache["cos_time_of_day"][_abs]),
                "sin_day_of_year": float(_cache["sin_day_of_year"][_abs]),
                "cos_day_of_year": float(_cache["cos_day_of_year"][_abs]),
            }
        else:
            time_features = self._time_features(row)
        last_history_row = self._history_rows[-1] if self._history_rows else {}
        observation = {}

        for name in self.active_observations:
            if name in self.OBSERVATION_COLUMN_MAPPING:
                observation[name] = float(row[self.OBSERVATION_COLUMN_MAPPING[name]])
            elif name in time_features:
                observation[name] = float(time_features[name])
            elif name == "dni_clear_sky_ratio":
                observation[name] = self._safe_ratio(
                    row[self.OBSERVATION_COLUMN_MAPPING["dni"]],
                    row[self.OBSERVATION_COLUMN_MAPPING["clearsky_dni"]],
                )
            elif name == "dhi_clear_sky_ratio":
                observation[name] = self._safe_ratio(
                    row[self.OBSERVATION_COLUMN_MAPPING["dhi"]],
                    row[self.OBSERVATION_COLUMN_MAPPING["clearsky_dhi"]],
                )
            elif name == "current_panel_tilt_angle":
                observation[name] = float(self.panel_tilt_deg)
            elif name == "current_panel_azimuth_angle":
                observation[name] = float(self.panel_azimuth_deg)
            elif name == "previous_action_tilt_angle":
                observation[name] = float(self._previous_panel_tilt_deg)
            elif name == "previous_action_azimuth_angle":
                observation[name] = float(self._previous_panel_azimuth_deg)
            elif name == "previous_generated_energy_kwh":
                observation[name] = float(self._previous_generated_energy_kwh)
            elif name in {"generated_energy_kwh", "current_generated_energy_kwh"}:
                observation[name] = float(last_history_row.get("generated_energy_kwh", 0.0))
            elif name in {
                "panel_azimuth_delta_deg",
                "panel_tilt_delta_deg",
                "tracking_error_deg",
                "cos_incidence",
                "poa_w_per_m2",
            }:
                observation[name] = float(last_history_row.get(name, 0.0))
            else:
                raise KeyError(f"Unknown observation key in schema: {name}")

        return observation

    def _append_current_time_step_row(
        self,
        azimuth_delta_deg: float,
        tilt_delta_deg: float,
        automatic_orientation_reset: bool = False,
        actual_azimuth_delta_deg: float = None,
        actual_tilt_delta_deg: float = None,
    ):
        row = self._row_at_time_step(self.time_step)
        solar_azimuth = float(row[self.OBSERVATION_COLUMN_MAPPING["solar_azimuth_angle"]])
        solar_zenith = float(row[self.OBSERVATION_COLUMN_MAPPING["solar_zenith_angle"]])
        dni = float(row[self.OBSERVATION_COLUMN_MAPPING["dni"]])
        dhi = float(row[self.OBSERVATION_COLUMN_MAPPING["dhi"]])
        ghi = float(row[self.OBSERVATION_COLUMN_MAPPING["ghi"]])
        temperature = float(row[self.OBSERVATION_COLUMN_MAPPING["temperature"]])
        wind_speed = float(row[self.OBSERVATION_COLUMN_MAPPING["wind_speed"]])
        cloud_type = float(row[self.OBSERVATION_COLUMN_MAPPING["cloud_type"]])
        clearsky_dni = float(row[self.OBSERVATION_COLUMN_MAPPING["clearsky_dni"]])
        clearsky_dhi = float(row[self.OBSERVATION_COLUMN_MAPPING["clearsky_dhi"]])
        panel_azimuth_360 = self.wrap_azimuth(self.panel_azimuth_deg)
        solar_azimuth_360 = self.wrap_azimuth(solar_azimuth)
        poa = compute_plane_of_array_irradiance(
            dni=dni,
            dhi=dhi,
            solar_zenith_deg=solar_zenith,
            solar_azimuth_deg=solar_azimuth_360,
            panel_tilt_deg=self.panel_tilt_deg,
            panel_azimuth_deg=panel_azimuth_360,
            dni_min_threshold=self.dni_min_threshold,
        )
        power = compute_power_and_energy(
            poa_irradiance_w_per_m2=poa["poa_w_per_m2"],
            ambient_temperature_c=temperature,
            panel=self.panel_model,
            seconds_per_time_step=self.seconds_per_time_step,
        )
        # Use pre-computed cache for static ideal values and time features when available.
        _cache = self._step_cache
        if _cache is not None:
            _abs = self.episode_tracker.episode_start_time_step + self.time_step
            time_features = {
                "sin_time_of_day": float(_cache["sin_time_of_day"][_abs]),
                "cos_time_of_day": float(_cache["cos_time_of_day"][_abs]),
                "sin_day_of_year": float(_cache["sin_day_of_year"][_abs]),
                "cos_day_of_year": float(_cache["cos_day_of_year"][_abs]),
            }
            ideal_panel_azimuth = float(_cache["ideal_azimuth"][_abs])
            ideal_panel_tilt    = float(_cache["ideal_tilt"][_abs])
            ideal_poa = {
                "poa_w_per_m2":          float(_cache["ideal_poa_w_per_m2"][_abs]),
                "poa_direct_w_per_m2":   float(_cache["ideal_poa_direct_w_per_m2"][_abs]),
                "poa_diffuse_w_per_m2":  float(_cache["ideal_poa_diffuse_w_per_m2"][_abs]),
                "poa_ground_w_per_m2":   0.0,
            }
            ideal_power = {
                "power_kw":   float(_cache["ideal_power_kw"][_abs]),
                "energy_kwh": float(_cache["ideal_energy_kwh"][_abs]),
            }
        else:
            time_features = self._time_features(row)
            ideal_panel_azimuth, ideal_panel_tilt = self._ideal_panel_orientation(
                dni=dni,
                dhi=dhi,
                solar_zenith_deg=solar_zenith,
                solar_azimuth_deg=solar_azimuth_360,
            )
            ideal_poa = compute_plane_of_array_irradiance(
                dni=dni,
                dhi=dhi,
                solar_zenith_deg=solar_zenith,
                solar_azimuth_deg=solar_azimuth_360,
                panel_tilt_deg=ideal_panel_tilt,
                panel_azimuth_deg=ideal_panel_azimuth,
                dni_min_threshold=self.dni_min_threshold,
            )
            ideal_power = compute_power_and_energy(
                poa_irradiance_w_per_m2=ideal_poa["poa_w_per_m2"],
                ambient_temperature_c=temperature,
                panel=self.panel_model,
                seconds_per_time_step=self.seconds_per_time_step,
            )
        cos_incidence = incidence_cosine(
            solar_zenith_deg=solar_zenith,
            solar_azimuth_deg=solar_azimuth_360,
            panel_tilt_deg=self.panel_tilt_deg,
            panel_azimuth_deg=panel_azimuth_360,
        )
        tracking_error = np.degrees(
            np.arccos(
                np.clip(
                    cos_incidence,
                    -1.0,
                    1.0,
                )
            )
        )
        self._history_rows.append(
            {
                "time_step": float(self.time_step),
                "month": float(row[self.OBSERVATION_COLUMN_MAPPING["month"]]),
                "day": float(row[self.OBSERVATION_COLUMN_MAPPING["day"]]),
                "hour": float(row[self.OBSERVATION_COLUMN_MAPPING["hour"]]),
                "minute": float(row[self.OBSERVATION_COLUMN_MAPPING["minute"]]),
                **time_features,
                "panel_azimuth_deg": float(self.panel_azimuth_deg),
                "panel_tilt_deg": float(self.panel_tilt_deg),
                "panel_azimuth_delta_deg": float(azimuth_delta_deg),
                "panel_tilt_delta_deg": float(tilt_delta_deg),
                "actual_panel_azimuth_delta_deg": float(
                    azimuth_delta_deg if actual_azimuth_delta_deg is None else actual_azimuth_delta_deg
                ),
                "actual_panel_tilt_delta_deg": float(
                    tilt_delta_deg if actual_tilt_delta_deg is None else actual_tilt_delta_deg
                ),
                "automatic_orientation_reset": bool(automatic_orientation_reset),
                "generated_power_kw": float(power["power_kw"]),
                "generated_energy_kwh": float(power["energy_kwh"]),
                "ideal_generated_power_kw": float(ideal_power["power_kw"]),
                "ideal_generated_energy_kwh": float(ideal_power["energy_kwh"]),
                "tracking_error_deg": float(tracking_error),
                "cos_incidence": float(cos_incidence),
                "poa_w_per_m2": float(poa["poa_w_per_m2"]),
                "poa_direct_w_per_m2": float(poa["poa_direct_w_per_m2"]),
                "poa_diffuse_w_per_m2": float(poa["poa_diffuse_w_per_m2"]),
                "poa_ground_w_per_m2": float(poa["poa_ground_w_per_m2"]),
                "ideal_poa_w_per_m2": float(ideal_poa["poa_w_per_m2"]),
                "ideal_poa_direct_w_per_m2": float(ideal_poa["poa_direct_w_per_m2"]),
                "ideal_poa_diffuse_w_per_m2": float(ideal_poa["poa_diffuse_w_per_m2"]),
                "solar_azimuth_deg": solar_azimuth_360,
                "solar_zenith_deg": solar_zenith,
                "dni": dni,
                "dhi": dhi,
                "ghi": ghi,
                "dni_clear_sky_ratio": self._safe_ratio(dni, clearsky_dni),
                "dhi_clear_sky_ratio": self._safe_ratio(dhi, clearsky_dhi),
                "wind_speed": wind_speed,
                "cloud_type": cloud_type,
            }
        )
        self._previous_generated_energy_kwh = float(power["energy_kwh"])

    def _estimate_observation_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        low: List[float] = []
        high: List[float] = []

        for name in self.active_observations:
            if name in self.OBSERVATION_COLUMN_MAPPING:
                series = self.data[self.OBSERVATION_COLUMN_MAPPING[name]]
                low.append(float(series.min()))
                high.append(float(series.max()))
            elif name in {"sin_time_of_day", "cos_time_of_day", "sin_day_of_year", "cos_day_of_year"}:
                low.append(-1.0)
                high.append(1.0)
            elif name in {"dni_clear_sky_ratio", "dhi_clear_sky_ratio"}:
                low.append(0.0)
                high.append(2.0)
            elif name == "current_panel_tilt_angle":
                low.append(float(self._panel_low[1]))
                high.append(float(self._panel_high[1]))
            elif name == "current_panel_azimuth_angle":
                low.append(float(self._panel_low[0]))
                high.append(float(self._panel_high[0]))
            elif name == "previous_action_tilt_angle":
                low.append(float(self._panel_low[1]))
                high.append(float(self._panel_high[1]))
            elif name == "previous_action_azimuth_angle":
                low.append(float(self._panel_low[0]))
                high.append(float(self._panel_high[0]))
            elif name == "panel_azimuth_delta_deg":
                low.append(float(self._action_low[0] if self.action_mode == "delta" else -(self._panel_high[0] - self._panel_low[0])))
                high.append(float(self._action_high[0] if self.action_mode == "delta" else self._panel_high[0] - self._panel_low[0]))
            elif name == "panel_tilt_delta_deg":
                low.append(float(self._action_low[1] if self.action_mode == "delta" else -(self._panel_high[1] - self._panel_low[1])))
                high.append(float(self._action_high[1] if self.action_mode == "delta" else self._panel_high[1] - self._panel_low[1]))
            elif name == "tracking_error_deg":
                low.append(0.0)
                high.append(180.0)
            elif name == "cos_incidence":
                low.append(-1.0)
                high.append(1.0)
            elif name in {"previous_generated_energy_kwh", "generated_energy_kwh", "current_generated_energy_kwh"}:
                high_limit = self.panel_model.effective_max_power_kw() * self.seconds_per_time_step / 3600.0
                low.append(0.0)
                high.append(float(high_limit))
            elif name == "poa_w_per_m2":
                dni_high = float(self.data[self.OBSERVATION_COLUMN_MAPPING["dni"]].max())
                dhi_high = float(self.data[self.OBSERVATION_COLUMN_MAPPING["dhi"]].max())
                low.append(0.0)
                high.append(max(1.0, dni_high + dhi_high))
            else:
                low.append(-1e6)
                high.append(1e6)

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    def _load(self, **kwargs):
        schema = deepcopy(self.schema)
        root_directory = kwargs.get("root_directory", schema["root_directory"])
        root_directory = schema["root_directory"] if root_directory in (None, "") else root_directory
        data_config = schema["data"]
        data_filename = data_config["filename"]
        data_path = os.path.join(root_directory, data_filename)
        data = pd.read_csv(data_path)
        simulation_start_time_step = (
            kwargs["simulation_start_time_step"]
            if kwargs.get("simulation_start_time_step") is not None
            else schema["simulation_start_time_step"]
        )
        simulation_end_time_step = (
            kwargs["simulation_end_time_step"]
            if kwargs.get("simulation_end_time_step") is not None
            else schema["simulation_end_time_step"]
        )
        assert 0 <= simulation_start_time_step < len(data), "simulation_start_time_step out of range."
        assert 0 <= simulation_end_time_step < len(data), "simulation_end_time_step out of range."
        assert simulation_end_time_step >= simulation_start_time_step, "simulation_end_time_step must be >= start."
        episode_tracker = EpisodeTracker(simulation_start_time_step, simulation_end_time_step)
        episode_time_steps = (
            kwargs["episode_time_steps"]
            if kwargs.get("episode_time_steps") is not None
            else schema.get("episode_time_steps")
        )
        if episode_time_steps is None:
            episode_time_steps = (simulation_end_time_step - simulation_start_time_step) + 1
        rolling_episode_split = (
            kwargs["rolling_episode_split"]
            if kwargs.get("rolling_episode_split") is not None
            else schema.get("rolling_episode_split", False)
        )
        random_episode_split = (
            kwargs["random_episode_split"]
            if kwargs.get("random_episode_split") is not None
            else schema.get("random_episode_split", False)
        )
        seconds_per_time_step = (
            kwargs["seconds_per_time_step"]
            if kwargs.get("seconds_per_time_step") is not None
            else schema["seconds_per_time_step"]
        )
        central_agent = kwargs["central_agent"] if kwargs.get("central_agent") is not None else schema.get("central_agent", False)
        observation_config = schema["observations"]
        action_config = schema["actions"]
        self.action_mode = schema.get("action_mode", "absolute")
        assert self.action_mode in {"absolute", "delta"}, "action_mode must be 'absolute' or 'delta'."
        tracking_window = schema.get("tracking_window", {})
        self.tracking_start_hour = float(tracking_window.get("start_hour", 7.0))
        self.tracking_end_hour = float(tracking_window.get("end_hour", 18.0))
        self.auto_reset_to_initial_orientation = bool(schema.get("auto_reset_to_initial_orientation", False))
        action_deadband = schema.get("action_deadband", {})
        self.action_deadband_azimuth_deg = float(action_deadband.get("azimuth_delta_deg", 0.0))
        self.action_deadband_tilt_deg = float(action_deadband.get("tilt_delta_deg", 0.0))
        self.panel_angle_bounds = schema.get(
            "panel_angle_bounds",
            {
                "azimuth": {"low": 70.0, "high": 290.0},
                "tilt": {"low": 10.0, "high": 85.0},
            },
        )
        panel_attributes = schema.get("panel", {}).get("attributes", {})

        # Notebook sessions and saved checkpoints can keep an old in-memory schema.
        # For the bundled solar-tracker dataset, refresh panel parameters from
        # the canonical schema on disk so fixed, RBC, and Meta-SAC use the same
        # currently documented PV module.
        canonical_schema_path = Path(__file__).resolve().parent / "data" / "schema.json"
        if data_filename == "2020.csv" and canonical_schema_path.exists():
            try:
                with canonical_schema_path.open("r", encoding="utf-8") as file:
                    canonical_schema = json.load(file)
                panel_attributes = canonical_schema.get("panel", {}).get("attributes", panel_attributes)
                schema.setdefault("panel", {})["attributes"] = panel_attributes
            except Exception:
                pass

        generation_settings = schema.get("generation", {})
        self.dni_min_threshold = float(generation_settings.get("dni_min_threshold", 0.0))
        panel = PanelModel(
            area_m2=float(panel_attributes.get("area_m2", 10.0)),
            efficiency=float(panel_attributes.get("efficiency", 0.2)),
            performance_ratio=float(panel_attributes.get("performance_ratio", 0.9)),
            temperature_coefficient=float(panel_attributes.get("temperature_coefficient", 0.004)),
            reference_temperature_c=float(panel_attributes.get("reference_temperature_c", 25.0)),
            max_power_kw=float(panel_attributes.get("max_power_kw", 0.0)),
        )
        orientation = schema.get("initial_orientation", {"tilt_deg": 30.0, "azimuth_deg": 180.0})
        initial_panel_orientation = {"tilt": float(orientation.get("tilt_deg", 30.0)), "azimuth": float(orientation.get("azimuth_deg", 180.0))}
        baseline_config = schema.get("baselines", {})

        if kwargs.get("reward_function") is not None:
            reward = kwargs["reward_function"](None)
        else:
            reward_config = schema["reward_function"]
            reward_type = reward_config["type"]
            reward_attributes = reward_config.get("attributes", {})
            reward_module = ".".join(reward_type.split(".")[:-1])
            reward_class = reward_type.split(".")[-1]
            reward_constructor = getattr(importlib.import_module(reward_module), reward_class)
            reward = reward_constructor(None, **reward_attributes)

        return (
            root_directory,
            data,
            episode_time_steps,
            rolling_episode_split,
            random_episode_split,
            seconds_per_time_step,
            central_agent,
            observation_config,
            action_config,
            panel,
            initial_panel_orientation,
            baseline_config,
            reward,
            episode_tracker,
        )
