"""Goal-conditioned SLM wrapper for Meta-SAC solar-tracker experiments."""

from __future__ import annotations

import copy
from typing import Mapping, Optional

import numpy as np

from st.utilities import SolarTrackerMetaSACWrapper
from .goal_guidance import GoalOutput, LLMGoalGuidance
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .regime_router import RegimeRouter


def _angular_error_deg(current: float, target: float) -> float:
    """Return signed shortest azimuth-like angular error in degrees."""

    return float((float(current) - float(target) + 180.0) % 360.0 - 180.0)


class LLMGoalConditionedMetaSACWrapper(SolarTrackerMetaSACWrapper):
    """Append hourly SLM goals and use a fixed goal-conditioned reward."""

    def __init__(
        self,
        schema_template,
        reward_scale: float = 50.0,
        guidance: Optional[LLMGoalGuidance] = None,
        query_interval_steps: int = 6,
        use_llm: bool = True,
        verbose: bool = False,
        alpha: float = 2.0,
        beta: float = 0.12,
        gamma: float = 0.35,
        delta: float = 0.20,
        regime_router: "Optional[RegimeRouter]" = None,
    ):
        self.guidance = guidance or LLMGoalGuidance(
            enabled=True,
            query_interval_steps=query_interval_steps,
            use_llm=use_llm,
            verbose=verbose,
        )
        self.context_names = self.guidance.context_names() + [
            "llm_goal_hour_motion_so_far",
            "llm_goal_time_until_next_call",
        ]
        self.last_goal: Optional[GoalOutput] = None
        self.hour_motion_so_far_deg = 0.0
        self.previous_goal_bucket = None
        self.previous_azimuth_delta_deg = 0.0
        self.previous_tilt_delta_deg = 0.0
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.query_interval_steps = int(query_interval_steps)
        self.regime_router = regime_router
        super().__init__(schema_template=copy.deepcopy(schema_template), reward_scale=reward_scale)

    def _build_env(self):
        super()._build_env()
        self.base_obs_low = self.obs_low.copy()
        self.base_obs_high = self.obs_high.copy()
        context_low = self._np.zeros(len(self.context_names), dtype=self._np.float32)
        context_high = self._np.ones(len(self.context_names), dtype=self._np.float32)
        self.observation_names = list(self.observation_names) + list(self.context_names)
        self.obs_low = self._np.concatenate([self.base_obs_low, context_low]).astype(self._np.float32)
        self.obs_high = self._np.concatenate([self.base_obs_high, context_high]).astype(self._np.float32)
        self.observation_space = self._spaces.Box(
            low=self._np.full_like(self.obs_low, -1.0, dtype=self._np.float32),
            high=self._np.full_like(self.obs_high, 1.0, dtype=self._np.float32),
            dtype=self._np.float32,
        )

    def reset(self):
        self.hour_motion_so_far_deg = 0.0
        self.previous_goal_bucket = None
        self.previous_azimuth_delta_deg = 0.0
        self.previous_tilt_delta_deg = 0.0
        obs = self.env.reset()[0]
        return self._enrich_normalized_obs(obs)

    def _normalize_base_obs(self, obs):
        obs = self._np.asarray(obs, dtype=self._np.float32)
        denom = self._np.where(
            self._np.abs(self.base_obs_high - self.base_obs_low) < 1e-6,
            1.0,
            self.base_obs_high - self.base_obs_low,
        )
        scaled = 2.0 * (obs - self.base_obs_low) / denom - 1.0
        return self._np.nan_to_num(self._np.clip(scaled, -1.0, 1.0)).astype(self._np.float32)

    def _enrich_normalized_obs(self, raw_obs):
        normalized = self._normalize_base_obs(raw_obs)
        context = self._context_vector()
        return self._np.concatenate([normalized, context]).astype(self._np.float32)

    def _row_value(self, row, key: str, default: float = 0.0) -> float:
        mapping = self.env.OBSERVATION_COLUMN_MAPPING
        column = mapping.get(key)
        if column is None or column not in row:
            return float(default)
        return float(row[column])

    def _ideal_orientation_at(self, time_step: int) -> Mapping[str, float]:
        row = self.env._row_at_time_step(time_step)
        azimuth, tilt = self.env._ideal_panel_orientation(
            dni=self._row_value(row, "dni"),
            dhi=self._row_value(row, "dhi"),
            solar_zenith_deg=self._row_value(row, "solar_zenith_angle"),
            solar_azimuth_deg=self._row_value(row, "solar_azimuth_angle"),
        )
        return {"azimuth": float(azimuth), "tilt": float(tilt)}

    def _future_time_step(self, horizon_min: int = 60) -> int:
        step_minutes = max(float(self.env.seconds_per_time_step) / 60.0, 1.0)
        horizon_steps = int(round(float(horizon_min) / step_minutes))
        end_step = int(self.env.episode_tracker.episode_time_steps) - 1
        return int(min(self.env.time_step + horizon_steps, end_step))

    def _dni_window(self, steps: int) -> list[float]:
        values = []
        end_step = int(self.env.episode_tracker.episode_time_steps) - 1
        for offset in range(1, steps + 1):
            row = self.env._row_at_time_step(min(int(self.env.time_step) + offset, end_step))
            values.append(self._row_value(row, "dni"))
        return values

    def _tracking_window_active(self, row) -> bool:
        return bool(self.env._is_tracking_window_active(row))

    def _time_until_tracking_end_min(self, row) -> float:
        hour = self._row_value(row, "hour")
        minute = self._row_value(row, "minute")
        current = hour * 60.0 + minute
        end = float(self.env.tracking_end_hour) * 60.0
        return float(max(0.0, end - current))

    def _telemetry(self) -> Mapping[str, object]:
        row = self.env._row_at_time_step(self.env.time_step)
        dataset_time_step = int(self.env.episode_tracker.episode_start_time_step) + int(self.env.time_step)
        current_target = self._ideal_orientation_at(int(self.env.time_step))
        future_30 = self._ideal_orientation_at(self._future_time_step(30))
        future_60 = self._ideal_orientation_at(self._future_time_step(60))
        dni_forecast = self._dni_window(6)
        next_30 = dni_forecast[:3] if dni_forecast else [self._row_value(row, "dni")]
        max_forecast_dni = max(self._row_value(row, "dni"), float(np.mean(next_30)))
        if max_forecast_dni >= 700:
            dni_bucket = "CLEAR_STRONG"
        elif max_forecast_dni >= 500:
            dni_bucket = "CLEAR_MODERATE"
        elif max_forecast_dni >= 300:
            dni_bucket = "PARTIAL_CLOUD"
        elif max_forecast_dni >= 100:
            dni_bucket = "OVERCAST_DIM"
        else:
            dni_bucket = "NIGHT_OR_HEAVY_OVERCAST"
        return {
            "dataset_time_step": dataset_time_step,
            "hour": self._row_value(row, "hour"),
            "minute": self._row_value(row, "minute"),
            "solar_azimuth_deg": self._row_value(row, "solar_azimuth_angle"),
            "solar_zenith_deg": self._row_value(row, "solar_zenith_angle"),
            "dni": self._row_value(row, "dni"),
            "dhi": self._row_value(row, "dhi"),
            "dni_forecast_next_60min_10min_steps": dni_forecast,
            "next_10min_dni": dni_forecast[0] if dni_forecast else self._row_value(row, "dni"),
            "next_30min_average_dni": float(np.mean(next_30)),
            "cloud_type": self._row_value(row, "cloud_type"),
            "wind_speed": self._row_value(row, "wind_speed"),
            "temperature": self._row_value(row, "temperature"),
            "panel_azimuth_deg": float(self.env.panel_azimuth_deg),
            "panel_tilt_deg": float(self.env.panel_tilt_deg),
            "hour_motion_so_far_deg": float(self.hour_motion_so_far_deg),
            "time_until_tracking_end_min": self._time_until_tracking_end_min(row),
            "tracking_window_active": self._tracking_window_active(row),
            "dni_bucket": dni_bucket,
            "max_forecast_dni": round(max_forecast_dni, 1),
            "current_target": current_target,
            "future_30min_target": future_30,
            "future_target": future_60,
            "constraints": {
                "azimuth_min_deg": float(self.env.panel_azimuth_bounds[0]),
                "azimuth_max_deg": float(self.env.panel_azimuth_bounds[1]),
                "tilt_min_deg": float(self.env.panel_tilt_bounds[0]),
                "tilt_max_deg": float(self.env.panel_tilt_bounds[1]),
                "max_delta_azimuth_per_step_deg": float(abs(self.action_space.high[0])),
                "max_delta_tilt_per_step_deg": float(abs(self.action_space.high[1])),
            },
        }

    def _context_vector(self):
        bucket = int(self.env.time_step) // max(1, self.query_interval_steps)
        if self.previous_goal_bucket is None or bucket != self.previous_goal_bucket:
            self.hour_motion_so_far_deg = 0.0
            self.previous_goal_bucket = bucket
        self.last_goal = self.guidance.guidance_for(self.env.time_step, self._telemetry())
        base = self.last_goal.context_vector()
        time_into_bucket = int(self.env.time_step) % max(1, self.query_interval_steps)
        time_until_next = 1.0 - (time_into_bucket / max(1, self.query_interval_steps))
        extra = self._np.array(
            [
                float(self._np.clip(self.hour_motion_so_far_deg / 120.0, 0.0, 1.0)),
                float(self._np.clip(time_until_next, 0.0, 1.0)),
            ],
            dtype=self._np.float32,
        )
        return self._np.concatenate([base, extra]).astype(self._np.float32)

    def _current_step_target(self, goal: GoalOutput) -> tuple[float, float]:
        step_minutes = max(float(self.env.seconds_per_time_step) / 60.0, 1.0)
        elapsed_min = (int(self.env.time_step) % max(1, self.query_interval_steps)) * step_minutes
        ratio = float(self._np.clip(elapsed_min / max(goal.horizon_min, step_minutes), 0.0, 1.0))
        az_delta = _angular_error_deg(goal.pre_position_az, goal.target_az)
        target_az = float((goal.target_az + ratio * az_delta) % 360.0)
        target_az = float(self._np.clip(target_az, self.env.panel_azimuth_bounds[0], self.env.panel_azimuth_bounds[1]))
        target_tilt = float(goal.target_tilt + ratio * (goal.pre_position_tilt - goal.target_tilt))
        target_tilt = float(self._np.clip(target_tilt, self.env.panel_tilt_bounds[0], self.env.panel_tilt_bounds[1]))
        return target_az, target_tilt

    def _goal_reward(self, previous_azimuth: float, previous_tilt: float, goal: Optional[GoalOutput]):
        if goal is None or len(self.env.history) == 0:
            return None, {}

        row = self.env.history.iloc[-1]
        azimuth_delta = float(row.get("panel_azimuth_delta_deg", 0.0))
        tilt_delta = float(row.get("panel_tilt_delta_deg", 0.0))
        motion = abs(azimuth_delta) + abs(tilt_delta)
        self.hour_motion_so_far_deg += motion

        if goal.hold:
            # During hold periods, Qwen intentionally defines the target pose.
            target_az, target_tilt = self._current_step_target(goal)
        else:
            # During active tracking, use physics for exact sun-facing angles so
            # Qwen provides strategy/budget while solar geometry provides tactics.
            target_az, target_tilt = self.env._ideal_panel_orientation(
                dni=float(row.get("dni", 0.0)),
                dhi=float(row.get("dhi", 0.0)),
                solar_zenith_deg=float(row.get("solar_zenith_deg", 180.0)),
                solar_azimuth_deg=float(row.get("solar_azimuth_deg", goal.target_az)),
            )

        az_error = abs(_angular_error_deg(float(row.get("panel_azimuth_deg", 0.0)), target_az))
        tilt_error = abs(float(row.get("panel_tilt_deg", 0.0)) - target_tilt)
        target_error = ((az_error / 220.0) ** 2 + (tilt_error / 75.0) ** 2) / 2.0
        dni_weight = float(self._np.clip(float(row.get("dni", 0.0)) / 1000.0, 0.0, 1.0))
        align = -self.alpha * dni_weight * target_error

        max_az_delta = max(float(abs(self.action_space.high[0])), 1e-6)
        max_tilt_delta = max(float(abs(self.action_space.high[1])), 1e-6)
        budget_fraction = float(np.clip(goal.motion_budget_deg / 30.0, 0.1, 1.0))
        motion_cost = -self.beta * ((abs(azimuth_delta) / max_az_delta + abs(tilt_delta) / max_tilt_delta) / 2.0) / budget_fraction

        over_budget = max(0.0, self.hour_motion_so_far_deg - float(goal.motion_budget_deg))
        budget_penalty = -self.gamma * (over_budget / 120.0)

        if goal.hold:
            anticipation = -self.delta * (motion / (max_az_delta + max_tilt_delta))
        else:
            prev_az_error = abs(_angular_error_deg(previous_azimuth, goal.pre_position_az))
            prev_tilt_error = abs(previous_tilt - goal.pre_position_tilt)
            new_az_error = abs(_angular_error_deg(float(row.get("panel_azimuth_deg", 0.0)), goal.pre_position_az))
            new_tilt_error = abs(float(row.get("panel_tilt_deg", 0.0)) - goal.pre_position_tilt)
            prev_distance = (prev_az_error / 220.0 + prev_tilt_error / 75.0) / 2.0
            new_distance = (new_az_error / 220.0 + new_tilt_error / 75.0) / 2.0
            anticipation = self.delta * (prev_distance - new_distance)

        reward = align + motion_cost + budget_penalty + anticipation
        components = {
            "llm_goal_target_az_deg": target_az,
            "llm_goal_target_tilt_deg": target_tilt,
            "llm_goal_motion_budget_deg": float(goal.motion_budget_deg),
            "llm_goal_hour_motion_so_far_deg": float(self.hour_motion_so_far_deg),
            "llm_goal_hold": bool(goal.hold),
            "llm_goal_dni_weight": float(dni_weight),
            "llm_goal_align_component": float(align),
            "llm_goal_motion_component": float(motion_cost),
            "llm_goal_budget_component": float(budget_penalty),
            "llm_goal_anticipation_component": float(anticipation),
            "llm_goal_reward_unscaled": float(reward),
        }
        self.previous_azimuth_delta_deg = azimuth_delta
        self.previous_tilt_delta_deg = tilt_delta
        return reward, components

    def step(self, action):
        action = self._np.asarray(action, dtype=self._np.float32)
        goal_for_action = self.last_goal
        previous_azimuth = float(self.env.panel_azimuth_deg)
        previous_tilt = float(self.env.panel_tilt_deg)
        obs, reward, done, info = self.env.step([action])
        self.last_history = self.env.history.copy()
        goal_reward, components = self._goal_reward(previous_azimuth, previous_tilt, goal_for_action)
        unscaled_reward = float(goal_reward if goal_reward is not None else reward[0])
        enriched_obs = self._enrich_normalized_obs(obs[0])
        info = dict(info or {})
        if goal_for_action is not None:
            info.update(
                {
                    "llm_goal_source": goal_for_action.source,
                    "llm_goal_raw_hold": goal_for_action.hold,
                    "llm_goal_raw_target_az": goal_for_action.target_az,
                    "llm_goal_raw_target_tilt": goal_for_action.target_tilt,
                    "llm_goal_raw_pre_position_az": goal_for_action.pre_position_az,
                    "llm_goal_raw_pre_position_tilt": goal_for_action.pre_position_tilt,
                    "llm_goal_raw_horizon_min": goal_for_action.horizon_min,
                    "llm_goal_raw_confidence": goal_for_action.confidence,
                }
            )
        info.update(components)
        return enriched_obs, unscaled_reward * self.reward_scale, bool(done), info
