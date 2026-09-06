"""Mode-A+B SLM wrapper for Meta-SAC solar-tracker experiments."""

from __future__ import annotations

import copy
from typing import Mapping, Optional

import numpy as np

from st.utilities import SolarTrackerMetaSACWrapper
from .guidance import LLMGuidance


class LLMStateEnrichedMetaSACWrapper(SolarTrackerMetaSACWrapper):
    """Append hourly SLM context and optionally reshape rewards.

    This implements Mode B state enrichment and Mode A reward shaping.
    The appended context contains 7 regime one-hot values, 4 strategy one-hot
    values, two scalar guidance values, and three Mode-A reward weights.
    Guidance is cached for six 10-minute environment steps by default.
    """

    def __init__(
        self,
        schema_template,
        reward_scale: float = 50.0,
        guidance: Optional[LLMGuidance] = None,
        query_interval_steps: int = 6,
        use_llm: bool = True,
        use_mode_a_reward: bool = True,
        verbose: bool = False,
    ):
        self.use_mode_a_reward = bool(use_mode_a_reward)
        self.previous_azimuth_delta_deg = 0.0
        self.previous_tilt_delta_deg = 0.0
        self.guidance = guidance or LLMGuidance(
            enabled=True,
            query_interval_steps=query_interval_steps,
            use_llm=use_llm,
            verbose=verbose,
        )
        self.context_names = self.guidance.context_names()
        self.last_guidance = None
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

    def _telemetry(self) -> Mapping[str, float]:
        row = self.env._row_at_time_step(self.env.time_step)
        mapping = self.env.OBSERVATION_COLUMN_MAPPING

        def value(key: str, default: float = 0.0) -> float:
            column = mapping.get(key)
            if column is None or column not in row:
                return float(default)
            return float(row[column])

        clearsky_dni = max(value("clearsky_dni"), 1e-6)
        return {
            "hour": value("hour"),
            "minute": value("minute"),
            "solar_zenith_deg": value("solar_zenith_angle"),
            "solar_azimuth_deg": value("solar_azimuth_angle"),
            "dni": value("dni"),
            "dhi": value("dhi"),
            "next_10min_dni": value("next_10min_DNI", value("dni")),
            "next_10min_dhi": value("next_10min_DHI", value("dhi")),
            "next_30min_average_dni": value("next_30min_average_DNI", value("dni")),
            "cloud_type": value("cloud_type"),
            "wind_speed": value("wind_speed"),
            "temperature": value("temperature"),
            "dni_clear_sky_ratio": value("dni") / clearsky_dni,
        }

    def _context_vector(self):
        self.last_guidance = self.guidance.guidance_for(self.env.time_step, self._telemetry())
        return self.last_guidance.context_vector().astype(self._np.float32)

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

    def reset(self):
        self.previous_azimuth_delta_deg = 0.0
        self.previous_tilt_delta_deg = 0.0
        obs = self.env.reset()[0]
        return self._enrich_normalized_obs(obs)

    def _rated_power_kw(self) -> float:
        panel = self.env.get_metadata().get("panel", {})
        return max(float(panel.get("rated_power_kw", 0.0)), 1e-9)

    def _action_delta_bounds(self):
        high = self._np.asarray(self.action_space.high, dtype=self._np.float32)
        if high.shape[0] < 2:
            return 1.0, 1.0
        return max(float(abs(high[0])), 1e-6), max(float(abs(high[1])), 1e-6)

    def _mode_a_reward(self, guidance):
        """Compute Qwen-weighted reward from the latest environment history row.

        The positive term is energy-aware alignment:
            irradiance_factor * max(cos_incidence, 0)
        This avoids a separate direct-energy weight while still making alignment
        more valuable during high available-power periods.
        """

        if guidance is None or len(self.env.history) == 0:
            return None, {}

        row = self.env.history.iloc[-1]
        rated_power_kw = self._rated_power_kw()
        ideal_power_kw = float(row.get("ideal_generated_power_kw", 0.0))
        irradiance_factor = float(self._np.clip(ideal_power_kw / rated_power_kw, 0.0, 1.25))
        alignment_quality = float(self._np.clip(row.get("cos_incidence", 0.0), 0.0, 1.0))
        alignment_component = irradiance_factor * alignment_quality

        azimuth_delta = float(row.get("panel_azimuth_delta_deg", 0.0))
        tilt_delta = float(row.get("panel_tilt_delta_deg", 0.0))
        max_azimuth_delta, max_tilt_delta = self._action_delta_bounds()
        movement_component = (
            abs(azimuth_delta) / max_azimuth_delta + abs(tilt_delta) / max_tilt_delta
        ) / 2.0

        azimuth_jerk = azimuth_delta - self.previous_azimuth_delta_deg
        tilt_jerk = tilt_delta - self.previous_tilt_delta_deg
        jerk_component = (
            (azimuth_jerk / max_azimuth_delta) ** 2 + (tilt_jerk / max_tilt_delta) ** 2
        ) / 2.0

        azimuth_sign_flip = (
            abs(azimuth_delta) > 1e-6
            and abs(self.previous_azimuth_delta_deg) > 1e-6
            and azimuth_delta * self.previous_azimuth_delta_deg < 0.0
        )
        tilt_sign_flip = (
            abs(tilt_delta) > 1e-6
            and abs(self.previous_tilt_delta_deg) > 1e-6
            and tilt_delta * self.previous_tilt_delta_deg < 0.0
        )
        sign_flip_count = float(azimuth_sign_flip) + float(tilt_sign_flip)
        # Delta-action policies need small corrective reversals. Keep a
        # direction-change cost for wear reduction, but avoid making the agent
        # afraid to correct alignment during useful irradiance periods.
        direction_change_component = 0.10 * sign_flip_count
        smoothness_component = jerk_component + direction_change_component

        peak_relief = 1.0
        if guidance.strategy in {"peak_track", "track"}:
            relief_strength = float(self._np.clip((irradiance_factor - 0.50) / 0.75, 0.0, 1.0))
            peak_relief = 1.0 - 0.35 * relief_strength

        effective_movement_component = movement_component * peak_relief
        effective_smoothness_component = smoothness_component * peak_relief

        reward = (
            float(guidance.alignment_weight) * alignment_component
            - float(guidance.movement_weight) * effective_movement_component
            - float(guidance.smoothness_weight) * effective_smoothness_component
        )

        self.previous_azimuth_delta_deg = azimuth_delta
        self.previous_tilt_delta_deg = tilt_delta

        components = {
            "llm_reward_alignment_component": alignment_component,
            "llm_reward_movement_component": movement_component,
            "llm_reward_effective_movement_component": effective_movement_component,
            "llm_reward_jerk_component": jerk_component,
            "llm_reward_direction_change_component": direction_change_component,
            "llm_reward_sign_flip_count": sign_flip_count,
            "llm_reward_smoothness_component": smoothness_component,
            "llm_reward_effective_smoothness_component": effective_smoothness_component,
            "llm_reward_peak_relief_factor": peak_relief,
            "llm_reward_unscaled": reward,
        }
        return reward, components

    def step(self, action):
        action = self._np.asarray(action, dtype=self._np.float32)
        guidance_for_action = self.last_guidance
        obs, reward, done, info = self.env.step([action])
        self.last_history = self.env.history.copy()
        mode_a_components = {}
        if self.use_mode_a_reward:
            mode_a_reward, mode_a_components = self._mode_a_reward(guidance_for_action)
            unscaled_reward = float(mode_a_reward)
        else:
            unscaled_reward = float(reward[0])
        scaled_reward = unscaled_reward * self.reward_scale
        enriched_obs = self._enrich_normalized_obs(obs[0])
        info = dict(info or {})
        if guidance_for_action is not None:
            info["llm_regime"] = guidance_for_action.regime
            info["llm_strategy"] = guidance_for_action.strategy
            info["llm_tilt_priority"] = guidance_for_action.tilt_priority
            info["llm_movement_budget"] = guidance_for_action.movement_budget
            info["llm_alignment_weight"] = guidance_for_action.alignment_weight
            info["llm_movement_weight"] = guidance_for_action.movement_weight
            info["llm_smoothness_weight"] = guidance_for_action.smoothness_weight
            info["llm_source"] = guidance_for_action.source
        info["llm_mode_a_reward_enabled"] = self.use_mode_a_reward
        info.update(mode_a_components)
        return enriched_obs, scaled_reward, bool(done), info
