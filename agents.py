"""Baseline controllers for the dual-axis solar tracker environment."""

from typing import Any, List

import numpy as np

from citylearn.base import Environment


class SolarAgent(Environment):
    """Minimal agent base with CityLearn-like ``predict`` and action history."""

    def __init__(self, env, **kwargs: Any):
        self.env = env
        self.observation_names = env.observation_names
        self.action_names = env.action_names
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.actions = [[[]] for _ in self.action_space]
        super().__init__(
            seconds_per_time_step=env.seconds_per_time_step,
            random_seed=env.random_seed,
            episode_tracker=env.episode_tracker,
        )
        self.reset()

    def reset(self):
        super().reset()
        self.actions = [[[]] for _ in self.action_space]

    def next_time_step(self):
        super().next_time_step()

        for action_history in self.actions:
            action_history.append([])

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        action = [self.action_space[0].sample().tolist()]
        self.actions[0][self.time_step] = action[0]
        self.next_time_step()
        return action


class FixedSolarAgent(SolarAgent):
    """Baseline agent that always outputs a fixed panel azimuth and tilt.

    Default baseline assumption is a south-facing panel (azimuth = 180 degrees).
    """

    def __init__(self, env, fixed_azimuth_deg: float = 180.0, fixed_tilt_deg: float = 30.0, **kwargs: Any):
        self.fixed_azimuth_deg = fixed_azimuth_deg
        self.fixed_tilt_deg = fixed_tilt_deg
        super().__init__(env, **kwargs)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        action = self.env.action_from_target_angles(self.fixed_azimuth_deg, self.fixed_tilt_deg).tolist()
        actions = [action]
        self.actions[0][self.time_step] = actions[0]
        self.next_time_step()
        return actions


class RuleBasedSolarTrackerAgent(SolarAgent):
    """Deterministic baseline that tracks sun position with optional slew-rate limit."""

    def __init__(
        self,
        env,
        night_tilt_deg: float = 10.0,
        max_delta_deg_per_step: float = 15.0,
        daylight_tilt_offset_deg: float = 0.0,
        **kwargs: Any,
    ):
        self.night_tilt_deg = float(night_tilt_deg)
        self.max_delta_deg_per_step = float(max_delta_deg_per_step)
        self.daylight_tilt_offset_deg = float(daylight_tilt_offset_deg)
        super().__init__(env, **kwargs)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        observation = observations[0]
        names = self.observation_names[0]
        sun_azimuth = float(observation[names.index("solar_azimuth_angle")])
        sun_zenith = float(observation[names.index("solar_zenith_angle")])
        az_low, az_high = self.env.panel_azimuth_bounds
        tilt_low, tilt_high = self.env.panel_tilt_bounds

        target_azimuth = np.clip(self.env.wrap_azimuth(sun_azimuth), az_low, az_high)
        if sun_zenith >= 90.0:
            target_tilt = np.clip(self.night_tilt_deg, tilt_low, tilt_high)
        else:
            target_tilt = np.clip(90.0 - sun_zenith + self.daylight_tilt_offset_deg, tilt_low, tilt_high)

        current_azimuth = self.env.panel_azimuth_deg
        current_tilt = self.env.panel_tilt_deg
        delta_azimuth = np.clip(target_azimuth - current_azimuth, -self.max_delta_deg_per_step, self.max_delta_deg_per_step)
        delta_tilt = np.clip(target_tilt - current_tilt, -self.max_delta_deg_per_step, self.max_delta_deg_per_step)
        action = self.env.action_from_target_angles(
            current_azimuth + float(delta_azimuth),
            current_tilt + float(delta_tilt),
        ).tolist()
        actions = [action]
        self.actions[0][self.time_step] = actions[0]
        self.next_time_step()
        return actions
