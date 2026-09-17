"""Reward functions for solar tracker control."""

from typing import Any, List, Mapping, Optional, Union


class SolarTrackerReward:
    """Reward function based on peak-weighted relative tracking performance.

    The main terms are raw/generated energy and a peak-weighted tracking
    efficiency bonus:

        step_tracking_efficiency = generated_energy_kwh / ideal_generated_energy_kwh
        peak_weight = ideal_generated_power_kw / rated_power_kw

    This emphasizes accurate pointing when energy availability is high, exactly
    where small angle errors cost the most.
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        energy_weight: float = 1.0,
        tracking_efficiency_weight: float = 0.5,
        peak_tracking_exponent: float = 1.0,
        azimuth_movement_penalty: float = 0.001,
        tilt_movement_penalty: float = 0.0015,
        azimuth_squared_movement_penalty: float = 0.000002,
        tilt_squared_movement_penalty: float = 0.000003,
        smoothness_penalty: float = 0.000002,
        ideal_energy_epsilon: float = 1e-6,
        energy_reward_mode: str = "normalized",
        reward_curriculum: Optional[List[Mapping[str, Union[int, float, str]]]] = None,
    ):
        self.env_metadata = env_metadata
        self.energy_weight = energy_weight
        self.tracking_efficiency_weight = tracking_efficiency_weight
        self.peak_tracking_exponent = peak_tracking_exponent
        self.azimuth_movement_penalty = azimuth_movement_penalty
        self.tilt_movement_penalty = tilt_movement_penalty
        self.azimuth_squared_movement_penalty = azimuth_squared_movement_penalty
        self.tilt_squared_movement_penalty = tilt_squared_movement_penalty
        self.smoothness_penalty = smoothness_penalty
        self.ideal_energy_epsilon = ideal_energy_epsilon
        self.energy_reward_mode = energy_reward_mode
        self.reward_curriculum = sorted(
            reward_curriculum or [],
            key=lambda stage: int(stage.get("start_episode", 1)),
        )
        self.current_stage_name = "default"
        self._previous_azimuth_deltas: List[float] = []
        self._previous_tilt_deltas: List[float] = []

    @property
    def central_agent(self) -> bool:
        return bool(self.env_metadata.get("central_agent", False))

    def reset(self):
        """Hook for episode-reset stateful rewards."""
        self._previous_azimuth_deltas = []
        self._previous_tilt_deltas = []

    def set_episode(self, episode: int) -> None:
        """Apply curriculum reward coefficients for a 1-based training episode."""

        selected_stage = None

        for stage in self.reward_curriculum:
            if int(episode) >= int(stage.get("start_episode", 1)):
                selected_stage = stage
            else:
                break

        if selected_stage is not None:
            self.set_weights(**selected_stage)

    def set_weights(self, **weights: Union[int, float, str]) -> None:
        """Update reward weights without reconstructing the environment."""

        self.current_stage_name = str(weights.get("name", self.current_stage_name))
        for name in [
            "energy_weight",
            "tracking_efficiency_weight",
            "peak_tracking_exponent",
            "azimuth_movement_penalty",
            "tilt_movement_penalty",
            "azimuth_squared_movement_penalty",
            "tilt_squared_movement_penalty",
            "smoothness_penalty",
            "ideal_energy_epsilon",
        ]:
            if name in weights:
                setattr(self, name, float(weights[name]))

        if "energy_reward_mode" in weights:
            self.energy_reward_mode = str(weights["energy_reward_mode"])

    def get_weights(self) -> Mapping[str, float]:
        """Return current reward weights for logging/debugging."""

        return {
            "stage": self.current_stage_name,
            "energy_weight": self.energy_weight,
            "tracking_efficiency_weight": self.tracking_efficiency_weight,
            "peak_tracking_exponent": self.peak_tracking_exponent,
            "azimuth_movement_penalty": self.azimuth_movement_penalty,
            "tilt_movement_penalty": self.tilt_movement_penalty,
            "azimuth_squared_movement_penalty": self.azimuth_squared_movement_penalty,
            "tilt_squared_movement_penalty": self.tilt_squared_movement_penalty,
            "smoothness_penalty": self.smoothness_penalty,
            "ideal_energy_epsilon": self.ideal_energy_epsilon,
            "energy_reward_mode": self.energy_reward_mode,
        }

    def _max_step_energy_kwh(self) -> float:
        """Return rated one-step energy for a small absolute-energy bonus."""

        metadata = self.env_metadata or {}
        panel_metadata = metadata.get("panel", {})
        rated_power_kw = float(panel_metadata.get("rated_power_kw", 0.0))
        seconds_per_time_step = float(metadata.get("seconds_per_time_step", 3600.0))
        return max(rated_power_kw * seconds_per_time_step / 3600.0, self.ideal_energy_epsilon)

    def _rated_power_kw(self) -> float:
        """Return configured panel rated power for peak weighting."""

        metadata = self.env_metadata or {}
        panel_metadata = metadata.get("panel", {})
        return max(float(panel_metadata.get("rated_power_kw", 0.0)), self.ideal_energy_epsilon)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list: List[float] = []
        max_step_energy_kwh = self._max_step_energy_kwh()
        rated_power_kw = self._rated_power_kw()

        while len(self._previous_azimuth_deltas) < len(observations):
            self._previous_azimuth_deltas.append(0.0)
            self._previous_tilt_deltas.append(0.0)

        for index, observation in enumerate(observations):
            energy_kwh = float(observation.get("generated_energy_kwh", 0.0))
            ideal_energy_kwh = float(observation.get("ideal_generated_energy_kwh", 0.0))
            ideal_power_kw = float(observation.get("ideal_generated_power_kw", 0.0))
            signed_delta_azimuth = float(observation.get("panel_azimuth_delta_deg", 0.0))
            signed_delta_tilt = float(observation.get("panel_tilt_delta_deg", 0.0))
            delta_azimuth = abs(signed_delta_azimuth)
            delta_tilt = abs(signed_delta_tilt)
            azimuth_jerk = signed_delta_azimuth - self._previous_azimuth_deltas[index]
            tilt_jerk = signed_delta_tilt - self._previous_tilt_deltas[index]

            if ideal_energy_kwh > self.ideal_energy_epsilon:
                tracking_efficiency = min(max(energy_kwh / ideal_energy_kwh, 0.0), 1.25)
                normalized_energy = min(max(energy_kwh / max_step_energy_kwh, 0.0), 1.25)
                peak_weight = min(max(ideal_power_kw / rated_power_kw, 0.0), 1.25) ** self.peak_tracking_exponent
                reward = self.tracking_efficiency_weight * peak_weight * tracking_efficiency
                energy_term = energy_kwh if self.energy_reward_mode == "raw" else normalized_energy
                reward += self.energy_weight * energy_term
            else:
                # At night or during no-useful-irradiance periods, only movement
                # matters. This avoids confusing the agent with impossible energy.
                reward = self.energy_weight * energy_kwh if self.energy_reward_mode == "raw" else 0.0

            reward -= self.azimuth_movement_penalty * delta_azimuth
            reward -= self.tilt_movement_penalty * delta_tilt
            reward -= self.azimuth_squared_movement_penalty * (delta_azimuth**2)
            reward -= self.tilt_squared_movement_penalty * (delta_tilt**2)
            reward -= self.smoothness_penalty * ((azimuth_jerk**2) + (tilt_jerk**2))
            reward_list.append(reward)
            self._previous_azimuth_deltas[index] = signed_delta_azimuth
            self._previous_tilt_deltas[index] = signed_delta_tilt

        if self.central_agent:
            return [sum(reward_list)]

        return reward_list
