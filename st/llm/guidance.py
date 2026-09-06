"""Mode-B guidance orchestration and deterministic fallback classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .client import QwenClient
from .parser import extract_choice_from_text, extract_json_object, validate_choice
from .prompts import build_mode_b_prompt

REGIMES = [
    "clear_stable",
    "clear_variable",
    "partly_cloudy",
    "overcast_stable",
    "overcast_variable",
    "rain_storm",
    "extreme_event",
]

STRATEGIES = ["peak_track", "track", "low_motion_track", "stow"]


def _clip01(value: Any, fallback: float) -> float:
    """Return a finite scalar clipped to [0, 1]."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)

    if not np.isfinite(parsed):
        parsed = float(fallback)

    return float(np.clip(parsed, 0.0, 1.0))


def _normalized_reward_weights(
    alignment: Any,
    movement: Any,
    smoothness: Any,
    fallback: Tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> Tuple[float, float, float]:
    """Return safe Mode-A weights for alignment, movement, and smoothness."""

    weights = np.array(
        [
            _clip01(alignment, fallback[0]),
            _clip01(movement, fallback[1]),
            _clip01(smoothness, fallback[2]),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 1e-9:
        weights = np.asarray(fallback, dtype=np.float64)

    # Keep every objective present so Qwen cannot collapse the reward into a
    # single unstable term. The values are normalized again after clipping.
    weights = np.clip(weights, 0.05, 0.90)
    weights = weights / weights.sum()
    return float(weights[0]), float(weights[1]), float(weights[2])


def _strategy_adjusted_reward_weights(
    strategy: str,
    alignment_weight: float,
    movement_weight: float,
    smoothness_weight: float,
) -> Tuple[float, float, float]:
    """Apply simple safety bounds so strategies behave as named.

    Qwen occasionally labels a period as ``low_motion_track`` while returning a
    high alignment weight or very low smoothness weight. This post-processor
    keeps the language-model output useful without letting it contradict the
    controller semantics.
    """

    weights = np.array([alignment_weight, movement_weight, smoothness_weight], dtype=np.float64)

    if strategy == "low_motion_track":
        # Low-motion tracking should reduce unnecessary actuator activity, not
        # prevent useful alignment when irradiance is still worth collecting.
        weights[1] = max(weights[1], 0.25)
        weights[2] = max(weights[2], 0.25)
        weights[0] = min(weights[0], 0.45)
    elif strategy == "stow":
        weights = np.array([0.05, 0.45, 0.50], dtype=np.float64)
    elif strategy == "peak_track":
        # During high-value sunlight, alignment should dominate while movement
        # and smoothness remain present to prevent jittery policies.
        weights[0] = max(weights[0], 0.65)
        weights[1] = max(weights[1], 0.08)
        weights[2] = max(weights[2], 0.08)

    weights = np.clip(weights, 0.05, 0.90)
    weights = weights / weights.sum()
    return float(weights[0]), float(weights[1]), float(weights[2])


def _useful_irradiance(telemetry: Mapping[str, float]) -> bool:
    """Return whether the next guidance hour has enough DNI to justify tracking."""

    dni = float(telemetry.get("dni", 0.0) or 0.0)
    avg_dni = float(telemetry.get("next_30min_average_dni", dni) or dni)
    return dni >= 150.0 or avg_dni >= 150.0


def _peak_irradiance(telemetry: Mapping[str, float]) -> bool:
    """Return whether alignment should dominate because direct irradiance is high."""

    dni = float(telemetry.get("dni", 0.0) or 0.0)
    avg_dni = float(telemetry.get("next_30min_average_dni", dni) or dni)
    return dni >= 450.0 or avg_dni >= 450.0


def _project_weights(
    weights: np.ndarray,
    min_alignment: float,
    max_movement: float,
    max_smoothness: float,
) -> np.ndarray:
    """Project weights so final normalized values respect key guardrails."""

    movement = float(np.clip(weights[1], 0.05, max_movement))
    smoothness = float(np.clip(weights[2], 0.05, max_smoothness))
    remaining_for_penalties = max(0.10, 1.0 - float(min_alignment))
    penalty_sum = movement + smoothness

    if penalty_sum > remaining_for_penalties:
        scale = remaining_for_penalties / penalty_sum
        movement *= scale
        smoothness *= scale

    alignment = max(float(min_alignment), 1.0 - movement - smoothness)
    projected = np.array([alignment, movement, smoothness], dtype=np.float64)
    projected = np.clip(projected, 0.05, 0.90)
    projected = projected / projected.sum()
    return projected


def _irradiance_adjusted_strategy(strategy: str, telemetry: Mapping[str, float]) -> str:
    """Avoid low-motion labels when useful DNI is available.

    In partly cloudy daylight, low-motion tracking can become too conservative
    because the tracker still needs to recover alignment with bounded delta
    actions. Promote those periods to normal tracking and let reward weights
    manage movement instead of suppressing alignment.
    """

    if strategy == "low_motion_track" and _useful_irradiance(telemetry):
        return "track"

    return strategy


def _irradiance_adjusted_reward_weights(
    strategy: str,
    alignment_weight: float,
    movement_weight: float,
    smoothness_weight: float,
    telemetry: Mapping[str, float],
) -> Tuple[float, float, float]:
    """Apply irradiance-aware guardrails to preserve energy during useful sun."""

    weights = np.array([alignment_weight, movement_weight, smoothness_weight], dtype=np.float64)
    useful = _useful_irradiance(telemetry)
    peak = _peak_irradiance(telemetry)

    if strategy == "stow":
        weights = np.array([0.05, 0.45, 0.50], dtype=np.float64)

    elif strategy == "peak_track" or peak:
        weights = _project_weights(weights, min_alignment=0.70, max_movement=0.20, max_smoothness=0.20)

    elif strategy == "track" and useful:
        weights = _project_weights(weights, min_alignment=0.55, max_movement=0.30, max_smoothness=0.30)

    elif strategy == "low_motion_track":
        weights = np.array([0.35, 0.325, 0.325], dtype=np.float64)

    weights = np.clip(weights, 0.05, 0.90)
    weights = weights / weights.sum()
    return float(weights[0]), float(weights[1]), float(weights[2])


@dataclass(frozen=True)
class GuidanceOutput:
    """Validated hourly SLM Mode-B context and Mode-A reward weights."""

    regime: str
    strategy: str
    tilt_priority: float = 0.5
    movement_budget: float = 0.5
    alignment_weight: float = 0.5
    movement_weight: float = 0.25
    smoothness_weight: float = 0.25
    source: str = "fallback"
    raw_response: str = ""

    def context_vector(self) -> np.ndarray:
        """Return one-hot context, scalar priorities, and Mode-A weights."""

        vector = np.zeros(len(REGIMES) + len(STRATEGIES) + 5, dtype=np.float32)
        vector[REGIMES.index(self.regime)] = 1.0
        vector[len(REGIMES) + STRATEGIES.index(self.strategy)] = 1.0
        vector[-5] = _clip01(self.tilt_priority, 0.5)
        vector[-4] = _clip01(self.movement_budget, 0.5)
        weights = _normalized_reward_weights(self.alignment_weight, self.movement_weight, self.smoothness_weight)
        vector[-3:] = weights
        return vector


class LLMGuidance:
    """Hourly Mode-B regime/strategy provider with cache and fallback.

    The cache is keyed by the simulated-hour bucket. This enforces one reusable
    daily/hourly guidance decision across the six 10-minute SAC steps in that
    hour instead of querying the local LLM at every fast-loop step.
    """

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        enabled: bool = True,
        query_interval_steps: int = 6,
        use_llm: bool = True,
        verbose: bool = False,
    ):
        self.client = client or QwenClient()
        self.enabled = bool(enabled)
        self.query_interval_steps = int(query_interval_steps)
        self.use_llm = bool(use_llm)
        self.verbose = bool(verbose)
        self.cache: Dict[Tuple, GuidanceOutput] = {}
        self.request_count = 0
        self.call_count = 0
        self.fallback_count = 0
        self.parse_failure_count = 0
        self.last_raw_response = ""
        self.last_error = ""

    def context_names(self) -> list[str]:
        return (
            [f"llm_regime_{name}" for name in REGIMES]
            + [f"llm_strategy_{name}" for name in STRATEGIES]
            + [
                "llm_tilt_priority",
                "llm_movement_budget",
                "llm_alignment_weight",
                "llm_movement_weight",
                "llm_smoothness_weight",
            ]
        )

    def guidance_for(self, time_step: int, telemetry: Mapping[str, float]) -> GuidanceOutput:
        if not self.enabled:
            return GuidanceOutput(
                "clear_stable",
                "track",
                tilt_priority=0.5,
                movement_budget=0.5,
                alignment_weight=0.5,
                movement_weight=0.25,
                smoothness_weight=0.25,
                source="disabled",
            )

        key = self._cache_key(time_step, telemetry)
        if key in self.cache:
            cached = self.cache[key]

            # If this object is now configured to use Qwen, do not let older
            # fallback-only cache entries silently prevent real SLM calls.
            # This can happen in notebooks when fallback seeding/diagnostics are
            # run before enabling Qwen-guided training.
            if not self.use_llm or str(cached.source).startswith("qwen"):
                return cached

        fallback = self.fallback_guidance(telemetry)

        # Nighttime and hard safety stow are deterministic; no expensive LLM
        # inference is needed. This preserves the slow-loop budget for daylight
        # decisions where semantic guidance can change tracking behavior.
        if self._is_deterministic_stow(telemetry):
            self.cache[key] = fallback
            return fallback

        output = fallback

        if self.use_llm:
            raw = ""
            try:
                self.request_count += 1
                raw = self.client.chat_json(build_mode_b_prompt(telemetry), temperature=0.1, max_tokens=512)
                self.last_raw_response = raw
                parsed = extract_json_object(raw)
                reward_weights = parsed.get("reward_weights", {}) or {}
                fallback_weights = (
                    fallback.alignment_weight,
                    fallback.movement_weight,
                    fallback.smoothness_weight,
                )
                alignment_weight, movement_weight, smoothness_weight = _normalized_reward_weights(
                    reward_weights.get("alignment", parsed.get("alignment_weight")),
                    reward_weights.get("movement", parsed.get("movement_weight")),
                    reward_weights.get("smoothness", parsed.get("smoothness_weight")),
                    fallback=fallback_weights,
                )
                parsed_strategy = validate_choice(parsed.get("strategy"), STRATEGIES, fallback.strategy)
                parsed_strategy = _irradiance_adjusted_strategy(parsed_strategy, telemetry)
                alignment_weight, movement_weight, smoothness_weight = _strategy_adjusted_reward_weights(
                    parsed_strategy,
                    alignment_weight,
                    movement_weight,
                    smoothness_weight,
                )
                alignment_weight, movement_weight, smoothness_weight = _irradiance_adjusted_reward_weights(
                    parsed_strategy,
                    alignment_weight,
                    movement_weight,
                    smoothness_weight,
                    telemetry,
                )
                output = GuidanceOutput(
                    regime=validate_choice(parsed.get("regime"), REGIMES, fallback.regime),
                    strategy=parsed_strategy,
                    tilt_priority=_clip01(parsed.get("tilt_priority"), fallback.tilt_priority),
                    movement_budget=_clip01(parsed.get("movement_budget"), fallback.movement_budget),
                    alignment_weight=alignment_weight,
                    movement_weight=movement_weight,
                    smoothness_weight=smoothness_weight,
                    source="qwen",
                    raw_response=raw,
                )
                self.call_count += 1
            except Exception as exc:  # keep RL training alive if LM Studio is unavailable
                self.last_error = f"{type(exc).__name__}: {exc}"
                recovered_regime = extract_choice_from_text(raw, REGIMES, fallback.regime)
                recovered_strategy = extract_choice_from_text(raw, STRATEGIES, fallback.strategy)
                recovered = (recovered_regime != fallback.regime) or (recovered_strategy != fallback.strategy)
                source = "qwen_text_recovered" if recovered and raw else f"fallback:{type(exc).__name__}"
                output = GuidanceOutput(
                    recovered_regime,
                    recovered_strategy,
                    tilt_priority=fallback.tilt_priority,
                    movement_budget=fallback.movement_budget,
                    alignment_weight=fallback.alignment_weight,
                    movement_weight=fallback.movement_weight,
                    smoothness_weight=fallback.smoothness_weight,
                    source=source,
                    raw_response=raw,
                )
                if source == "qwen_text_recovered":
                    self.call_count += 1
                else:
                    self.parse_failure_count += 1
                self.fallback_count += 1
        else:
            self.fallback_count += 1

        self.cache[key] = output
        if self.verbose:
            hour = telemetry.get("hour")
            minute = telemetry.get("minute")
            print(f"Mode B @ step {time_step} ({hour}:{minute}): {output.regime}, {output.strategy}, source={output.source}")
        return output

    def _is_deterministic_stow(self, telemetry: Mapping[str, float]) -> bool:
        zenith = float(telemetry.get("solar_zenith_deg", 180.0) or 180.0)
        wind = float(telemetry.get("wind_speed", 0.0) or 0.0)
        return zenith >= 90.0 or wind >= 19.44

    def fallback_guidance(self, telemetry: Mapping[str, float]) -> GuidanceOutput:
        """Rule-based fallback used when Qwen is disabled or returns invalid JSON."""

        zenith = float(telemetry.get("solar_zenith_deg", 180.0) or 180.0)
        dni = float(telemetry.get("dni", 0.0) or 0.0)
        next_dni = float(telemetry.get("next_10min_dni", dni) or dni)
        avg_dni = float(telemetry.get("next_30min_average_dni", dni) or dni)
        wind = float(telemetry.get("wind_speed", 0.0) or 0.0)
        csi = float(telemetry.get("dni_clear_sky_ratio", 0.0) or 0.0)

        if zenith >= 90.0 or wind >= 19.44:
            return GuidanceOutput(
                "extreme_event" if wind >= 19.44 else "overcast_stable",
                "stow",
                tilt_priority=0.0,
                movement_budget=0.0,
                alignment_weight=0.05,
                movement_weight=0.45,
                smoothness_weight=0.50,
                source="deterministic",
            )

        variability = abs(next_dni - dni) + abs(avg_dni - dni)
        if (dni >= 450.0 or avg_dni >= 450.0) and zenith <= 65.0 and csi >= 0.55:
            regime = "clear_variable" if variability > 180.0 else "clear_stable"
            return GuidanceOutput(
                regime,
                "peak_track",
                tilt_priority=1.0,
                movement_budget=0.9,
                alignment_weight=0.70,
                movement_weight=0.15,
                smoothness_weight=0.15,
            )
        if dni >= 500.0 and csi >= 0.65:
            regime = "clear_variable" if variability > 180.0 else "clear_stable"
            return GuidanceOutput(
                regime,
                "track",
                tilt_priority=0.8,
                movement_budget=0.7,
                alignment_weight=0.55,
                movement_weight=0.25,
                smoothness_weight=0.20,
            )
        if dni >= 150.0 or avg_dni >= 150.0:
            regime = "partly_cloudy" if variability > 120.0 else "overcast_variable"
            strategy = "track"
            tilt_priority = 0.65 if avg_dni >= 250.0 else 0.50
            movement_budget = 0.55 if avg_dni >= 250.0 else 0.40
            weights = (0.55, 0.25, 0.20) if avg_dni >= 250.0 else (0.50, 0.25, 0.25)
            return GuidanceOutput(
                regime,
                strategy,
                tilt_priority=tilt_priority,
                movement_budget=movement_budget,
                alignment_weight=weights[0],
                movement_weight=weights[1],
                smoothness_weight=weights[2],
            )
        return GuidanceOutput(
            "overcast_stable",
            "low_motion_track",
            tilt_priority=0.2,
            movement_budget=0.15,
            alignment_weight=0.35,
            movement_weight=0.325,
            smoothness_weight=0.325,
        )

    def _cache_key(self, time_step: int, telemetry: Mapping[str, float]) -> Tuple:
        bucket = int(time_step) // max(1, self.query_interval_steps)
        return (bucket,)
