"""Goal-conditioned hourly SLM guidance for solar tracking."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .client import QwenClient
from .local_client import LocalQwenClient
from .goal_prompts import (build_goal_prompt, build_goal_prompt_v2,
                           build_goal_retry_prompt)
from .parser import extract_json_object


def _finite_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not np.isfinite(parsed):
        parsed = float(fallback)
    return float(parsed)


def _bool_value(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "hold"}:
            return True
        if normalized in {"false", "0", "no", "track"}:
            return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return bool(fallback)


@dataclass(frozen=True)
class GoalOutput:
    """Validated hourly SLM target, pre-position, and motion-budget plan."""

    target_tilt: float
    target_az: float
    motion_budget_deg: float
    pre_position_tilt: float
    pre_position_az: float
    hold: bool
    horizon_min: float
    confidence: float = 0.5
    source: str = "fallback"
    raw_response: str = ""

    def context_vector(self) -> np.ndarray:
        """Return compact normalized goal vector for the SAC observation."""

        return np.array(
            [
                _scale(self.target_az, 70.0, 290.0),
                _scale(self.target_tilt, 10.0, 85.0),
                _scale(self.pre_position_az, 70.0, 290.0),
                _scale(self.pre_position_tilt, 10.0, 85.0),
                _scale(self.motion_budget_deg, 0.0, 120.0, output_low=0.0, output_high=1.0),
                1.0 if self.hold else 0.0,
                _scale(self.horizon_min, 10.0, 60.0, output_low=0.0, output_high=1.0),
                float(np.clip(self.confidence, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )


def _scale(value: float, low: float, high: float, output_low: float = -1.0, output_high: float = 1.0) -> float:
    if abs(high - low) < 1e-9:
        return float(output_low)
    value = float(np.clip(value, low, high))
    scaled = (value - low) / (high - low)
    return float(output_low + scaled * (output_high - output_low))


class LLMGoalGuidance:
    """Hourly goal provider with Qwen, validation, cache, and fallback."""

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        enabled: bool = True,
        query_interval_steps: int = 6,
        use_llm: bool = True,
        verbose: bool = False,
        reuse_cached_fallbacks: bool = True,
        max_retries: int = 5,
        retry_delay_seconds: float = 2.0,
        guardrail_mode: str = "full",
        prompt_version: int = 1,
        operator_instruction: str = "",
    ):
        self.client = client or LocalQwenClient()
        self.enabled = bool(enabled)
        self.query_interval_steps = int(query_interval_steps)
        self.use_llm = bool(use_llm)
        self.verbose = bool(verbose)
        self.reuse_cached_fallbacks = bool(reuse_cached_fallbacks)
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        # Guardrail level for _validate_goal (PROJECT_HANDOFF.md 11, phase 3):
        #   'full'    - paper behaviour: forced budget + angle override + budget blend
        #   'partial' - drop forced budget and budget blend; KEEP the angle override
        #   'loose'   - drop all three; SLM decisions stand (safety limits still apply)
        if str(guardrail_mode) not in ("full", "partial", "loose"):
            raise ValueError("guardrail_mode must be full|partial|loose")
        self.guardrail_mode = str(guardrail_mode)
        # 1 = paper prompt (DNI-threshold ladder); 2 = ladder-free autonomous
        # planner prompt used with the oracle-trained adapter.
        if int(prompt_version) not in (1, 2):
            raise ValueError("prompt_version must be 1 or 2")
        self.prompt_version = int(prompt_version)
        # Plain-language directive injected into the v2 prompt (programmability).
        self.operator_instruction = str(operator_instruction or "")
        self.cache: Dict[Tuple, GoalOutput] = {}
        self.request_count = 0
        self.call_count = 0
        self.fallback_count = 0
        self.parse_failure_count = 0
        self.last_raw_response = ""
        self.last_error = ""

    def context_names(self) -> list[str]:
        return [
            "llm_goal_target_az",
            "llm_goal_target_tilt",
            "llm_goal_pre_position_az",
            "llm_goal_pre_position_tilt",
            "llm_goal_motion_budget",
            "llm_goal_hold",
            "llm_goal_horizon",
            "llm_goal_confidence",
        ]

    def guidance_for(self, time_step: int, telemetry: Mapping[str, object]) -> GoalOutput:
        if not self.enabled:
            return self.fallback_goal(telemetry, source="disabled")

        key = self._cache_key(time_step, telemetry)
        if key in self.cache:
            cached = self.cache[key]
            if not self.use_llm or cached.source == "qwen" or self.reuse_cached_fallbacks:
                return cached

        fallback = self.fallback_goal(telemetry)
        output = fallback

        # Outside the active tracking window, no strategic SLM call is needed.
        # The tracker should simply hold/reset according to the deterministic
        # environment policy, and this cached fallback still provides a valid
        # context vector for the SAC observation.
        if not bool(telemetry.get("tracking_window_active", True)):
            self.cache[key] = output
            if self.verbose:
                print(
                    f"Goal guidance @ step {time_step}: outside tracking window, "
                    f"using fallback hold target=({output.target_az:.1f},{output.target_tilt:.1f})"
                )
            return output

        if self.use_llm:
            self.request_count += 1
            last_exception = None
            for attempt in range(1, self.max_retries + 1):
                raw = ""
                try:
                    if attempt == 1:
                        messages = (build_goal_prompt_v2(telemetry, self.operator_instruction)
                                    if self.prompt_version == 2
                                    else build_goal_prompt(telemetry))
                        temperature = 0.35
                        # 160 tokens fits the reasoning line (~30 t) + all JSON
                        # fields (~80 t) with margin. Capping here cuts warmup
                        # time from ~30 s/call to ~12 s/call on the local adapter.
                        max_tokens = 160
                    else:
                        # Compact JSON-only retry for malformed first responses.
                        # A v1 retry prompt would re-inject the DNI-threshold
                        # ladder that v2 deliberately removes.
                        messages = (build_goal_prompt_v2(telemetry, self.operator_instruction)
                                    if self.prompt_version == 2
                                    else build_goal_retry_prompt(telemetry))
                        temperature = 0.0
                        max_tokens = 100

                    raw = self.client.chat_json(messages, temperature=temperature, max_tokens=max_tokens)
                    self.last_raw_response = raw
                    if not str(raw).strip():
                        raise ValueError("empty model response")
                    parsed = extract_json_object(raw)
                    output = self._validate_goal(parsed, fallback, raw, telemetry)
                    self.call_count += 1
                    last_exception = None
                    break
                except Exception as exc:
                    last_exception = exc
                    self.last_error = f"attempt {attempt}/{self.max_retries} {type(exc).__name__}: {exc}"
                    if attempt < self.max_retries and self.retry_delay_seconds > 0.0:
                        time.sleep(self.retry_delay_seconds * attempt)
            if last_exception is not None:
                self.parse_failure_count += 1
                self.fallback_count += 1
                output = fallback
        else:
            self.fallback_count += 1

        self.cache[key] = output
        if self.verbose:
            print(
                f"Goal guidance @ step {time_step}: hold={output.hold}, "
                f"target=({output.target_az:.1f},{output.target_tilt:.1f}), "
                f"pre=({output.pre_position_az:.1f},{output.pre_position_tilt:.1f}), "
                f"budget={output.motion_budget_deg:.1f}, source={output.source}"
            )
        return output

    def fallback_goal(self, telemetry: Mapping[str, object], source: str = "fallback") -> GoalOutput:
        current = telemetry.get("current_target", {}) or {}
        future = telemetry.get("future_target", {}) or current
        dni = _finite_float(telemetry.get("dni"), 0.0)
        avg_dni = _finite_float(telemetry.get("next_30min_average_dni"), dni)
        zenith = _finite_float(telemetry.get("solar_zenith_deg"), 180.0)
        tracking_window_active = bool(telemetry.get("tracking_window_active", True))
        hold = zenith >= 90.0 or not tracking_window_active or (dni < 100.0 and avg_dni < 100.0)
        max_d = max(dni, avg_dni)
        budget = 0.0 if hold else (
            30.0 if max_d >= 700.0 else (
            20.0 if max_d >= 500.0 else (
            12.0 if max_d >= 300.0 else (
            6.0 if max_d >= 100.0 else 3.0))))
        panel_az = _finite_float(telemetry.get("panel_azimuth_deg"), _finite_float(current.get("azimuth"), 180.0))
        panel_tilt = _finite_float(telemetry.get("panel_tilt_deg"), _finite_float(current.get("tilt"), 10.0))
        target_az = panel_az if hold else _finite_float(current.get("azimuth"), 180.0)
        target_tilt = panel_tilt if hold else _finite_float(current.get("tilt"), 10.0)
        pre_az = panel_az if hold else _finite_float(future.get("azimuth"), target_az)
        pre_tilt = panel_tilt if hold else _finite_float(future.get("tilt"), target_tilt)
        return GoalOutput(
            target_tilt=target_tilt,
            target_az=target_az,
            motion_budget_deg=budget,
            pre_position_tilt=pre_tilt,
            pre_position_az=pre_az,
            hold=hold,
            horizon_min=60.0,
            confidence=0.5,
            source=source,
        )

    def _validate_goal(self, parsed: Mapping[str, Any], fallback: GoalOutput, raw: str, telemetry: Mapping[str, object]) -> GoalOutput:
        target_az = float(np.clip(_finite_float(parsed.get("target_az"), fallback.target_az), 70.0, 290.0))
        target_tilt = float(np.clip(_finite_float(parsed.get("target_tilt"), fallback.target_tilt), 10.0, 85.0))
        pre_az = float(np.clip(_finite_float(parsed.get("pre_position_az"), fallback.pre_position_az), 70.0, 290.0))
        pre_tilt = float(np.clip(_finite_float(parsed.get("pre_position_tilt"), fallback.pre_position_tilt), 10.0, 85.0))
        budget = float(np.clip(_finite_float(parsed.get("motion_budget_deg"), fallback.motion_budget_deg), 0.0, 120.0))
        horizon = float(np.clip(_finite_float(parsed.get("horizon_min"), fallback.horizon_min), 10.0, 60.0))
        confidence = float(np.clip(_finite_float(parsed.get("confidence"), fallback.confidence), 0.0, 1.0))
        hold = _bool_value(parsed.get("hold"), fallback.hold)

        # Physical daylight guardrail: Qwen may be conservative, but useful
        # irradiance should not be converted into a hold/near-zero-budget plan.
        # This preserves Qwen guidance in cloudy periods while preventing missed
        # high-DNI tracking opportunities.
        dni = _finite_float(telemetry.get("dni"), 0.0)
        avg_dni = _finite_float(telemetry.get("next_30min_average_dni"), dni)
        zenith = _finite_float(telemetry.get("solar_zenith_deg"), 180.0)
        tracking_window_active = bool(telemetry.get("tracking_window_active", True))
        max_dni = max(dni, avg_dni)

        if self.guardrail_mode == "full":
            if tracking_window_active and zenith < 85.0 and max_dni >= 700.0:
                hold = False
                budget = max(budget, 25.0)
            elif tracking_window_active and zenith < 85.0 and max_dni >= 500.0:
                hold = False
                budget = max(budget, 15.0)
            elif tracking_window_active and zenith < 80.0 and max_dni >= 200.0 and hold:
                hold = False
                budget = max(budget, 10.0)

        # In tracking/pre-position mode, solar geometry gives the exact target.
        # Qwen decides the strategic mode and movement budget, but it should not
        # invent sun-facing angles. This prevents inconsistent outputs like
        # hold=False with target=(100, 85), which is just the morning hold pose.
        if not hold and self.guardrail_mode in ("full", "partial"):
            current = telemetry.get("current_target", {}) or {}
            future = telemetry.get("future_target", {}) or current
            target_az = float(np.clip(_finite_float(current.get("azimuth"), target_az), 70.0, 290.0))
            target_tilt = float(np.clip(_finite_float(current.get("tilt"), target_tilt), 10.0, 85.0))
            pre_az = float(np.clip(_finite_float(future.get("azimuth"), target_az), 70.0, 290.0))
            pre_tilt = float(np.clip(_finite_float(future.get("tilt"), target_tilt), 10.0, 85.0))

        # Blend Qwen's budget with a physics estimate to smooth bimodal outputs.
        # Weight Qwen's contribution by its confidence: at confidence=1.0 Qwen
        # controls 80% of the budget; at the fallback confidence=0.5 it controls
        # 60%, keeping the physics anchor while giving high-confidence plans more
        # influence over strategic motion decisions.
        if not hold and self.guardrail_mode == "full":
            max_dni_blend = max(
                _finite_float(telemetry.get("dni"), 0.0),
                _finite_float(telemetry.get("next_30min_average_dni"), 0.0),
            )
            physics_budget = float(np.clip(max_dni_blend * 0.043, 3.0, 30.0))
            llm_weight = float(np.clip(0.4 + 0.4 * confidence, 0.4, 0.8))
            budget = float(np.clip(llm_weight * budget + (1.0 - llm_weight) * physics_budget, 0.0, 30.0))

        # Deterministic guardrail: hold means keep the current panel pose.
        # This prevents Qwen from turning a hold/stow decision into an artificial
        # jump such as azimuth=100 deg, tilt=85 deg during useful daylight.
        if hold:
            panel_az = _finite_float(telemetry.get("panel_azimuth_deg"), fallback.target_az)
            panel_tilt = _finite_float(telemetry.get("panel_tilt_deg"), fallback.target_tilt)
            target_az = float(np.clip(panel_az, 70.0, 290.0))
            target_tilt = float(np.clip(panel_tilt, 10.0, 85.0))
            pre_az = target_az
            pre_tilt = target_tilt
            budget = min(budget, 3.0)

        # Environment-defined night/outside-window holds remain authoritative.
        if fallback.hold:
            hold = True
            budget = min(budget, 3.0)
            target_az = fallback.target_az
            target_tilt = fallback.target_tilt
            pre_az = fallback.pre_position_az
            pre_tilt = fallback.pre_position_tilt

        return GoalOutput(
            target_tilt=target_tilt,
            target_az=target_az,
            motion_budget_deg=budget,
            pre_position_tilt=pre_tilt,
            pre_position_az=pre_az,
            hold=hold,
            horizon_min=horizon,
            confidence=confidence,
            source="qwen",
            raw_response=raw,
        )

    def _cache_key(self, time_step: int, telemetry: Mapping[str, object]) -> Tuple:
        """Return a stable per-dataset-hour cache key.

        Training repeatedly rolls out the same selected weather period. Reusing
        the same hourly SLM plan across episodes avoids thousands of duplicate
        LM Studio calls and makes the experiment reproducible. If the wrapper
        provides an absolute dataset time step, cache by that value; otherwise
        fall back to the relative environment time step.
        """

        absolute_time_step = telemetry.get("dataset_time_step", time_step)
        bucket = int(absolute_time_step) // max(1, self.query_interval_steps)
        return ("dataset_hour_bucket", bucket)
