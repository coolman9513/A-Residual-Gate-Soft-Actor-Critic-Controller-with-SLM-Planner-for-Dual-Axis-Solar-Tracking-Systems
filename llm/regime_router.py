"""Weather-regime-based policy router for SLM-SAC-Auto.

The SLM (Qwen2.5-0.5B) already classifies every hour into a weather regime
via its motion_budget_deg and hold fields. RegimeRouter uses these fields to
dispatch actions to one of three specialized SAC-Auto policies:

    clear  — high-DNI tracking  (motion_budget_deg >= 20)
    cloud  — moderate/variable  (3 < motion_budget_deg < 20)
    hold   — stow / night       (hold=True or motion_budget_deg <= 3)

Each policy is trained on homogeneous regime data using the seasonal curriculum
(summer → spring+autumn → full 4-season), so it specialises without being
confused by out-of-distribution irradiance levels.

Usage:
    router = RegimeRouter(clear_agent, cloud_agent, hold_agent)
    action, log_prob = router.select_action(obs, goal, eval=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .goal_guidance import GoalOutput


def route_goal(goal: Optional[GoalOutput], clear_threshold: float = 20.0) -> str:
    """Return the weather regime ('hold' / 'clear' / 'cloud') for an SLM goal.

    Shared by RegimeRouter (policy dispatch) and authority_for (residual
    authority). The SLM was fine-tuned so motion_budget_deg scales with DNI,
    so the budget doubles as a regime classifier:

        hold  — goal missing, hold=True, or motion_budget_deg <= 3
        clear — motion_budget_deg >= clear_threshold (default 20)
        cloud — everything in between
    """
    if goal is None or goal.hold or goal.motion_budget_deg <= 3.0:
        return "hold"
    if goal.motion_budget_deg >= float(clear_threshold):
        return "clear"
    return "cloud"


@dataclass(frozen=True)
class Authority:
    """Hard residual-authority limits for one hour.

    azimuth_deg / tilt_deg bound the per-step residual the RL policy may add
    on top of the RBC base action. hour_budget_deg bounds the cumulative
    residual motion (|az| + |tilt|) within the goal hour. All are enforced by
    clipping/projection in ResidualTrackerWrapper — not by reward.
    """

    regime: str
    azimuth_deg: float
    tilt_deg: float
    hour_budget_deg: float


# Sized from the measured RBC-vs-ideal gap on the 56-day dataset (1.85 kWh):
#   clear : 0.97 kWh gap at ~31 deg pointing error — the year-averaged RBC
#           schedule is far off on individual days (winter especially), so
#           the residual needs real correction room even in clear sky.
#   cloud : 0.56 kWh gap at ~55 deg error — diffuse light favors flatter
#           tilt than the beam-tracking schedule; full action range.
#   hold  : 0.30 kWh of the gap is sun still up outside the 07-18 tracking
#           window (evening hour 18-19) — a small nonzero authority lets the
#           residual harvest it; at true night the reward teaches zero.
# The hour budget still bounds cumulative residual motion, and cloud > clear
# keeps the semantic inversion vs the SLM's raw motion budget.
DEFAULT_AUTHORITY_TABLE: Mapping[str, Authority] = {
    "hold":  Authority("hold",  2.0, 1.5, 12.0),
    "clear": Authority("clear", 6.0, 4.0, 60.0),
    "cloud": Authority("cloud", 8.0, 5.0, 90.0),
}


def authority_for(
    goal: Optional[GoalOutput],
    table: Optional[Mapping[str, Authority]] = None,
    clear_threshold: float = 20.0,
) -> Authority:
    """Map an SLM goal to the hard residual authority for this hour."""
    table = table or DEFAULT_AUTHORITY_TABLE
    return table[route_goal(goal, clear_threshold=clear_threshold)]


class RegimeRouter:
    """Dispatches actions to one of three SAC-Auto policies based on SLM goal.

    Parameters
    ----------
    clear_agent : SAC_Auto
        Policy trained on CLEAR_STRONG / CLEAR_MODERATE days.
    cloud_agent : SAC_Auto
        Policy trained on PARTIAL_CLOUD / OVERCAST_DIM days.
    hold_agent  : SAC_Auto
        Policy trained on night / hold / heavy-overcast periods.
    clear_threshold : float
        motion_budget_deg threshold above which 'clear' policy is active (default 20°).
    """

    def __init__(
        self,
        clear_agent,
        cloud_agent,
        hold_agent,
        clear_threshold: float = 20.0,
    ):
        self._agents = {
            "clear": clear_agent,
            "cloud": cloud_agent,
            "hold":  hold_agent,
        }
        self._clear_threshold = float(clear_threshold)
        self._last_regime: Optional[str] = None

    def route(self, goal: Optional[GoalOutput]) -> str:
        """Return the regime key for the given SLM goal."""
        return route_goal(goal, clear_threshold=self._clear_threshold)

    @property
    def last_regime(self) -> Optional[str]:
        """Regime selected on the most recent select_action call."""
        return self._last_regime

    def select_action(self, obs, goal: Optional[GoalOutput], eval: bool = False):
        """Select an action using the appropriate sub-policy.

        Parameters
        ----------
        obs   : array-like observation (already normalized).
        goal  : GoalOutput from LLMGoalGuidance.guidance_for(), or None.
        eval  : If True use deterministic (zero-noise) action.

        Returns
        -------
        (action, log_prob) — same shape as SAC_Auto.select_action().
        """
        regime = self.route(goal)
        self._last_regime = regime
        mode = "zero" if eval else None
        return self._agents[regime].select_action(obs, eval=eval, mode=mode)

    def agents(self) -> dict:
        """Return the {regime: agent} mapping (for checkpointing)."""
        return dict(self._agents)
