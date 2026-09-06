"""CityLearn-style rule-based controller utilities for solar tracker control.

This module provides:
1. A deterministic 10-minute schedule representation for 07:00-18:00.
2. A rule-based controller class that maps current (hour, minute) to angles.
3. A compact interactive schedule editor for notebook workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from st.agents import SolarAgent

try:  # pragma: no cover
    import ipywidgets as widgets
    from IPython.display import clear_output, display
except Exception:  # pragma: no cover
    widgets = None
    clear_output = None
    display = None


TRACKING_START_HOUR = 7
TRACKING_END_HOUR = 18
TIME_STEP_MINUTES = 10


def tracking_time_slots(
    start_hour: int = TRACKING_START_HOUR,
    end_hour: int = TRACKING_END_HOUR,
    step_minutes: int = TIME_STEP_MINUTES,
    include_end: bool = False,
) -> List[Tuple[int, int]]:
    """Return ordered active tracking slots.

    Default is 66 slots from 07:00 to 17:50 (end-hour excluded).
    """

    assert 1 <= step_minutes <= 60 and 60 % step_minutes == 0
    slots: List[Tuple[int, int]] = []
    end_bound = end_hour + (1 if include_end else 0)

    for hour in range(start_hour, end_bound):
        minute_values = range(0, 60, step_minutes)
        for minute in minute_values:
            if include_end and hour == end_hour and minute > 0:
                break
            if not include_end and hour >= end_hour:
                break
            slots.append((hour, minute))

    if include_end:
        if (end_hour, 0) not in slots:
            slots.append((end_hour, 0))

    return slots


def _default_tilt_profile(n: int, morning_tilt: float, noon_tilt: float, evening_tilt: float) -> np.ndarray:
    """Piecewise-linear morning->noon->evening tilt profile."""

    if n == 0:
        return np.array([], dtype=float)

    midpoint = n // 2
    left = np.linspace(morning_tilt, noon_tilt, max(midpoint, 1), endpoint=False)
    right = np.linspace(noon_tilt, evening_tilt, n - len(left))
    return np.concatenate([left, right])


def default_tracker_schedule(
    azimuth_start: float = 120.0,
    azimuth_end: float = 240.0,
    morning_tilt: float = 85.0,
    noon_tilt: float = 35.0,
    evening_tilt: float = 80.0,
    azimuth_bounds: Tuple[float, float] = (70.0, 290.0),
    tilt_bounds: Tuple[float, float] = (10.0, 85.0),
    include_end: bool = False,
) -> pd.DataFrame:
    """Return deterministic default 10-minute RBC schedule."""

    slots = tracking_time_slots(include_end=include_end)
    n = len(slots)
    azimuth = np.linspace(azimuth_start, azimuth_end, n)
    tilt = _default_tilt_profile(n, morning_tilt, noon_tilt, evening_tilt)
    azimuth = np.clip(azimuth, azimuth_bounds[0], azimuth_bounds[1])
    tilt = np.clip(tilt, tilt_bounds[0], tilt_bounds[1])

    return pd.DataFrame(
        {
            "hour": [h for h, _ in slots],
            "minute": [m for _, m in slots],
            "azimuth": azimuth.astype(float),
            "tilt": tilt.astype(float),
        }
    )


def _circular_mean_degrees(values: Sequence[float]) -> float:
    """Return circular mean angle in degrees."""

    radians = np.radians(np.asarray(values, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)


def average_tracker_schedule(
    env,
    include_end: bool = False,
    use_episode_window: bool = False,
) -> pd.DataFrame:
    """Return daily schedule averaged from ideal angles in the environment data.

    The schedule is indexed only by time-of-day, e.g. 07:00, 07:10, ..., 17:50.
    For each slot, ideal panel azimuth/tilt are computed from the same POA model
    used by the environment and averaged over the full dataset by default.
    This creates one fixed daily RBC schedule that is not tuned to the selected
    evaluation episode.
    """

    azimuth_bounds = env.panel_azimuth_bounds
    tilt_bounds = env.panel_tilt_bounds
    fallback = default_tracker_schedule(
        azimuth_bounds=azimuth_bounds,
        tilt_bounds=tilt_bounds,
        include_end=include_end,
    )
    fallback_lookup = schedule_lookup(fallback)

    if use_episode_window:
        start = int(env.episode_tracker.episode_start_time_step)
        end = int(env.episode_tracker.episode_end_time_step)
        data = env.data.iloc[start : end + 1]
    else:
        data = env.data

    rows = []
    column = env.OBSERVATION_COLUMN_MAPPING

    for hour, minute in tracking_time_slots(include_end=include_end):
        subset = data[(data[column["hour"]] == hour) & (data[column["minute"]] == minute)]
        azimuth_values = []
        tilt_values = []

        for _, row in subset.iterrows():
            solar_zenith = float(row[column["solar_zenith_angle"]])
            dni = float(row[column["dni"]])
            dhi = float(row[column["dhi"]])

            if solar_zenith >= 90.0 or (dni + dhi) <= 0.0:
                continue

            azimuth, tilt = env._ideal_panel_orientation(
                dni=dni,
                dhi=dhi,
                solar_zenith_deg=solar_zenith,
                solar_azimuth_deg=float(row[column["solar_azimuth_angle"]]),
            )
            azimuth_values.append(azimuth)
            tilt_values.append(tilt)

        if azimuth_values:
            azimuth = np.clip(_circular_mean_degrees(azimuth_values), azimuth_bounds[0], azimuth_bounds[1])
            tilt = np.clip(float(np.mean(tilt_values)), tilt_bounds[0], tilt_bounds[1])
        else:
            azimuth, tilt = fallback_lookup[(hour, minute)]

        rows.append({"hour": hour, "minute": minute, "azimuth": float(azimuth), "tilt": float(tilt)})

    return pd.DataFrame(rows)


def normalize_schedule(
    schedule: Union[pd.DataFrame, Sequence[Mapping[str, Union[int, float]]]],
    azimuth_bounds: Tuple[float, float],
    tilt_bounds: Tuple[float, float],
) -> pd.DataFrame:
    """Validate, clip and sort schedule into a canonical dataframe."""

    schedule_df = pd.DataFrame(schedule).copy()
    required = {"hour", "minute", "azimuth", "tilt"}
    missing = required.difference(schedule_df.columns)

    if missing:
        raise KeyError(f"Schedule is missing required columns: {sorted(missing)}")

    schedule_df["hour"] = schedule_df["hour"].astype(int)
    schedule_df["minute"] = schedule_df["minute"].astype(int)
    schedule_df["azimuth"] = schedule_df["azimuth"].astype(float)
    schedule_df["tilt"] = schedule_df["tilt"].astype(float)
    schedule_df["azimuth"] = np.clip(schedule_df["azimuth"], azimuth_bounds[0], azimuth_bounds[1])
    schedule_df["tilt"] = np.clip(schedule_df["tilt"], tilt_bounds[0], tilt_bounds[1])
    schedule_df = schedule_df.sort_values(["hour", "minute"]).reset_index(drop=True)
    return schedule_df


def schedule_lookup(schedule_df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Return mapping: (hour, minute) -> (azimuth, tilt)."""

    return {
        (int(r.hour), int(r.minute)): (float(r.azimuth), float(r.tilt))
        for r in schedule_df.itertuples(index=False)
    }


class SolarTrackerRBC(SolarAgent):
    """Rule-based solar tracker controller with per-10-minute angle schedule.

    Outside tracking slots policy:
    - ``hold_last`` (default): hold current panel pose.
    - ``stow``: move to configured stow pose.
    """

    def __init__(
        self,
        env,
        schedule: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Union[int, float]]]]] = None,
        outside_tracking_policy: str = "hold_last",
        stow_azimuth_deg: float = 180.0,
        stow_tilt_deg: float = 10.0,
        include_end: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        self.outside_tracking_policy = outside_tracking_policy
        self.stow_azimuth_deg = float(stow_azimuth_deg)
        self.stow_tilt_deg = float(stow_tilt_deg)
        self.include_end = include_end
        self.debug = debug
        super().__init__(env, **kwargs)
        az_bounds = self.env.panel_azimuth_bounds
        tilt_bounds = self.env.panel_tilt_bounds
        default_schedule = average_tracker_schedule(env, include_end=include_end)
        self.schedule = normalize_schedule(
            default_schedule
            if schedule is None
            else schedule,
            azimuth_bounds=az_bounds,
            tilt_bounds=tilt_bounds,
        )
        self._lookup = schedule_lookup(self.schedule)
        self._schedule_slots_count = len(self.schedule)

    def set_schedule(
        self, schedule: Union[pd.DataFrame, Sequence[Mapping[str, Union[int, float]]]]
    ) -> None:
        """Replace controller schedule and refresh lookup."""

        az_bounds = self.env.panel_azimuth_bounds
        tilt_bounds = self.env.panel_tilt_bounds
        self.schedule = normalize_schedule(schedule, az_bounds, tilt_bounds)
        self._lookup = schedule_lookup(self.schedule)
        self._schedule_slots_count = len(self.schedule)

    def _scheduled_action(self, hour: int, minute: int) -> List[float]:
        key = (int(hour), int(minute))

        if key in self._lookup:
            return [self._lookup[key][0], self._lookup[key][1]]

        if self.outside_tracking_policy == "stow":
            return [self.stow_azimuth_deg, self.stow_tilt_deg]

        # default: hold last known state
        return [self.env.panel_azimuth_deg, self.env.panel_tilt_deg]

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        observation = observations[0]
        names = self.observation_names[0]

        if "hour" in names and "minute" in names:
            hour = int(observation[names.index("hour")])
            minute = int(observation[names.index("minute")])
        else:
            # The RL observation set may use cyclic time features instead of raw
            # hour/minute. RBC still needs exact wall-clock slots, so read them
            # from the environment data rather than forcing hour/minute active.
            row = self.env._row_at_time_step(self.env.time_step)
            hour = int(row[self.env.OBSERVATION_COLUMN_MAPPING["hour"]])
            minute = int(row[self.env.OBSERVATION_COLUMN_MAPPING["minute"]])

        target_azimuth, target_tilt = self._scheduled_action(hour, minute)
        action = self.env.action_from_target_angles(target_azimuth, target_tilt).tolist()
        actions = [action]
        self.actions[0][self.time_step] = actions[0]
        self.next_time_step()
        if self.debug and self.time_step == 1:
            print(f"RBC loaded daily schedule once with {self._schedule_slots_count} slots.")
        return actions


@dataclass
class RBCSimulationResult:
    """Container for simple notebook simulation outputs."""

    history: pd.DataFrame
    evaluation: pd.DataFrame


def rollout_rbc(env, agent: SolarTrackerRBC) -> RBCSimulationResult:
    """Run full episode with RBC agent and return history + KPIs."""

    observations = env.reset()
    done = False

    while not done:
        action = agent.predict(observations, deterministic=True)
        observations, _, done, _ = env.step(action)

    return RBCSimulationResult(history=env.history.copy(), evaluation=env.evaluate(include_baselines=False))


class SolarTrackerRBCEditor:
    """Compact interactive schedule editor for 66 active 10-minute slots."""

    def __init__(
        self,
        schedule: Optional[pd.DataFrame] = None,
        azimuth_bounds: Tuple[float, float] = (70.0, 290.0),
        tilt_bounds: Tuple[float, float] = (10.0, 85.0),
        include_end: bool = False,
        auto_render_schedule_table: bool = True,
        render_schedule_on_each_update: bool = False,
    ):
        if widgets is None:  # pragma: no cover
            raise ImportError("ipywidgets is required for SolarTrackerRBCEditor.")

        self.azimuth_bounds = azimuth_bounds
        self.tilt_bounds = tilt_bounds
        self.include_end = include_end
        self.auto_render_schedule_table = auto_render_schedule_table
        self.render_schedule_on_each_update = render_schedule_on_each_update
        self._rendered_once = False
        self._last_status_text: Optional[str] = None
        self._last_event_signature: Optional[Tuple[str, int, float, float]] = None
        self._last_event_time: float = 0.0
        self.schedule = (
            default_tracker_schedule(
                azimuth_bounds=azimuth_bounds,
                tilt_bounds=tilt_bounds,
                include_end=include_end,
            )
            if schedule is None
            else normalize_schedule(schedule, azimuth_bounds, tilt_bounds)
        )
        self._slots = [(int(h), int(m)) for h, m in zip(self.schedule["hour"], self.schedule["minute"])]
        self._build_widgets()
        if self.auto_render_schedule_table:
            self._render_table()

    def _build_widgets(self) -> None:
        options = [(f"{h:02d}:{m:02d}", idx) for idx, (h, m) in enumerate(self._slots)]
        self.slot = widgets.Dropdown(options=options, description="Slot", layout=widgets.Layout(width="220px"))
        self.azimuth = widgets.FloatSlider(
            value=float(self.schedule.loc[0, "azimuth"]),
            min=self.azimuth_bounds[0],
            max=self.azimuth_bounds[1],
            step=1.0,
            description="Azimuth",
            readout_format=".0f",
            layout=widgets.Layout(width="420px"),
        )
        self.tilt = widgets.FloatSlider(
            value=float(self.schedule.loc[0, "tilt"]),
            min=self.tilt_bounds[0],
            max=self.tilt_bounds[1],
            step=1.0,
            description="Tilt",
            readout_format=".0f",
            layout=widgets.Layout(width="420px"),
        )
        self.set_slot = widgets.Button(description="Set Slot", button_style="success")
        self.set_all = widgets.Button(description="Set All Active Slots", button_style="")
        self.reset_default = widgets.Button(description="Reset Default", button_style="warning")
        self.show_table = widgets.Button(description="Show Schedule Table", button_style="")
        self.table_html = widgets.HTML(value="")
        self.status_html = widgets.HTML(value="")

        # Bind callbacks once per editor instance.
        if not getattr(self, "_callbacks_bound", False):
            self.slot.observe(self._on_slot_change, names="value")
            self.set_slot.on_click(self._on_set_slot)
            self.set_all.on_click(self._on_set_all)
            self.reset_default.on_click(self._on_reset_default)
            self.show_table.on_click(self._on_show_table)
            self._callbacks_bound = True

    def _on_slot_change(self, change) -> None:
        idx = int(change["new"])
        self.azimuth.value = float(self.schedule.loc[idx, "azimuth"])
        self.tilt.value = float(self.schedule.loc[idx, "tilt"])

    def _on_set_slot(self, _=None) -> None:
        idx = int(self.slot.value)
        event_signature = ("slot", idx, float(self.azimuth.value), float(self.tilt.value))

        if self._is_duplicate_event(event_signature):
            return

        current_azimuth = float(self.schedule.loc[idx, "azimuth"])
        current_tilt = float(self.schedule.loc[idx, "tilt"])
        if np.isclose(current_azimuth, float(self.azimuth.value)) and np.isclose(current_tilt, float(self.tilt.value)):
            # No-op update; avoid repeated status spam from duplicate callbacks.
            return

        self.schedule.loc[idx, "azimuth"] = float(self.azimuth.value)
        self.schedule.loc[idx, "tilt"] = float(self.tilt.value)
        if self.render_schedule_on_each_update:
            self._render_table()
        self._status(f"Updated slot {self.slot.label} -> azimuth={self.azimuth.value:.0f}, tilt={self.tilt.value:.0f}")

    def _on_set_all(self, _=None) -> None:
        event_signature = ("all", -1, float(self.azimuth.value), float(self.tilt.value))

        if self._is_duplicate_event(event_signature):
            return

        if self.schedule["azimuth"].eq(float(self.azimuth.value)).all() and self.schedule["tilt"].eq(float(self.tilt.value)).all():
            # No-op update; avoid repeated status spam from duplicate callbacks.
            return

        self.schedule["azimuth"] = float(self.azimuth.value)
        self.schedule["tilt"] = float(self.tilt.value)
        if self.render_schedule_on_each_update:
            self._render_table()
        self._status(f"Updated all slots -> azimuth={self.azimuth.value:.0f}, tilt={self.tilt.value:.0f}")

    def _on_reset_default(self, _=None) -> None:
        self.schedule = default_tracker_schedule(
            azimuth_bounds=self.azimuth_bounds,
            tilt_bounds=self.tilt_bounds,
            include_end=self.include_end,
        )
        self.azimuth.value = float(self.schedule.loc[int(self.slot.value), "azimuth"])
        self.tilt.value = float(self.schedule.loc[int(self.slot.value), "tilt"])
        self._rendered_once = False
        self.show_table.disabled = False
        self.table_html.value = ""
        if self.render_schedule_on_each_update:
            self._render_table()
        self._status("Reset schedule to default profile.")

    def _on_show_table(self, _=None) -> None:
        event_signature = ("show_table", -1, float(self.azimuth.value), float(self.tilt.value))

        # Guard against duplicated front-end click events.
        if self._is_duplicate_event(event_signature):
            return

        # Render once; avoid repeated schedule dumps unless explicitly reset.
        if not self._rendered_once:
            self._render_table()
            self.show_table.disabled = True
            self._status("Rendered schedule table.")
        else:
            self._status("Schedule table already shown.")

    def _render_table(self) -> None:
        # Render into a single HTML widget so repeated callbacks cannot stack table outputs.
        table = self.schedule.copy().to_html(index=False, border=0)
        self.table_html.value = (
            "<div style='max-height:420px; overflow:auto; border:1px solid #ddd; padding:6px;'>"
            f"{table}</div>"
        )
        self._rendered_once = True

    def _status(self, text: str) -> None:
        if text == self._last_status_text:
            return

        self.status_html.value = f"<pre style='margin:0'>{html.escape(text)}</pre>"
        self._last_status_text = text

    def _is_duplicate_event(self, signature: Tuple[str, int, float, float], time_window_s: float = 0.35) -> bool:
        """Guard against repeated callback triggers from UI reruns/front-end glitches."""

        now = time.monotonic()
        is_dup = signature == self._last_event_signature and (now - self._last_event_time) < time_window_s
        self._last_event_signature = signature
        self._last_event_time = now
        return is_dup

    def get_schedule(self) -> pd.DataFrame:
        """Return edited schedule dataframe."""

        return self.schedule.copy()

    def display(self) -> None:
        """Render editor UI in notebook."""

        controls = widgets.VBox(
            [
                widgets.HTML("<b>Solar Tracker RBC Schedule Editor (07:00-17:50, 10 min)</b>"),
                self.slot,
                self.azimuth,
                self.tilt,
                widgets.HBox([self.set_slot, self.set_all, self.reset_default, self.show_table]),
            ]
        )
        display(controls, self.status_html, self.table_html)
        if self.auto_render_schedule_table and not self._rendered_once:
            self._render_table()

    def close(self) -> None:
        """Close widget objects to prevent orphaned callbacks/views across reruns."""

        for widget_name in [
            "slot",
            "azimuth",
            "tilt",
            "set_slot",
            "set_all",
            "reset_default",
            "show_table",
            "table_html",
            "status_html",
        ]:
            w = getattr(self, widget_name, None)
            if w is not None and hasattr(w, "close"):
                w.close()
