"""Visualisation helpers for SLM-SAC-Auto experiments.

Provides clean, reusable plotting functions so experiment2.ipynb
stays lean — one call per figure, no inline matplotlib boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── internal helpers ───────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {path}")


# ── training curves ────────────────────────────────────────────────────────────

def make_training_energy_curve(training_df, label: str) -> pd.DataFrame:
    """Return a tidy DataFrame with columns [episode, energy_kwh, controller]."""
    df = pd.DataFrame(training_df).copy()
    if df.empty:
        return pd.DataFrame(columns=["episode", "energy_kwh", "controller"])
    if "episode" not in df.columns:
        df["episode"] = (
            df["continued_episode"] if "continued_episode" in df.columns
            else np.arange(1, len(df) + 1)
        )
    if "total_harvested_energy_kwh" not in df.columns:
        return pd.DataFrame(columns=["episode", "energy_kwh", "controller"])
    return pd.DataFrame({
        "episode":    pd.to_numeric(df["episode"], errors="coerce"),
        "energy_kwh": pd.to_numeric(df["total_harvested_energy_kwh"], errors="coerce"),
        "controller": label,
    }).dropna()


def plot_training_curves(
    training_df,
    label: str = "SAC-Auto",
    rbc_energy: Optional[float] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """4-panel training curve: energy, movement, jerk, log-alpha.

    Parameters
    ----------
    training_df : DataFrame or list[dict] from training history.
    label       : Controller name shown in legend and title.
    rbc_energy  : If provided, draw a dashed RBC baseline on the energy panel.
    save_path   : Optional file path to save the figure.
    """
    df = pd.DataFrame(training_df).copy()
    if df.empty:
        raise ValueError("training_df is empty")

    ep_col = "episode" if "episode" in df.columns else (
        "continued_episode" if "continued_episode" in df.columns else None
    )
    if ep_col is None:
        df["episode"] = np.arange(1, len(df) + 1)
        ep_col = "episode"

    x = pd.to_numeric(df[ep_col], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{label} — Training Curves", fontsize=13)

    panels = [
        ("total_harvested_energy_kwh", "Total Energy (kWh)", axes[0, 0]),
        ("movement_magnitude_deg",     "Movement (°)",       axes[0, 1]),
        ("tracking_smoothness_jerk_deg2", "Jerk (°²)",       axes[1, 0]),
        ("log_alpha",                  "log α",              axes[1, 1]),
    ]

    for col, ylabel, ax in panels:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, linewidth=1.2, alpha=0.8, label=label)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Episode", fontsize=9)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if col == "total_harvested_energy_kwh" and rbc_energy is not None:
                ax.axhline(rbc_energy, color="red", linestyle="--", linewidth=1.0, label="RBC")
                ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── comparison plots ───────────────────────────────────────────────────────────

def plot_energy_comparison(
    histories: Sequence,
    labels: Sequence[str],
    seconds_per_time_step: float = 600.0,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Overlay cumulative energy curves for multiple controllers.

    histories : sequence of DataFrames (one per controller).
    labels    : controller names in same order as histories.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for history, label in zip(histories, labels):
        h = pd.DataFrame(history)
        if h.empty or "generated_energy_kwh" not in h.columns:
            continue
        cumulative = h["generated_energy_kwh"].cumsum()
        ax.plot(cumulative.values, label=label, linewidth=1.4)

    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Cumulative Energy (kWh)", fontsize=10)
    ax.set_title("Cumulative Energy — Controller Comparison", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_daily_energy(
    histories: Sequence,
    labels: Sequence[str],
    seconds_per_time_step: float = 600.0,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Grouped bar chart of daily energy per controller."""
    steps_per_day = int(round(24 * 3600 / seconds_per_time_step))
    all_series = []
    for history, label in zip(histories, labels):
        h = pd.DataFrame(history)
        if h.empty or "generated_energy_kwh" not in h.columns:
            continue
        if "time_step" in h.columns:
            first_step = float(h["time_step"].min())
            day_idx = np.floor((h["time_step"].astype(float) - first_step) / steps_per_day).astype(int)
        else:
            day_idx = np.arange(len(h)) // steps_per_day
        series = h.groupby(day_idx)["generated_energy_kwh"].sum()
        all_series.append((series, label))

    if not all_series:
        raise ValueError("No valid histories to plot")

    n_days = max(len(s) for s, _ in all_series)
    n_controllers = len(all_series)
    x = np.arange(n_days)
    width = 0.8 / n_controllers

    fig, ax = plt.subplots(figsize=(max(10, n_days * 0.6), 5))
    for i, (series, label) in enumerate(all_series):
        vals = [series.get(d, 0.0) for d in range(n_days)]
        ax.bar(x + i * width - width * (n_controllers - 1) / 2, vals, width, label=label)

    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel("Energy (kWh)", fontsize=10)
    ax.set_title("Daily Energy by Controller", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Day {d + 1}" for d in range(n_days)], rotation=45, fontsize=7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_log_alpha_comparison(
    training_dfs: Sequence,
    labels: Sequence[str],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot log_alpha per episode for multiple runs (shows alpha collapse vs stability)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for df_raw, label in zip(training_dfs, labels):
        df = pd.DataFrame(df_raw)
        if "log_alpha" not in df.columns:
            continue
        ep_col = next((c for c in ("episode", "continued_episode") if c in df.columns), None)
        x = pd.to_numeric(df[ep_col], errors="coerce") if ep_col else np.arange(len(df))
        ax.plot(x, pd.to_numeric(df["log_alpha"], errors="coerce"), label=label, linewidth=1.2)

    ax.axhline(-2.0, color="grey", linestyle=":", linewidth=0.9, label="log_alpha_min floor")
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("log α", fontsize=10)
    ax.set_title("Entropy Temperature per Episode", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_energy_movement_pareto(
    points: dict,
    save_path: Optional[str | Path] = None,
    title: str = "Energy vs. Movement — Pareto View",
) -> plt.Figure:
    """Scatter each controller in the movement/energy plane.

    points: {label: (movement_per_step_deg, energy_kwh)} — the ideal region is
    the upper-left (high energy, low movement). Pass every controller you want
    compared; the residual controllers should sit above-left of RBC.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 3)))
    for (label, (movement, energy)), color in zip(points.items(), colors):
        ax.scatter(movement, energy, s=130, color=color, zorder=3, label=label)
        ax.annotate(
            label,
            xy=(movement, energy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Movement per step (deg/step)  →  more wear", fontsize=10)
    ax.set_ylabel("Harvested energy (kWh)  →  better", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig

