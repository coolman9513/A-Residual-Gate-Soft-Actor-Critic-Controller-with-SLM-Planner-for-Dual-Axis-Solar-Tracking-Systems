# -*- coding: utf-8 -*-
"""Regenerate manuscript figures 8-12 and 14 from captured rollout histories.

Companion to make_paper_figures_v2.py, which covers figures 7 and 13 from summary
metrics. These six need per-step data, produced by plots/capture_histories.py.

  fig_cumulative_v2   cumulative harvested energy over the 56 days      (Figure 8)
  fig_daily_v2        daily harvested energy                            (Figure 9)
  fig_contribution_v2 energy contributed by each architecture layer     (Figure 10)
  fig_stepsize_v2     distribution of step sizes on moving steps        (Figure 11)
  fig_hourly_v2       mean movement per step by hour of day             (Figure 12)
  fig_weather_v2      tracking efficiency by irradiance regime          (Figure 14)

The oracle-tuned SLM is included automatically once its history exists; until then
the figures are drawn with the controllers that are available, so the script is
useful before the (GPU-bound) SLM rollout completes.

Palette and marker conventions match make_paper_figures_v2.py.

Usage:
    python plots/make_paper_figures_v2b.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "plots" / "paper" / "histories"
OUT = ROOT / "plots" / "paper"

# file stem -> (label, colour, linestyle)
SERIES = [
    ("fixed",      "Fixed panel",               "#999999", (0, (1, 2))),
    ("rbc",        "RBC",                       "#56B4E9", (0, (4, 2))),
    ("nosllm",     "SAC-Auto (no planner)",     "#E69F00", (0, (6, 2))),
    ("fsm",        "Rule-threshold (FSM)",      "#009E73", "-"),
    ("oracle_slm", "RG-SAC (proposed)", "#0072B2", "-"),
]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load():
    out = []
    for stem, label, col, ls in SERIES:
        f = HIST / ("%s.csv" % stem)
        if not f.exists():
            print("  (missing, skipped: %s)" % stem)
            continue
        d = pd.read_csv(f)
        d["move"] = d["panel_azimuth_delta_deg"].abs() + d["panel_tilt_delta_deg"].abs()
        out.append((label, col, ls, d))
    return out


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("%s.%s" % (stem, ext)))
    plt.close(fig)
    print("  wrote: %s.pdf / .png" % stem)


def fig_cumulative(S):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for label, col, ls, d in S:
        day = np.arange(len(d)) / 144.0
        ax.plot(day, d["generated_energy_kwh"].cumsum(), color=col, ls=ls, lw=1.6,
                label=label, zorder=3)
    for label, col, ls, d in S:
        tot = d["generated_energy_kwh"].sum()
        ax.annotate("%.2f" % tot, xy=(56.3, tot), fontsize=7.5, color=col,
                    va="center", ha="left")
    ax.set_xlabel("Day of the 56-day evaluation")
    ax.set_ylabel("Cumulative harvested energy (kWh)")
    ax.set_xlim(0, 60.5)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_cumulative_v2")


def fig_daily(S):
    """Absolute daily energy as grouped bars, one group per evaluation day."""
    fig, ax = plt.subplots(figsize=(19.5, 4.4))
    series = [(label, col, d.groupby(["month", "day"])["generated_energy_kwh"].sum().to_numpy())
              for label, col, _, d in S]
    n_day = len(series[0][2])
    w = 0.8 / len(series)
    for k, (label, col, g) in enumerate(series):
        x = np.arange(n_day) + (k - (len(series) - 1) / 2.0) * w
        ax.bar(x, g, width=w * 0.94, color=col, edgecolor="#333333", linewidth=0.35,
               label=label, zorder=3)
    ax.set_xticks(np.arange(n_day))
    ax.set_xticklabels(["Day %d" % (i + 1) for i in range(n_day)], rotation=90, fontsize=7)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily energy (kWh)")
    ax.set_xlim(-0.8, n_day - 0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_daily_v2")


def fig_contribution(S):
    """Cumulative energy as each architectural layer is added on top of the RBC base.

    The axis is deliberately zoomed to the region the layers occupy; the fixed panel
    at 24.08 kWh would otherwise flatten every increment into invisibility. Absolute
    values are printed above each bar so the zoom cannot mislead.
    """
    tot = {label: d["generated_energy_kwh"].sum() for label, _, _, d in S}
    order = [("Fixed panel\n(no tracking)", "Fixed panel", "#999999"),
             ("RBC\n(base schedule)", "RBC", "#E69F00"),
             ("+ Gate\n(residual RL)", "SAC-Auto (no planner)", "#7BAFD4"),
             ("+ Rule-threshold\nplanner", "Rule-threshold (FSM)", "#6FC7A1"),
             ("+ Oracle-tuned\nplanner (RG-SAC)", "RG-SAC (proposed)", "#009E73")]
    stages = [(n, tot[k], c) for n, k, c in order if k in tot]
    if len(stages) < 2:
        print("  (not enough data for the contribution figure)")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    lo = min(v for _, v, _ in stages) - 0.35
    hi = max(v for _, v, _ in stages) + 0.22
    for i, (name, val, c) in enumerate(stages):
        ax.bar(i, val - lo, bottom=lo, color=c, edgecolor="#333333", linewidth=0.7,
               width=0.66, zorder=3)
        ax.text(i, val + (hi - lo) * 0.022, "%.2f" % val, ha="center", fontsize=10,
                fontweight="bold", color="#111111")
        if i:
            ax.text(i, val - (hi - lo) * 0.075, "+%.2f" % (val - stages[i - 1][1]),
                    ha="center", fontsize=9, color="#333333")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([n for n, _, _ in stages], fontsize=9)
    ax.set_ylabel("Harvested energy (kWh)")
    ax.set_ylim(lo, hi)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_contribution_v2")


def fig_stepsize(S):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bins = np.linspace(0, 12, 49)
    handles = []
    for label, col, ls, d in S:
        m = d["move"][d["move"] > 0.5]
        if len(m):
            # linestyle matters here: RBC and the proposed controller are both blues,
            # so the dash pattern is what separates them in print and for CVD readers.
            ax.hist(m, bins=bins, histtype="step", lw=1.6, color=col, ls=ls, zorder=3)
        # the fixed panel never moves; it still gets an entry so its absence is
        # stated rather than silently omitted
        handles.append(plt.Line2D([0], [0], color=col, ls=ls, lw=1.6,
                                  label="%s  (n=%d)" % (label, len(m))))
    ax.set_xlabel("Step size on moving steps, $|\\Delta_{az}| + |\\Delta_{tilt}|$ (deg)")
    ax.set_ylabel("Number of steps")
    ax.legend(handles=handles, loc="upper right", frameon=False)
    save(fig, "fig_stepsize_v2")


def fig_hourly(S):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for label, col, ls, d in S:
        g = d.groupby("hour")["move"].mean()
        ax.plot(g.index, g.to_numpy(), color=col, ls=ls, lw=1.8, marker="o",
                markersize=3.2, label=label, zorder=3)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean movement per step (deg)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_hourly_v2")


def fig_weather(S):
    """Tracking efficiency by irradiance regime, over daylight steps only."""
    regimes = [("Clear\n(DNI $\\geq$ 700)", lambda x: x >= 700),
               ("Partly cloudy\n(300-700)", lambda x: (x >= 300) & (x < 700)),
               ("Overcast\n(DNI < 300)", lambda x: x < 300)]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    w = 0.8 / max(len(S), 1)
    for k, (label, col, ls, d) in enumerate(S):
        day = d[d["ideal_generated_energy_kwh"] > 1e-9]
        vals = []
        for _, test in regimes:
            sel = day[test(day["dni"])]
            vals.append(sel["generated_energy_kwh"].sum() / sel["ideal_generated_energy_kwh"].sum()
                        if len(sel) else np.nan)
        x = np.arange(len(regimes)) + (k - (len(S) - 1) / 2.0) * w
        ax.bar(x, vals, width=w * 0.92, color=col, edgecolor="white", linewidth=0.8,
               label=label, zorder=3)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.012, "%.3f" % v, ha="center", fontsize=6.4,
                        rotation=90, color="#222222")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([r[0] for r in regimes])
    ax.set_ylabel("Tracking efficiency")
    ax.set_ylim(0, 1.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=7.5)
    save(fig, "fig_weather_v2")


if __name__ == "__main__":
    S = load()
    print("controllers: %d" % len(S))
    fig_cumulative(S)
    fig_daily(S)
    fig_contribution(S)
    fig_stepsize(S)
    fig_hourly(S)
    fig_weather(S)
    print("done -> %s" % OUT)
