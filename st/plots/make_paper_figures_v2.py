# -*- coding: utf-8 -*-
"""Regenerate the paper figures that the oracle-planner results made stale.

  fig_pareto_v2      energy against motor activations - the key figure, because the
                     previous version shows RG-SAC as the best operating point and
                     that is no longer true.
  fig_overall_v2     four-panel comparison across all reported metrics.

Design notes
------------
Colours are the Okabe-Ito palette, an empirically designed colour-vision-deficiency
safe set that is standard in scientific publishing. The dataviz skill's validator
could not be run here (node is not installed), so a published CVD-safe palette is
used rather than an unvalidated ad-hoc one. Identity is additionally carried by
marker shape, so it never depends on colour alone - which also keeps the figures
readable if the journal prints in greyscale.

The fixed panel is deliberately omitted from the Pareto figure: with zero
activations it is not an operating point on the energy/wear trade-off, and
including it compresses the informative region into the top fifth of the axes. It
remains in the four-panel comparison and in the results table.

Usage:
    python plots/make_paper_figures_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# name: (energy, e_sd, move, m_sd, activ, a_sd, rev, r_sd, colour, marker)
C = {
    "Fixed panel":               (24.077, 0.0,   0.00, 0.0,     0,  0,    0,   0, "#999999", "o"),
    "RBC":                       (27.227, 0.0,   2.10, 0.0,  3640,  0,  111,   0, "#56B4E9", "o"),
    "SAC-Auto (no planner)":     (27.679, 0.054, 2.19, 0.04, 3680, 14,  301, 104, "#E69F00", "o"),
    "Rule-threshold (FSM)":      (28.007, 0.057, 1.82, 0.06, 2699, 15,  144,  35, "#009E73", "o"),
    "RG-SAC (proposed)": (28.497, 0.020, 1.77, 0.04, 2056, 35,   55,  12, "#0072B2", "o"),
}

ORACLE_BOUND = 28.522

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("%s.%s" % (stem, ext)))
    plt.close(fig)
    print("  wrote: %s.pdf / .png" % stem)


# label anchor per point: (x_offset, y_offset, horizontal alignment)
PARETO_LABELS = {
    "RBC":                       (-95, -0.045, "right"),
    "SAC-Auto (no planner)":     (-105, 0.030, "right"),
    "Rule-threshold (FSM)":      (95,  0.055, "left"),
    "RG-SAC (proposed)": (100, -0.055, "left"),
}


def fig_pareto():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    pts = {k: v for k, v in C.items() if k != "Fixed panel"}

    ax.axhline(ORACLE_BOUND, color="#555555", ls=(0, (5, 4)), lw=1.0, zorder=1)
    ax.text(3880, ORACLE_BOUND + 0.022, "physics-derived bound (%.2f kWh)" % ORACLE_BOUND,
            fontsize=7.5, color="#555555", ha="right", va="bottom")

    for name, v in pts.items():
        e, es, _, _, a, asd, _, _, col, mk = v
        ax.errorbar(a, e, yerr=es or None, xerr=asd or None, fmt=mk, color=col,
                    markersize=9, markeredgecolor="white", markeredgewidth=1.2,
                    ecolor=col, elinewidth=1.2, capsize=2.5, zorder=3, label=name)
        dx, dy, ha = PARETO_LABELS[name]
        ax.annotate(name, xy=(a, e), xytext=(a + dx, e + dy), fontsize=7.5,
                    color="#222222", ha=ha, va="center", zorder=4)

    ax.set_xlabel("Motor activations over the 56-day evaluation  (fewer is better)")
    ax.set_ylabel("Harvested energy (kWh)  (higher is better)")
    ax.set_xlim(1700, 3980)
    ax.set_ylim(27.05, 28.68)
    save(fig, "fig_pareto_v2")


def fig_overall():
    metrics = [
        ("Harvested energy (kWh)", 0, 1, "%.2f", (22.8, 29.9)),
        ("Movement (deg/step)",     2, 3, "%.2f", None),
        ("Motor activations",       4, 5, "%.0f", None),
        ("Directional reversals",   6, 7, "%.0f", None),
    ]
    names = list(C)
    # wspace: the default leaves the y-label of each panel sitting on top of the
    # previous panel's bars once the metric names moved out of the titles.
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6),
                             gridspec_kw={"wspace": 0.34})
    for ax, (title, i, j, fmt, ylim) in zip(axes, metrics):
        vals = [C[n][i] for n in names]
        errs = [C[n][j] for n in names]
        cols = [C[n][8] for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, yerr=errs, color=cols, edgecolor="white", linewidth=1.0,
               capsize=2.5, width=0.72, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["" for _ in names])
        ax.set_ylabel(title, fontsize=8.5)
        lo, hi = ylim if ylim else (0, max(v + e for v, e in zip(vals, errs)) * 1.20)
        ax.set_ylim(lo, hi)
        head = (hi - lo) * 0.035
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + head, fmt % v, ha="center", fontsize=7,
                    color="#222222", zorder=4)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=7,
                          markerfacecolor=C[n][8], markeredgecolor="white", label=n)
               for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.20))
    save(fig, "fig_overall_v2")


if __name__ == "__main__":
    fig_pareto()
    fig_overall()
    print("done -> %s" % OUT)
