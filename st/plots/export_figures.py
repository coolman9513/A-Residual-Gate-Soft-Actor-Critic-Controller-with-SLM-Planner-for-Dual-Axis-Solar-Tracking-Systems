# -*- coding: utf-8 -*-
"""Copy the regenerated figures into revision/figures_v2/ under the names main.tex
already uses, so they can be uploaded to Overleaf without renaming anything.

plots/paper/ holds both the old and the new figures side by side, which makes the
new ones awkward to pick out. This produces a clean folder containing exactly the
eight figures that changed, named "figure 7.pdf" ... "figure 14.pdf" to match the
\\includegraphics calls in the manuscript.

Usage:
    python plots/export_figures.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "plots" / "paper"
DST = ROOT / "revision" / "figures_v2"

# manuscript figure number -> generated stem
MAP = {
    7:  ("fig_overall_v2",      "Overall comparison across all reported metrics"),
    8:  ("fig_cumulative_v2",   "Cumulative harvested energy over the 56 days"),
    9:  ("fig_daily_v2",        "Daily harvested energy by controller"),
    10: ("fig_contribution_v2", "Cumulative energy as each layer is added"),
    11: ("fig_stepsize_v2",     "Step-size distribution on moving steps"),
    12: ("fig_hourly_v2",       "Mean movement per step by hour of day"),
    13: ("fig_pareto_v2",       "Energy against motor activations"),
    14: ("fig_weather_v2",      "Tracking efficiency by irradiance regime"),
}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for num, (stem, desc) in sorted(MAP.items()):
        for ext in ("pdf", "png"):
            src = SRC / ("%s.%s" % (stem, ext))
            if not src.exists():
                missing.append(src.name)
                continue
            dst = DST / ("figure %d.%s" % (num, ext))
            shutil.copy2(src, dst)
        rows.append((num, stem, desc))
        print("  figure %-2d  <- %-22s  %s" % (num, stem, desc))

    readme = ["# Regenerated figures (2026-09 revision round)", "",
              "Upload the .pdf files to Overleaf. They already carry the names used by",
              "main.tex, so no \\includegraphics edits are needed. The .png copies are",
              "for quick viewing only.", "",
              "Controllers shown: Fixed panel, RBC, SAC-Auto (no SLM), FSM + SAC,",
              "RG-SAC (proposed).", "",
              "| manuscript | source stem | content |",
              "|---|---|---|"]
    for num, stem, desc in rows:
        readme.append("| figure %d | %s | %s |" % (num, stem, desc))
    readme += ["", "Figures 1-6 are unchanged and are not included here.", "",
               "Regenerate with:", "",
               "    python plots/capture_histories.py --with-slm",
               "    python plots/make_paper_figures_v2.py     # 7, 13",
               "    python plots/make_paper_figures_v2b.py    # 8-12, 14",
               "    python plots/export_figures.py            # copy here", ""]
    (DST / "README.md").write_text("\n".join(readme), encoding="utf-8")

    if missing:
        print("\n  MISSING: %s" % ", ".join(missing))
    print("\ndone -> %s" % DST)


if __name__ == "__main__":
    main()
