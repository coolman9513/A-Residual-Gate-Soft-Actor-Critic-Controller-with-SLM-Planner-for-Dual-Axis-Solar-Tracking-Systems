# -*- coding: utf-8 -*-
"""Wear-cost sensitivity sweep for the hindsight-optimal regime oracle.

PROJECT_HANDOFF.md §11, Phase 1. Answers: is the oracle's Pareto win over the
DNI-threshold FSM (more energy AND fewer motor activations) an artifact of the
break-even wear coefficient, or does it hold across the plausible range?

c_act is the kWh-equivalent charged per motor activation (start/stop cycle).
3.79e-4 is the break-even implied by the paper's own multi-seed results: the
value at which the FSM's 997 saved activations exactly pay for the 0.378 kWh
of energy it gives up versus the no-SLM ablation.

Every run uses the identical harness and a fully open gate (g=1), so the FSM
baseline below is directly comparable.

Detached run:
    C:/Users/mrcoo/anaconda3/envs/sllm_rl/python.exe run_oracle_sweep.py
Monitor:  Get-Content oracle_sweep.log -Wait      Done marker: oracle_sweep_DONE.txt
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "finetune"))
import oracle_goals as og  # noqa: E402

LOG = ROOT / "oracle_sweep.log"
DONE = ROOT / "oracle_sweep_DONE.txt"
OUT = og.OUT

# c_act (kWh-equivalent per activation); c_deg held at 0 — see eval_utils.movement_profile_stats
SWEEP = [
    ("c_act_0",       0.0),
    ("c_act_1e-4",    1.0e-4),
    ("breakeven",     og.C_ACT_BREAKEVEN),   # 3.79e-4, already computed
    ("c_act_1e-3",    1.0e-3),
    ("c_act_3e-3",    3.0e-3),
]


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fsm_baseline():
    """Same harness, gate fully open, regimes from the rule-based router."""
    env = og.build_env()
    env.reset()
    env.authority_mode = "goal"
    done = False
    while not done:
        _, _, done, _ = env.step(env.oracle_action())
    h = env.history
    num = lambda c: pd.to_numeric(h.get(c), errors="coerce").fillna(0.0).to_numpy(float)
    mv = np.abs(num("panel_azimuth_delta_deg")) + np.abs(num("panel_tilt_delta_deg"))
    return dict(energy_kwh=round(float(num("generated_energy_kwh").sum()), 3),
                activations=int((mv > og.MOVE_EPS).sum()), degrees=round(float(mv.sum()), 1))


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== oracle wear-cost sweep started ===========")
    blob = json.loads(OUT.read_text()) if OUT.exists() else {}
    blob.setdefault("runs", {})

    if "fsm_baseline" not in blob:
        blob["fsm_baseline"] = fsm_baseline()
        OUT.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    log("FSM baseline (g=1): %s" % blob["fsm_baseline"])

    for tag, c_act in SWEEP:
        if tag in blob["runs"]:
            log("skip %s (already present)" % tag)
            continue
        log("running %s  c_act=%.3e" % (tag, c_act))
        og.run_and_save(c_act=c_act, c_deg=0.0, seed=2021, tag=tag)
        blob = json.loads(OUT.read_text())

    log("\n===== SWEEP SUMMARY (gate g=1, identical harness) =====")
    b = blob["fsm_baseline"]
    log("  %-14s %10s %12s %10s" % ("run", "energy", "activations", "degrees"))
    log("  %-14s %10.3f %12d %10.0f   <- FSM rule router" %
        ("fsm", b["energy_kwh"], b["activations"], b["degrees"]))
    for tag, c_act in SWEEP:
        r = blob["runs"].get(tag)
        if not r:
            continue
        s = r["summary"]
        rc = s["regime_counts"]
        log("  %-14s %10.3f %12d %10.0f   c_act=%.1e  hold/cloud/clear=%d/%d/%d" %
            (tag, s["energy_kwh"], s["activations"], s["degrees"], c_act,
             rc["hold"], rc["cloud"], rc["clear"]))
    log("=========== SWEEP DONE ===========")
    DONE.write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e))
        log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8")
        raise
