# -*- coding: utf-8 -*-
"""Hindsight-optimal ("oracle") hourly regime decisions for the residual-gate tracker.

Phase 1 of the "authentic autonomous SLM" experiment (PROJECT_HANDOFF.md §11).

The current fine-tuning targets are DNI-threshold rules, which is *why* the SLM
reproduces an FSM (§7). This module replaces them with decisions derived from the
actual simulator using known future irradiance.

What is actually being decided
------------------------------
`motion_budget_deg` does not move the panel. Via llm/regime_router.route_goal it
selects a regime -> Authority(azimuth, tilt, hour_budget) that bounds only the
RESIDUAL correction; at zero authority the panel still tracks on the RBC base
schedule (the guaranteed floor). The regime additionally switches the base pose
(flat-hold / anticipatory pre-position). So the oracle's real choice per hour is
which of the three authorities to spend.

Objective
---------
For each hour we roll the REAL environment forward six 10-min steps under each
regime with the gate fully open, and score:

    net = generated_energy_kwh - c_act * activations - c_deg * degrees_travelled

Wear is charged per activation (start/stop cycle) rather than per degree, matching
the project's own stated wear model in eval_utils.movement_profile_stats: total
angular distance is largely fixed by the sun's path, so start/stop cycles and jerk
are what wear the hardware. c_deg is kept as a parameter for sensitivity analysis.

Selection is greedy forward: the winning regime is committed before moving to the
next hour, so each decision sees the pose its predecessors actually produced.
"""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT), str(ROOT / "meta_sac")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm.regime_router import DEFAULT_AUTHORITY_TABLE          # noqa: E402
from train_utils import build_schema, make_residual_env        # noqa: E402

CSV = "2020_4months_2weeks.csv"
STEPS_PER_HOUR = 6
MOVE_EPS = 0.5                      # same activation threshold as movement_profile_stats
REGIMES = ("hold", "cloud", "clear")

# Break-even wear costs implied by the paper's own multi-seed results
# (no-SLM -> FSM: 0.378 kWh of energy given up for 997 fewer activations
#  / 2871 fewer degrees over the 56-day episode).
C_ACT_BREAKEVEN = 3.79e-4           # kWh-equivalent per activation
C_DEG_BREAKEVEN = 1.32e-4           # kWh-equivalent per degree

FINAL_KW = dict(action_mode="gate", authority_mode="fixed", hold_tilt_deg=10.0,
                cloud_anticipate_lookahead=12, move_deadband_deg=2.0)


GROW_LISTS = [
    ("wrapper", "wind_metric_log"),
    ("env", "_history_rows"),
    ("env", "_SolarTrackerEnv__rewards"),
]
SCALARS = [
    ("wrapper", "hour_residual_used_deg"),
    ("wrapper", "hour_motion_so_far_deg"),
    ("wrapper", "previous_goal_bucket"),
    ("wrapper", "last_history"),
    ("wrapper", "last_goal"),
    ("wrapper", "_active_authority"),
    ("wrapper", "fixed_authority"),
    ("env", "_Environment__time_step"),
    ("env", "_panel_azimuth_deg"),
    ("env", "_panel_tilt_deg"),
    ("env", "_previous_panel_azimuth_deg"),
    ("env", "_previous_panel_tilt_deg"),
    ("env", "_previous_generated_energy_kwh"),
]


def _targets(env):
    return {"wrapper": env, "env": env.env}


def snapshot(env):
    """O(1) capture of every field that mutates during a step (see statediff probe)."""
    t = _targets(env)
    snap = {"lens": {}, "vals": {}}
    for owner, name in GROW_LISTS:
        snap["lens"][(owner, name)] = len(getattr(t[owner], name))
    for owner, name in SCALARS:
        snap["vals"][(owner, name)] = getattr(t[owner], name, None)
    rf = getattr(env.env, "reward_function", None)
    if rf is not None:
        snap["rw"] = (list(getattr(rf, "_previous_azimuth_deltas", [])),
                      list(getattr(rf, "_previous_tilt_deltas", [])))
    return snap


def restore(env, snap):
    """Rewind the environment to a snapshot taken by snapshot()."""
    t = _targets(env)
    for (owner, name), n in snap["lens"].items():
        lst = getattr(t[owner], name)
        del lst[n:]
    for (owner, name), v in snap["vals"].items():
        setattr(t[owner], name, v)
    rf = getattr(env.env, "reward_function", None)
    if rf is not None and "rw" in snap:
        rf._previous_azimuth_deltas = list(snap["rw"][0])
        rf._previous_tilt_deltas = list(snap["rw"][1])


def build_env(seed: int = 2021, csv: str = None, start: int = 0, end: int = 8063):
    """Residual-gate env over [start, end] of `csv` (defaults to the 56-day study set)."""
    base = json.loads((ROOT / "data" / "schema.json").read_text())
    base["root_directory"] = str(ROOT / "data")
    schema = build_schema(base, csv or CSV, start, end)
    return make_residual_env(schema, use_llm=False, seed=seed, **FINAL_KW)


def _run_hour(env, regime, n_steps):
    """Step one hour under `regime`; return (energy_kwh, activations, degrees, done)."""
    env.fixed_authority = DEFAULT_AUTHORITY_TABLE[regime]
    start = len(env.history)
    done = False
    for _ in range(n_steps):
        if done:
            break
        _, _, done, _ = env.step(env.oracle_action())
    h = env.history.iloc[start:]
    if h.empty:
        return 0.0, 0.0, 0.0, done
    num = lambda c: pd.to_numeric(h.get(c), errors="coerce").fillna(0.0).to_numpy(float)
    e = float(num("generated_energy_kwh").sum())
    move = np.abs(num("panel_azimuth_delta_deg")) + np.abs(num("panel_tilt_delta_deg"))
    return e, float((move > MOVE_EPS).sum()), float(move.sum()), done


def generate(c_act=C_ACT_BREAKEVEN, c_deg=0.0, seed=2021, verbose=True, env=None):
    """Greedy per-hour hindsight-optimal regime sequence. Returns list of hour records."""
    env = env if env is not None else build_env(seed=seed)
    env.reset()

    records, done = [], False
    while not done:
        telemetry = copy.deepcopy(dict(env._telemetry()))
        snap = snapshot(env)

        scored, components = {}, {}
        for regime in REGIMES:
            e, act, deg, d = _run_hour(env, regime, STEPS_PER_HOUR)
            scored[regime] = dict(energy=e, activations=act, degrees=deg,
                                  net=e - c_act * act - c_deg * deg, done=d)
            # Raw components are kept so the optimal regime can be re-derived
            # offline for ANY wear coefficient without re-running the oracle.
            components[regime] = dict(e=round(e, 8), act=int(act), deg=round(deg, 4))
            restore(env, snap)

        best = max(REGIMES, key=lambda r: scored[r]["net"])
        e, act, deg, done = _run_hour(env, best, STEPS_PER_HOUR)
        records.append(dict(
            dataset_time_step=int(telemetry.get("dataset_time_step", len(records) * STEPS_PER_HOUR)),
            hour=int(telemetry.get("hour", 0)), regime=best,
            energy_kwh=e, activations=int(act), degrees=deg,
            scores={r: round(scored[r]["net"], 6) for r in REGIMES},
            components=components,
            telemetry=telemetry,
        ))
        if verbose and len(records) % 200 == 0:
            print("  %d hours ..." % len(records), flush=True)
    return records


def summarize(records, label=""):
    n = len(records)
    counts = {r: sum(1 for x in records if x["regime"] == r) for r in REGIMES}
    tot_e = sum(x["energy_kwh"] for x in records)
    tot_a = sum(x["activations"] for x in records)
    tot_d = sum(x["degrees"] for x in records)
    print("\n=== oracle %s ===" % label)
    print("  hours %d | energy %.3f kWh | activations %d | degrees %.0f" % (n, tot_e, tot_a, tot_d))
    for r in REGIMES:
        print("    %-6s %5d hours (%.1f%%)" % (r, counts[r], 100.0 * counts[r] / max(n, 1)))
    return dict(hours=n, energy_kwh=round(tot_e, 3), activations=int(tot_a),
                degrees=round(tot_d, 1), regime_counts=counts)


OUT = ROOT / "models" / "oracle_regimes.json"


def run_and_save(c_act=C_ACT_BREAKEVEN, c_deg=0.0, seed=2021, tag="breakeven",
                 env=None, out=None):
    import time
    t0 = time.time()
    print("oracle run [%s]: c_act=%.3e c_deg=%.3e seed=%d" % (tag, c_act, c_deg, seed), flush=True)
    recs = generate(c_act=c_act, c_deg=c_deg, seed=seed, env=env)
    summ = summarize(recs, tag)
    summ.update(c_act=c_act, c_deg=c_deg, seed=seed, seconds=round(time.time() - t0, 1))
    target = out or OUT
    blob = {}
    if target.exists():
        blob = json.loads(target.read_text())
    blob.setdefault("runs", {})[tag] = {
        "summary": summ,
        "hours": [{k: r[k] for k in ("dataset_time_step", "hour", "regime",
                                     "energy_kwh", "activations", "degrees",
                                     "scores", "components")}
                  for r in recs],
    }
    target.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print("saved -> %s  (%.0fs)" % (target, summ["seconds"]), flush=True)
    return recs, summ


if __name__ == "__main__":
    run_and_save()
