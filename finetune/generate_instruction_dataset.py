# -*- coding: utf-8 -*-
"""Instruction-conditioned planner dataset (Version B, handoff section 11).

Version A failed because every training example came from ONE wear setting, so the
directive carried no information and the model learned to ignore it. Here the SAME
telemetry is paired with THREE directives and THREE different correct answers, so
the instruction is the only thing that distinguishes the examples. The model has to
read it to be right.

How the three label sets are obtained without three oracle runs
--------------------------------------------------------------
The oracle records, per hour and per regime, the raw (energy, activations, degrees)
of rolling that regime forward. A wear coefficient only re-weights those numbers:

    net(regime | c_act) = energy - c_act * activations

So the optimal regime for ANY c_act is recoverable offline from a single pass. This
also means all three label sets sit on the SAME committed trajectory, so telemetry
is byte-identical across the three directives. That is essential: if telemetry
differed, the model could tell the arms apart without reading the instruction.

Evaluation days (months 1/4/7/10, days 9-22) are held out, as before, because the
oracle targets are hindsight-derived.

Oversampling is deliberately milder than Version A (clear x3 rather than x10).
Version A's aggressive oversampling skewed the class prior far from the true
77/21/1 distribution and left the SLM below the majority-class floor.

Usage:
    python finetune/generate_instruction_dataset.py
Outputs:
    finetune/data_instruct/{train,val}.jsonl, stats.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT), str(ROOT / "meta_sac"), str(ROOT / "finetune")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oracle_goals as og                                                # noqa: E402
from llm.goal_prompts import (AUTHORITY_BUDGETS, WEAR_DIRECTIVES,        # noqa: E402
                              build_goal_prompt_v2)

ORACLE_JSON = ROOT / "models" / "oracle_regimes_2020full_components.json"
TAG = "fullyear_components"
CSV = "2020.csv"
N_ROWS = 52560
OUT_DIR = ROOT / "finetune" / "data_instruct"

EVAL_MONTHS = {1, 4, 7, 10}
EVAL_DAYS = set(range(9, 23))
OVERSAMPLE = {"clear": 3, "cloud": 1, "hold": 1}
REGIMES = ("hold", "cloud", "clear")


def best_regime(components, c_act):
    """Optimal regime for a given per-activation wear cost."""
    return max(REGIMES, key=lambda r: components[r]["e"] - c_act * components[r]["act"])


def _calendar_index():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df["Month"].to_numpy(int), df["Day"].to_numpy(int)


def build_goal(tel, regime):
    hold = regime == "hold"
    cur = tel.get("current_target", {}) or {}
    fut = tel.get("future_target", {}) or cur
    if hold:
        t_az = float(tel.get("panel_azimuth_deg", cur.get("azimuth", 180.0)))
        t_tilt = float(tel.get("panel_tilt_deg", cur.get("tilt", 10.0)))
        p_az, p_tilt = t_az, t_tilt
    else:
        t_az, t_tilt = float(cur.get("azimuth", 180.0)), float(cur.get("tilt", 10.0))
        p_az, p_tilt = float(fut.get("azimuth", t_az)), float(fut.get("tilt", t_tilt))
    return {
        "target_tilt": round(float(np.clip(t_tilt, 10.0, 85.0)), 1),
        "target_az": round(float(np.clip(t_az, 70.0, 290.0)), 1),
        "motion_budget_deg": AUTHORITY_BUDGETS[regime],
        "pre_position_tilt": round(float(np.clip(p_tilt, 10.0, 85.0)), 1),
        "pre_position_az": round(float(np.clip(p_az, 70.0, 290.0)), 1),
        "hold": bool(hold),
        "horizon_min": 60.0,
        "confidence": 0.9,
    }


def _prompt_telemetry(tel):
    drop = {"dataset_time_step", "constraints"}
    return {k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in tel.items() if k not in drop}


def main():
    random.seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(exist_ok=True)

    if not ORACLE_JSON.exists():
        raise FileNotFoundError("run stage 1 first: %s" % ORACLE_JSON)
    records = json.loads(ORACLE_JSON.read_text())["runs"][TAG]["hours"]
    print("oracle hours: %d" % len(records))
    if "components" not in records[0]:
        raise RuntimeError("oracle JSON lacks per-regime components; re-run stage 1")

    months, days = _calendar_index()

    # Replay the committed path to recover hour-start telemetry (identical for all
    # three directives, which is the whole point).
    env = og.build_env(seed=2021, csv=CSV, start=0, end=N_ROWS - 1)
    env.reset()
    pairs = []
    for i, rec in enumerate(records):
        tel = dict(env._telemetry())
        pairs.append((tel, rec))
        og._run_hour(env, rec["regime"], og.STEPS_PER_HOUR)
        if (i + 1) % 2000 == 0:
            print("  replayed %d/%d" % (i + 1, len(records)), flush=True)

    kept, n_night, n_eval = [], 0, 0
    for tel, rec in pairs:
        ts = min(max(int(rec.get("dataset_time_step", 0)), 0), len(months) - 1)
        if int(months[ts]) in EVAL_MONTHS and int(days[ts]) in EVAL_DAYS:
            n_eval += 1
            continue
        if not bool(tel.get("tracking_window_active", False)):
            n_night += 1
            continue
        kept.append((tel, rec))
    print("dropped %d night, %d evaluation-day hours -> %d kept" % (n_night, n_eval, len(kept)))

    # how often the directives actually disagree - the learnable signal
    lab = {k: [best_regime(r["components"], c) for _, r in kept]
           for k, (c, _) in WEAR_DIRECTIVES.items()}
    diff = sum(1 for i in range(len(kept))
               if lab["max_energy"][i] != lab["min_movement"][i])
    print("max_energy vs min_movement disagree on %d/%d hours (%.1f%%)"
          % (diff, len(kept), 100.0 * diff / max(len(kept), 1)))
    for k in WEAR_DIRECTIVES:
        c = {r: lab[k].count(r) for r in REGIMES}
        print("  %-13s %s" % (k, c))

    random.shuffle(kept)
    n_val = int(len(kept) * 0.1)
    splits = {"val": kept[:n_val], "train": kept[n_val:]}

    written = {}
    for name, hours in splits.items():
        rows = []
        for tel, rec in hours:
            tstr = json.dumps(_prompt_telemetry(tel), separators=(", ", ": "))
            for key, (c_act, directive) in WEAR_DIRECTIVES.items():
                regime = best_regime(rec["components"], c_act)
                msgs = build_goal_prompt_v2(tstr, directive)
                msgs = list(msgs) + [{"role": "assistant",
                                      "content": json.dumps(build_goal(tel, regime))}]
                n = OVERSAMPLE.get(regime, 1) if name == "train" else 1
                rows.extend([{"messages": msgs}] * n)
        random.shuffle(rows)
        written[name] = len(rows)
        with open(OUT_DIR / ("%s.jsonl" % name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + chr(10))
        print("wrote %6d samples -> %s" % (len(rows), OUT_DIR / ("%s.jsonl" % name)))

    stats = {"source": str(ORACLE_JSON), "tag": TAG, "hours_kept": len(kept),
             "dropped_night": n_night, "dropped_eval_days": n_eval,
             "directives": {k: v[0] for k, v in WEAR_DIRECTIVES.items()},
             "disagreement_max_vs_min": diff, "oversample": OVERSAMPLE,
             "train": written["train"], "val": written["val"]}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("stats -> %s" % (OUT_DIR / "stats.json"))


if __name__ == "__main__":
    main()
