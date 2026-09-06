# -*- coding: utf-8 -*-
"""Build the oracle-target SLM fine-tuning dataset (PROJECT_HANDOFF.md §11, Phase 1).

Replaces finetune/generate_dataset.py, whose targets are DNI-threshold rules and
whose prompt spells out a 7-step decision ladder. Training on that is precisely
why the fine-tuned SLM reproduces an FSM (handoff §7). Here the targets come from
run_oracle_fullyear.py: per-hour regime choices made by scoring the REAL simulator
with known future irradiance.

Held-out evaluation
-------------------
Oracle targets are hindsight-derived, so any day used for training leaks its
future into the model. Every hour falling on a study-set day (months 1/4/7/10,
days 9-22) is therefore DROPPED here, leaving the paper's 56-day evaluation set
genuinely unseen.

Granularity
-----------
One sample per PLANNING HOUR, not per 10-min row. LLMGoalGuidance caches by hour
bucket, so the model is actually invoked once per hour on hour-start telemetry;
training at that same granularity keeps the training and serving distributions
aligned. (The old generator emitted one sample per row, giving 6 near-duplicate
copies of each decision.)

Usage:
    python finetune/generate_oracle_dataset.py
Outputs:
    finetune/data_oracle/train.jsonl
    finetune/data_oracle/val.jsonl
    finetune/data_oracle/stats.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT.parent), str(ROOT), str(ROOT / "meta_sac"), str(ROOT / "finetune")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oracle_goals as og                                    # noqa: E402
from llm.goal_prompts import AUTHORITY_BUDGETS, build_goal_prompt_v2   # noqa: E402

ORACLE_JSON = ROOT / "models" / "oracle_regimes_2020full.json"
TAG = "fullyear_breakeven"
CSV = "2020.csv"
N_ROWS = 52560
OUT_DIR = ROOT / "finetune" / "data_oracle"

EVAL_MONTHS = {1, 4, 7, 10}
EVAL_DAYS = set(range(9, 23))

# The oracle picks 'clear' in only ~2% of daylight hours; without rebalancing the
# model would collapse onto the majority class.
OVERSAMPLE = {"clear": 10, "cloud": 3, "hold": 1}


def _is_eval_day(month, day):
    return int(month) in EVAL_MONTHS and int(day) in EVAL_DAYS


def replay_for_telemetry(records, verbose=True):
    """Re-run the env along the oracle's committed regime path, capturing telemetry.

    Hour-start telemetry depends on the pose the previous hours produced, so it can
    only be recovered by replaying the same regime sequence the oracle chose.
    """
    env = og.build_env(seed=2021, csv=CSV, start=0, end=N_ROWS - 1)
    env.reset()
    out = []
    for i, rec in enumerate(records):
        tel = dict(env._telemetry())
        out.append((tel, rec))
        og._run_hour(env, rec["regime"], og.STEPS_PER_HOUR)
        if verbose and (i + 1) % 1000 == 0:
            print("  replayed %d/%d hours" % (i + 1, len(records)), flush=True)
    return out


def build_goal(tel, rec, conf_scale):
    """Oracle regime + physics geometry -> the goal JSON the SLM must learn to emit."""
    regime = rec["regime"]
    hold = regime == "hold"
    budget = AUTHORITY_BUDGETS[regime]
    hour = int(float(tel.get("hour", 0)))
    horizon = 60.0 if hour < 16 else 30.0

    cur = tel.get("current_target", {}) or {}
    fut = (tel.get("future_target", {}) if horizon == 60 else tel.get("future_30min_target", {})) or cur
    if hold:
        # A hold means "stay where you are" - the pose the system actually adopts.
        t_az = float(tel.get("panel_azimuth_deg", cur.get("azimuth", 180.0)))
        t_tilt = float(tel.get("panel_tilt_deg", cur.get("tilt", 10.0)))
        p_az, p_tilt = t_az, t_tilt
    else:
        t_az, t_tilt = float(cur.get("azimuth", 180.0)), float(cur.get("tilt", 10.0))
        p_az, p_tilt = float(fut.get("azimuth", t_az)), float(fut.get("tilt", t_tilt))

    # Confidence = how decisively the winning regime beat the runner-up.
    sc = sorted(rec["scores"].values(), reverse=True)
    margin = (sc[0] - sc[1]) if len(sc) > 1 else 0.0
    confidence = float(np.clip(margin / conf_scale, 0.2, 1.0)) if conf_scale > 0 else 0.5

    return {
        "target_tilt": round(float(np.clip(t_tilt, 10.0, 85.0)), 1),
        "target_az": round(float(np.clip(t_az, 70.0, 290.0)), 1),
        "motion_budget_deg": budget,
        "pre_position_tilt": round(float(np.clip(p_tilt, 10.0, 85.0)), 1),
        "pre_position_az": round(float(np.clip(p_az, 70.0, 290.0)), 1),
        "hold": bool(hold),
        "horizon_min": horizon,
        "confidence": round(confidence, 2),
    }


def _prompt_telemetry(tel):
    """Telemetry as shown to the model - drop control-plane-only keys."""
    drop = {"dataset_time_step", "constraints"}
    out = {}
    for k, v in tel.items():
        if k in drop:
            continue
        out[k] = round(v, 2) if isinstance(v, float) else v
    return out


def _calendar_index():
    """dataset_time_step -> (month, day), read straight from the source CSV."""
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return (df["Month"].to_numpy(int), df["Day"].to_numpy(int))


def generate(val_fraction=0.1, seed=42, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    OUT_DIR.mkdir(exist_ok=True)

    if not ORACLE_JSON.exists():
        raise FileNotFoundError("Run run_oracle_fullyear.py first: %s" % ORACLE_JSON)
    blob = json.loads(ORACLE_JSON.read_text())
    records = blob["runs"][TAG]["hours"]
    print("oracle hours: %d" % len(records))

    months, days = _calendar_index()
    pairs = replay_for_telemetry(records, verbose=verbose)

    # confidence scale: 75th percentile of the non-zero decision margins
    margins = [sc[0] - sc[1] for sc in
               (sorted(r["scores"].values(), reverse=True) for _, r in pairs)
               if len(sc) > 1 and sc[0] - sc[1] > 0]
    conf_scale = float(np.percentile(margins, 75)) if margins else 1.0
    print("confidence scale (p75 decision margin): %.6g" % conf_scale)

    kept, n_night, n_eval = [], 0, 0
    for tel, rec in pairs:
        ts = int(rec.get("dataset_time_step", 0))
        ts = min(max(ts, 0), len(months) - 1)
        if _is_eval_day(months[ts], days[ts]):
            n_eval += 1
            continue
        if not bool(tel.get("tracking_window_active", False)):
            n_night += 1
            continue
        kept.append((tel, rec))
    print("dropped: %d night/out-of-window, %d evaluation-day hours" % (n_night, n_eval))
    print("kept for training: %d hours" % len(kept))

    counts = {}
    for _, rec in kept:
        counts[rec["regime"]] = counts.get(rec["regime"], 0) + 1
    print("regime balance before oversampling: %s" % counts)

    # Split by HOUR before oversampling. Oversampling first would put duplicate
    # copies of the same hour into both train and val, inflating val scores.
    random.shuffle(kept)
    n_val = int(len(kept) * val_fraction)
    hour_splits = {"val": kept[:n_val], "train": kept[n_val:]}

    def _to_sample(tel, rec):
        goal = build_goal(tel, rec, conf_scale)
        msgs = build_goal_prompt_v2(json.dumps(_prompt_telemetry(tel), separators=(", ", ": ")))
        return {"messages": list(msgs) + [{"role": "assistant", "content": json.dumps(goal)}]}

    splits = {}
    for name, hours in hour_splits.items():
        rows = []
        for tel, rec in hours:
            n = OVERSAMPLE.get(rec["regime"], 1) if name == "train" else 1
            rows.extend([_to_sample(tel, rec)] * n)
        random.shuffle(rows)
        splits[name] = rows

    for name, rows in splits.items():
        path = OUT_DIR / ("%s.jsonl" % name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + chr(10))
        print("wrote %5d samples -> %s" % (len(rows), path))

    stats = {
        "source_oracle": str(ORACLE_JSON), "tag": TAG,
        "hours_total": len(records), "hours_kept": len(kept),
        "dropped_night": n_night, "dropped_eval_days": n_eval,
        "regime_counts_pre_oversample": counts,
        "oversample": OVERSAMPLE,
        "confidence_scale_p75_margin": conf_scale,
        "train": len(splits["train"]), "val": len(splits["val"]),
        "held_out_eval": {"months": sorted(EVAL_MONTHS),
                          "days": [min(EVAL_DAYS), max(EVAL_DAYS)]},
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("stats -> %s" % (OUT_DIR / "stats.json"))
    return stats


if __name__ == "__main__":
    generate()
