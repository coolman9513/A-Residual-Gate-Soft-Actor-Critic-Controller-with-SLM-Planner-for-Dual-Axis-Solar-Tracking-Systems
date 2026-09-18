# -*- coding: utf-8 -*-
"""Full-year hindsight-optimal regime oracle -> SLM fine-tuning source.

PROJECT_HANDOFF.md §11, Phase 1 (training-set generation).

Why the full year and not the 56-day study set
----------------------------------------------
The oracle's targets are HINDSIGHT-optimal: they encode the actual future
irradiance of the days they are computed on. The 56-day set
(2020_4months_2weeks.csv) is the paper's EVALUATION set, so fine-tuning the SLM
on oracle targets for those days would be training on the test set - the model
would memorise the answers and Phase 5 would be meaningless.

This did not apply to the original rule-based targets: a DNI threshold carries
no information about the future, so train/eval overlap was harmless there. It
becomes fatal only once the targets are hindsight-derived.

So the oracle is run over all of 2020 here, and generate_oracle_dataset.py then
drops every hour falling on an evaluation day (months 1/4/7/10, days 9-22),
leaving the evaluation days genuinely unseen.

Writes to models/oracle_regimes_2020full.json - a DIFFERENT file from the
56-day sweep, so the two can run concurrently without clobbering each other.

Detached run:
    python run_oracle_fullyear.py
Monitor:  Get-Content oracle_fullyear.log -Wait   Done: oracle_fullyear_DONE.txt
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "finetune"))
import oracle_goals as og  # noqa: E402

CSV = "2020.csv"
N_ROWS = 52560                      # full year at 10-min resolution
OUT = ROOT / "models" / "oracle_regimes_2020full.json"
LOG = ROOT / "oracle_fullyear.log"
DONE = ROOT / "oracle_fullyear_DONE.txt"

# Evaluation days held out of training (the 56-day study set).
EVAL_MONTHS = (1, 4, 7, 10)
EVAL_DAYS = tuple(range(9, 23))


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== full-year oracle started (%s, %d rows) ===========" % (CSV, N_ROWS))
    log("wear cost: c_act=%.3e per activation (break-even), c_deg=0" % og.C_ACT_BREAKEVEN)
    log("evaluation days held out later: months %s days %d-%d"
        % (EVAL_MONTHS, EVAL_DAYS[0], EVAL_DAYS[-1]))
    env = og.build_env(seed=2021, csv=CSV, start=0, end=N_ROWS - 1)
    og.run_and_save(c_act=og.C_ACT_BREAKEVEN, c_deg=0.0, seed=2021,
                    tag="fullyear_breakeven", env=env, out=OUT)
    log("=========== FULL-YEAR ORACLE DONE -> %s ===========" % OUT.name)
    DONE.write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e))
        log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8")
        raise
