# -*- coding: utf-8 -*-
"""Seed replication of the programmability result (handoff section 11, Version B).

The headline Version B finding - a plain-language directive shifting motor
activations by +106 against a 7-activation prompt-noise floor - rests on a SINGLE
seed (2021). The earlier multi-seed study showed reversals swinging 307/194/402 on
an identical setup, so a single-seed movement claim is not publishable.

This replays the same instruction-conditioned adapter against the seed-2022 and
seed-2023 gate policies already on disk.

Arms are limited to the three that carry the argument:
    neutral      - semantically empty sentence; the noise floor and the fairest
                   reference, since it is matched for prompt length
    max_energy   - the directive that produced the effect
    min_movement - expected null, since the oracle policy saturates: c_act of
                   3e-3, 1e-2, 3e-2 and 1e-1 all yield identical labels

Scope note: these seeds vary the SAC gate, not the SLM. The adapter is a single
training run, so this tests whether the instruction effect survives different
gate policies - not whether a differently-seeded adapter would behave the same.
Retraining the adapter per seed would cost ~14.6 h each and is out of scope here.

Detached run:
    python run_prog_seeds.py
Monitor:  Get-Content prog_seeds.log -Wait   Done: prog_seeds_DONE.txt
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Interpreters for the two environments. The RL stack and the language model have
# incompatible dependencies (see requirements.txt / requirements-llm.txt), so each
# stage is dispatched to its own interpreter. RL_PYTHON defaults to the
# interpreter running this script; SLM_PYTHON must point at the Python 3.10
# environment that has transformers and peft.
PY_RL = os.environ.get("RL_PYTHON") or sys.executable
PY_FT = os.environ.get("SLM_PYTHON") or sys.executable
ADAPTER = ROOT / "finetune" / "output_instruct" / "adapter"
ARMS = "neutral,max_energy,min_movement"

SEEDS = {2022: "rev_final_2022", 2023: "rev_final_2023"}

LOG = ROOT / "prog_seeds.log"
DONE = ROOT / "prog_seeds_DONE.txt"


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                        str(m).encode("ascii", "replace").decode("ascii"))
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== programmability seed replication started ===========")
    log("adapter: %s" % ADAPTER)
    log("arms: %s" % ARMS)

    for seed, policy in SEEDS.items():
        out = ROOT / "models" / ("programmability_seed%d.json" % seed)
        log("---- seed %d (policy %s) ----" % (seed, policy))
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(
            [PY_RL, str(ROOT / "run_programmability.py"),
             "--adapter", str(ADAPTER), "--out", str(out),
             "--tag", "seed%d" % seed, "--seed", str(seed),
             "--policy", policy, "--arms", ARMS],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip()
            if not line or line.startswith(("Gym has been", "Please upgrade",
                                            "Users of this", "See the migration",
                                            "Loading weights", "serve_inference")):
                continue
            print("    " + line.encode("ascii", "replace").decode("ascii"), flush=True)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("seed %d failed (exit %d)" % (seed, proc.returncode))
        log("  seed %d OK (%.0f min)" % (seed, (time.time() - t0) / 60.0))

    # ---- combined verdict across seeds ----
    log("")
    log("===== PROGRAMMABILITY ACROSS SEEDS =====")
    log("  %-6s %10s %10s %12s %10s" % ("seed", "neutral", "max_energy", "min_movement", "spread"))
    rows = []
    seed2021 = ROOT / "models" / "programmability_version_b.json"
    files = [(2021, seed2021)] + [(s, ROOT / "models" / ("programmability_seed%d.json" % s))
                                  for s in SEEDS]
    for seed, f in files:
        if not f.exists():
            continue
        arms = json.loads(f.read_text()).get("arms", {})
        if not all(k in arms for k in ("neutral", "max_energy", "min_movement")):
            continue
        n = arms["neutral"]["activations"]
        hi = arms["max_energy"]["activations"]
        lo = arms["min_movement"]["activations"]
        rows.append((seed, n, hi, lo, hi - lo))
        log("  %-6d %10d %10d %12d %10d" % (seed, n, hi, lo, hi - lo))
    if len(rows) > 1:
        import statistics as st
        sp = [r[4] for r in rows]
        gain = [r[2] - r[1] for r in rows]   # max_energy vs neutral
        log("")
        log("  spread (max_energy - min_movement): mean %.0f, min %d, max %d"
            % (st.mean(sp), min(sp), max(sp)))
        log("  effect  (max_energy - neutral)    : %s" % gain)
        if min(gain) > 30:
            log("  VERDICT: instruction effect REPLICATES - max_energy raises activations")
            log("           on every seed, well above the ~7 activation prompt-noise floor.")
        elif min(gain) > 0:
            log("  VERDICT: direction consistent but magnitude varies across seeds;")
            log("           report as mean +/- std rather than a single figure.")
        else:
            log("  VERDICT: DOES NOT REPLICATE - the seed-2021 result was not robust.")
    log("=========== DONE ===========")
    DONE.write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e))
        log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8")
        raise
