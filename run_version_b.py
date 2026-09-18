# -*- coding: utf-8 -*-
"""Version B - instruction-conditioned planner training (PROJECT_HANDOFF.md section 11).

Why
---
Version A showed the planner IGNORES plain-language directives: the semantic
spread (max_energy vs min_movement) was 12 activations while merely adding a
meaningless sentence moved behaviour by 26. That is not a capacity problem - the
adapter fit its training data well (loss 0.301 -> 0.031) and a tiny decision tree
matches its control performance. It is a DATA problem: every training example was
generated at a single wear setting, so the instruction carried no information and
the model correctly learned to ignore it.

Version B fixes the data. The same telemetry is paired with three different
directives and three different correct answers, so the instruction becomes the
only way to tell the examples apart. The model must read it.

Key efficiency
--------------
The oracle scores all three regimes per hour and records raw (energy, activations,
degrees). Different wear coefficients only re-weight those same numbers, so ONE
oracle pass yields the optimal regime for ANY coefficient. This halves the compute
versus re-running the oracle per setting, and - more importantly - all three label
sets then lie on the SAME committed trajectory, so telemetry is identical across
directives. Without that, the model could distinguish the arms from telemetry
alone and would not need the instruction.

Divergence check (56-day sweep, daylight hours) confirms the signal is real:
    c_act=0 vs c_act=3e-3 : 289/616 hours differ (46.9%)

Expected outcome if it works: the oracle's own spread between those settings is
2512 vs 1921 activations (591 apart), versus a 26-activation prompt-noise floor.
So a successful result should be unambiguous.

Stages (resumable; each skipped if its output exists):
    1. full-year oracle recording raw components      ~2.5 h
    2. instruction-conditioned dataset                ~30 min
    3. LoRA fine-tune (2 epochs, 3x data)             ~13 h
    4. programmability evaluation                     ~4.3 h

Detached run:
    python run_version_b.py
Monitor:  Get-Content version_b.log -Wait     Done: version_b_DONE.txt
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
sys.path.insert(0, str(ROOT / "finetune"))

# Interpreters for the two environments. The RL stack and the language model have
# incompatible dependencies (see requirements.txt / requirements-llm.txt), so each
# stage is dispatched to its own interpreter. RL_PYTHON defaults to the
# interpreter running this script; SLM_PYTHON must point at the Python 3.10
# environment that has transformers and peft.
PY_RL = os.environ.get("RL_PYTHON") or sys.executable
PY_FT = os.environ.get("SLM_PYTHON") or sys.executable

ORACLE_JSON = ROOT / "models" / "oracle_regimes_2020full_components.json"
ORACLE_TAG = "fullyear_components"
DATA_DIR = ROOT / "finetune" / "data_instruct"
OUT_DIR = ROOT / "finetune" / "output_instruct"

LOG = ROOT / "version_b.log"
DONE = ROOT / "version_b_DONE.txt"

CSV = "2020.csv"
N_ROWS = 52560
EPOCHS = 2          # loss converged well inside epoch 1 last time; 3x data now
MAX_SEQ_LEN = 1216  # v2 prompt + directive line + reply; measured max was ~975 + directive


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), str(m).encode("ascii", "replace").decode("ascii"))
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(step, exe, script, *args):
    log("---- %s ----" % step)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([exe, str(script)] + [str(a) for a in args],
                            cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip()
        if not line or line.startswith(("Gym has been", "Please upgrade", "Users of this",
                                        "See the migration", "Loading weights")):
            continue
        print("    " + line.encode("ascii", "replace").decode("ascii"), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("%s failed (exit %d)" % (step, proc.returncode))
    log("  %s OK (%.0f min)" % (step, (time.time() - t0) / 60.0))


def stage1_oracle():
    if ORACLE_JSON.exists():
        blob = json.loads(ORACLE_JSON.read_text())
        if ORACLE_TAG in blob.get("runs", {}):
            log("stage 1: oracle with components already present - skipping")
            return
    log("---- stage 1/4: full-year oracle recording raw components ----")
    t0 = time.time()
    import oracle_goals as og
    env = og.build_env(seed=2021, csv=CSV, start=0, end=N_ROWS - 1)
    og.run_and_save(c_act=og.C_ACT_BREAKEVEN, c_deg=0.0, seed=2021,
                    tag=ORACLE_TAG, env=env, out=ORACLE_JSON)
    log("  stage 1 OK (%.0f min)" % ((time.time() - t0) / 60.0))


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== VERSION B started ===========")
    log("goal: teach the planner to follow plain-language wear directives")

    stage1_oracle()

    if (DATA_DIR / "train.jsonl").exists():
        log("stage 2: instruction dataset already built - skipping")
    else:
        run("stage 2/4: instruction-conditioned dataset", PY_RL,
            ROOT / "finetune" / "generate_instruction_dataset.py")

    if (OUT_DIR / "adapter" / "adapter_model.safetensors").exists():
        log("stage 3: adapter already trained - skipping")
    else:
        run("stage 3/4: LoRA fine-tune (instruction-conditioned)", PY_FT,
            ROOT / "finetune" / "train.py",
            "--data_dir", DATA_DIR, "--out_dir", OUT_DIR,
            "--epochs", EPOCHS, "--lr", 2e-4, "--max_seq_len", MAX_SEQ_LEN)

    run("stage 4/4: programmability evaluation", PY_RL,
        ROOT / "run_programmability.py", "--adapter", OUT_DIR / "adapter",
        "--out", ROOT / "models" / "programmability_version_b.json",
        "--tag", "version_b")

    log("=========== VERSION B DONE ===========")
    DONE.write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e))
        log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8")
        raise
