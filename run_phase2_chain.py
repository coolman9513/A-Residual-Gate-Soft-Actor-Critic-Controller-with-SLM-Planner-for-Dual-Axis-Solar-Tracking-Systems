# -*- coding: utf-8 -*-
"""Chain Phase 1b -> Phase 2 -> control baseline (PROJECT_HANDOFF.md §11).

Waits for the full-year oracle to finish, then runs, in order:

  1. generate_oracle_dataset.py  - oracle regimes -> prompt/JSON pairs,
                                   evaluation days held out       [sllm_rl]
  2. finetune/train.py           - LoRA fine-tune Qwen2.5-0.5B on
                                   the oracle targets             [sllm_finetune]

The steps span two conda envs (the RL stack is py3.7 / no transformers; the
fine-tune stack is py3.10 / no gym), so each is dispatched to its own interpreter.

The existing paper adapter in finetune/output/ is NOT touched - the new adapter
goes to finetune/output_oracle/.

Detached run:
    python run_phase2_chain.py
Monitor:  Get-Content phase2_chain.log -Wait     Done: phase2_chain_DONE.txt
"""
from __future__ import annotations

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

ORACLE_DONE = ROOT / "oracle_fullyear_DONE.txt"
ORACLE_JSON = ROOT / "models" / "oracle_regimes_2020full.json"
DATA_DIR = ROOT / "finetune" / "data_oracle"
OUT_DIR = ROOT / "finetune" / "output_oracle"

LOG = ROOT / "phase2_chain.log"
DONE = ROOT / "phase2_chain_DONE.txt"

EPOCHS = 4
LR = 2e-4
# Prompt+reply runs to ~975 tokens; 768 silently truncated away every answer.
MAX_SEQ_LEN = 1152
WAIT_POLL_S = 30
WAIT_MAX_S = 6 * 3600


def _safe(text):
    """Redirected stdout is cp1252 here; strip anything it cannot encode."""
    return str(text).encode("ascii", "replace").decode("ascii")


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), _safe(m))
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(step, exe, script, *args):
    log("---- %s ----" % step)
    log("  %s %s %s" % (Path(exe).parent.parent.name, script, " ".join(map(str, args))))
    t0 = time.time()
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([exe, str(script)] + [str(a) for a in args],
                            cwd=str(ROOT), env=child_env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line or line.startswith(("Gym has been", "Please upgrade", "Users of this", "See the migration")):
            continue
        tail.append(line)
        print("    " + _safe(line), flush=True)
    proc.wait()
    with open(LOG, "a", encoding="utf-8") as f:
        for l in tail[-40:]:
            f.write("    " + l + "\n")
    if proc.returncode != 0:
        raise RuntimeError("%s failed (exit %d)" % (step, proc.returncode))
    log("  %s OK (%.0fs)" % (step, time.time() - t0))


def wait_for_oracle():
    if ORACLE_DONE.exists() and ORACLE_JSON.exists():
        log("oracle already complete")
        return
    log("waiting for full-year oracle ...")
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_S:
        if ORACLE_DONE.exists():
            marker = ORACLE_DONE.read_text(encoding="utf-8").strip()
            if marker != "done":
                raise RuntimeError("oracle finished with marker %r" % marker)
            if not ORACLE_JSON.exists():
                raise RuntimeError("oracle marked done but %s missing" % ORACLE_JSON)
            log("oracle complete after %.0f min of waiting" % ((time.time() - t0) / 60.0))
            return
        time.sleep(WAIT_POLL_S)
    raise TimeoutError("oracle did not finish within %.1f h" % (WAIT_MAX_S / 3600.0))


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== phase 2 chain started ===========")
    wait_for_oracle()

    if (DATA_DIR / "train.jsonl").exists():
        log("1/2 dataset already built - skipping")
    else:
        run("1/2 build oracle dataset", PY_RL, ROOT / "finetune" / "generate_oracle_dataset.py")
        if not (DATA_DIR / "train.jsonl").exists():
            raise RuntimeError("dataset step produced no train.jsonl")

    run("2/2 LoRA fine-tune Qwen2.5-0.5B", PY_FT, ROOT / "finetune" / "train.py",
        "--data_dir", DATA_DIR, "--out_dir", OUT_DIR, "--epochs", EPOCHS, "--lr", LR,
        "--max_seq_len", MAX_SEQ_LEN)

    log("=========== PHASE 2 CHAIN DONE ===========")
    log("  adapter -> %s" % (OUT_DIR / "adapter"))
    log("  next: Phase 3/4 - retrain the SAC gate under guardrail_mode partial|loose")
    DONE.write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e))
        log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8")
        raise
