# -*- coding: utf-8 -*-
"""Programmability experiment (Version A) - does a plain-language instruction re-task
the planner WITHOUT retraining?

Why this matters
----------------
The paper's title claims a "Programmable Small-Language-Model Planner" and
contribution #2 claims a "natural-language-programmable planning layer", but no
experiment in the paper tests either. Re-taskability is a capability a planner
that consumes a fixed numeric feature vector cannot match, since it has no
language input, so it is worth measuring rather than asserting.

Design
------
The SAME adapter, the SAME trained gate, the SAME evaluation set. Only one
sentence in the prompt changes. No retraining anywhere.

    baseline      - no instruction (reproduces the earlier partial run)
    neutral       - a semantically empty sentence. CONTROL: if this alone shifts
                    behaviour, any difference between the other arms is prompt
                    perturbation rather than instruction following.
    max_energy    - "wear is not a concern, maximise energy"
    min_movement  - "actuators are aging, minimise movement"

Ground truth for the expected direction comes from the wear-cost sweep
(models/oracle_regimes.json), where the oracle was re-solved at several
per-activation wear costs:

    c_act = 0       -> 28.571 kWh, 2512 activations   (ignore wear)
    c_act = 3.79e-4 -> 28.522 kWh, 1945 activations   (break-even, what we trained on)
    c_act = 3e-3    -> 28.422 kWh, 1921 activations   (wear dominates)

So "maximise energy" should move energy UP and activations UP; "minimise movement"
should move both DOWN. Absolute values are not comparable to the sweep (it used a
fully open gate), only the DIRECTION is.

Honest expectation: the adapter was fine-tuned on targets from a single wear
setting, so it may have learned that one policy firmly enough to ignore the
instruction entirely. A null result here is informative - it would mean
instruction-conditioned training (Version B) is required.

Detached run:
    python run_programmability.py
Monitor:  Get-Content programmability.log -Wait   Done: programmability_DONE.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in [str(ROOT), str(ROOT / "meta_sac")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ap = argparse.ArgumentParser()
_ap.add_argument("--adapter", default=str(ROOT / "finetune" / "output_oracle" / "adapter"))
_ap.add_argument("--out", default=str(ROOT / "models" / "programmability_results.json"))
_ap.add_argument("--tag", default="version_a")
_ap.add_argument("--seed", type=int, default=2021)
_ap.add_argument("--policy", default="sac_auto_gate_final")
_ap.add_argument("--arms", default="",
                 help="comma-separated subset of arm names; empty = all")
_ARGS = _ap.parse_args()
os.environ["SOLAR_SLM_ADAPTER"] = str(_ARGS.adapter)

import torch  # noqa: E402

import checkpointing  # noqa: E402
from sacauto import SAC_Auto  # noqa: E402
from utilities import load_flat_config  # noqa: E402
from train_utils import build_schema, make_residual_env  # noqa: E402
from eval_utils import (rollout_residual_for_schema, total_energy_from_history,  # noqa: E402
                        movement_stats, movement_profile_stats)
from llm.goal_guidance import LLMGoalGuidance  # noqa: E402
from llm.local_client import LocalQwenClient  # noqa: E402

CSV = "2020_4months_2weeks.csv"
SEED = _ARGS.seed
POLICY = _ARGS.policy
FINAL_KW = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                cloud_anticipate_lookahead=12, move_deadband_deg=2.0)

from llm.goal_prompts import WEAR_DIRECTIVES  # noqa: E402

# Directives come from the shared table, so the evaluation uses byte-identical
# wording to the instruction-conditioned training set.
ARMS = [
    ("baseline", ""),
    ("neutral", "Operate the tracker according to your standard policy."),
    ("max_energy", WEAR_DIRECTIVES["max_energy"][1]),
    ("balanced", WEAR_DIRECTIVES["balanced"][1]),
    ("min_movement", WEAR_DIRECTIVES["min_movement"][1]),
    # --- unseen paraphrases: same meaning, wording never used in training. ---
    # Tests whether the planner generalises over the MEANING of a directive or
    # merely recognises the three exact strings it was fine-tuned on. Word overlap
    # with the trained max_energy directive is limited to the word "whenever"
    # (para1) and to "not"/"motors" (para2, deliberately colloquial).
    ("max_energy_para1",
     "Disregard actuator lifetime at this installation. Electricity output is the "
     "sole objective; permit frequent repositioning whenever it increases yield."),
    ("max_energy_para2",
     "Do not worry about the motors here. Just get as much power as you can, and "
     "move the panel as often as you need to."),
]

if _ARGS.arms:
    _want = {a.strip() for a in _ARGS.arms.split(",") if a.strip()}
    ARMS = [a for a in ARMS if a[0] in _want]

# Direction the wear-cost sweep says each instruction should push behaviour.
EXPECTED = {"baseline": "reference", "neutral": "no change (control)",
            "balanced": "between the two extremes",
            "max_energy": "energy UP, activations UP",
            "min_movement": "energy DOWN, activations DOWN",
            "max_energy_para1": "same as max_energy if meaning generalises",
            "max_energy_para2": "same as max_energy if meaning generalises"}

LOG = ROOT / ("programmability_%s.log" % _ARGS.tag)
DONE = ROOT / ("programmability_%s_DONE.txt" % _ARGS.tag)
OUT = Path(_ARGS.out)


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== programmability (version A) started ===========")
    log("same adapter, same gate, same eval set; only one prompt sentence changes")

    config = load_flat_config(ROOT / "meta_sac" / "configs" / "solar_tracker.yml")
    config["cuda"] = 0 if torch.cuda.is_available() else -1
    base_schema = json.loads((ROOT / "data" / "schema.json").read_text())
    base_schema["root_directory"] = str(ROOT / "data")
    schema = build_schema(base_schema, CSV, 0, 8063)

    env = make_residual_env(schema, use_llm=False, seed=SEED, **FINAL_KW)
    agent = SAC_Auto(env.observation_space.shape[0], env.action_space,
                     {**config, "target_entropy": -3.0})
    _, ckpt = checkpointing.load_checkpoint(POLICY)
    checkpointing.load_agent_state(agent, ckpt, load_optimizer_states=False)
    log("loaded policy %s (seed %d)" % (POLICY, SEED))
    log("adapter: %s" % _ARGS.adapter)

    client = LocalQwenClient(max_new_tokens=120)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results.setdefault("expected_direction", EXPECTED)
    results.setdefault("arms", {})

    for name, instruction in ARMS:
        if name in results["arms"]:
            log("skip %s (already present)" % name)
            continue
        log("---- arm '%s' ----" % name)
        log("   instruction: %s" % (instruction or "(none)"))
        t0 = time.time()
        guidance = LLMGoalGuidance(client=client, enabled=True, use_llm=True,
                                   guardrail_mode="partial", prompt_version=2,
                                   operator_instruction=instruction)
        hist, _, _ = rollout_residual_for_schema(
            agent, schema, guidance=guidance, use_llm=True, seed=SEED,
            label="prog[%s]" % name, **FINAL_KW)
        ms, mp = movement_stats(hist), movement_profile_stats(hist)
        row = dict(instruction=instruction,
                   energy_kwh=round(total_energy_from_history(hist), 3),
                   movement_per_step_deg=round(ms["movement_per_step_deg"], 3),
                   activations=int(mp["activations"]),
                   reversals=int(mp["az_reversals"]),
                   minutes=round((time.time() - t0) / 60.0, 1),
                   parse_failures=int(guidance.parse_failure_count))
        results["arms"][name] = row
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log("  %-13s %.3f kWh | act %4d | rev %3d  (%.1f min, parse-fail %d)"
            % (name, row["energy_kwh"], row["activations"], row["reversals"],
               row["minutes"], row["parse_failures"]))

    base = results["arms"].get("baseline")
    log("")
    log("===== PROGRAMMABILITY (seed %d, same adapter, no retraining) =====" % SEED)
    log("  %-13s %9s %7s %7s   %s" % ("arm", "energy", "act", "rev", "expected"))
    for name, _ in ARMS:
        r = results["arms"].get(name)
        if not r:
            continue
        if base and name != "baseline":
            log("  %-13s %9.3f %7d %7d   %s   (dE %+.3f, dAct %+d)"
                % (name, r["energy_kwh"], r["activations"], r["reversals"], EXPECTED[name],
                   r["energy_kwh"] - base["energy_kwh"], r["activations"] - base["activations"]))
        else:
            log("  %-13s %9.3f %7d %7d   %s"
                % (name, r["energy_kwh"], r["activations"], r["reversals"], EXPECTED[name]))

    # verdict
    if base and all(k in results["arms"] for k in ("neutral", "max_energy", "min_movement")):
        n, hi, lo = (results["arms"][k] for k in ("neutral", "max_energy", "min_movement"))
        drift = abs(n["activations"] - base["activations"])
        spread = abs(hi["activations"] - lo["activations"])
        log("")
        log("  neutral-vs-baseline drift (prompt perturbation): %d activations" % drift)
        log("  max_energy-vs-min_movement spread (semantics)  : %d activations" % spread)
        if spread > max(3 * max(drift, 1), 30) and hi["activations"] > lo["activations"]:
            log("  VERDICT: instruction following DEMONSTRATED - behaviour moves in the")
            log("           direction the wear-cost sweep predicts, beyond prompt noise.")
        elif spread <= max(drift, 1) * 1.5:
            log("  VERDICT: NO instruction following. The adapter was fine-tuned on a single")
            log("           wear setting and ignores the directive. Version B")
            log("           (instruction-conditioned training) would be required.")
        else:
            log("  VERDICT: WEAK/AMBIGUOUS - some movement but not cleanly above prompt noise.")
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
