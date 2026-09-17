# -*- coding: utf-8 -*-
"""Control test: does the ORACLE-TUNED SLM beat the FSM on actual control?

The regime-accuracy numbers (models/slm_regime_eval.json) are a poor proxy: most
hours are near-ties where a wrong label costs almost no energy. This measures what
actually matters - energy and motor activations on the 56-day evaluation set.

It reuses the EXISTING trained policy (sac_auto_gate_final), so no SAC retraining
is needed for a first signal. Phase 4 (retraining the gate under the new planner)
is only worth doing if this shows promise.

Reference points on the same evaluation set with the same trained policy (seed 2021):
    no-SLM        27.646 kWh | act 3682 | rev 307
    FSM + SAC     28.024 kWh | act 2685 | rev 122     <- the number to beat
    RG-SAC (old Qwen, rule-tuned)
                  28.024 kWh | act 2703 | rev 126
Upper bound with a fully open gate (not directly comparable, g=1):
    oracle regimes 28.522 kWh | act 1945

Guardrail modes evaluated (handoff §11 phase 3):
    partial - forced-budget and budget-blend removed, angle override kept
    loose   - all three removed; the SLM's decisions stand

Detached run:
    C:/Users/mrcoo/anaconda3/envs/sllm_rl/python.exe run_control_test.py
Monitor:  Get-Content control_test.log -Wait    Done: control_test_DONE.txt
"""
from __future__ import annotations

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

# Serve the oracle-tuned adapter, not the paper one.
os.environ["SOLAR_SLM_ADAPTER"] = str(ROOT / "finetune" / "output_oracle" / "adapter")

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
SEED = 2021
POLICY = "sac_auto_gate_final"
FINAL_KW = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                cloud_anticipate_lookahead=12, move_deadband_deg=2.0)
MODES = ["partial", "loose"]

LOG = ROOT / "control_test.log"
DONE = ROOT / "control_test_DONE.txt"
OUT = ROOT / "models" / "control_test_results.json"

REFERENCE = {
    "no-SLM":                 dict(energy_kwh=27.646, activations=3682, reversals=307),
    "FSM+SAC":                dict(energy_kwh=28.024, activations=2685, reversals=122),
    "RG-SAC (old Qwen)":      dict(energy_kwh=28.024, activations=2703, reversals=126),
}


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== control test started ===========")
    log("adapter: %s" % os.environ["SOLAR_SLM_ADAPTER"])

    config = load_flat_config(ROOT / "meta_sac" / "configs" / "solar_tracker.yml")
    config["cuda"] = 0 if torch.cuda.is_available() else -1
    base = json.loads((ROOT / "data" / "schema.json").read_text())
    base["root_directory"] = str(ROOT / "data")
    schema = build_schema(base, CSV, 0, 8063)

    env = make_residual_env(schema, use_llm=False, seed=SEED, **FINAL_KW)
    agent = SAC_Auto(env.observation_space.shape[0], env.action_space,
                     {**config, "target_entropy": -3.0})
    _, ckpt = checkpointing.load_checkpoint(POLICY)
    checkpointing.load_agent_state(agent, ckpt, load_optimizer_states=False)
    log("loaded policy %s" % POLICY)

    # One model process shared across both guardrail modes.
    client = LocalQwenClient(max_new_tokens=120)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results.setdefault("reference", REFERENCE)

    for mode in MODES:
        if mode in results.get("modes", {}):
            log("skip %s (already present)" % mode)
            continue
        log("---- guardrail_mode=%s, prompt_version=2 ----" % mode)
        t0 = time.time()
        guidance = LLMGoalGuidance(client=client, enabled=True, use_llm=True,
                                   guardrail_mode=mode, prompt_version=2)
        hist, _, _ = rollout_residual_for_schema(
            agent, schema, guidance=guidance, use_llm=True, seed=SEED,
            label="oracle-SLM[%s]" % mode, **FINAL_KW)
        e = total_energy_from_history(hist)
        ms = movement_stats(hist)
        mp = movement_profile_stats(hist)
        row = dict(energy_kwh=round(e, 3),
                   movement_per_step_deg=round(ms["movement_per_step_deg"], 3),
                   activations=int(mp["activations"]), reversals=int(mp["az_reversals"]),
                   minutes=round((time.time() - t0) / 60.0, 1),
                   llm_calls=int(guidance.call_count),
                   parse_failures=int(guidance.parse_failure_count),
                   fallbacks=int(guidance.fallback_count))
        results.setdefault("modes", {})[mode] = row
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log("  %s: %.3f kWh | act %d | rev %d | %.0f min | llm %d, parse-fail %d, fallback %d"
            % (mode, row["energy_kwh"], row["activations"], row["reversals"],
               row["minutes"], row["llm_calls"], row["parse_failures"], row["fallbacks"]))

    log("")
    log("===== CONTROL TEST SUMMARY (56-day eval set, trained gate, seed 2021) =====")
    log("  %-22s %10s %7s %7s" % ("controller", "energy", "act", "rev"))
    for k, v in REFERENCE.items():
        log("  %-22s %10.3f %7d %7d" % (k, v["energy_kwh"], v["activations"], v["reversals"]))
    for mode, v in results.get("modes", {}).items():
        log("  %-22s %10.3f %7d %7d   <- oracle-SLM"
            % ("oracle-SLM [%s]" % mode, v["energy_kwh"], v["activations"], v["reversals"]))
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
