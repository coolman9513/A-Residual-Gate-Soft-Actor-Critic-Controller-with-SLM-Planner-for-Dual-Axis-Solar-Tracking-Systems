# -*- coding: utf-8 -*-
"""Planner forecast sensitivity for the oracle-tuned SLM.

run_planner_forecast_sensitivity.py covers the rule-threshold planner. This adds
the SLM planner, so the sensitivity table shows that the proposed planner also
tolerates forecast error - the substance of the reviewer's idealized-nowcast
objection.

Noise is injected into the PLANNER's irradiance forecast only, exactly as before,
leaving the dataset, the power model and the solar geometry untouched. The FSM
column already exists in the results file and is not recomputed.

Adapter: finetune/output_oracle/adapter - the same one behind the oracle-tuned SLM
row of the multi-seed table. (The instruction-conditioned adapter is used only for
the programmability experiment and is deliberately not used here.)

Seed 2021 only: each rollout costs roughly an hour of GPU time, so the four noise
levels are about four and a half hours. Resumable - tags already in the JSON are
skipped.

Detached run:
    C:/Users/mrcoo/anaconda3/envs/sllm_rl/python.exe run_slm_forecast_sens.py
Monitor:  Get-Content slm_fc_sens.log -Wait   Done: slm_fc_sens_DONE.txt
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in [str(ROOT.parent), str(ROOT), str(ROOT / "meta_sac")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

# reuse the noise wrapper so the perturbation is identical to the earlier sweep
from run_planner_forecast_sensitivity import NoisyForecastGuidance  # noqa: E402

CSV = "2020_4months_2weeks.csv"
SEED = 2021
POLICY = "sac_auto_gate_final"
FINAL_KW = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                cloud_anticipate_lookahead=12, move_deadband_deg=2.0)
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]

LOG = ROOT / "slm_fc_sens.log"
DONE = ROOT / "slm_fc_sens_DONE.txt"
OUT = ROOT / "models" / "planner_forecast_sensitivity.json"


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                        str(m).encode("ascii", "replace").decode("ascii"))
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== SLM planner forecast sensitivity started ===========")
    log("adapter: %s" % os.environ["SOLAR_SLM_ADAPTER"])

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

    results = json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}}
    results.setdefault("runs", {})
    client = LocalQwenClient(max_new_tokens=120)

    for sigma in NOISE_LEVELS:
        tag = "%d/SLM/%d%%" % (SEED, int(sigma * 100))
        if tag in results["runs"]:
            log("skip %s" % tag)
            continue
        base = LLMGoalGuidance(client=client, enabled=True, use_llm=True,
                               guardrail_mode="partial", prompt_version=2)
        guidance = NoisyForecastGuidance(base, sigma, seed=SEED)
        t0 = time.time()
        hist, _, _ = rollout_residual_for_schema(
            agent, schema, guidance=guidance, use_llm=True, seed=SEED,
            label=tag, **FINAL_KW)
        ms, mp = movement_stats(hist), movement_profile_stats(hist)
        row = dict(energy_kwh=round(total_energy_from_history(hist), 3),
                   movement_per_step_deg=round(ms["movement_per_step_deg"], 3),
                   activations=int(mp["activations"]),
                   reversals=int(mp["az_reversals"]),
                   minutes=round((time.time() - t0) / 60.0, 1),
                   parse_failures=int(base.parse_failure_count))
        results["runs"][tag] = row
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log("  %-16s %.3f kWh | act %4d | rev %3d  (%.0f min, parse-fail %d)"
            % (tag, row["energy_kwh"], row["activations"], row["reversals"],
               row["minutes"], row["parse_failures"]))

    log("")
    log("===== SLM PLANNER FORECAST SENSITIVITY (seed %d) =====" % SEED)
    log("  %-8s %10s %10s" % ("noise", "energy", "activ."))
    for sigma in NOISE_LEVELS:
        r = results["runs"].get("%d/SLM/%d%%" % (SEED, int(sigma * 100)))
        if r:
            log("  %-8s %10.3f %10d" % ("%d%%" % int(sigma * 100),
                                        r["energy_kwh"], r["activations"]))
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
