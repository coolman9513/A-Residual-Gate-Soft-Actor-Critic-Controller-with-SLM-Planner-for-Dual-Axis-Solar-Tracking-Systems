# -*- coding: utf-8 -*-
"""Forecast sensitivity of the PLANNER (reviewer point #3, done properly).

Why this exists
---------------
run_forecast_sensitivity.py injects noise into the CSV columns next_10min_DNI,
next_10min_DHI and next_30min_average_DNI. Those are active env observations, so
the SAC gate's inputs really are corrupted there. But the PLANNER does not read
those columns: goal_wrapper._dni_window() reads the actual `dni` value at future
time steps, and telemetry's dni_forecast / next_10min_dni / next_30min_average_dni
are all derived from it. The planner therefore retained PERFECT foresight at every
noise level in that analysis, which is very likely why it looked so robust.

The professor's point was about the idealized nowcast. This corrupts exactly that:
the planner's view of the future, leaving the physics that generates power, and the
solar geometry, untouched.

What it compares
----------------
Both planners under identical corruption, so the question "does the oracle planner's
advantage survive realistic forecast error?" gets a direct answer:

    FSM - DNI-threshold rule planner (the paper's baseline)

Noise is multiplicative Gaussian at 0/5/10/20% relative RMSE, matching the
definition already used in the paper.

Detached run:
    C:/Users/mrcoo/anaconda3/envs/sllm_rl/python.exe run_planner_forecast_sensitivity.py
Monitor:  Get-Content planner_fc_sens.log -Wait   Done: planner_fc_sens_DONE.txt
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
for _p in [str(ROOT), str(ROOT / "meta_sac")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import checkpointing  # noqa: E402
from sacauto import SAC_Auto  # noqa: E402
from utilities import load_flat_config  # noqa: E402
from train_utils import build_schema, make_residual_env  # noqa: E402
from eval_utils import (rollout_residual_for_schema, total_energy_from_history,  # noqa: E402
                        movement_stats, movement_profile_stats)
from llm.goal_guidance import LLMGoalGuidance  # noqa: E402

CSV = "2020_4months_2weeks.csv"
FINAL_KW = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                cloud_anticipate_lookahead=12, move_deadband_deg=2.0)
POLICIES = {2021: "sac_auto_gate_final", 2022: "rev_final_2022", 2023: "rev_final_2023"}
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]

LOG = ROOT / "planner_fc_sens.log"
DONE = ROOT / "planner_fc_sens_DONE.txt"
OUT = ROOT / "models" / "planner_forecast_sensitivity.json"


def log(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _bucket(max_dni):
    if max_dni >= 700:
        return "CLEAR_STRONG"
    if max_dni >= 500:
        return "CLEAR_MODERATE"
    if max_dni >= 300:
        return "PARTIAL_CLOUD"
    if max_dni >= 100:
        return "OVERCAST_DIM"
    return "NIGHT_OR_HEAVY_OVERCAST"


class NoisyForecastGuidance:
    """Corrupts the planner's view of the future, then delegates.

    Only the forecast FIELDS OF TELEMETRY are perturbed. The dataset, the power
    model and the solar geometry are untouched, so generation physics is exactly
    as before and only the planner's information is degraded.
    """

    def __init__(self, base, sigma: float, seed: int = 0):
        self.base = base
        self.sigma = float(sigma)
        self._seed = int(seed)

    def context_names(self):
        return self.base.context_names()

    def _perturb(self, tel):
        if self.sigma <= 0.0:
            return tel
        t = dict(tel)
        # Deterministic per (seed, sigma, step) so runs are reproducible.
        step = int(t.get("dataset_time_step", 0) or 0)
        rng = np.random.default_rng((self._seed * 100003 + step) * 1009
                                    + int(self.sigma * 1000))
        fc = list(t.get("dni_forecast_next_60min_10min_steps", []) or [])
        if fc:
            noisy = [float(max(0.0, v * (1.0 + rng.normal(0.0, self.sigma)))) for v in fc]
            t["dni_forecast_next_60min_10min_steps"] = noisy
            t["next_10min_dni"] = noisy[0]
            head = noisy[:3] if len(noisy) >= 3 else noisy
            t["next_30min_average_dni"] = float(np.mean(head))
            max_fc = max(float(t.get("dni", 0.0)), t["next_30min_average_dni"])
            t["max_forecast_dni"] = round(max_fc, 1)
            t["dni_bucket"] = _bucket(max_fc)
        return t

    def guidance_for(self, time_step, telemetry):
        return self.base.guidance_for(time_step, self._perturb(telemetry))


def main():
    if DONE.exists():
        DONE.unlink()
    log("=========== planner forecast sensitivity started ===========")
    log("noise: multiplicative Gaussian on the PLANNER's dni forecast; physics untouched")

    config = load_flat_config(ROOT / "meta_sac" / "configs" / "solar_tracker.yml")
    config["cuda"] = 0 if torch.cuda.is_available() else -1
    base_schema = json.loads((ROOT / "data" / "schema.json").read_text())
    base_schema["root_directory"] = str(ROOT / "data")
    schema = build_schema(base_schema, CSV, 0, 8063)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results.setdefault("noise_definition",
                       "multiplicative Gaussian on planner telemetry dni forecast; "
                       "sigma = relative RMSE; dataset/power model unchanged")
    results.setdefault("runs", {})

    for seed, folder in POLICIES.items():
        env = make_residual_env(schema, use_llm=False, seed=seed, **FINAL_KW)
        agent = SAC_Auto(env.observation_space.shape[0], env.action_space,
                         {**config, "target_entropy": -3.0})
        _, ckpt = checkpointing.load_checkpoint(folder)
        checkpointing.load_agent_state(agent, ckpt, load_optimizer_states=False)
        log("seed %d: loaded %s" % (seed, folder))

        for sigma in NOISE_LEVELS:
            for planner in ("FSM",):
                tag = "%d/%s/%d%%" % (seed, planner, int(sigma * 100))
                if tag in results["runs"]:
                    log("skip %s" % tag)
                    continue
                # use_llm=False -> the deterministic DNI-threshold rule planner
                base = LLMGoalGuidance(client=None, enabled=True, use_llm=False,
                                       guardrail_mode="full")
                guidance = NoisyForecastGuidance(base, sigma, seed=seed)
                t0 = time.time()
                hist, _, _ = rollout_residual_for_schema(
                    agent, schema, guidance=guidance, use_llm=True, seed=seed,
                    label=tag, **FINAL_KW)
                ms, mp = movement_stats(hist), movement_profile_stats(hist)
                row = dict(energy_kwh=round(total_energy_from_history(hist), 3),
                           movement_per_step_deg=round(ms["movement_per_step_deg"], 3),
                           activations=int(mp["activations"]),
                           reversals=int(mp["az_reversals"]),
                           minutes=round((time.time() - t0) / 60.0, 1))
                results["runs"][tag] = row
                OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
                log("  %-22s %.3f kWh | act %4d | rev %3d  (%.1f min)"
                    % (tag, row["energy_kwh"], row["activations"],
                       row["reversals"], row["minutes"]))

    log("")
    log("===== PLANNER FORECAST SENSITIVITY (mean over seeds) =====")
    log("  %-6s %-12s %10s %8s %8s" % ("noise", "planner", "energy", "act", "rev"))
    for sigma in NOISE_LEVELS:
        for planner in ("FSM",):
            rows = [results["runs"].get("%d/%s/%d%%" % (s, planner, int(sigma * 100)))
                    for s in POLICIES]
            rows = [r for r in rows if r]
            if not rows:
                continue
            e = float(np.mean([r["energy_kwh"] for r in rows]))
            a = float(np.mean([r["activations"] for r in rows]))
            v = float(np.mean([r["reversals"] for r in rows]))
            log("  %-6s %-12s %10.3f %8.0f %8.0f" % ("%d%%" % int(sigma * 100), planner, e, a, v))
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
