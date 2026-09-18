# -*- coding: utf-8 -*-
"""Reviewer #3 — Forecast Sensitivity Analysis.

Loads the paper's trained 56-day models (sac_auto_gate_final = RG-SAC,
sac_auto_gate_nosllm = no-SLM) and re-evaluates them with synthetic Gaussian
noise injected into the look-ahead IRRADIANCE forecast features
(next_10min_DNI, next_10min_DHI, next_30min_average_DNI) at 0 / 5 / 10 / 20 %
relative RMSE. The ACTUAL DNI/DHI used for power generation and the deterministic
solar-geometry look-ahead are left unchanged, so only the controller's forecast
inputs are corrupted. Fixed/RBC are forecast-independent and evaluated once.

Detached run:
    python run_forecast_sensitivity.py
Monitor: Get-Content forecast_sens.log -Wait   Done: forecast_sens_DONE.txt
"""
from __future__ import annotations
import json, random, time, sys, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent
for p in [str(ROOT), str(ROOT / "meta_sac")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import checkpointing
from sacauto import SAC_Auto
from utilities import load_flat_config
from train_utils import build_schema, make_residual_env
from eval_utils import (rollout_fixed_for_schema, rollout_rbc_for_schema,
                        rollout_residual_for_schema, total_energy_from_history,
                        movement_stats, movement_profile_stats)

CSV = "2020_4months_2weeks.csv"
NOISE_COLS = ["next_10min_DNI", "next_10min_DHI", "next_30min_average_DNI"]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
FINAL_KW  = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                 cloud_anticipate_lookahead=12, move_deadband_deg=2.0)
NOSLLM_KW = dict(action_mode="gate", authority_mode="fixed",
                 fixed_authority=(6.0, 4.0, float("inf")), hold_base=False,
                 cloud_anticipate_lookahead=0, move_deadband_deg=0.0, disable_guidance=True)
LOG, DONE = ROOT / "forecast_sens.log", ROOT / "forecast_sens_DONE.txt"
OUT = ROOT / "models" / "forecast_sensitivity_results.json"

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")

def main():
    if DONE.exists(): DONE.unlink()
    log("=========== forecast sensitivity started ===========")
    config = load_flat_config(ROOT / "meta_sac" / "configs" / "solar_tracker.yml")
    config["cuda"] = 0 if torch.cuda.is_available() else -1
    with open(ROOT / "data" / "schema.json") as f:
        base_schema = json.load(f)
    base_schema["root_directory"] = str(ROOT / "data")
    SEED = int(config.get("seed", 2021))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    base_df = pd.read_csv(ROOT / "data" / CSV)

    def noisy_csv(sigma):
        if sigma == 0.0:
            return CSV
        rng = np.random.default_rng(1000 + int(sigma * 100))
        df = base_df.copy()
        for c in NOISE_COLS:
            v = df[c].to_numpy(float)
            df[c] = np.clip(v * (1.0 + rng.normal(0.0, sigma, size=len(v))), 0.0, None)
        name = f"_noisy_fc_{int(sigma*100)}.csv"
        df.to_csv(ROOT / "data" / name, index=False)
        return name

    def make_agent(env_kw):
        env = make_residual_env(build_schema(base_schema, CSV, 0, 8063),
                                use_llm=False, seed=SEED, **env_kw)
        return SAC_Auto(env.observation_space.shape[0], env.action_space,
                        {**config, "target_entropy": -3.0})

    def load_saved(folder, env_kw):
        agent = make_agent(env_kw)
        _, ckpt = checkpointing.load_checkpoint(folder)
        checkpointing.load_agent_state(agent, ckpt, load_optimizer_states=False)
        log(f"loaded {folder}"); return agent

    def eval_ctrl(name, agent, env_kw, schema):
        hist, _, _ = rollout_residual_for_schema(agent, schema, use_llm=False,
                                                 seed=SEED, label=name, **env_kw)
        e = total_energy_from_history(hist); ms = movement_stats(hist); mp = movement_profile_stats(hist)
        row = dict(energy_kwh=round(e, 3),
                   movement_per_step_deg=round(ms["movement_per_step_deg"], 3),
                   activations=mp["activations"], reversals=mp["az_reversals"])
        log(f"    {name}: {e:.2f} kWh @ {ms['movement_per_step_deg']:.2f} | act {mp['activations']} | rev {mp['az_reversals']}")
        return row

    rl_agent = load_saved("sac_auto_gate_nosllm", NOSLLM_KW)
    final_agent = load_saved("sac_auto_gate_final", FINAL_KW)

    results = {"noise_definition": "multiplicative Gaussian on look-ahead DNI/DHI, sigma = relative RMSE",
               "levels": {}}
    # forecast-independent baselines (evaluated once on clean data)
    sch0 = build_schema(base_schema, CSV, 0, 8063)
    hfx, _ = rollout_fixed_for_schema(sch0)
    hrbc, _ = rollout_rbc_for_schema(sch0)
    mprbc = movement_profile_stats(hrbc)
    baselines = {
        "Fixed panel": dict(energy_kwh=round(total_energy_from_history(hfx), 3),
                            movement_per_step_deg=0.0, activations=0, reversals=0),
        "RBC": dict(energy_kwh=round(total_energy_from_history(hrbc), 3),
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in movement_stats(hrbc).items()},
                    activations=mprbc["activations"], reversals=mprbc["az_reversals"])}
    results["baselines_forecast_independent"] = baselines
    log(f"baselines: Fixed {baselines['Fixed panel']['energy_kwh']} | RBC {baselines['RBC']['energy_kwh']}")

    for sigma in NOISE_LEVELS:
        pct = f"{int(sigma*100)}%"
        log(f"--- noise level {pct} ---")
        schema = build_schema(base_schema, noisy_csv(sigma), 0, 8063)
        results["levels"][pct] = {
            "SAC-Auto (no SLM)": eval_ctrl("no-SLM", rl_agent, NOSLLM_KW, schema),
            "RG-SAC (with SLM)": eval_ctrl("RG-SAC", final_agent, FINAL_KW, schema)}
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")  # persist per level

    log("=========== ALL DONE -> forecast_sensitivity_results.json ===========")
    DONE.write_text("done", encoding="utf-8")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e)); log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8"); raise
