# -*- coding: utf-8 -*-
"""Reviewer #1 (multi-seed) + #2 (FSM vs SLM) — residual-gate architecture.

For each random seed it trains the two learned controllers on the 56-day dataset
(no-SLM ablation and the FINAL residual-gate policy, both trained with the
deterministic rule-based regime router, exactly as in the paper), then evaluates:

    no-SLM              : residual gate, no regime constraints          (use_llm=False)
    FSM + SAC-Auto (#2) : residual gate + rule-threshold regime router  (use_llm=False)
    RG-SAC (real Qwen)  : residual gate + fine-tuned Qwen2.5-0.5B        (use_llm=True)

FSM and RG-SAC share the SAME trained policy; they differ only in the regime
source at evaluation, which isolates the value of the Qwen transformer over an
explicit rule-threshold FSM. Fixed/RBC are deterministic (one evaluation).

Results (per seed + mean/std) -> models/revision_seeds_results.json.
Resumable: seeds already present in the JSON are skipped.

Detached run:
    python run_revision_seeds.py
Monitor: Get-Content revision_seeds.log -Wait     Done: revision_seeds_DONE.txt
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
from replay_memory import ReplayMemoryKL as ReplayMemory
from utilities import load_flat_config
from train_utils import build_schema, make_residual_env, train_residual_sac
from eval_utils import (rollout_fixed_for_schema, rollout_rbc_for_schema,
                        rollout_residual_for_schema, total_energy_from_history,
                        movement_stats, movement_profile_stats)

# ── configuration ───────────────────────────────────────────────────────────────
SEEDS = [2021, 2022, 2023]          # extend to [.., 2024, 2025] for a 5-seed study
EPISODES = 40                        # matches the paper
CSV = "2020_4months_2weeks.csv"      # 56-day dataset, 8,064 steps (cached)
TARGET_ENTROPY = -3.0

FINAL_KW  = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                 cloud_anticipate_lookahead=12, move_deadband_deg=2.0)
NOSLLM_KW = dict(action_mode="gate", authority_mode="fixed",
                 fixed_authority=(6.0, 4.0, float("inf")), hold_base=False,
                 cloud_anticipate_lookahead=0, move_deadband_deg=0.0, disable_guidance=True)

# seed 2021 was the paper run — reuse those checkpoints instead of retraining
EXISTING = {2021: {"nosllm": "sac_auto_gate_nosllm", "final": "sac_auto_gate_final"}}

RUN_QWEN = True                      # set False to skip the (slow) real-Qwen eval
LOG, DONE = ROOT / "revision_seeds.log", ROOT / "revision_seeds_DONE.txt"
OUT = ROOT / "models" / "revision_seeds_results.json"

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")

def main():
    if DONE.exists(): DONE.unlink()
    log("=========== revision seeds (#1 multi-seed + #2 FSM) started ===========")
    config = load_flat_config(ROOT / "meta_sac" / "configs" / "solar_tracker.yml")
    config["cuda"] = 0 if torch.cuda.is_available() else -1
    with open(ROOT / "data" / "schema.json") as f:
        base_schema = json.load(f)
    base_schema["root_directory"] = str(ROOT / "data")
    schema = build_schema(base_schema, CSV, 0, 8063)

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

    def make_agent(env_kw, seed):
        env = make_residual_env(schema, use_llm=False, seed=seed, **env_kw)
        agent = SAC_Auto(env.observation_space.shape[0], env.action_space,
                         {**config, "target_entropy": TARGET_ENTROPY})
        return env, agent

    def load_saved(folder, env_kw, seed):
        _, agent = make_agent(env_kw, seed)
        _, ckpt = checkpointing.load_checkpoint(folder)
        checkpointing.load_agent_state(agent, ckpt, load_optimizer_states=False)
        log(f"  loaded existing {folder}"); return agent

    def train(name, folder, env_kw, seed):
        cfg = {**config, "num_episodes": EPISODES, "target_entropy": TARGET_ENTROPY,
               "eval_interval": max(1, EPISODES)}
        env, agent = make_agent(env_kw, seed)
        mem = ReplayMemory(int(cfg["replay_size"])); kl = ReplayMemory(int(cfg["kl_replay_size"]))
        for policy in ("oracle", "floor"):
            s = env.reset(); done = False
            while not done:
                a = env.oracle_action() if policy == "oracle" else env.floor_action()
                ns, r, done, _ = env.step(a)
                mem.push(s, a, np.zeros(1, dtype=np.float32), r, ns, 0.0 if done else 1.0)
                kl.push(s, a, np.zeros(1, dtype=np.float32), r, ns, 0.0 if done else 1.0)
                s = ns
        res = train_residual_sac(agent, env, mem, kl, cfg, label=name)
        checkpointing.save_controller_results(
            folder_name=folder, controller=name, agent=agent, config=cfg, schema=schema,
            training_df=res["history"], replay_memory=mem, kl_replay_memory=kl,
            best_metric_name=res["best_metric_name"], best_metric_value=res["best_metric_value"],
            best_episode=res["best_episode"], best_policy_state=res["best_policy_state"],
            best_log_alpha=res["best_log_alpha"], total_numsteps=res["total_numsteps"],
            updates=res["updates"], overwrite=True,
            extra_metadata={"dataset": "56-day", "episodes": EPISODES, "seed": seed})
        log(f"  trained {name} (seed {seed}) -> {folder}"); return agent

    def ev(name, agent, env_kw, seed, use_llm):
        hist, _, _ = rollout_residual_for_schema(agent, schema, use_llm=use_llm,
                                                 seed=seed, label=name, **env_kw)
        e = total_energy_from_history(hist); ms = movement_stats(hist); mp = movement_profile_stats(hist)
        row = dict(energy_kwh=round(e, 3),
                   movement_per_step_deg=round(ms["movement_per_step_deg"], 3),
                   activations=mp["activations"], reversals=mp["az_reversals"])
        log(f"    {name}: {e:.3f} kWh @ {ms['movement_per_step_deg']:.2f} | act {mp['activations']} | rev {mp['az_reversals']}")
        return row

    results = json.loads(OUT.read_text()) if OUT.exists() else {"per_seed": {}, "baselines": None}
    done_seeds = set(int(s) for s in results["per_seed"])
    if done_seeds: log(f"resuming — seeds done: {sorted(done_seeds)}")

    # deterministic, seed-independent baselines (once)
    if not results.get("baselines"):
        hfx, _ = rollout_fixed_for_schema(schema); hrbc, _ = rollout_rbc_for_schema(schema)
        mprbc = movement_profile_stats(hrbc)
        results["baselines"] = {
            "Fixed panel": dict(energy_kwh=round(total_energy_from_history(hfx), 3),
                                movement_per_step_deg=0.0, activations=0, reversals=0),
            "RBC": dict(energy_kwh=round(total_energy_from_history(hrbc), 3),
                        movement_per_step_deg=round(movement_stats(hrbc)["movement_per_step_deg"], 3),
                        activations=mprbc["activations"], reversals=mprbc["az_reversals"])}
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"baselines: Fixed {results['baselines']['Fixed panel']['energy_kwh']} | RBC {results['baselines']['RBC']['energy_kwh']}")

    for seed in SEEDS:
        if seed in done_seeds:
            continue
        log(f"===== SEED {seed} =====")
        set_seed(seed)
        # no-SLM policy
        if seed in EXISTING:
            rl = load_saved(EXISTING[seed]["nosllm"], NOSLLM_KW, seed)
        else:
            rl = train("no-SLM", f"rev_nosllm_{seed}", NOSLLM_KW, seed)
        # FINAL residual-gate policy (trained with rule-router)
        if seed in EXISTING:
            fin = load_saved(EXISTING[seed]["final"], FINAL_KW, seed)
        else:
            fin = train("RG-SAC", f"rev_final_{seed}", FINAL_KW, seed)
        # evaluations
        row = {"no-SLM": ev("no-SLM", rl, NOSLLM_KW, seed, use_llm=False),
               "FSM+SAC": ev("FSM+SAC", fin, FINAL_KW, seed, use_llm=False)}
        if RUN_QWEN:
            try:
                row["RG-SAC (Qwen)"] = ev("RG-SAC-Qwen", fin, FINAL_KW, seed, use_llm=True)
            except Exception as e:
                log(f"    RG-SAC (Qwen) eval failed: {e!r}")
        results["per_seed"][str(seed)] = row
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"  seed {seed} recorded")

    # ── aggregate mean ± std ────────────────────────────────────────────────────
    def agg(key, metric):
        vals = [results["per_seed"][s][key][metric] for s in results["per_seed"] if key in results["per_seed"][s]]
        vals = np.array(vals, float)
        return dict(mean=round(float(vals.mean()), 3),
                    std=round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 3), n=len(vals))
    summary = {}
    for key in ("no-SLM", "FSM+SAC", "RG-SAC (Qwen)"):
        if any(key in results["per_seed"][s] for s in results["per_seed"]):
            summary[key] = {m: agg(key, m) for m in ("energy_kwh", "movement_per_step_deg", "activations", "reversals")}
    results["summary_mean_std"] = summary
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("SUMMARY (mean +/- std over seeds):")
    for k, v in summary.items():
        log(f"  {k:16s} energy {v['energy_kwh']['mean']}+/-{v['energy_kwh']['std']} | "
            f"move {v['movement_per_step_deg']['mean']}+/-{v['movement_per_step_deg']['std']} | "
            f"rev {v['reversals']['mean']}+/-{v['reversals']['std']}  (n={v['energy_kwh']['n']})")
    log("=========== ALL DONE -> revision_seeds_results.json ===========")
    DONE.write_text("done", encoding="utf-8")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + repr(e)); log(traceback.format_exc())
        DONE.write_text("error", encoding="utf-8"); raise
