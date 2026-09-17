# -*- coding: utf-8 -*-
"""Capture per-step rollout histories so the results figures can be regenerated.

Figures 8, 9, 11, 12 and 14 of the manuscript are built from per-step data
(cumulative energy, daily energy, step-size distribution, hourly movement,
efficiency by irradiance regime). Only summary metrics were persisted by the
evaluation scripts, so the rollouts have to be replayed once with the history
written to disk.

The four non-LLM controllers cost a few minutes each and need no GPU, so they run
first and do not contend with any SLM job. The oracle-tuned SLM rollout costs about
an hour of GPU time and is gated behind --with-slm so it can be deferred until the
GPU is free.

Usage:
    python plots/capture_histories.py              # four cheap controllers
    python plots/capture_histories.py --with-slm   # add the oracle-tuned SLM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT), str(ROOT / "meta_sac")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("SOLAR_SLM_ADAPTER", str(ROOT / "finetune" / "output_oracle" / "adapter"))

import torch  # noqa: E402

import checkpointing  # noqa: E402
from sacauto import SAC_Auto  # noqa: E402
from utilities import load_flat_config  # noqa: E402
from train_utils import build_schema, make_residual_env  # noqa: E402
from eval_utils import (rollout_fixed_for_schema, rollout_rbc_for_schema,  # noqa: E402
                        rollout_residual_for_schema)

CSV = "2020_4months_2weeks.csv"
SEED = 2021
OUT = ROOT / "plots" / "paper" / "histories"
FINAL_KW = dict(action_mode="gate", authority_mode="goal", hold_tilt_deg=10.0,
                cloud_anticipate_lookahead=12, move_deadband_deg=2.0)
NOSLLM_KW = dict(action_mode="gate", authority_mode="fixed",
                 fixed_authority=(6.0, 4.0, float("inf")), hold_base=False,
                 cloud_anticipate_lookahead=0, move_deadband_deg=0.0,
                 disable_guidance=True)


def ctx():
    config = load_flat_config(ROOT / "meta_sac" / "configs" / "solar_tracker.yml")
    config["cuda"] = 0 if torch.cuda.is_available() else -1
    base = json.loads((ROOT / "data" / "schema.json").read_text())
    base["root_directory"] = str(ROOT / "data")
    return config, build_schema(base, CSV, 0, 8063)


def agent_for(config, schema, folder, env_kw):
    env = make_residual_env(schema, use_llm=False, seed=SEED, **env_kw)
    a = SAC_Auto(env.observation_space.shape[0], env.action_space,
                 {**config, "target_entropy": -3.0})
    _, ck = checkpointing.load_checkpoint(folder)
    checkpointing.load_agent_state(a, ck, load_optimizer_states=False)
    return a


def dump(hist, name):
    import pandas as pd
    df = pd.DataFrame(hist)
    path = OUT / ("%s.csv" % name)
    df.to_csv(path, index=False)
    print("  %-14s %5d rows -> %s" % (name, len(df), path.name), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-slm", action="store_true")
    ap.add_argument("--only-slm", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    config, schema = ctx()

    if not args.only_slm:
        t0 = time.time()
        h, _ = rollout_fixed_for_schema(schema)
        dump(h, "fixed")
        h, _ = rollout_rbc_for_schema(schema)
        dump(h, "rbc")

        a = agent_for(config, schema, "sac_auto_gate_nosllm", NOSLLM_KW)
        h, _, _ = rollout_residual_for_schema(a, schema, use_llm=False, seed=SEED,
                                              label="no-SLM", **NOSLLM_KW)
        dump(h, "nosllm")

        a = agent_for(config, schema, "sac_auto_gate_final", FINAL_KW)
        h, _, _ = rollout_residual_for_schema(a, schema, use_llm=False, seed=SEED,
                                              label="FSM+SAC", **FINAL_KW)
        dump(h, "fsm")
        print("  (four non-LLM controllers in %.0f min)" % ((time.time() - t0) / 60.0))

    if args.with_slm or args.only_slm:
        from llm.goal_guidance import LLMGoalGuidance
        from llm.local_client import LocalQwenClient
        t0 = time.time()
        g = LLMGoalGuidance(client=LocalQwenClient(max_new_tokens=120), enabled=True,
                            use_llm=True, guardrail_mode="partial", prompt_version=2)
        a = agent_for(config, schema, "sac_auto_gate_final", FINAL_KW)
        h, _, _ = rollout_residual_for_schema(a, schema, guidance=g, use_llm=True,
                                              seed=SEED, label="oracle-SLM", **FINAL_KW)
        dump(h, "oracle_slm")
        print("  (SLM in %.0f min)" % ((time.time() - t0) / 60.0))

    print("done -> %s" % OUT)


if __name__ == "__main__":
    main()
