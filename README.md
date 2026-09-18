# RG-SAC: Residual-Gate Soft Actor-Critic with a Small-Language-Model Planner

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22809319.svg)](https://doi.org/10.5281/zenodo.22809319)

Code for *A Residual-Gate Soft Actor-Critic Controller with a Small-Language-Model
Planner for Dual-Axis Solar Tracking*.

A dual-axis solar tracker is controlled in three layers. An astronomical schedule
gives a smooth base orientation, a sun-position model computes the correction
toward the ideal orientation, and a SAC-Auto agent learns a per-axis gate
`g ∈ [0,1]` deciding how much of that correction to apply. Because `g = 0`
reproduces the schedule exactly, the classical controller is a structural lower
bound: the learned policy can add energy but never fall below the baseline.

An hourly planner classifies the weather regime and sets hard constraints
(per-hour authority limits, a deadband, and the base-pose mode). The planner is a small language model fine-tuned with LoRA.

## Environments

Two conda environments are needed. The RL stack (Python 3.7, torch 1.13 + CUDA
11.7, old `gym`) and the language model (Python 3.10, torch 2.5 + CUDA 12.1,
transformers 5.x) have mutually incompatible pins, so the model is served in a
subprocess rather than imported in-process.

```bash
# RL / simulation                      -> requirements.txt
conda create -n solar-rl python=3.7 -y
conda activate solar-rl
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# language model / planner             -> requirements-llm.txt
conda create -n solar-llm python=3.10 -y
conda activate solar-llm
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-llm.txt
```

In both cases install `torch` first so pip resolves the CUDA build rather than
the CPU one.

`llm/local_client.py` starts `finetune/serve_inference.py` in the second
environment. Point it there with an environment variable rather than editing any
source file:

```bash
# Windows
set SLM_PYTHON=C:\path	o\envs\solar-llm\python.exe
# Linux / macOS
export SLM_PYTHON=~/miniconda3/envs/solar-llm/bin/python
```

If it is unset, the current interpreter is used when it can import
`transformers`; otherwise the client raises an error naming the variable.

| variable | purpose |
|---|---|
| `SLM_PYTHON` | Python 3.10 interpreter with transformers and peft |
| `RL_PYTHON` | Python 3.7 RL interpreter (chain scripts only; defaults to the running one) |
| `SOLAR_SLM_ADAPTER` | which fine-tuned adapter `serve_inference.py` loads |

## Data

Irradiance data is NREL NSRDB for Gwangju, South Korea, and is **not
redistributed here**. Download it from <https://nsrdb.nrel.gov/> and place the CSV
in `data/`. `data/schema.json` defines the panel, the bounds and the reward
weights. `finetune/preprocess_csv.py` adds the look-ahead columns and
`finetune/precompute_env_cache.py` builds the step cache:

```
python finetune/precompute_env_cache.py data/2020.csv --schema data/schema.json
```

The evaluation set is 56 days (four two-week blocks: January, April, July,
October). Training of the planner excludes those days entirely, because the
oracle targets encode the realised future of the day they are computed on.

## Reproducing the results

Each script is resumable and writes a `*_DONE.txt` marker when it finishes.

| Result | Script | Cost |
|---|---|---|
| Multi-seed comparison, FSM baseline | `run_revision_seeds.py` | long (trains 2 seeds) |
| Hindsight-optimal targets, full year | `run_oracle_fullyear.py` | ~2.5 h |
| Wear-cost sensitivity sweep | `run_oracle_sweep.py` | ~2.5 h |
| Dataset build + LoRA fine-tune | `run_phase2_chain.py` | ~9 h |
| Control test (SLM planner) | `run_control_test.py` | ~2 h |
| Planner forecast sensitivity | `run_planner_forecast_sensitivity.py`, `run_slm_forecast_sens.py` | ~4 h |
| Instruction-conditioned training | `run_version_b.py` | ~17 h |
| Programmability evaluation | `run_programmability.py`, `run_prog_seeds.py` | ~2 h per arm |

Figures:

```
python plots/capture_histories.py --with-slm
python plots/make_paper_figures_v2.py      # figures 7 and 13
python plots/make_paper_figures_v2b.py     # figures 8-12 and 14
python plots/export_figures.py
```

## How the hindsight-optimal targets work

For every planning hour, each of the three candidate regimes (`hold`, `cloud`,
`clear`) is rolled forward in the simulator and scored as

```
net = harvested_energy − c_act × motor_activations
```

with `c_act = 3.79e-4` kWh per activation, the break-even value implied by the
rule-threshold planner's own energy/wear trade-off. The highest-scoring regime
becomes the training target for that hour. `finetune/oracle_goals.py` records the
raw components per regime, so the optimum can be re-derived offline for any wear
cost without re-running the oracle.

Hindsight is used **only** to build training labels. At inference the planner sees
telemetry alone, and the evaluation days are held out of training.

## Layout

```
environment.py, tracker.py, reward.py   simulation and PV model
train_utils.py                          residual-gate wrapper, training loop
eval_utils.py                           rollouts and metrics
meta_sac/                               SAC-Auto agent and config
llm/                                    planner: prompts, guardrails, routing
  goal_guidance.py                        guardrail modes, prompt versions
finetune/                               target generation and LoRA fine-tuning
  oracle_goals.py                         hindsight-optimal regime search
plots/                                  figure generation
```

## Notes

`llm/goal_guidance.py` exposes `guardrail_mode` (`full` / `partial` / `loose`) and
`prompt_version` (1 = the rule-threshold prompt, 2 = the ladder-free prompt used
with oracle targets). The defaults reproduce the earlier published behaviour.
