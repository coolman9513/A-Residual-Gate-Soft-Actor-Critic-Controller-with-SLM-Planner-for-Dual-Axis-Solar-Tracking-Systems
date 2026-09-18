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

### Source

Irradiance and meteorological data come from the **NREL National Solar Radiation
Database (NSRDB)**, Physical Solar Model v3. The files are **not redistributed
here**; download them from <https://nsrdb.nrel.gov/> and place them in `data/`.

| field | value |
|---|---|
| Database | NSRDB, Physical Solar Model (PSM) v3 |
| NSRDB version string | `3.1.1` (as recorded in the file header) |
| Location ID | `5771095` |
| Latitude, longitude | 35.20° N, 126.85° E |
| Elevation | 25 m |
| Region | Jeollanam-do, South Korea (grid cell adjacent to Gwangju) |
| Time zone | UTC+9 |
| Year | 2020 |
| Temporal resolution | 10 minutes |
| Records | 52,560 (full year) |

The raw download is named `5771095_35.20_126.85_2020.csv` and carries the two
NSRDB metadata header rows above the column header.

NSRDB is published under a Creative Commons Attribution licence; credit
DOE/NREL/ALLIANCE when reusing it. Cite the database as:

> Sengupta, M., Xie, Y., Lopez, A., Habte, A., Maclaurin, G., Shelby, J. (2018).
> The National Solar Radiation Data Base (NSRDB). *Renewable and Sustainable
> Energy Reviews*, 89, 51–60. https://doi.org/10.1016/j.rser.2018.03.003

### Columns

Taken directly from the NSRDB download:

`Year`, `Month`, `Day`, `Hour`, `Minute`, `Temperature`, `Dew Point`, `DHI`,
`DNI`, `GHI`, `Surface Albedo`, `Pressure`, `Wind Direction`, `Wind Speed`,
`Relative Humidity`, `Solar Zenith Angle`, `Cloud Type`, `Fill Flag`, `Ozone`,
`Clearsky GHI`, `Clearsky DNI`, `Clearsky DHI`

Added by `finetune/prepare_nsrdb.py`:

| column | meaning |
|---|---|
| `Solar Azimuth Angle` | computed from the timestamp and site coordinates; NSRDB supplies zenith but not azimuth |
| `Datetime` | parsed timestamp |
| `next_10min_solar_azimuth`, `next_10min_solar_zenith` | deterministic solar geometry one step ahead |
| `next_10min_DNI`, `next_10min_DHI` | look-ahead irradiance, one step |
| `next_30min_average_DNI` | mean DNI over the next three steps |

The look-ahead irradiance columns are an idealised nowcast taken from the
realised series; `run_forecast_sensitivity.py` and
`run_planner_forecast_sensitivity.py` quantify how much the results depend on
them.

### Preparation

Three steps, in order:

```bash
# 1. raw NSRDB download -> data/2020.csv and the 56-day evaluation subset
python finetune/prepare_nsrdb.py 5771095_35.20_126.85_2020.csv --subset

# 2. optional, planner fine-tuning only: adds noisy 20-60 min horizons
#    -> data/2020_enriched.csv
python finetune/preprocess_csv.py

# 3. optional: precompute per-step constants, roughly halves env runtime
python finetune/precompute_env_cache.py data/2020.csv --schema data/schema.json
```

Step 1 is required; the environment reads `data/2020.csv`. Step 2 is needed only
to regenerate the planner training set. `data/schema.json` defines the panel
model, the mechanical bounds and the reward weights.

### Evaluation subset

`data/2020_4months_2weeks.csv` holds **56 days / 8,064 records**: four two-week
blocks, days 9–22 of January, April, July and October, chosen to sample all four
seasons. Planner training excludes those days entirely, because the oracle
targets encode the realised future of the day they are computed on.

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

## Code Information

All code sits at the repository root; there is no package directory to `cd` into.
Scripts are run from the root, for example `python run_control_test.py`.

### Simulation environment

| file | contents |
|---|---|
| `environment.py` | `SolarTrackerEnv`, a CityLearn-style environment for one dual-axis tracker: observations, stepping, the PV power model and the tracking-window logic |
| `tracker.py` | solar geometry and irradiance: incidence cosine, plane-of-array irradiance from DNI plus isotropic DHI, and the panel power/energy conversion |
| `reward.py` | `SolarTrackerReward`: energy, peak-weighted tracking efficiency, and the linear, quadratic and smoothness movement penalties |
| `schema.py` | loads and normalises `data/schema.json` into the environment configuration |
| `data.py` | dataset helpers that resolve and read the CSV assets under `data/` |
| `metrics.py` | tabular evaluation metrics in CityLearn's format |
| `utilities.py` | JSON I/O, project/model path resolution and schema refresh helpers |

### Controllers

| file | contents |
|---|---|
| `agents.py` | baseline controllers: the fixed panel and the rule-based tracker |
| `rbc.py` | the astronomical rule-based schedule that forms the residual-gate base, plus `SolarTrackerRBCEditor` for inspecting it interactively |
| `train_utils.py` | `ResidualTrackerWrapper` (base action, gate, authority limits, safety layer), the environment factory and the SAC training loop |
| `eval_utils.py` | rollouts for every controller and the energy, movement, activation and reversal metrics |
| `checkpointing.py` | saving and restoring agent state, config, schema and training history |
| `meta_sac/` | the SAC-Auto agent: `meta_sac/sacauto.py` (automatic entropy tuning), `meta_sac/modelmeta.py` (networks), `meta_sac/replay_memory.py` (uniform and KL replay buffers), `meta_sac/sacmeta.py`, `meta_sac/utils.py` |

### Planner

| file | contents |
|---|---|
| `llm/goal_guidance.py` | `LLMGoalGuidance`: hourly planning calls, `guardrail_mode` (`full`/`partial`/`loose`), `prompt_version`, the operator directive, caching and the rule fallback |
| `llm/goal_prompts.py` | prompt builders. v1 carries the DNI-threshold decision ladder; v2 is ladder-free and takes an operator directive. `WEAR_DIRECTIVES` holds the directives used for both training and evaluation |
| `llm/goal_wrapper.py` | telemetry construction and the goal-conditioned observation channels |
| `llm/regime_router.py` | `route_goal` and the authority table mapping a regime to per-step limits and an hourly budget |
| `llm/parser.py` | JSON extraction and validation of planner responses |
| `llm/local_client.py` | serves the fine-tuned adapter through a subprocess; interpreter from `SLM_PYTHON` |
| `llm/client.py` | optional OpenAI-compatible client for an LM Studio endpoint |

### Data preparation and fine-tuning

| file | contents |
|---|---|
| `finetune/prepare_nsrdb.py` | raw NSRDB export to `data/2020.csv`, plus the 56-day evaluation subset |
| `finetune/preprocess_csv.py` | adds noisy 20-60 minute horizons for planner training (`data/2020_enriched.csv`) |
| `finetune/precompute_env_cache.py` | precomputes per-step constants into an `.npz` cache |
| `finetune/oracle_goals.py` | the hindsight-optimal regime search, including the exact environment snapshot and restore |
| `finetune/generate_dataset.py` | rule-threshold targets (the original planner) |
| `finetune/generate_oracle_dataset.py` | oracle targets, evaluation days held out |
| `finetune/generate_instruction_dataset.py` | instruction-conditioned targets: identical telemetry, three directives, three answers |
| `finetune/train.py` | LoRA fine-tuning loop |
| `finetune/serve_inference.py` | stdin/stdout inference server; adapter from `SOLAR_SLM_ADAPTER` |
| `finetune/eval_slm_regime.py` | planner regime accuracy on the held-out split |
| `finetune/evaluate.py` | adapter diagnostics: parse rate, accuracy, inference speed |

### Experiment scripts

| file | produces |
|---|---|
| `run_revision_seeds.py` | multi-seed comparison and the rule-threshold baseline |
| `run_oracle_fullyear.py` | full-year oracle targets |
| `run_oracle_sweep.py` | wear-cost sensitivity sweep |
| `run_phase2_chain.py` | dataset build and LoRA fine-tune, chained |
| `run_version_b.py` | instruction-conditioned training, end to end |
| `run_control_test.py` | oracle-supervised planner versus the rule-threshold planner |
| `run_programmability.py` | directive-switching evaluation, one seed |
| `run_prog_seeds.py` | the same across seeds |
| `run_forecast_sensitivity.py` | noise in the low-level policy's look-ahead inputs |
| `run_planner_forecast_sensitivity.py` | noise in the planner's own forecast |
| `run_slm_forecast_sens.py` | the same for the fine-tuned planner |

### Figures

`plots/capture_histories.py` records rollout histories;
`plots/make_paper_figures_v2.py` and `plots/make_paper_figures_v2b.py` draw the
figures; `plots/export_figures.py` writes them out.

### Configuration

| file | contents |
|---|---|
| `data/schema.json` | panel model, mechanical bounds, observation set, reward weights, baselines |
| `meta_sac/configs/solar_tracker.yml` | SAC hyperparameters |
| `requirements.txt`, `requirements-llm.txt` | the two environments |
| `setup.py` | optional editable install; reads `requirements.txt` |
| `llm/__init__.py` | re-exports the planner classes |

## Notes

`llm/goal_guidance.py` exposes `guardrail_mode` (`full` / `partial` / `loose`) and
`prompt_version` (1 = the rule-threshold prompt, 2 = the ladder-free prompt used
with oracle targets). The defaults reproduce the earlier published behaviour.
