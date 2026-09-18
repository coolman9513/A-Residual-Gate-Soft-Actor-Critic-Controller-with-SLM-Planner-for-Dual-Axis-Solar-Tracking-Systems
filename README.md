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

## Citation

### This software

Archived on Zenodo with a DOI that resolves to the latest version:

```bibtex
@software{esanov_rgsac_code,
  author    = {Esanov, Alibek Rustamovich and Lim, Chang Gyoon},
  title     = {A Residual-Gate Soft Actor-Critic Controller with Small Language
               Model Planner for Dual-Axis Solar Tracking Systems},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22809319},
  url       = {https://doi.org/10.5281/zenodo.22809319}
}
```

`CITATION.cff` carries the same metadata, so GitHub's *Cite this repository*
button produces this automatically.

### The paper

```bibtex
@article{esanov_rgsac_paper,
  author  = {Esanov, Alibek Rustamovich and Lim, Chang Gyoon},
  title   = {A Residual-Gate Soft Actor-Critic Controller with a
             Small-Language-Model Planner for Dual-Axis Solar Tracking},
  journal = {PeerJ Computer Science},
  note    = {Under review}
}
```

Update the volume, pages and DOI here and in `CITATION.cff` once it is published.

### Data

The irradiance and meteorological data are NREL/NSRDB and are not redistributed
with this code. Credit DOE/NREL/ALLIANCE and cite:

```bibtex
@article{sengupta2018nsrdb,
  author  = {Sengupta, Manajit and Xie, Yu and Lopez, Anthony and Habte, Aron
             and Maclaurin, Galen and Shelby, James},
  title   = {The National Solar Radiation Data Base ({NSRDB})},
  journal = {Renewable and Sustainable Energy Reviews},
  volume  = {89},
  pages   = {51--60},
  year    = {2018},
  doi     = {10.1016/j.rser.2018.03.003}
}
```

### Base model

The planner fine-tunes `Qwen/Qwen2.5-0.5B-Instruct`, licensed Apache-2.0. The
LoRA adapters are derivative works of it.

```bibtex
@article{qwen2025qwen25,
  author  = {{Qwen Team}},
  title   = {Qwen2.5 Technical Report},
  journal = {arXiv preprint arXiv:2412.15115},
  year    = {2025},
  url     = {https://arxiv.org/abs/2412.15115}
}
```

The simulation environment builds on CityLearn, which supplies the `Environment`
and `EpisodeTracker` base classes; see its own repository for its citation.

## Requirements

### Hardware

Measured on the machine the reported results were produced on: Intel i7-12700H
(20 logical cores), 48 GB RAM, NVIDIA RTX 3070 Ti Laptop (8 GB VRAM), Windows 11.

| stage | VRAM | notes |
|---|---|---|
| LoRA fine-tuning | **8 GB** | peaks at 7.98 GB with batch size 2 at 1216 tokens; a smaller card needs a shorter sequence, a smaller batch, or 4-bit quantisation |
| Planner inference | ~2 GB | one 0.5B model in fp16 |
| SAC training and evaluation | <1 GB | the policy is a 2x512 MLP |

RAM is modest: the full-year environment peaks around 280 MB, and the simulation
is single-threaded, so more cores only help when running experiments in parallel.
A GPU is required for fine-tuning and makes planner evaluation practical;
everything else runs on CPU, slower.

Disk: about 30 MB for the NSRDB download and its derived CSVs, and about 400 MB
for the checkpoints, adapters and fine-tuning datasets a full run produces.

### Software

Two environments, because the RL stack and the language model have mutually
incompatible pins. Exact versions are pinned in the two files; the table lists the
ones that matter.

| | RL / simulation | language model / planner |
|---|---|---|
| file | [`requirements.txt`](requirements.txt) | [`requirements-llm.txt`](requirements-llm.txt) |
| Python | 3.7 | 3.10 |
| PyTorch | 1.13.1 + CUDA 11.7 | 2.5.1 + CUDA 12.1 |
| key packages | CityLearn 2.1.2, gym 0.26.2, gymnasium 0.28.1, numpy 1.21.6, pandas 1.3.5, scikit-learn 1.0.2, pvlib 0.10.4 | transformers 5.9.0, peft 0.19.1, accelerate 1.13.0, tokenizers 0.22.2, numpy 1.26.4 |
| used by | environment, controller, all `run_*.py` | `finetune/train.py`, `finetune/serve_inference.py` |

`llm/client.py` can also talk to an LM Studio endpoint; the `openai` packages for
that are listed, commented out, at the end of `requirements.txt`.

Installation commands are in [Environments](#environments).

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
set SLM_PYTHON=C:\path\to\envs\solar-llm\python.exe
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
DOE/NREL/ALLIANCE when reusing it. The citation is in [Citation](#citation).

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

Three scripts turn the download into what the code reads. The commands, in order,
are in [Usage](#usage).

| script | produces | needed for |
|---|---|---|
| `finetune/prepare_nsrdb.py` | `data/2020.csv` and the 56-day subset | everything |
| `finetune/preprocess_csv.py` | `data/2020_enriched.csv`, noisy 20-60 min horizons | regenerating the planner training set only |
| `finetune/precompute_env_cache.py` | `*_precomputed.npz` step cache | optional; roughly halves environment runtime |

`data/schema.json` defines the panel model, the mechanical bounds and the reward
weights.

### Evaluation subset

`data/2020_4months_2weeks.csv` holds **56 days / 8,064 records**: four two-week
blocks, days 9–22 of January, April, July and October, chosen to sample all four
seasons. Planner training excludes those days entirely, because the oracle
targets encode the realised future of the day they are computed on.

## Usage

End to end, from a fresh clone. Timings are wall-clock on an RTX 3070 Ti Laptop
(8 GB) with a 20-core CPU.

### 1. Create the two environments

See [Environments](#environments). The RL stack and the planner have
incompatible pins, so both are needed. Point the planner interpreter at the
second one:

```bash
export SLM_PYTHON=/path/to/envs/solar-llm/bin/python   # Windows: set SLM_PYTHON=...
```

### 2. Get the data

Download the NSRDB grid cell described in [Data](#data) from
<https://nsrdb.nrel.gov/> — location `5771095`, 35.20 N / 126.85 E, year 2020,
10-minute resolution — then build the files the code reads:

```bash
python finetune/prepare_nsrdb.py 5771095_35.20_126.85_2020.csv --subset
```

This writes `data/2020.csv` (full year) and `data/2020_4months_2weeks.csv`
(the 56-day evaluation set).

### 3. Build the step cache (optional, recommended)

```bash
python finetune/precompute_env_cache.py data/2020.csv --schema data/schema.json
python finetune/precompute_env_cache.py data/2020_4months_2weeks.csv --schema data/schema.json
```

Precomputes the per-step constants and roughly halves environment runtime. Every
script below works without it, just slower.

### 4. Train

**Controller.** Trains the residual-gate policy and the no-planner ablation for
three seeds, then evaluates both against the rule-threshold planner:

```bash
python run_revision_seeds.py            # ~11 h per seed
```

Checkpoints already present under `models/` are reused instead of retrained, so
restoring `policies/` from the Zenodo archive skips this step entirely.

**Planner.** Derive hindsight-optimal targets and fine-tune the adapter:

```bash
python run_oracle_fullyear.py           # ~3.1 h   full-year oracle targets
python run_phase2_chain.py              # ~12 h    dataset build + LoRA fine-tune
```

For the instruction-conditioned planner, `run_version_b.py` does the whole chain
in one go (~24 h): oracle, dataset, fine-tune and evaluation.

Restoring `adapters/` from the Zenodo archive skips the fine-tuning.

### 5. Evaluate

```bash
python run_control_test.py                      # ~3.6 h  planner vs rule-threshold
python run_programmability.py                   # ~1.2 h per directive
python run_prog_seeds.py                        # ~6.8 h  the same across seeds
python run_planner_forecast_sensitivity.py      # ~1.5 h  forecast robustness
python run_oracle_sweep.py                      # ~2.5 h  wear-cost sweep
```

Each writes JSON into `models/` and a `*_DONE.txt` marker, and skips work that is
already recorded, so an interrupted run can simply be restarted.

### 6. Figures

```bash
python plots/capture_histories.py --with-slm
python plots/make_paper_figures_v2.py      # figures 7 and 13
python plots/make_paper_figures_v2b.py     # figures 8-12 and 14
python plots/export_figures.py
```

### Shortcut

Running everything is roughly **50 GPU-hours**. The supplementary Zenodo archive
holds the result JSONs, the fine-tuned adapters and the trained policies, so a
reader can verify the reported tables without running anything, or re-run a
single stage without repeating the ones before it. See the archive's `MANIFEST.md`.

## Methodology

### Data processing

The NSRDB export gives irradiance, weather and solar **zenith** at 10-minute
resolution, but no solar azimuth and no look-ahead columns.
`finetune/prepare_nsrdb.py` adds both:

1. `Datetime` from the Year/Month/Day/Hour/Minute columns.
2. `Solar Azimuth Angle` from `pvlib.solarposition.get_solarposition` at
   35.20 N / 126.85 E, timestamps localised to `Asia/Seoul`.
3. Look-ahead columns by shifting the realised series one step
   (`next_10min_solar_azimuth`, `next_10min_solar_zenith`, `next_10min_DNI`,
   `next_10min_DHI`) and three steps (`next_30min_average_DNI`), forward-filled
   at the tail where no successor exists.

These look-ahead columns are an **idealised nowcast** taken from the realised
series rather than a forecast model.
`run_forecast_sensitivity.py` perturbs the low-level policy's look-ahead inputs
and `run_planner_forecast_sensitivity.py` perturbs the planner's own forecast, so
the dependence on that idealisation is measured rather than assumed.

For planner fine-tuning only, `finetune/preprocess_csv.py` additionally derives
20–60 minute horizons with ±2 % uniform noise.

### Evaluation set

`data/2020_4months_2weeks.csv`: **56 days, 8,064 records** — days 9–22 of
January, April, July and October, one block per season.

The split matters because the oracle targets are hindsight-derived: they encode
the realised future of the day they are computed on. Training the planner on an
evaluation day would therefore leak that day's future into the model. The dataset
builders drop every hour falling on those 56 days (1,344 hours), so the evaluation
period is unseen. Hours outside the 07:00–18:00 tracking window are also dropped,
since the planner is not consulted then.

### Oracle targets

For every planning hour, each of the three candidate regimes (`hold`, `cloud`,
`clear`) is rolled forward six 10-minute steps in the real simulator with the gate
fully open, and scored as

```
net = harvested_energy − c_act × motor_activations
```

`c_act = 3.79e-4` kWh per activation is the break-even value implied by the
rule-threshold planner's own energy/wear trade-off. The highest-scoring regime
becomes that hour's target; the winner is committed before moving to the next
hour, so each decision sees the pose its predecessors actually produced.

Stepping forward and rewinding requires an exact environment snapshot and
restore, verified to reproduce an uninterrupted rollout to full float precision
in energy, activations, degrees travelled and final pose.

`finetune/oracle_goals.py` records the raw per-regime components, so the optimum
for **any** wear coefficient can be re-derived offline without re-running the
oracle — which is how the instruction-conditioned targets at three wear settings
are produced from a single pass, on one committed trajectory, so telemetry is
identical across directives.

Hindsight is used **only** to build training labels. At inference the planner sees
telemetry alone.

### Planner fine-tuning

Base model: **Qwen2.5-0.5B-Instruct** (`Qwen/Qwen2.5-0.5B-Instruct`), 0.5 B
parameters, served in fp16. Not Qwen3, and not a larger variant: the size was
chosen for edge deployability.

| LoRA setting | value |
|---|---|
| rank `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| bias | none |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| trainable parameters | 8.8 M of 502.8 M (1.75 %) |

| training | oracle-supervised | instruction-conditioned |
|---|---|---|
| epochs | 4 | 2 |
| max sequence length | 1152 | 1216 |
| learning rate | 2e-4 | 2e-4 |
| optimiser | AdamW, weight decay 0 | AdamW, weight decay 0 |
| schedule | cosine, 5 % warmup | cosine, 5 % warmup |
| batch size x grad accumulation | 2 x 8 | 2 x 8 |
| train / val samples | 4,697 / 339 | 10,160 / 1,017 |
| oversampling | `clear` x10, `cloud` x3 | `clear` x3 |
| final training loss | 0.031 | 0.0098 |

The validation split is 10 %, taken **per planning hour before oversampling**, so
duplicated copies of an hour cannot straddle the split.

Sequence length is not a free parameter: the prompt plus reply runs to about 975
tokens, and the loss is masked to the assistant turn only. At the old 768-token
default every sample was truncated before its answer, leaving nothing supervised;
`finetune/train.py` now raises rather than training on such a batch.

Only `motion_budget_deg` and `hold` reach the controller, through the regime
router. The pose fields in the planner's output are observation context; the
residual always aims at the physics-derived ideal orientation.

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

## License

This code is released under the **MIT License**; see [`LICENSE`](LICENSE).

Three licences apply to different parts of a working setup, so they are listed
together:

| component | licence | note |
|---|---|---|
| this repository | MIT | code, scripts and documentation |
| `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | downloaded at runtime, not redistributed here; the LoRA adapters are derivative works of it |
| NREL/NSRDB data | CC BY | not redistributed here; credit DOE/NREL/ALLIANCE, see [Citation](#citation) |

## Contributing

This is research code accompanying a paper, so the priority is that the published
numbers stay reproducible rather than that the code keeps growing.

**Reporting a problem.** Open an issue and include:

- which environment you used, `solar-rl` or `solar-llm`, and its Python and
  PyTorch versions
- the exact command you ran
- the relevant part of the `*.log` file the script writes
- for a reproduction mismatch, the JSON the script wrote into `models/` next to
  the figure or table you expected

**Pull requests.** Fixes, portability improvements and documentation are welcome.
Two things to keep in mind:

- Results in the paper correspond to the tagged release. A change that alters a
  reported number is not a bug fix; please open an issue first so the difference
  can be understood before it is merged.
- The defaults in `llm/goal_guidance.py` (`guardrail_mode="full"`,
  `prompt_version=1`) reproduce the earlier published behaviour and should stay
  as they are; new behaviour belongs behind a new option.

Before submitting, check that the scripts you touched still import and start:

```bash
python -m compileall -q .
python run_programmability.py --help
```

## Notes

`llm/goal_guidance.py` exposes `guardrail_mode` (`full` / `partial` / `loose`) and
`prompt_version` (1 = the rule-threshold prompt, 2 = the ladder-free prompt used
with oracle targets). The defaults reproduce the earlier published behaviour.
