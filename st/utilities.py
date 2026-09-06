try:
    import simplejson as json
except Exception:  # pragma: no cover
    import json

import copy
import inspect
import random
import sys
from datetime import datetime
from pathlib import Path

def read_json(filepath: str, **kwargs):
    """Return json document as dictionary.
    
    Parameters
    ----------
    filepath : str
       pathname of JSON document.

    Other Parameters
    ----------------
    **kwargs : dict
        Other infrequently used keyword arguments to be parsed to `simplejson.load`.
       
    Returns
    -------
    dict
        JSON document converted to dictionary.
    """

    with open(filepath) as f:
        json_file = json.load(f,**kwargs)

    return json_file

def write_json(filepath: str, dictionary: dict, **kwargs):
    """Write dictionary to json file. Returns boolen value of operation success. 
    
    Parameters
    ----------
    filepath : str
        pathname of JSON document.
    dictionary: dict
        dictionary to convert to JSON.

    Other Parameters
    ----------------
    **kwargs : dict
        Other infrequently used keyword arguments to be parsed to `simplejson.dump`.
    """

    kwargs = {'ignore_nan':True,'sort_keys':False,'default':str,'indent':2,**kwargs}
    with open(filepath,'w') as f:
        json.dump(dictionary,f,**kwargs)


def package_root() -> Path:
    """Return the ``st`` package directory."""

    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the parent directory that contains the movable ``st`` folder."""

    return package_root().parent


def models_root() -> Path:
    """Return the canonical checkpoint directory, ``st/models``."""

    return package_root() / "models"


def current_schema() -> dict:
    """Return the current canonical solar-tracker schema from ``st/data``."""

    schema_path = package_root() / "data" / "schema.json"
    schema = read_json(str(schema_path))
    schema["root_directory"] = str(schema_path.parent)
    return schema


def refresh_schema_from_current(schema_template: dict) -> dict:
    """Update a saved/checkpoint schema with current data and panel settings.

    Older checkpoints may contain stale panel parameters. This helper preserves
    observations, actions, reward settings, and period values from the supplied
    schema while replacing data root and panel specs with the current project
    schema so loaded models evaluate on the same PV module as new runs.
    """

    schema_run = copy.deepcopy(schema_template)
    canonical = current_schema()
    schema_run["root_directory"] = canonical["root_directory"]
    schema_run["data"] = copy.deepcopy(canonical.get("data", schema_run.get("data", {})))
    schema_run["panel"] = copy.deepcopy(canonical.get("panel", schema_run.get("panel", {})))
    return schema_run


def list_model_folders(root: str = None):
    """Print and return saved model folder names under ``st/models``."""

    root_path = Path(root) if root is not None else models_root()
    root_path.mkdir(parents=True, exist_ok=True)
    folders = sorted(path.name for path in root_path.iterdir() if path.is_dir())

    if folders:
        print("Saved model folders:")
        for name in folders:
            print(f"- {name}")
    else:
        print(f"No saved model folders found in {root_path}")

    return folders


def _caller_globals():
    return inspect.currentframe().f_back.f_back.f_globals


def _json_safe(value):
    _ensure_training_imports()
    np = sys.modules["numpy"]
    torch = sys.modules["torch"]

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _unique_model_dir(folder_name: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / str(folder_name)

    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{folder_name}_{stamp}"
    candidate.mkdir(parents=True)
    return candidate


def _replay_state(memory):
    if memory is None:
        return None

    return {
        "capacity": int(memory.capacity),
        "position": int(memory.position),
        "buffer": memory.buffer,
    }


def _active_observation_names(schema_template):
    return [
        name for name, metadata in schema_template.get("observations", {}).items()
        if metadata.get("active", False)
    ]


def _agent_policy_input_dim(agent):
    first_parameter = next(agent.policy.parameters())
    return int(first_parameter.shape[1])


def _validate_checkpoint_observation_dim(agent, schema_template, context):
    active_observations = _active_observation_names(schema_template)
    expected_dim = len(active_observations)
    actual_dim = _agent_policy_input_dim(agent)

    if expected_dim != actual_dim:
        raise RuntimeError(
            f"{context} observation mismatch: schema has {expected_dim} active observations "
            f"but the Meta-SAC policy expects {actual_dim} inputs. "
            "Rebuild the environment/agent from the same schema before saving or loading. "
            f"Active observations: {active_observations}"
        )


def _meta_sac_state_dict(agent):
    state = {
        "policy": agent.policy.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "policy_optim": agent.policy_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict(),
        "alpha_optim": agent.alpha_optim.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "history_meta_grad": getattr(agent, "history_meta_grad", None),
    }

    if getattr(agent, "meta_Q", False):
        state.update(
            {
                "meta_critic": agent.meta_critic.state_dict(),
                "meta_critic_optim": agent.meta_critic_optim.state_dict(),
                "meta_v": agent.meta_v.state_dict(),
                "meta_v_optim": agent.meta_v_optim.state_dict(),
                "meta_v_target": agent.meta_v_target.state_dict(),
            }
        )

    return state


def save_model(folder_name: str):
    """Save the current notebook Meta-SAC run to ``st/models/<folder_name>``.

    This convenience function reads the common training variables from the
    calling notebook: ``tracker_agent``, ``meta_sac_tracker_config``,
    ``schema_run``, replay buffers, counters, and training history.
    """

    _ensure_training_imports()
    pd = sys.modules["pandas"]
    torch = sys.modules["torch"]
    caller = _caller_globals()

    required = ["tracker_agent", "meta_sac_tracker_config", "schema_run"]
    missing = [name for name in required if name not in caller]
    if missing:
        raise RuntimeError("Cannot save model. Missing notebook variables: " + ", ".join(missing))

    run_dir = _unique_model_dir(folder_name, models_root())
    config = copy.deepcopy(dict(caller["meta_sac_tracker_config"]))
    schema_template = copy.deepcopy(caller["schema_run"])
    _validate_checkpoint_observation_dim(caller["tracker_agent"], schema_template, "save_model")
    training_history = caller.get("tracker_training_history")
    if training_history is None and "tracker_training_df" in caller:
        training_history = caller["tracker_training_df"]

    checkpoint = {
        "format_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "agent_state": _meta_sac_state_dict(caller["tracker_agent"]),
        "config": config,
        "schema_template": schema_template,
        "simulation_period": {
            "simulation_start_time_step": int(caller.get("SIM_START_TIME_STEP", schema_template.get("simulation_start_time_step", 0))),
            "simulation_end_time_step": int(caller.get("SIM_END_TIME_STEP", schema_template.get("simulation_end_time_step", 0))),
            "sim_days": int(caller.get("SIM_DAYS", 0)),
        },
        "training_counters": {
            "total_numsteps": int(caller.get("total_numsteps", 0)),
            "updates": int(caller.get("updates", 0)),
            "best_episode": caller.get("best_episode"),
            "best_metric_name": caller.get("best_metric_name"),
            "best_metric_value": caller.get("best_metric_value"),
        },
        "best_policy_state": caller.get("best_policy_state"),
        "best_log_alpha": caller.get("best_log_alpha"),
        "replay_memory": _replay_state(caller.get("tracker_memory")),
        "kl_replay_memory": _replay_state(caller.get("tracker_kl_memory")),
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")

    active_observations = _active_observation_names(schema_template)
    metadata = {
        "created_at": checkpoint["created_at"],
        "folder_name": run_dir.name,
        "config": config,
        "simulation_period": checkpoint["simulation_period"],
        "training_counters": checkpoint["training_counters"],
        "active_observations": active_observations,
        "observation_count": len(active_observations),
        "policy_input_dim": _agent_policy_input_dim(caller["tracker_agent"]),
        "action_space": schema_template.get("actions", {}),
        "saved_replay_buffers": True,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_safe), encoding="utf-8")
    (run_dir / "schema.json").write_text(json.dumps(schema_template, indent=2, default=_json_safe), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=_json_safe), encoding="utf-8")

    if training_history is not None:
        if hasattr(training_history, "to_csv"):
            training_history.to_csv(run_dir / "training_history.csv", index=False)
        else:
            pd.DataFrame(training_history).to_csv(run_dir / "training_history.csv", index=False)

    caller["checkpoint_dir"] = run_dir
    print(f"Meta-SAC checkpoint saved to: {run_dir}")
    return run_dir


def _coerce_config_value(value: str):
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value.strip("\"").strip("'")


def load_flat_config(filepath):
    """Load the flat YAML config files used by the Meta-SAC examples."""

    config = {}
    for raw_line in Path(filepath).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        config[key.strip()] = _coerce_config_value(value)

    return config


def _ensure_training_imports():
    """Import training dependencies and local Meta-SAC modules lazily."""

    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import torch  # noqa: F401

    meta_sac_dir = package_root() / "meta_sac"
    if str(meta_sac_dir) in sys.path:
        sys.path.remove(str(meta_sac_dir))
    sys.path.insert(0, str(meta_sac_dir))

    for module_name in ["sacmeta", "modelmeta", "replay_memory", "utils"]:
        if module_name in sys.modules and getattr(sys.modules[module_name], "__file__", "").find(str(meta_sac_dir)) == -1:
            sys.modules.pop(module_name, None)


class SolarTrackerMetaSACWrapper:
    """Gym-like adapter around the CityLearn-style SolarTrackerEnv."""

    def __init__(self, schema_template, reward_scale=50.0):
        _ensure_training_imports()
        np = sys.modules["numpy"]
        from gym import spaces
        from st.environment import SolarTrackerEnv

        self.schema_template = copy.deepcopy(schema_template)
        self.reward_scale = float(reward_scale)
        self.env = None
        self.observation_names = None
        self.obs_low = None
        self.obs_high = None
        self.action_space = None
        self.observation_space = None
        self.last_history = None
        self._np = np
        self._spaces = spaces
        self._SolarTrackerEnv = SolarTrackerEnv
        self._build_env()

    def _build_env(self):
        schema_episode = copy.deepcopy(self.schema_template)
        self.env = self._SolarTrackerEnv(schema=schema_episode)
        self.observation_names = self.env.observation_names[0]
        self.obs_low = self._np.asarray(self.env.observation_space[0].low, dtype=self._np.float32)
        self.obs_high = self._np.asarray(self.env.observation_space[0].high, dtype=self._np.float32)
        action_low = self._np.asarray(self.env.action_space[0].low, dtype=self._np.float32)
        action_high = self._np.asarray(self.env.action_space[0].high, dtype=self._np.float32)
        self.action_space = self._spaces.Box(low=action_low, high=action_high, dtype=self._np.float32)
        self.observation_space = self._spaces.Box(
            low=self._np.full_like(self.obs_low, -1.0, dtype=self._np.float32),
            high=self._np.full_like(self.obs_high, 1.0, dtype=self._np.float32),
            dtype=self._np.float32,
        )

    def _normalize_obs(self, obs):
        obs = self._np.asarray(obs, dtype=self._np.float32)
        denom = self._np.where(self._np.abs(self.obs_high - self.obs_low) < 1e-6, 1.0, self.obs_high - self.obs_low)
        scaled = 2.0 * (obs - self.obs_low) / denom - 1.0
        return self._np.nan_to_num(self._np.clip(scaled, -1.0, 1.0)).astype(self._np.float32)

    def reset(self):
        obs = self.env.reset()[0]
        return self._normalize_obs(obs)

    def set_reward_episode(self, episode):
        if hasattr(self.env.reward_function, "set_episode"):
            self.env.reward_function.set_episode(int(episode))

    def reward_stage(self):
        if hasattr(self.env.reward_function, "get_weights"):
            return self.env.reward_function.get_weights()
        return {"stage": "default"}

    def step(self, action):
        action = self._np.asarray(action, dtype=self._np.float32)
        obs, reward, done, info = self.env.step([action])
        self.last_history = self.env.history.copy()
        scaled_reward = float(reward[0]) * self.reward_scale
        return self._normalize_obs(obs[0]), scaled_reward, bool(done), info

    def evaluate(self, include_baselines=False):
        return self.env.evaluate(include_baselines=include_baselines)

    @property
    def history(self):
        return self.env.history.copy()


def _daily_time_steps(schema_template):
    seconds_per_time_step = float(schema_template.get("seconds_per_time_step", 600.0))
    if seconds_per_time_step <= 0:
        seconds_per_time_step = 600.0
    return int(round(24.0 * 60.0 * 60.0 / seconds_per_time_step))


def set_tracker_simulation_period(schema_template, start_time_step=None, end_time_step=None):
    schema_run = copy.deepcopy(schema_template)
    if start_time_step is None:
        start_time_step = int(schema_run.get("simulation_start_time_step", 0))
    if end_time_step is None:
        end_time_step = int(schema_run.get("simulation_end_time_step", start_time_step))

    schema_run["simulation_start_time_step"] = int(start_time_step)
    schema_run["simulation_end_time_step"] = int(end_time_step)
    schema_run["episode_time_steps"] = int(end_time_step) - int(start_time_step) + 1
    return schema_run


def _restore_replay_memory(memory_class, state, fallback_capacity):
    memory = memory_class(int(state.get("capacity", fallback_capacity)) if state else int(fallback_capacity))
    if state:
        memory.buffer = state.get("buffer", [])
        memory.position = int(state.get("position", 0))
    return memory


def _move_optimizer_state_to_device(optimizer, device):
    torch = sys.modules["torch"]
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _apply_optimizer_lrs(agent, config):
    if "lr" in config:
        for optim in [agent.policy_optim, agent.critic_optim]:
            for group in optim.param_groups:
                group["lr"] = float(config["lr"])
    if "meta_lr" in config:
        for group in agent.alpha_optim.param_groups:
            group["lr"] = float(config["meta_lr"])
        if getattr(agent, "meta_Q", False):
            for optim in [agent.meta_critic_optim, agent.meta_v_optim]:
                for group in optim.param_groups:
                    group["lr"] = float(config["meta_lr"])


def _load_meta_sac_state_dict(agent, checkpoint, load_optimizer_states=True):
    torch = sys.modules["torch"]
    state = checkpoint["agent_state"]
    agent.policy.load_state_dict(state["policy"])
    agent.critic.load_state_dict(state["critic"])
    agent.critic_target.load_state_dict(state["critic_target"])
    agent.log_alpha.data.copy_(state["log_alpha"].to(agent.device))

    if state.get("history_meta_grad") is not None:
        agent.history_meta_grad = state["history_meta_grad"]

    if getattr(agent, "meta_Q", False) and "meta_critic" in state:
        agent.meta_critic.load_state_dict(state["meta_critic"])
        agent.meta_v.load_state_dict(state["meta_v"])
        agent.meta_v_target.load_state_dict(state["meta_v_target"])

    if load_optimizer_states:
        agent.policy_optim.load_state_dict(state["policy_optim"])
        agent.critic_optim.load_state_dict(state["critic_optim"])
        agent.alpha_optim.load_state_dict(state["alpha_optim"])
        _move_optimizer_state_to_device(agent.policy_optim, agent.device)
        _move_optimizer_state_to_device(agent.critic_optim, agent.device)
        _move_optimizer_state_to_device(agent.alpha_optim, agent.device)

        if getattr(agent, "meta_Q", False) and "meta_critic_optim" in state:
            agent.meta_critic_optim.load_state_dict(state["meta_critic_optim"])
            agent.meta_v_optim.load_state_dict(state["meta_v_optim"])
            _move_optimizer_state_to_device(agent.meta_critic_optim, agent.device)
            _move_optimizer_state_to_device(agent.meta_v_optim, agent.device)

    return torch


def _print_hyperparameters(config):
    print("Saved training hyperparameters:")
    for key in sorted(config):
        print(f"{key}: {config[key]}")


def load_model(folder_name: str):
    """Load ``st/models/<folder_name>`` and prepare notebook variables."""

    _ensure_training_imports()
    np = sys.modules["numpy"]
    torch = sys.modules["torch"]
    from sacmeta import SAC_META
    from replay_memory import ReplayMemoryKL as ReplayMemory

    caller = _caller_globals()
    checkpoint_dir = models_root() / str(folder_name)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    saved_config = copy.deepcopy(checkpoint["config"])
    saved_config["cuda"] = 0 if torch.cuda.is_available() else -1
    schema_template = refresh_schema_from_current(checkpoint["schema_template"])
    period = checkpoint.get("simulation_period", {})
    schema_run = set_tracker_simulation_period(
        schema_template,
        start_time_step=period.get("simulation_start_time_step"),
        end_time_step=period.get("simulation_end_time_step"),
    )
    train_env = SolarTrackerMetaSACWrapper(schema_run, reward_scale=50.0)
    agent = SAC_META(train_env.observation_space.shape[0], train_env.action_space, saved_config)
    _validate_checkpoint_observation_dim(agent, schema_run, "load_model")
    _load_meta_sac_state_dict(agent, checkpoint, load_optimizer_states=True)
    _apply_optimizer_lrs(agent, saved_config)

    memory = _restore_replay_memory(ReplayMemory, checkpoint.get("replay_memory"), saved_config["replay_size"])
    kl_memory = _restore_replay_memory(ReplayMemory, checkpoint.get("kl_replay_memory"), saved_config["kl_replay_size"])
    counters = checkpoint.get("training_counters", {})

    caller.update(
        {
            "loaded_model_folder": str(folder_name),
            "loaded_checkpoint": checkpoint,
            "loaded_counters": counters,
            "saved_meta_sac_tracker_config": saved_config,
            "meta_sac_tracker_config": copy.deepcopy(saved_config),
            "schema_run": schema_run,
            "tracker_train_env": train_env,
            "tracker_agent": agent,
            "loaded_tracker_agent": agent,
            "loaded_schema_run": schema_run,
            "SolarTrackerMetaSACWrapper": SolarTrackerMetaSACWrapper,
            "evaluate_tracker_policy": evaluate_tracker_policy,
            "tracker_memory": memory,
            "tracker_kl_memory": kl_memory,
            "SIM_START_TIME_STEP": int(schema_run["simulation_start_time_step"]),
            "SIM_END_TIME_STEP": int(schema_run["simulation_end_time_step"]),
            "SIM_DAYS": max(1, int(round(schema_run["episode_time_steps"] / _daily_time_steps(schema_run)))),
            "total_numsteps": int(counters.get("total_numsteps", 0)),
            "updates": int(counters.get("updates", 0)),
        }
    )

    _print_hyperparameters(saved_config)
    print(f"Loaded model from: {checkpoint_dir}")
    print(f"Restored replay buffers: memory={len(memory)}, kl_memory={len(kl_memory)}")
    return checkpoint_dir


def metric_from_eval(evaluation, metric_name):
    np = sys.modules["numpy"]
    rows = evaluation[evaluation["cost_function"] == metric_name]
    return float(rows["value"].iloc[0]) if len(rows) else np.nan


def evaluate_tracker_policy(agent, schema_template):
    eval_env = SolarTrackerMetaSACWrapper(schema_template, reward_scale=50.0)
    state = eval_env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(state, eval=True, mode="running")
        state, _, done, _ = eval_env.step(action)
    evaluation = eval_env.evaluate(include_baselines=False)
    return eval_env.history, evaluation


def continue_loaded_model(
    use_saved_hyperparameters=True,
    simulation_start_time_step=None,
    simulation_end_time_step=None,
):
    """Continue training the loaded model using saved or current YAML hyperparameters."""

    _ensure_training_imports()
    np = sys.modules["numpy"]
    pd = sys.modules["pandas"]
    torch = sys.modules["torch"]
    from replay_memory import ReplayMemoryKL as ReplayMemory
    from sacmeta import SAC_META

    caller = _caller_globals()
    if "loaded_checkpoint" not in caller:
        raise RuntimeError("Load a model first with load_model(folder_name).")

    checkpoint = caller["loaded_checkpoint"]
    saved_config = copy.deepcopy(checkpoint["config"])
    if use_saved_hyperparameters:
        config = copy.deepcopy(saved_config)
    else:
        config = load_flat_config(package_root() / "meta_sac" / "configs" / "solar_tracker.yml")
        for locked_key in ["hidden_size", "meta_Q", "alpha_embedding"]:
            if locked_key in saved_config:
                config[locked_key] = saved_config[locked_key]

    config["cuda"] = 0 if torch.cuda.is_available() else -1
    schema_run = set_tracker_simulation_period(
        refresh_schema_from_current(checkpoint["schema_template"]),
        start_time_step=simulation_start_time_step or caller.get("SIM_START_TIME_STEP"),
        end_time_step=simulation_end_time_step or caller.get("SIM_END_TIME_STEP"),
    )

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    train_env = SolarTrackerMetaSACWrapper(schema_run, reward_scale=50.0)
    agent = SAC_META(train_env.observation_space.shape[0], train_env.action_space, config)
    _validate_checkpoint_observation_dim(agent, schema_run, "continue_loaded_model")
    _load_meta_sac_state_dict(agent, checkpoint, load_optimizer_states=use_saved_hyperparameters)
    _apply_optimizer_lrs(agent, config)

    memory = _restore_replay_memory(ReplayMemory, checkpoint.get("replay_memory"), config["replay_size"])
    kl_memory = _restore_replay_memory(ReplayMemory, checkpoint.get("kl_replay_memory"), config["kl_replay_size"])
    local_total_numsteps = int(caller.get("total_numsteps", 0))
    local_updates = int(caller.get("updates", 0))
    best_metric_name = str(config.get("best_metric", "total_harvested_energy_kwh"))
    best_metric_value = -np.inf
    best_policy_state = None
    best_log_alpha = None
    best_episode = None
    history = []

    for episode in range(1, int(config["num_episodes"]) + 1):
        train_env.set_reward_episode(episode)
        reward_stage = train_env.reward_stage()
        state = train_env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        indicators = np.zeros(6, dtype=np.float32)

        while not done:
            if local_total_numsteps < int(config.get("start_steps", 0)):
                action = train_env.action_space.sample()
                log_prob = np.array([1.0 / (2 ** train_env.action_space.shape[0])], dtype=np.float32)
            else:
                action, log_prob = agent.select_action(state)

            if len(memory) > int(config["batch_size"]):
                for _ in range(int(config["updates_per_step"])):
                    indicators += agent.update_parameters(memory, kl_memory, int(config["batch_size"]), local_updates)
                    local_updates += 1
            else:
                indicators[0] += agent.log_alpha.item()

            next_state, reward, done, _ = train_env.step(action)
            episode_steps += 1
            local_total_numsteps += 1
            episode_reward += reward
            mask = 0.0 if done else 1.0
            memory.push(state, action, log_prob, reward, next_state, mask)
            kl_memory.push(state, action, log_prob, reward, next_state, mask)
            state = next_state

        indicators = indicators / max(episode_steps, 1)
        episode_metrics = train_env.evaluate()
        metric_values = dict(zip(episode_metrics["cost_function"], episode_metrics["value"]))
        row = {
            "continued_episode": episode,
            "total_numsteps": local_total_numsteps,
            "episode_steps": episode_steps,
            "scaled_reward": episode_reward,
            "reward_stage": reward_stage.get("stage", "default"),
            "total_harvested_energy_kwh": metric_values.get("total_harvested_energy_kwh", np.nan),
            "tracking_efficiency": metric_values.get("tracking_efficiency", np.nan),
            "movement_magnitude_deg": metric_values.get("movement_magnitude_deg", np.nan),
            "tracking_smoothness_jerk_deg2": metric_values.get("tracking_smoothness_jerk_deg2", np.nan),
            "mean_tracking_error_deg": metric_values.get("mean_tracking_error_deg", np.nan),
            "log_alpha": float(indicators[0]),
            "critic_1_loss": float(indicators[1]),
            "critic_2_loss": float(indicators[2]),
            "policy_loss": float(indicators[3]),
            "meta_policy_loss": float(indicators[4]),
            "entropy": float(indicators[5]),
        }

        eval_interval = int(config.get("eval_interval", 10))
        if episode == 1 or episode % eval_interval == 0 or episode == int(config["num_episodes"]):
            _, deterministic_eval = evaluate_tracker_policy(agent, schema_run)
            deterministic_metrics = dict(zip(deterministic_eval["cost_function"], deterministic_eval["value"]))
            deterministic_score = metric_from_eval(deterministic_eval, best_metric_name)
            row[f"eval_{best_metric_name}"] = deterministic_score
            row["eval_total_harvested_energy_kwh"] = deterministic_metrics.get("total_harvested_energy_kwh", np.nan)
            row["eval_tracking_efficiency"] = deterministic_metrics.get("tracking_efficiency", np.nan)
            row["eval_movement_magnitude_deg"] = deterministic_metrics.get("movement_magnitude_deg", np.nan)
            row["eval_tracking_smoothness_jerk_deg2"] = deterministic_metrics.get("tracking_smoothness_jerk_deg2", np.nan)
            if deterministic_score > best_metric_value:
                best_metric_value = deterministic_score
                best_episode = episode
                best_policy_state = copy.deepcopy(agent.policy.state_dict())
                best_log_alpha = agent.log_alpha.detach().clone()

        history.append(row)
        print(
            f"Continue episode {episode:02d}/{int(config['num_episodes'])} | "
            f"energy={row['total_harvested_energy_kwh']:.3f} kWh | "
            f"movement={row['movement_magnitude_deg']:.1f} deg | "
            f"log_alpha={row['log_alpha']:.3f}"
        )

    if best_policy_state is not None:
        agent.policy.load_state_dict(best_policy_state)
        agent.log_alpha.data.copy_(best_log_alpha)
        print(f"Loaded best continued policy from episode {best_episode} based on {best_metric_name}={best_metric_value:.3f}.")

    continued_training_df = pd.DataFrame(history)
    caller.update(
        {
            "tracker_agent": agent,
            "tracker_train_env": train_env,
            "tracker_memory": memory,
            "tracker_kl_memory": kl_memory,
            "meta_sac_tracker_config": config,
            "schema_run": schema_run,
            "continued_training_df": continued_training_df,
            "tracker_training_df": continued_training_df,
            "tracker_training_history": history,
            "total_numsteps": local_total_numsteps,
            "updates": local_updates,
            "SIM_START_TIME_STEP": int(schema_run["simulation_start_time_step"]),
            "SIM_END_TIME_STEP": int(schema_run["simulation_end_time_step"]),
            "SIM_DAYS": max(1, int(round(schema_run["episode_time_steps"] / _daily_time_steps(schema_run)))),
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_policy_state": best_policy_state,
            "best_log_alpha": best_log_alpha,
            "best_episode": best_episode,
        }
    )
    return continued_training_df
