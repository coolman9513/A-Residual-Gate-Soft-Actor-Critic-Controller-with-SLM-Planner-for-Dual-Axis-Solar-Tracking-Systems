"""Checkpoint helpers for the solar-tracker experiment notebook.

The functions in this module are intentionally notebook-friendly: they read
and write explicit objects passed from ``experiment.ipynb`` and save enough
artifacts to resume training or load previous results into comparison plots.
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


MODELS_ROOT = Path(__file__).resolve().parent / "models"


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def make_save_dir(folder_name: str, overwrite: bool = False) -> Path:
    """Create a model result directory under ``st/models``."""

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    save_dir = MODELS_ROOT / str(folder_name)

    if save_dir.exists():
        if overwrite:
            shutil.rmtree(save_dir)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = MODELS_ROOT / f"{folder_name}_{stamp}"

    save_dir.mkdir(parents=True, exist_ok=False)
    return save_dir


def active_observation_names(schema: dict) -> list:
    """Return active observation names from schema."""

    return [
        name for name, metadata in schema.get("observations", {}).items()
        if metadata.get("active", False)
    ]


def episodes_trained(training_df: Optional[pd.DataFrame]) -> int:
    """Return the highest episode number represented by a training dataframe."""

    if training_df is None or len(training_df) == 0:
        return 0

    if "episode" in training_df.columns:
        return int(pd.to_numeric(training_df["episode"], errors="coerce").max())

    if "continued_episode" in training_df.columns:
        return int(pd.to_numeric(training_df["continued_episode"], errors="coerce").max())

    return int(len(training_df))


def memory_state(memory: Any) -> Optional[dict]:
    """Serialize replay memory state."""

    if memory is None:
        return None

    return {
        "capacity": int(memory.capacity),
        "position": int(memory.position),
        "buffer": memory.buffer,
    }


def restore_memory(memory_cls: type, state: Optional[dict], fallback_capacity: int) -> Any:
    """Restore replay memory state."""

    memory = memory_cls(int(state.get("capacity", fallback_capacity)) if state else int(fallback_capacity))

    if state:
        memory.buffer = state.get("buffer", [])
        memory.position = int(state.get("position", 0))

    return memory


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensors to model device after loading state dicts."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def agent_state(agent: Any) -> dict:
    """Return trainable Meta-SAC state."""

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


def load_agent_state(agent: Any, checkpoint: dict, load_optimizer_states: bool = True) -> None:
    """Load trainable Meta-SAC state into an agent."""

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
        optimizer_to_device(agent.policy_optim, agent.device)
        optimizer_to_device(agent.critic_optim, agent.device)
        optimizer_to_device(agent.alpha_optim, agent.device)

        if getattr(agent, "meta_Q", False) and "meta_critic_optim" in state:
            agent.meta_critic_optim.load_state_dict(state["meta_critic_optim"])
            agent.meta_v_optim.load_state_dict(state["meta_v_optim"])
            optimizer_to_device(agent.meta_critic_optim, agent.device)
            optimizer_to_device(agent.meta_v_optim, agent.device)


def load_checkpoint(folder_name: str) -> tuple:
    """Load checkpoint from ``st/models/<folder_name>``."""

    checkpoint_dir = MODELS_ROOT / str(folder_name)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    return checkpoint_dir, checkpoint


def save_training_curves(training_df: pd.DataFrame, title: str, save_path: Path) -> Optional[Path]:
    """Save a compact training-curve figure."""

    df = pd.DataFrame(training_df).copy()

    if df.empty:
        return None

    if "episode" not in df.columns:
        df["episode"] = np.arange(1, len(df) + 1)

    def _series(column_name: str):
        if column_name in df.columns:
            return pd.to_numeric(df[column_name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(df["episode"], _series("total_harvested_energy_kwh"), marker="o", label="Training")

    if "eval_total_harvested_energy_kwh" in df.columns:
        eval_df = df.dropna(subset=["eval_total_harvested_energy_kwh"])
        axes[0].plot(
            eval_df["episode"],
            pd.to_numeric(eval_df["eval_total_harvested_energy_kwh"], errors="coerce"),
            "--",
            marker="D",
            color="black",
            label="Deterministic eval",
        )

    axes[0].set_ylabel("Energy (kWh)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["episode"], _series("movement_magnitude_deg"), marker="o", color="tab:orange")
    axes[1].set_ylabel("Movement (deg)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["episode"], _series("tracking_efficiency"), marker="o", color="tab:green")
    axes[2].set_ylabel("Tracking efficiency")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["episode"], _series("log_alpha"), marker="o", color="tab:red")
    axes[3].set_ylabel("log_alpha")
    axes[3].set_xlabel("Episode")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def save_controller_results(
    *,
    folder_name: str,
    controller: str,
    agent: Any,
    config: dict,
    schema: dict,
    training_df: pd.DataFrame,
    evaluation_df: Optional[pd.DataFrame] = None,
    rollout_history_df: Optional[pd.DataFrame] = None,
    replay_memory: Any = None,
    kl_replay_memory: Any = None,
    total_numsteps: int = 0,
    updates: int = 0,
    best_policy_state: Any = None,
    best_log_alpha: Any = None,
    best_episode: Any = None,
    best_metric_name: Optional[str] = None,
    best_metric_value: Any = None,
    overwrite: bool = False,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Save model, training curves, evaluation table, and rollout table."""

    save_dir = make_save_dir(folder_name, overwrite=overwrite)
    training_df = pd.DataFrame(training_df)
    evaluation_df = None if evaluation_df is None else pd.DataFrame(evaluation_df)
    rollout_history_df = None if rollout_history_df is None else pd.DataFrame(rollout_history_df)
    trained = episodes_trained(training_df)

    checkpoint = {
        "format_version": 4,
        "controller": controller,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "agent_state": agent_state(agent),
        "config": copy.deepcopy(config),
        "schema_template": copy.deepcopy(schema),
        "simulation_period": {
            "simulation_start_time_step": int(schema.get("simulation_start_time_step", 0)),
            "simulation_end_time_step": int(schema.get("simulation_end_time_step", 0)),
            "episode_time_steps": int(schema.get("episode_time_steps", 0)),
        },
        "training_counters": {
            "episodes_trained": trained,
            "total_numsteps": int(total_numsteps),
            "updates": int(updates),
            "best_episode": best_episode,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
        },
        "best_policy_state": best_policy_state,
        "best_log_alpha": best_log_alpha,
        "replay_memory": memory_state(replay_memory),
        "kl_replay_memory": memory_state(kl_replay_memory),
        "training_history": training_df.to_dict(orient="records"),
        "metadata": {
            "controller": controller,
            "episodes_trained": trained,
            "active_observations": active_observation_names(schema),
            **(extra_metadata or {}),
        },
    }

    torch.save(checkpoint, save_dir / "checkpoint.pt")
    (save_dir / "metadata.json").write_text(
        json.dumps(checkpoint["metadata"], indent=2, default=json_safe),
        encoding="utf-8",
    )
    (save_dir / "config.json").write_text(
        json.dumps(checkpoint["config"], indent=2, default=json_safe),
        encoding="utf-8",
    )
    (save_dir / "schema.json").write_text(
        json.dumps(checkpoint["schema_template"], indent=2, default=json_safe),
        encoding="utf-8",
    )

    training_df.to_csv(save_dir / "training_history.csv", index=False)

    if evaluation_df is not None:
        evaluation_df.to_csv(save_dir / "evaluation.csv", index=False)

    if rollout_history_df is not None:
        rollout_history_df.to_csv(save_dir / "rollout_history.csv", index=False)

    save_training_curves(training_df, f"{controller} training curves", save_dir / "training_curves.png")

    print(f"Saved {controller} checkpoint/results to: {save_dir}")
    return save_dir


def load_results_for_comparison(folder_name: str) -> dict[str, Any]:
    """Load saved tables for plotting without recreating the model."""

    results_dir = MODELS_ROOT / str(folder_name)

    if not results_dir.exists():
        raise FileNotFoundError(f"Saved results folder not found: {results_dir}")

    result = {"results_dir": results_dir}

    for key, filename, required in [
        ("training", "training_history.csv", True),
        ("history", "rollout_history.csv", False),
        ("evaluation", "evaluation.csv", False),
    ]:
        path = results_dir / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Expected saved artifact missing: {path}")
            result[key] = None
        else:
            result[key] = pd.read_csv(path)

    print(f"Loaded saved results from: {results_dir}")
    return result


def validate_target_episodes(checkpoint: dict, target_total_episodes: int) -> int:
    """Validate continuation target and return trained episode count."""

    trained = int(checkpoint.get("training_counters", {}).get("episodes_trained", 0))

    if int(target_total_episodes) <= trained:
        raise ValueError(
            f"This checkpoint already has {trained} trained episodes. "
            f"Enter a target above {trained}; for example {trained + 50}."
        )

    return trained


def append_checkpoint_training_history(checkpoint_dir: Path, new_training_df: pd.DataFrame) -> pd.DataFrame:
    """Append new continuation rows to the saved training history if available."""

    old_path = checkpoint_dir / "training_history.csv"

    if old_path.exists():
        old_df = pd.read_csv(old_path)
        return pd.concat([old_df, pd.DataFrame(new_training_df)], ignore_index=True)

    return pd.DataFrame(new_training_df)


def continue_meta_sac_training(
    *,
    checkpoint_folder: str,
    target_total_episodes: int,
    sac_class: type,
    memory_class: type,
    env_class: type,
    evaluate_fn: Callable,
    metric_fn: Callable,
    load_flat_config_fn: Callable,
    use_saved_hyperparameters: bool = True,
    config_path: str | Path = None,
    reward_scale: float = 50.0,
    override_schema: Optional[dict] = None,
    reset_log_alpha: Optional[float] = None,
) -> dict[str, Any]:
    """Continue a plain Meta-SAC checkpoint to a target total episode count.

    override_schema: if provided, replaces the schema saved in the checkpoint —
        useful when continuing Phase 1 (clear-day) checkpoints on Phase 2 (4-month) data.
    reset_log_alpha: if provided, overwrite agent.log_alpha after loading — set to 0.0
        to reset alpha=1.0 (full entropy) when continuing on a harder dataset.
    """

    checkpoint_dir, checkpoint = load_checkpoint(checkpoint_folder)
    trained_so_far = validate_target_episodes(checkpoint, target_total_episodes)

    if use_saved_hyperparameters:
        config = copy.deepcopy(checkpoint["config"])
    else:
        config = load_flat_config_fn(config_path)
        for key in ["hidden_size", "meta_Q", "alpha_embedding"]:
            if key in checkpoint["config"]:
                config[key] = checkpoint["config"][key]

    config["cuda"] = 0 if torch.cuda.is_available() else -1
    schema = copy.deepcopy(override_schema) if override_schema is not None else copy.deepcopy(checkpoint["schema_template"])
    env = env_class(schema, reward_scale=reward_scale)
    agent = sac_class(env.observation_space.shape[0], env.action_space, config)
    load_agent_state(agent, checkpoint, load_optimizer_states=use_saved_hyperparameters)
    if reset_log_alpha is not None:
        agent.log_alpha.data.fill_(float(reset_log_alpha))
        print(f"log_alpha reset to {reset_log_alpha:.4f} (alpha={float(np.exp(reset_log_alpha)):.4f})")
    memory = restore_memory(memory_class, checkpoint.get("replay_memory"), config["replay_size"])
    kl_memory = restore_memory(memory_class, checkpoint.get("kl_replay_memory"), config["kl_replay_size"])

    total_numsteps = int(checkpoint.get("training_counters", {}).get("total_numsteps", 0))
    updates = int(checkpoint.get("training_counters", {}).get("updates", 0))
    best_metric_name = str(config.get("best_metric", checkpoint.get("training_counters", {}).get("best_metric_name", "total_harvested_energy_kwh")))
    best_metric_value = float(checkpoint.get("training_counters", {}).get("best_metric_value", -np.inf))
    best_episode = checkpoint.get("training_counters", {}).get("best_episode")
    best_policy_state = checkpoint.get("best_policy_state")
    best_log_alpha = checkpoint.get("best_log_alpha")
    rows = []

    for episode in range(trained_so_far + 1, int(target_total_episodes) + 1):
        env.set_reward_episode(episode)
        reward_stage = env.reward_stage()
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        indicators = np.zeros(6, dtype=np.float32)

        while not done:
            if total_numsteps < int(config.get("start_steps", 0)):
                action = env.action_space.sample()
                log_prob = np.array([1.0 / (2 ** env.action_space.shape[0])], dtype=np.float32)
            else:
                action, log_prob = agent.select_action(state)

            if len(memory) > int(config["batch_size"]):
                for _ in range(int(config["updates_per_step"])):
                    indicators += agent.update_parameters(memory, kl_memory, int(config["batch_size"]), updates)
                    updates += 1
            else:
                indicators[0] += agent.log_alpha.item()

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 0.0 if done else 1.0
            memory.push(state, action, log_prob, reward, next_state, mask)
            kl_memory.push(state, action, log_prob, reward, next_state, mask)
            state = next_state

        indicators = indicators / max(episode_steps, 1)
        episode_metrics = env.evaluate()
        metric_values = dict(zip(episode_metrics["cost_function"], episode_metrics["value"]))
        row = {
            "episode": episode,
            "total_numsteps": total_numsteps,
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
        if episode == trained_so_far + 1 or episode % eval_interval == 0 or episode == int(target_total_episodes):
            _, deterministic_eval = evaluate_fn(agent, schema)
            det_metrics = dict(zip(deterministic_eval["cost_function"], deterministic_eval["value"]))
            deterministic_score = metric_fn(deterministic_eval, best_metric_name)
            row[f"eval_{best_metric_name}"] = deterministic_score
            row["eval_total_harvested_energy_kwh"] = det_metrics.get("total_harvested_energy_kwh", np.nan)
            row["eval_tracking_efficiency"] = det_metrics.get("tracking_efficiency", np.nan)
            row["eval_movement_magnitude_deg"] = det_metrics.get("movement_magnitude_deg", np.nan)
            row["eval_tracking_smoothness_jerk_deg2"] = det_metrics.get("tracking_smoothness_jerk_deg2", np.nan)

            if deterministic_score > best_metric_value:
                best_metric_value = deterministic_score
                best_episode = episode
                best_policy_state = copy.deepcopy(agent.policy.state_dict())
                best_log_alpha = agent.log_alpha.detach().clone()

        rows.append(row)
        print(
            f"Meta-SAC continued episode {episode}/{target_total_episodes} | "
            f"energy={row['total_harvested_energy_kwh']:.3f} kWh | "
            f"move={row['movement_magnitude_deg']:.1f} deg | "
            f"log_alpha={row['log_alpha']:.3f}"
        )

    continued_df = append_checkpoint_training_history(checkpoint_dir, pd.DataFrame(rows))

    return {
        "agent": agent,
        "env": env,
        "memory": memory,
        "kl_memory": kl_memory,
        "config": config,
        "schema": schema,
        "training_df": continued_df,
        "total_numsteps": total_numsteps,
        "updates": updates,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_episode": best_episode,
        "best_policy_state": best_policy_state,
        "best_log_alpha": best_log_alpha,
        "trained_so_far": trained_so_far,
        "target_total_episodes": target_total_episodes,
    }


def continue_sllm_training(
    *,
    checkpoint_folder: str,
    target_total_episodes: int,
    sac_class: type,
    memory_class: type,
    env_class: type,
    guidance: Any,
    evaluate_fn: Callable,
    metric_fn: Callable,
    load_flat_config_fn: Callable,
    use_saved_hyperparameters: bool = True,
    config_path: str | Path = None,
    reward_scale: float = 50.0,
    override_schema: Optional[dict] = None,
    reset_log_alpha: Optional[float] = None,
) -> dict[str, Any]:
    """Continue a goal-conditioned sLLM Meta-SAC checkpoint.

    override_schema: if provided, replaces the schema saved in the checkpoint.
    reset_log_alpha: if provided, overwrite agent.log_alpha after loading.
    """

    checkpoint_dir, checkpoint = load_checkpoint(checkpoint_folder)
    trained_so_far = validate_target_episodes(checkpoint, target_total_episodes)

    if use_saved_hyperparameters:
        config = copy.deepcopy(checkpoint["config"])
    else:
        config = load_flat_config_fn(config_path)
        for key in ["hidden_size", "meta_Q", "alpha_embedding"]:
            if key in checkpoint["config"]:
                config[key] = checkpoint["config"][key]

    config["cuda"] = 0 if torch.cuda.is_available() else -1
    schema = copy.deepcopy(override_schema) if override_schema is not None else copy.deepcopy(checkpoint["schema_template"])
    env = env_class(schema, guidance=guidance, reward_scale=reward_scale, verbose=False)
    agent = sac_class(env.observation_space.shape[0], env.action_space, config)
    load_agent_state(agent, checkpoint, load_optimizer_states=use_saved_hyperparameters)
    if reset_log_alpha is not None:
        agent.log_alpha.data.fill_(float(reset_log_alpha))
        print(f"log_alpha reset to {reset_log_alpha:.4f} (alpha={float(np.exp(reset_log_alpha)):.4f})")
    memory = restore_memory(memory_class, checkpoint.get("replay_memory"), config["replay_size"])
    kl_memory = restore_memory(memory_class, checkpoint.get("kl_replay_memory"), config["kl_replay_size"])

    total_numsteps = int(checkpoint.get("training_counters", {}).get("total_numsteps", 0))
    updates = int(checkpoint.get("training_counters", {}).get("updates", 0))
    best_metric_name = str(config.get("best_metric", checkpoint.get("training_counters", {}).get("best_metric_name", "total_harvested_energy_kwh")))
    best_metric_value = float(checkpoint.get("training_counters", {}).get("best_metric_value", -np.inf))
    best_episode = checkpoint.get("training_counters", {}).get("best_episode")
    best_policy_state = checkpoint.get("best_policy_state")
    best_log_alpha = checkpoint.get("best_log_alpha")
    rows = []

    for episode in range(trained_so_far + 1, int(target_total_episodes) + 1):
        env.set_reward_episode(episode)
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        indicators = np.zeros(6, dtype=np.float32)

        while not done:
            if total_numsteps < int(config.get("start_steps", 0)):
                action = env.action_space.sample()
                log_prob = np.array([1.0 / (2 ** env.action_space.shape[0])], dtype=np.float32)
            else:
                action, log_prob = agent.select_action(state)

            if len(memory) > int(config["batch_size"]):
                for _ in range(int(config["updates_per_step"])):
                    indicators += agent.update_parameters(memory, kl_memory, int(config["batch_size"]), updates)
                    updates += 1
            else:
                indicators[0] += agent.log_alpha.item()

            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 0.0 if done else 1.0
            memory.push(state, action, log_prob, reward, next_state, mask)
            kl_memory.push(state, action, log_prob, reward, next_state, mask)
            state = next_state

        indicators = indicators / max(episode_steps, 1)
        episode_metrics = env.evaluate()
        metric_values = dict(zip(episode_metrics["cost_function"], episode_metrics["value"]))
        row = {
            "episode": episode,
            "total_numsteps": total_numsteps,
            "episode_steps": episode_steps,
            "scaled_reward": episode_reward,
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
            "qwen_requests": int(getattr(guidance, "request_count", getattr(guidance, "call_count", 0))),
            "qwen_successes": int(getattr(guidance, "call_count", 0)),
            "fallback_count": int(getattr(guidance, "fallback_count", 0)),
        }

        eval_interval = int(config.get("eval_interval", 10))
        if episode == trained_so_far + 1 or episode % eval_interval == 0 or episode == int(target_total_episodes):
            _, deterministic_eval, _ = evaluate_fn(agent, schema, shared_guidance=guidance, include_baselines=False)
            det_metrics = dict(zip(deterministic_eval["cost_function"], deterministic_eval["value"]))
            deterministic_score = metric_fn(deterministic_eval, best_metric_name)
            row[f"eval_{best_metric_name}"] = deterministic_score
            row["eval_total_harvested_energy_kwh"] = det_metrics.get("total_harvested_energy_kwh", np.nan)
            row["eval_tracking_efficiency"] = det_metrics.get("tracking_efficiency", np.nan)
            row["eval_movement_magnitude_deg"] = det_metrics.get("movement_magnitude_deg", np.nan)
            row["eval_tracking_smoothness_jerk_deg2"] = det_metrics.get("tracking_smoothness_jerk_deg2", np.nan)

            if deterministic_score > best_metric_value:
                best_metric_value = deterministic_score
                best_episode = episode
                best_policy_state = copy.deepcopy(agent.policy.state_dict())
                best_log_alpha = agent.log_alpha.detach().clone()

        rows.append(row)
        print(
            f"sLLM continued episode {episode}/{target_total_episodes} | "
            f"energy={row['total_harvested_energy_kwh']:.3f} kWh | "
            f"TE={row['tracking_efficiency']:.3f} | "
            f"move={row['movement_magnitude_deg']:.1f} deg | "
            f"Qwen successes={row['qwen_successes']}"
        )

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    continued_df = append_checkpoint_training_history(checkpoint_dir, pd.DataFrame(rows))

    return {
        "agent": agent,
        "env": env,
        "memory": memory,
        "kl_memory": kl_memory,
        "config": config,
        "schema": schema,
        "training_df": continued_df,
        "total_numsteps": total_numsteps,
        "updates": updates,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_episode": best_episode,
        "best_policy_state": best_policy_state,
        "best_log_alpha": best_log_alpha,
        "trained_so_far": trained_so_far,
        "target_total_episodes": target_total_episodes,
    }
