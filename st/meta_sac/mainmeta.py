import sys


def _coerce_config_value(value):
    value = value.strip()
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_config(filepath):
    """Load a flat YAML config without requiring venv changes.

    The original project depends on PyYAML.  If it is unavailable, this small
    fallback supports the flat ``key: value`` config files used in this folder.
    """

    try:
        import yaml

        with open(filepath) as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        config = {}

        with open(filepath) as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if not line or ':' not in line:
                    continue

                key, value = line.split(':', 1)
                config[key.strip()] = _coerce_config_value(value)

        return config


if len(sys.argv) < 2:
    raise SystemExit('Usage: python mainmeta.py configs/{env}.yml')


config = _load_config(sys.argv[1]) # custom hyperparams

# Support num_episodes as alias for num_steps (computed after first env reset)
_num_episodes = config.get('num_episodes', None)
if 'num_steps' not in config and _num_episodes is None:
    raise SystemExit('Config must specify num_steps or num_episodes')

import os

if config['exp_id'] != 'debug':
    if not os.path.exists('logs/' + config['exp_id']):
        os.makedirs('logs/' + config['exp_id'])

    log_file = 'logs/' + config['exp_id'] + '/' + '{}.txt'.format(config['seed'])
    sys.stdout = open(log_file, 'w')

print(config)
print(os.getpid())

cores = str(config['cores'])
os.environ["OMP_NUM_THREADS"] = cores # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = cores # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = cores # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = cores # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = cores # export NUMEXPR_NUM_THREADS=6

import datetime
import gym
import numpy as np
import itertools
import torch, random
from sacmeta import SAC_META as SAC
from replay_memory import ReplayMemoryKL as ReplayMemory

# Environment
_base_env = gym.make(config['env_name'])

use_llm = bool(config.get('use_llm', False))
if use_llm:
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from llm.goal_wrapper import LLMGoalConditionedMetaSACWrapper
    from llm.goal_guidance import LLMGoalGuidance
    _guidance = LLMGoalGuidance(use_llm=False)  # deterministic fallback during training
    env = LLMGoalConditionedMetaSACWrapper(_base_env, guidance=_guidance)
else:
    env = _base_env
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])

def reset_env(env, seed=None):
    """Reset an environment across old and new Gym APIs."""

    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
            result = env.reset()

    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    """Step an environment across old and new Gym APIs."""

    result = env.step(action)

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info

    return result


# Compute num_steps from num_episodes now that env is available
if 'num_steps' not in config:
    try:
        _eps = env.spec.max_episode_steps or env._max_episode_steps
    except AttributeError:
        _eps = getattr(env, '_max_episode_steps', 1008)
    config['num_steps'] = int(_num_episodes) * int(_eps)
    print(f"num_steps computed: {_num_episodes} episodes x {_eps} steps = {config['num_steps']}")

reset_env(env, config['seed'])
try:
    env.action_space.seed(config['seed'])
except AttributeError:
    env.action_space.np_random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, config)

# Memory
memory = ReplayMemory(config['replay_size'])
kl_memory = ReplayMemory(config['kl_replay_size'])

# Training Loop
total_numsteps = 0
updates = 0
training_history = []
evaluation_history = []

# make model save path
import os
from os import path
import time

def test(env, mode):
    '''
    mode='zero': use alpha = 0 in policy net.
    mode='running': use alpha as meta alpha in policy net.
    '''
    avg_reward = 0.
    episodes = 10
    for _  in range(episodes):
        state = reset_env(env)
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(state, eval=True, mode = mode)

            next_state, reward, done, _ = step_env(env, action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {} Mode: {}".format(episodes, round(avg_reward, 2), mode))
    print("----------------------------------------")

if config['meta_obj_s0']:
    s0_list = np.stack([reset_env(env) for _ in range(config['batch_size'])])

test_step = 10000
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = reset_env(env)
    indicators = np.zeros((6), np.float32) # log_alpha, critic_1_loss, critic_2_loss, policy_loss, new_policy_loss, entropy

    while not done:
        if config['start_steps'] > total_numsteps:
            action, log_prob = env.action_space.sample(), [1 / 2**(env.action_space.shape[0])]  # Sample random action
        else:
            action, log_prob = agent.select_action(state)  # Sample action from policy


        if len(memory) > config['batch_size']:
            # Number of updates per step in environment
            for i in range(config['updates_per_step']):
                # Update parameters of all the networks
                if config['meta_obj_s0']:
                    indicators += agent.update_parameters(memory, kl_memory, config['batch_size'], updates, s0_list)
                else:
                    indicators += agent.update_parameters(memory, kl_memory, config['batch_size'], updates)
                updates += 1
        else:
            indicators[0] += agent.log_alpha.item()
            
        next_state, reward, done, _ = step_env(env, action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        max_episode_steps = getattr(env, '_max_episode_steps', None)
        if max_episode_steps is None and getattr(env, 'spec', None) is not None:
            max_episode_steps = env.spec.max_episode_steps
        mask = 1 if max_episode_steps is not None and episode_steps == max_episode_steps else float(not done)

        memory.push(state, action, log_prob, reward, next_state, mask) # Append transition to memory
        kl_memory.push(state, action, log_prob, reward, next_state, mask)

        state = next_state

        # if total_numsteps in save_numsteps:
        #     agent.save_model(save_path, env_name=config['env_name'], suffix="{}".format(str(total_numsteps)))

    if total_numsteps > config['num_steps']:
        break
    
    indicators /= episode_steps
    training_history.append({
        'episode': i_episode,
        'total_numsteps': total_numsteps,
        'episode_steps': episode_steps,
        'reward': float(episode_reward),
        'log_alpha': float(indicators[0]),
        'critic_1_loss': float(indicators[1]),
        'critic_2_loss': float(indicators[2]),
        'policy_loss': float(indicators[3]),
        'meta_policy_loss': float(indicators[4]),
        'entropy': float(indicators[5]),
    })
    print("Episode: {} total numsteps: {} episode steps: {} reward: {} log alpha: {} c1: {} c2: {} p: {} mp: {} ent: {}".format(
        i_episode, total_numsteps, episode_steps, str(round(episode_reward, 2)), str(round(indicators[0], 3)), 
        str(round(indicators[1], 2)), str(round(indicators[2], 2)), str(round(indicators[3], 2)), 
        str(round(indicators[4], 2)), str(round(indicators[5], 2))
        ))

    if total_numsteps > test_step and config['eval'] == True:
        test_step += 10000
        if config['alpha_embedding']:
            test(env, 'zero')
        test(env, 'running')
        evaluation_history.append({
            'total_numsteps': total_numsteps,
            'log_alpha': float(agent.log_alpha.item()),
        })
        print("Test Log Alpha: {}".format(agent.log_alpha.item()))
        print("---------------------------------------")

env.close()
