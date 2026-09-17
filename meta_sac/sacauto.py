"""Standard SAC with automatic entropy tuning (SAC-Auto).

Drop-in replacement for SAC_META. Fixes the alpha collapse problem on the
multi-season dataset: the Meta-SAC meta-objective always rewards deterministic
policies, pushing log_alpha to its floor within the first few episodes.
SAC-Auto instead uses a fixed entropy target H = -dim(A) = -2, which keeps
alpha from collapsing when Q-values are still noisy.

Usage:
    from sacauto import SAC_Auto
    agent = SAC_Auto(obs_dim, action_space, config)
    # identical API to SAC_META — works with mainmeta.py and checkpointing.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from sacmeta import SAC_META
from utils import soft_update


class SAC_Auto(SAC_META):
    """Standard SAC with automatic entropy tuning.

    Overrides only update_parameters. All networks, optimisers, replay
    interface, select_action, and checkpointing stay identical to SAC_META.
    kl_memory and s0_list are accepted but unused so the training loop
    in mainmeta.py and checkpointing.py need zero changes.
    """

    def __init__(self, num_inputs, action_space, config):
        super().__init__(num_inputs, action_space, config)
        # H* = -|A| is the SAC-Auto default (Haarnoja et al. 2018).
        # For precision tasks (solar tracking) a lower value reduces action
        # variance and mechanical movement without sacrificing tracking quality.
        # Override via config key "target_entropy" (e.g. -0.5).
        self.target_entropy = float(
            config.get("target_entropy", -float(action_space.shape[0]))
        )

    def update_parameters(self, memory, kl_memory, batch_size, updates, s0_list=None):
        # ── sample ────────────────────────────────────────────────────────────
        state_batch, action_batch, _, reward_batch, next_state_batch, mask_batch = (
            memory.sample(batch_size=batch_size)
        )
        state_batch      = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch     = torch.FloatTensor(action_batch).to(self.device)
        reward_batch     = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch       = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Detached alpha used as a constant in critic and policy updates
        alpha = torch.exp(self.log_alpha).detach().repeat((len(state_batch), 1))

        # ── critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(
                next_state_batch, alpha, self.alpha_embedding
            )
            qf1_next, qf2_next = self.critic_target(
                next_state_batch, next_action, alpha, self.alpha_embedding
            )
            min_next_q = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_next_q

        qf1, qf2 = self.critic(state_batch, action_batch, alpha, self.alpha_embedding)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        # ── policy update ─────────────────────────────────────────────────────
        pi, log_pi, _ = self.policy.sample(state_batch, alpha, self.alpha_embedding)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, alpha, self.alpha_embedding)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (alpha * log_pi - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ── entropy temperature update (SAC-Auto) ─────────────────────────────
        # Gradient flows only through log_alpha; log_pi is treated as constant.
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.log_alpha.data.clamp_(min=self.log_alpha_min, max=self.log_alpha_max)

        # ── soft target update ────────────────────────────────────────────────
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # Return same tuple shape as SAC_META for logging compatibility
        return np.array([
            self.log_alpha.item(),
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            -log_pi.mean().item(),
        ])
