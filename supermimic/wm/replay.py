"""Fixed-size replay buffer of simulator transitions (Table A.5: 1000 steps per environment).

Each slot holds, for every env, the state s_t the policy acted in, the executed noisy action
a_t + ε_t, the sim context ξ, the reference frame and an episode id. New steps overwrite the
oldest. Episodes are told apart by their id, so windows never cross a reset.
"""

from __future__ import annotations

import torch


class TransitionBuffer:
    def __init__(self, num_envs: int, capacity: int, state_dim: int, action_dim: int, context_dim: int, device):
        self.num_envs, self.capacity, self.device = num_envs, capacity, device
        self.state = torch.zeros(capacity, num_envs, state_dim, device=device)
        self.action = torch.zeros(capacity, num_envs, action_dim, device=device)
        self.context = torch.zeros(capacity, num_envs, context_dim, device=device)
        self.frame = torch.zeros(capacity, num_envs, dtype=torch.long, device=device)
        self.episode = torch.zeros(capacity, num_envs, dtype=torch.long, device=device)
        self._episode_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, context, frame, done):
        """Step t: ``state`` s_t (post-reset), executed ``action``, and whether the episode ended after it."""
        p = self.ptr
        self.state[p] = state
        self.action[p] = action
        self.context[p] = context
        self.frame[p] = frame
        self.episode[p] = self._episode_counter
        self._episode_counter += done.long()
        self.ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _slot(self, age):
        """Ring slot of a step ``age`` steps before the newest (age 0)."""
        return (self.ptr - 1 - age) % self.capacity

    def sample_windows(self, batch: int, k: int):
        """k-step windows within one episode: states (B, k+1, S), actions (B, k, A), context (B, C)."""
        assert self.size > k
        env = torch.randint(0, self.num_envs, (2 * batch,), device=self.device)
        age = torch.randint(k, self.size, (2 * batch,), device=self.device)
        valid = self.episode[self._slot(age), env] == self.episode[self._slot(age - k), env]
        # keep valid windows first; with short episodes a few invalid ones may remain at the end
        order = torch.argsort((~valid).to(torch.int8), stable=True)[:batch]
        env, age, valid = env[order], age[order], valid[order]
        steps = self._slot(age[:, None] - torch.arange(k + 1, device=self.device))  # (B, k+1), oldest first
        e = env[:, None]
        return {
            "state": self.state[steps, e],
            "action": self.action[steps[:, :k], e],
            "context": self.context[steps[:, 0], env],
            "valid": valid,
        }

    def sample_starts(self, batch: int, history: int):
        """Imagination start points with the policy's observation history.

        Returns the start state, context and frame, plus the ``history`` states up to and including
        the start (oldest first) and the action executed before each of them. Before the episode
        start, the first state is repeated with a zero previous action, as the env fills its history.
        """
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        age = torch.randint(0, self.size, (batch,), device=self.device)
        back = torch.arange(history, device=self.device)  # 0 = start step
        hist_age = (age[:, None] + back).clamp(max=self.size - 1)
        e = env[:, None]
        ep = self.episode[self._slot(age), env]
        same = self.episode[self._slot(hist_age), e] == ep[:, None]
        # same-episode steps form a prefix in age; clamp to the episode's first step
        # (when the buffer's oldest step is reached first, it stands in for the episode start)
        first = (age + same.sum(-1) - 1).clamp(max=self.size - 1)
        hist_age = torch.minimum(hist_age, first[:, None])
        prev_age = hist_age + 1
        has_prev = prev_age <= first[:, None]
        prev_action = self.action[self._slot(prev_age.clamp(max=self.size - 1)), e] * has_prev[..., None]
        slots = self._slot(hist_age).flip(-1)
        s0 = self._slot(age)
        return {
            "state": self.state[s0, env],
            "context": self.context[s0, env],
            "frame": self.frame[s0, env],
            "history_state": self.state[slots, e],
            "history_prev_action": prev_action.flip(1),
        }
