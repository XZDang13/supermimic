"""PPO-Sim networks, sized as SuperMimic's Table A.4 so the baseline matches its policy.

Actor: motion encoder over the K future reference frames -> tanh latent z (32), then a
decoder over z and the last H noisy proprioceptive frames -> Gaussian action.
Critic: MLP over privileged inputs (noise-free proprioception, root height and velocity,
goal, sim context xi).

Running observation normalizers live inside the modules, so a checkpoint carries them.
"""

from typing import Optional

import torch
import torch.nn as nn

from RLAlg.nn.layers import CriticHead, GaussianHead, NormPosition, make_mlp_layers
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.normalizer import Normalizer

from supermimic.envs import G1TrackingEnv
from supermimic.utils.math import quat_apply_inv


# Variance floor of the running normalizers: std >= 0.01 as in rsl_rl, so near-constant
# inputs are not amplified by up to 1e4 and the wide nets stay well conditioned.
NORM_EPS = 1e-4


@torch.no_grad()
def masked_update(normalizer: Normalizer, x: torch.Tensor, mask: torch.Tensor):
    """``normalizer.update(x[mask])`` for x (N, D), without the boolean index (a host sync).

    Batch statistics are reduced in x's dtype and merged in float64, as ``Normalizer.update`` merges.
    """
    w = mask.to(x.dtype)[:, None]
    n = w.sum()
    mean = (x * w).sum(0) / n.clamp_min(1.0)
    var = ((x - mean).square() * w).sum(0) / n.clamp_min(1.0)
    n, mean, var = n.double(), mean.double(), var.double()
    count = normalizer.count
    total = count + n
    delta = mean - normalizer.mean
    m2 = normalizer.var * count + var * n + delta.square() * count * n / total.clamp_min(1.0)
    normalizer.mean.add_(delta * n / total.clamp_min(1.0))
    normalizer.var.copy_(torch.where(total > 0, m2 / total.clamp_min(1.0), normalizer.var).clamp_min(0.0))
    normalizer.count.copy_(total)


def fixed_scale(normalizer: Normalizer, *groups: tuple[int, float]):
    """Set ``normalizer`` to zero mean and hand-picked std, from (size, std) groups of its dimensions."""
    std = torch.cat([torch.full((n, ), s, dtype=normalizer.var.dtype) for n, s in groups]).to(normalizer.var.device)
    normalizer.var.copy_(std.square())


def proprio_scales(j: int) -> list[tuple[int, float]]:
    return [(j, 0.5), (j, 3.0), (3, 1.0), (3, 1.0), (j, 1.0)]        # q, qd, angular velocity, gravity, prev action


def goal_scales(j: int) -> list[tuple[int, float]]:
    return [(j, 0.5), (j, 3.0), (3, 0.2), (6, 1.0)]                  # q, qd, relative root position, orientation


def context_scales(j: int) -> list[tuple[int, float]]:
    return [(1, 1.0), (j, 0.02), (3, 0.05)]                          # friction, joint offset, torso CoM offset


def critic_observation(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Privileged critic input (800,) built from the env's observation dict."""
    state = obs["state"]
    root_height = state[:, 2:3]
    root_lin_vel_b = quat_apply_inv(state[:, 3:7], state[:, 7:10])
    return torch.cat((obs["proprio"], root_height, root_lin_vel_b, obs["goal"].flatten(1), obs["context"]), dim=-1)


def critic_observation_dim(env: G1TrackingEnv) -> int:
    return env.PROPRIO_DIM + 4 + env.cfg.future_frames * env.GOAL_FRAME_DIM + env.CONTEXT_DIM


class MotionEncoder(nn.Module):
    def __init__(self, future_frames: int, frame_dim: int, hidden_dims: list[int], latent_dim: int,
                 norm_position: NormPosition = NormPosition.POST):
        super().__init__()
        self.normalizer = Normalizer(frame_dim, epsilon=NORM_EPS)
        self.layers, feature_dim = make_mlp_layers(future_frames * frame_dim, hidden_dims, activate_function=nn.SiLU(),
                                                   norm_position=norm_position)
        self.latent = nn.Linear(feature_dim, latent_dim)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(goal).flatten(1)
        return torch.tanh(self.latent(self.layers(x)))


class Actor(nn.Module):
    def __init__(self, history_len: int, proprio_dim: int, future_frames: int, goal_frame_dim: int, action_dim: int,
                 encoder_dims: list[int], decoder_dims: list[int], latent_dim: int = 32,
                 norm_position: NormPosition = NormPosition.POST, init_std: float = 1.0):
        super().__init__()
        self.encoder = MotionEncoder(future_frames, goal_frame_dim, encoder_dims, latent_dim, norm_position)
        self.proprio_normalizer = Normalizer(proprio_dim, epsilon=NORM_EPS)
        self.layers, feature_dim = make_mlp_layers(latent_dim + history_len * proprio_dim, decoder_dims,
                                                   activate_function=nn.SiLU(), norm_position=norm_position)
        # Unbounded Gaussian: actions are offsets of PD setpoints, as in BeyondMimic.
        self.head = GaussianHead(feature_dim, action_dim, max_action=None, state_dependent_std=False,
                                 init_std=init_std, validate_args=False)

    @torch.no_grad()
    def update_normalizers(self, obs: dict[str, torch.Tensor]):
        self.encoder.normalizer.update(obs["goal"])
        self.proprio_normalizer.update(obs["policy"])

    def forward(self, obs: dict[str, torch.Tensor], action: Optional[torch.Tensor] = None) -> StochasticContinuousPolicyStep:
        z = self.encoder(obs["goal"])
        x = torch.cat((z, self.proprio_normalizer(obs["policy"]).flatten(1)), dim=-1)
        return self.head(self.layers(x), action)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], norm_position: NormPosition = NormPosition.POST):
        super().__init__()
        self.normalizer = Normalizer(in_dim, epsilon=NORM_EPS)
        self.layers, feature_dim = make_mlp_layers(in_dim, hidden_dims, activate_function=nn.SiLU(),
                                                   norm_position=norm_position)
        self.head = CriticHead(feature_dim)

    @torch.no_grad()
    def update_normalizers(self, critic_obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            self.normalizer.update(critic_obs)
        else:
            masked_update(self.normalizer, critic_obs, mask)

    def forward(self, critic_obs: torch.Tensor) -> ValueStep:
        return self.head(self.layers(self.normalizer(critic_obs)))
