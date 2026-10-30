"""Deterministic SuperMimic policy a = dec(enc(g), o_{t-9:t}) (Table A.4, Appendix B).

Same motion encoder (tanh latent z, 32) and decoder sizes as the PPO-Sim actor, with a plain
linear action output: imagined rollouts differentiate the return with respect to the action.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers
from RLAlg.normalizer import Normalizer

from supermimic.ppo.model import NORM_EPS, MotionEncoder


class Policy(nn.Module):
    def __init__(self, history_len: int, proprio_dim: int, future_frames: int, goal_frame_dim: int, action_dim: int,
                 encoder_dims: list[int], decoder_dims: list[int], latent_dim: int = 32,
                 norm_position: NormPosition = NormPosition.POST):
        super().__init__()
        self.encoder = MotionEncoder(future_frames, goal_frame_dim, encoder_dims, latent_dim, norm_position)
        self.proprio_normalizer = Normalizer(proprio_dim, epsilon=NORM_EPS)
        self.layers, feature_dim = make_mlp_layers(latent_dim + history_len * proprio_dim, decoder_dims,
                                                   activate_function=nn.SiLU(), norm_position=norm_position)
        self.action = nn.Linear(feature_dim, action_dim)
        # start near the default pose
        nn.init.orthogonal_(self.action.weight, gain=0.01)
        nn.init.zeros_(self.action.bias)

    @torch.no_grad()
    def update_normalizers(self, obs: dict[str, torch.Tensor]):
        self.encoder.normalizer.update(obs["goal"])
        self.proprio_normalizer.update(obs["policy"])

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.encoder(obs["goal"])
        x = torch.cat((z, self.proprio_normalizer(obs["policy"]).flatten(1)), dim=-1)
        return self.action(self.layers(x))
