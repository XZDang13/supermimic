"""Learned dynamics f_ω(ŝ, a, ξ) -> ŝ' (Table A.4: MLP 2048 x 3) and its k-step loss (Eq. 1).

The network sees a heading-invariant view of the state (root height, projected gravity, root
velocities in the root frame, joint q and qd) with the action and sim context, and predicts
the change over one control step in the root frame of step t: root displacement, rotation
vector, change of root linear and angular velocity, Δq and Δq̇. Inputs and outputs are
normalized by running statistics; the loss weighs each dimension by the scale of its
one-step change, so positions, velocities and joints count alike.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RLAlg.nn.layers import NormPosition, make_mlp_layers
from RLAlg.normalizer import Normalizer

from supermimic.envs import G1TrackingEnv
from supermimic.envs.tracking import projected_gravity
from supermimic.ppo.model import NORM_EPS, masked_update
from supermimic.utils.math import (
    axis_angle_to_quat,
    quat_apply,
    quat_apply_inv,
    quat_conj,
    quat_mul,
    quat_normalize,
    quat_to_axis_angle,
)
from supermimic.wm.task import split_state

NA = G1TrackingEnv.NUM_ACTIONS
STATE_DIM = G1TrackingEnv.STATE_DIM
FEATURE_DIM = 1 + 3 + 3 + 3 + 2 * NA       # 68
DELTA_DIM = 12 + 2 * NA                    # 70


def state_features(state):
    pos, quat, lin, ang, q, qd = split_state(state)
    return torch.cat((pos[:, 2:3], projected_gravity(quat), quat_apply_inv(quat, lin), quat_apply_inv(quat, ang), q, qd), -1)


def state_delta(state, next_state):
    """One-step change in the root frame of ``state`` (the network's target), (N, 70)."""
    pos, quat, lin, ang, q, qd = split_state(state)
    npos, nquat, nlin, nang, nq, nqd = split_state(next_state)
    return torch.cat((
        quat_apply_inv(quat, npos - pos),
        quat_to_axis_angle(quat_mul(quat_conj(quat), nquat)),
        quat_apply_inv(quat, nlin - lin),
        quat_apply_inv(quat, nang - ang),
        nq - q,
        nqd - qd,
    ), -1)


def apply_delta(state, delta):
    pos, quat, lin, ang, q, qd = split_state(state)
    d_pos, d_rot, d_lin, d_ang = delta[:, 0:3], delta[:, 3:6], delta[:, 6:9], delta[:, 9:12]
    return torch.cat((
        pos + quat_apply(quat, d_pos),
        quat_normalize(quat_mul(quat, axis_angle_to_quat(d_rot))),
        lin + quat_apply(quat, d_lin),
        ang + quat_apply(quat, d_ang),
        q + delta[:, 12 : 12 + NA],
        qd + delta[:, 12 + NA :],
    ), -1)


def state_error(pred, target):
    """Per-dimension error of a predicted state, in the delta layout (N, 70)."""
    p_pos, p_quat, p_lin, p_ang, p_q, p_qd = split_state(pred)
    t_pos, t_quat, t_lin, t_ang, t_q, t_qd = split_state(target)
    return torch.cat((
        quat_apply_inv(t_quat, p_pos - t_pos),
        quat_to_axis_angle(quat_mul(quat_conj(t_quat), p_quat)),
        quat_apply_inv(t_quat, p_lin - t_lin),
        quat_apply_inv(t_quat, p_ang - t_ang),
        p_q - t_q,
        p_qd - t_qd,
    ), -1)


class WorldModel(nn.Module):
    def __init__(self, action_dim: int = NA, context_dim: int = G1TrackingEnv.CONTEXT_DIM,
                 hidden_dims: list[int] = (2048, 2048, 2048), norm_position: NormPosition = NormPosition.POST):
        super().__init__()
        in_dim = FEATURE_DIM + action_dim + context_dim
        self.in_normalizer = Normalizer(in_dim, epsilon=NORM_EPS)
        self.out_normalizer = Normalizer(DELTA_DIM, epsilon=1e-6)  # std floor 1e-3: one-step changes are small
        self.layers, feature_dim = make_mlp_layers(in_dim, list(hidden_dims), activate_function=nn.SiLU(),
                                                   norm_position=norm_position)
        self.head = nn.Linear(feature_dim, DELTA_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def compile_step(self, mode: str | None = None):
        self._step = torch.compile(self._step_impl, dynamic=False, mode=mode)
        # the whole k-step loss as one graph: the per-step errors and the k-fold weight-gradient sums fuse too
        self._rollout_loss = torch.compile(self._rollout_loss_impl, dynamic=False, mode=mode)

    def _out_std(self):
        return torch.sqrt(self.out_normalizer.var + self.out_normalizer.epsilon).float()

    def forward(self, state, action, context):
        return self._step(state, action, context) if hasattr(self, "_step") else self._step_impl(state, action, context)

    def _step_impl(self, state, action, context):
        x = self.in_normalizer(torch.cat((state_features(state), action, context), -1))
        delta = self.out_normalizer.mean.float() + self._out_std() * self.head(self.layers(x))
        return apply_delta(state, delta)

    @torch.no_grad()
    def update_normalizers(self, states, actions, context, valid=None):
        """From windows: states (B, k+1, S), actions (B, k, A), context (B, C); only ``valid`` (B,) windows count."""
        k = actions.shape[1]
        s, sn = states[:, :-1].flatten(0, 1), states[:, 1:].flatten(0, 1)
        c = context[:, None].expand(-1, k, -1).flatten(0, 1)
        x, d = torch.cat((state_features(s), actions.flatten(0, 1), c), -1), state_delta(s, sn)
        if valid is None:
            self.in_normalizer.update(x)
            self.out_normalizer.update(d)
        else:
            mask = valid[:, None].expand(-1, k).flatten()
            masked_update(self.in_normalizer, x, mask)
            masked_update(self.out_normalizer, d, mask)

    def rollout_loss(self, states, actions, context, valid=None):
        """Eq. 1 (dynamics term): k-step autoregressive rollout from the logged s_0 on logged actions."""
        if valid is None:
            valid = torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
        fn = self._rollout_loss if hasattr(self, "_rollout_loss") else self._rollout_loss_impl
        return fn(states, actions, context, valid)

    def _rollout_loss_impl(self, states, actions, context, valid):
        k = actions.shape[1]
        scale = self._out_std()
        s = states[:, 0]
        errors = []
        for j in range(k):
            s = self._step_impl(s, actions[:, j], context)
            errors.append(state_error(s, states[:, j + 1]) / scale)
        err = torch.stack(errors, 1)                          # (B, k, 70)
        per_window = err.square().mean((1, 2))
        w = valid.to(per_window.dtype)
        return (per_window * w).sum() / w.sum().clamp_min(1.0), err.detach()
