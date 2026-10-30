"""Motion-tracking task terms as pure torch functions of states and reference.

The simulator env and the world model's analytic heads (§3.2) share these, so imagined
rollouts see exactly the observation, goal, reward and termination the simulator computes,
and gradients flow through them.

Bodies are passed as ``(pos, quat, lin_vel, ang_vel)`` with shapes (N, B, 3|4): world-frame
link-origin positions and velocities, xyzw quaternions.
"""

from __future__ import annotations

import torch

from supermimic.utils.math import (
    quat_apply,
    quat_apply_inv,
    quat_conj,
    quat_error_sq,
    quat_mul,
    quat_to_tan_norm,
    yaw_quat,
)


def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    g = torch.tensor([0.0, 0.0, -1.0], device=quat.device, dtype=quat.dtype).expand_as(quat[..., :3])
    return quat_apply_inv(quat, g)


def proprio_obs(root_quat, root_ang_vel_w, q, qd, default_joint_pos, joint_offset, prev_actions) -> torch.Tensor:
    """Noise-free proprioception o-bar (Table A.1), (N, 93)."""
    return torch.cat(
        (
            q - default_joint_pos - joint_offset,
            qd,
            quat_apply_inv(root_quat, root_ang_vel_w),
            projected_gravity(root_quat),
            prev_actions,
        ),
        dim=-1,
    )


def goal_obs(ref_root_pos, ref_root_quat, ref_joint_pos, ref_joint_vel, root_pos, root_quat, default_joint_pos):
    """K future reference frames relative to the robot root, (N, K, 67). ``ref_*`` are (N, K, ...)."""
    rq = root_quat[:, None].expand_as(ref_root_quat)
    rel_pos = quat_apply_inv(rq, ref_root_pos - root_pos[:, None])
    rel_ori = quat_to_tan_norm(quat_mul(quat_conj(rq), ref_root_quat))
    return torch.cat((ref_joint_pos - default_joint_pos, ref_joint_vel, rel_pos, rel_ori), dim=-1)


def tracking_errors(bodies, ref_bodies, anchor_idx, tracked_idx):
    """Squared errors of the exp(-err / std^2) tracking terms, each (N,)."""
    pos, quat, lin_vel, ang_vel = bodies
    rpos, rquat, rlin_vel, rang_vel = ref_bodies
    a, t = anchor_idx, tracked_idx

    # reference bodies re-anchored at the robot's anchor (xy and yaw), as in BeyondMimic
    delta_pos = torch.cat((pos[:, a, :2], rpos[:, a, 2:]), dim=-1)
    delta_ori = yaw_quat(quat_mul(quat[:, a], quat_conj(rquat[:, a])))[:, None].expand(-1, len(t), -1)
    rel_pos = delta_pos[:, None] + quat_apply(delta_ori, rpos[:, t] - rpos[:, a : a + 1])
    rel_quat = quat_mul(delta_ori, rquat[:, t])

    return {
        "body_pos": ((rel_pos - pos[:, t]) ** 2).sum(-1).mean(-1),
        "body_ori": quat_error_sq(rel_quat, quat[:, t]).mean(-1),
        "body_lin_vel": ((rlin_vel[:, t] - lin_vel[:, t]) ** 2).sum(-1).mean(-1),
        "body_ang_vel": ((rang_vel[:, t] - ang_vel[:, t]) ** 2).sum(-1).mean(-1),
        "anchor_pos": ((rpos[:, a] - pos[:, a]) ** 2).sum(-1),
        "anchor_ori": quat_error_sq(rquat[:, a], quat[:, a]),
    }


def tracking_reward(cfg, dt, errs, q, actions, prev_actions, soft_lower, soft_upper):
    """BeyondMimic tracking reward with SuperMimic's changes (Table A.2), every term times dt.

    ``errs`` are the tracking errors from ``tracking_errors``.
    """
    terms = {k: getattr(cfg, k)[0] * torch.exp(-e / getattr(cfg, k)[1] ** 2) * dt for k, e in errs.items()}
    terms["action_rate"] = cfg.action_rate * ((actions - prev_actions) ** 2).sum(-1) * dt
    violation = (q - soft_upper).clamp_min(0.0) + (soft_lower - q).clamp_min(0.0)
    terms["joint_limit"] = cfg.joint_limit * violation.sum(-1) * dt
    return sum(terms.values()), terms


def termination_errors(root_pos, root_quat, ref_root_pos, ref_root_quat):
    """Root height error and tilt error (difference of gravity z in the two root frames)."""
    height_err = (ref_root_pos[..., 2] - root_pos[..., 2]).abs()
    tilt_err = (projected_gravity(ref_root_quat)[..., 2] - projected_gravity(root_quat)[..., 2]).abs()
    return height_err, tilt_err
