"""Differentiable forward kinematics of one robot in torch, read from a Newton model.

Gives link-origin poses and twists from the generalized state, so the world model's analytic
reward head can backpropagate into the predicted state. Handles the G1's tree: one free root
joint, revolute joints whose child frame is the joint frame (``joint_X_c`` = identity).
Bodies at the same tree depth are computed together.
"""

from __future__ import annotations

import newton
import numpy as np
import torch

from supermimic.utils.math import axis_angle_to_quat, quat_apply, quat_mul


class TorchFK:
    def __init__(self, model: newton.Model, device):
        nb = model.body_count // model.world_count
        parent = model.joint_parent.numpy()[:nb]
        child = model.joint_child.numpy()[:nb]
        jtype = model.joint_type.numpy()[:nb]
        X_p = model.joint_X_p.numpy()[:nb]
        X_c = model.joint_X_c.numpy()[:nb]
        axis = model.joint_axis.numpy()
        qd_start = model.joint_qd_start.numpy()[:nb]

        assert (child == np.arange(nb)).all() and jtype[0] == int(newton.JointType.FREE) and parent[0] == -1
        assert (jtype[1:] == int(newton.JointType.REVOLUTE)).all(), "only revolute joints below the root"
        assert np.allclose(X_c, [0, 0, 0, 0, 0, 0, 1]), "child joint frames must be identity"
        self.num_bodies = nb
        depth = np.zeros(nb, dtype=int)
        for b in range(1, nb):
            depth[b] = depth[parent[b]] + 1
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        # per level: bodies, parents, their joint offset, joint axis and actuated dof index (q[:, 7+i])
        self.levels = []
        for d in range(1, depth.max() + 1):
            bodies = np.nonzero(depth == d)[0]
            self.levels.append((
                t(bodies, torch.long), t(parent[bodies], torch.long), t(X_p[bodies, :3]), t(X_p[bodies, 3:]),
                t(axis[qd_start[bodies]]), t(qd_start[bodies] - 6, torch.long),
            ))
        # bodies are concatenated level by level; position of each body in that order, and back
        order = np.concatenate([[0]] + [np.nonzero(depth == d)[0] for d in range(1, depth.max() + 1)])
        self.inverse_order = t(np.argsort(order), torch.long)
        self.parent_slot = [self.inverse_order[parents] for _, parents, *_ in self.levels]

    def __call__(self, root_pos, root_quat, root_lin_vel, root_ang_vel, q, qd):
        """Link-origin (pos, quat, lin_vel, ang_vel), each (N, B, 3|4), world frame.

        ``root_lin_vel`` is the root link-origin velocity, as in the env state.
        """
        all_pos, all_quat, all_lin, all_ang = root_pos[:, None], root_quat[:, None], root_lin_vel[:, None], root_ang_vel[:, None]
        for (bodies, _, xp_pos, xp_quat, axis, dof), idx in zip(self.levels, self.parent_slot):
            p_pos, p_quat, p_lin, p_ang = all_pos[:, idx], all_quat[:, idx], all_lin[:, idx], all_ang[:, idx]
            n = len(bodies)
            joint_quat = quat_mul(p_quat, xp_quat.expand(p_quat.shape[0], n, 4))
            b_pos = p_pos + quat_apply(p_quat, xp_pos.expand_as(p_pos))
            b_quat = quat_mul(joint_quat, axis_angle_to_quat(axis * q[:, dof, None]))
            b_lin = p_lin + torch.cross(p_ang, b_pos - p_pos, dim=-1)
            b_ang = p_ang + quat_apply(joint_quat, axis.expand_as(p_pos)) * qd[:, dof, None]
            all_pos = torch.cat((all_pos, b_pos), 1)
            all_quat = torch.cat((all_quat, b_quat), 1)
            all_lin = torch.cat((all_lin, b_lin), 1)
            all_ang = torch.cat((all_ang, b_ang), 1)
        inv = self.inverse_order
        return all_pos[:, inv], all_quat[:, inv], all_lin[:, inv], all_ang[:, inv]
