"""Analytic heads of the world model (§3.2) for the motion-tracking interface.

The world model learns only the dynamics ŝ' = f(ŝ, a, ξ). Observation, goal, reward and
termination are computed from ŝ with the same functions the simulator env uses
(``envs.tracking``), so they have no loss term and are exact given ŝ. Termination is the
one change: the env's hard thresholds become a sigmoid, so the survival weight ĉ stays
differentiable.

State layout (71): root pos 0:3, root quat (xyzw) 3:7, root link-origin linear velocity 7:10,
root angular velocity 10:13 (world frame), joint q 13:42, joint qd 42:71.
Context ξ (33): friction, 29 joint calibration offsets, torso CoM offset.
"""

from __future__ import annotations

import torch

from supermimic.envs import G1TrackingEnv
from supermimic.envs.tracking import goal_obs, proprio_obs, termination_errors, tracking_errors, tracking_reward
from supermimic.ppo.model import critic_observation
from supermimic.wm.kinematics import TorchFK

def split_state(s: torch.Tensor):
    na = (s.shape[-1] - 13) // 2
    return s[:, :3], s[:, 3:7], s[:, 7:10], s[:, 10:13], s[:, 13 : 13 + na], s[:, 13 + na :]


class TrackingTask:
    def __init__(self, env: G1TrackingEnv, termination_temperature: tuple[float, float] = (0.02, 0.05)):
        cfg = env.cfg
        self.device = env.device
        self.motions = m = env.motions
        self.fk = TorchFK(env.model, env.device)
        self.reward_cfg = cfg.rewards
        self.dt = env.control_dt
        self.anchor_idx, self.tracked_idx = env.anchor_idx, env.tracked_idx
        self.default_joint_pos = env.default_joint_pos
        self.soft_lower, self.soft_upper = env.soft_lower, env.soft_upper
        self.noise_scale = env.noise_scale if cfg.add_noise else torch.zeros_like(env.noise_scale)
        self.history_len, self.future_frames = cfg.history_len, cfg.future_frames
        self.max_height_err, self.max_tilt_err = cfg.max_root_height_error, cfg.max_tilt_error
        self.temperature = termination_temperature
        self._future = torch.arange(1, cfg.future_frames + 1, device=env.device)

        # only the anchor and the tracked bodies enter the reward
        keep = torch.cat((torch.tensor([env.anchor_idx], device=env.device), env.tracked_idx))
        self._keep = keep
        self.anchor_idx_kept, self.tracked_idx_kept = 0, torch.arange(1, len(keep), device=env.device)

    # ------------------------------------------------------------------ heads

    def clip_end(self, frames):
        return self.motions.clip_end[self.motions.frame_clip[frames]]

    def proprio(self, state, prev_actions, context):
        _, quat, _, ang, q, qd = split_state(state)
        return proprio_obs(quat, ang, q, qd, self.default_joint_pos, context[:, 1 : 1 + q.shape[-1]], prev_actions)

    def add_noise(self, proprio):
        return proprio + (2.0 * torch.rand_like(proprio) - 1.0) * self.noise_scale

    def goal(self, state, frames, clip_end):
        m = self.motions
        f = torch.minimum(frames[:, None] + self._future, clip_end[:, None])
        return goal_obs(m.root_pos[f], m.root_quat[f], m.joint_pos[f], m.joint_vel[f], state[:, :3], state[:, 3:7],
                        self.default_joint_pos)

    def bodies(self, state):
        pos, quat, lin, ang = self.fk(*split_state(state))
        k = self._keep
        return pos[:, k], quat[:, k], lin[:, k], ang[:, k]

    def reference_bodies(self, frames):
        """Reference link states at library ``frames``, by FK of the reference (constant: no gradient).

        Computed per step rather than stored per library frame, which would take 13 floats per body
        per frame (about 35 GB for the full BONES-SEED set).
        """
        m = self.motions
        with torch.no_grad():
            pos, quat, lin, ang = self.fk(m.root_pos[frames], m.root_quat[frames], m.root_lin_vel[frames],
                                          m.root_ang_vel[frames], m.joint_pos[frames], m.joint_vel[frames])
        k = self._keep
        return pos[:, k], quat[:, k], lin[:, k], ang[:, k]

    def reward(self, state, actions, prev_actions, frames):
        """Reward of reaching ``state`` at reference ``frames`` after ``actions``, and the tracking errors."""
        ref = self.reference_bodies(frames)
        errs = tracking_errors(self.bodies(state), ref, self.anchor_idx_kept, self.tracked_idx_kept)
        reward, _ = tracking_reward(self.reward_cfg, self.dt, errs, split_state(state)[4], actions, prev_actions,
                                    self.soft_lower, self.soft_upper)
        return reward, errs

    def termination(self, state, frames, hard: bool = False):
        """Termination probability d̂ in [0, 1] (sigmoid of the env's height and tilt margins)."""
        m = self.motions
        h, tilt = termination_errors(state[:, :3], state[:, 3:7], m.root_pos[frames], m.root_quat[frames])
        if hard:
            return ((h > self.max_height_err) | (tilt > self.max_tilt_err)).float()
        th, tt = self.temperature
        alive = torch.sigmoid((self.max_height_err - h) / th) * torch.sigmoid((self.max_tilt_err - tilt) / tt)
        return 1.0 - alive

    def critic_obs(self, proprio, state, goal, context):
        return critic_observation({"proprio": proprio, "state": state, "goal": goal, "context": context})
