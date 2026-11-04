"""Vectorized G1 whole-body motion-tracking environment on Newton + MuJoCo Warp.

Implements the simulator side of SuperMimic (Appendix A): the BeyondMimic
tracking task with SuperMimic's changes (no self-collision penalty, joint-limit
weight -2, height/tilt-only termination, no root-velocity pushes, per-episode
sim context xi, return-based reset sampling).

Everything is batched torch on the GPU. Physics runs at 200 Hz (4 x 5 ms
substeps) under a 50 Hz policy, captured in one CUDA graph.

Observation dict (all float32, leading dim ``num_envs``):

* ``proprio``  (93,)       noise-free proprioception o-bar_t
* ``policy``   (H, 93)     last H noisy proprioceptive frames o_t (oldest first)
* ``goal``     (K, 67)     next K reference frames, relative to the robot root
* ``state``    (71,)       full generalized state s_t (root pose/twist, joint q/qd)
* ``context``  (33,)       sim context xi: friction, joint calibration offsets, torso CoM offset
"""

from __future__ import annotations

from dataclasses import dataclass, field

import newton
import torch
import warp as wp

from supermimic.motion.library import MotionLibrary
from supermimic.envs.tracking import goal_obs, proprio_obs, termination_errors, tracking_errors, tracking_reward
from supermimic.motion.sampler import ReturnBasedSampler
from supermimic.robots import g1
from supermimic.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, quat_normalize


@dataclass
class RewardCfg:
    # (weight, std) of the exp(-err / std^2) tracking terms, BeyondMimic values
    body_pos: tuple[float, float] = (1.0, 0.3)
    body_ori: tuple[float, float] = (1.0, 0.4)
    body_lin_vel: tuple[float, float] = (1.0, 1.0)
    body_ang_vel: tuple[float, float] = (1.0, 3.14)
    anchor_pos: tuple[float, float] = (0.5, 0.3)
    anchor_ori: tuple[float, float] = (0.5, 0.4)
    action_rate: float = -0.1
    joint_limit: float = -2.0


@dataclass
class G1TrackingEnvCfg:
    motion_files: list[str] | str = "data/motions"
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 0

    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 10.0
    use_cuda_graph: bool = True

    # MuJoCo Warp
    self_collisions: bool = True
    solver_iterations: int = 10
    ls_iterations: int = 20
    njmax: int = 600       # 150 contacts x 4 pyramidal rows; 300 overflowed on falls
    nconmax: int = 150
    cone: str = "pyramidal"

    history_len: int = 10
    future_frames: int = 10
    robot: str = "g1_29dof"                 # key of robots.g1.ROBOTS
    anchor_body: str = g1.ROOT_BODY
    tracked_bodies: list[str] | None = None  # None: the robot's tracked set
    soft_joint_limit_factor: float = 0.9
    rewards: RewardCfg = field(default_factory=RewardCfg)

    # termination
    max_root_height_error: float = 0.25
    max_tilt_error: float = 0.8

    # sensor noise eta_t ~ U[-n, n] (Table A.1)
    add_noise: bool = True
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.5
    noise_ang_vel: float = 0.2
    noise_gravity: float = 0.05

    # sim context xi (Table A.3), resampled every episode
    randomize: bool = True
    friction_range: tuple[float, float] = (0.3, 1.2)
    nominal_friction: float = 1.0
    joint_offset_range: float = 0.01
    torso_com_range: tuple[tuple[float, float], ...] = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))

    # reset perturbation around the sampled reference frame (BeyondMimic)
    reset_pose_range: tuple[tuple[float, float], ...] = (
        (-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01), (-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2),
    )  # x, y, z, roll, pitch, yaw
    reset_vel_range: tuple[tuple[float, float], ...] = (
        (-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2), (-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78),
    )
    reset_joint_pos_range: float = 0.1

    # return-based reset sampling (Appendix D.1)
    adaptive_sampling: bool = True
    sampler_gamma: float = 0.99
    sampler_window: int = 250


def env_dims(num_actions: int) -> dict[str, int]:
    """Observation, goal, state and context sizes for a robot with ``num_actions`` joints."""
    return {
        "NUM_ACTIONS": num_actions,
        "PROPRIO_DIM": 3 * num_actions + 6,
        "GOAL_FRAME_DIM": 2 * num_actions + 9,
        "STATE_DIM": 13 + 2 * num_actions,
        "CONTEXT_DIM": 1 + num_actions + 3,
    }


class G1TrackingEnv:
    def __init__(self, cfg: G1TrackingEnvCfg):
        self.cfg = cfg
        self.robot = g1.ROBOTS[cfg.robot]
        # per-robot sizes (NUM_ACTIONS, PROPRIO_DIM, GOAL_FRAME_DIM, STATE_DIM, CONTEXT_DIM)
        self.__dict__.update(env_dims(self.robot.num_joints))
        self.tracked_bodies = list(cfg.tracked_bodies or self.robot.tracked_bodies)
        self.num_envs = n = cfg.num_envs
        self.device = cfg.device
        self.control_dt = cfg.sim_dt * cfg.decimation
        self.max_episode_steps = int(round(cfg.episode_length_s / self.control_dt))
        if cfg.use_cuda_graph and cfg.decimation % 2:
            raise ValueError("CUDA-graph capture swaps state buffers, so decimation must be even")
        torch.manual_seed(cfg.seed)
        wp.config.quiet = True
        if wp.get_device(self.device).is_cuda:
            # Torch and Warp share buffers zero-copy, so they must share one stream. Torch
            # moves onto Warp's stream (not the reverse: graph capture cannot use the legacy
            # default stream). This sets the process-wide current torch stream.
            torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(self.device)))

        self.motions = MotionLibrary(cfg.motion_files, fps=1.0 / self.control_dt, device=self.device,
                                     joint_names=self.robot.joint_names)
        self._build_sim()
        self._init_buffers()

        r = cfg.rewards
        max_step_reward = (
            r.body_pos[0] + r.body_ori[0] + r.body_lin_vel[0] + r.body_ang_vel[0] + r.anchor_pos[0] + r.anchor_ori[0]
        ) * self.control_dt
        self.sampler = ReturnBasedSampler(
            self.motions, n, max_step_reward, gamma=cfg.sampler_gamma, window=cfg.sampler_window
        )

    # ------------------------------------------------------------------ build

    def _build_sim(self):
        cfg = self.cfg
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.default_shape_cfg.mu = cfg.nominal_friction
        robot.add_mjcf(self.robot.mjcf_path(), floating=True, enable_self_collisions=cfg.self_collisions)

        short = lambda label: label.rsplit("/", 1)[-1]
        joint_names = [short(lbl) for lbl in robot.joint_label]
        self.body_names = [short(lbl) for lbl in robot.body_label]
        self.bodies_per_world = robot.body_count
        self.coords_per_world = robot.joint_coord_count
        self.dofs_per_world = robot.joint_dof_count
        assert self.coords_per_world == 7 + self.NUM_ACTIONS and self.dofs_per_world == 6 + self.NUM_ACTIONS

        # Newton dof index of each actuated joint, in the robot's joint order
        dof_idx = [robot.joint_qd_start[joint_names.index(j)] for j in self.robot.joint_names]
        assert dof_idx == list(range(6, 6 + self.NUM_ACTIONS)), "expected MJCF joint order"
        act = self.robot.actuator_params()
        q0 = self.robot.default_joint_pos()
        for i, d in enumerate(dof_idx):
            robot.joint_target_ke[d] = act["stiffness"][i]
            robot.joint_target_kd[d] = act["damping"][i]
            robot.joint_armature[d] = act["armature"][i]
            robot.joint_effort_limit[d] = act["effort_limit"][i]
            robot.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)
            robot.joint_target_pos[d] = q0[i]
            robot.joint_q[7 + i] = q0[i]
        robot.joint_q[2] = self.robot.default_root_height

        builder = newton.ModelBuilder()
        builder.replicate(robot, self.num_envs)
        builder.default_shape_cfg.mu = cfg.friction_range[0]
        # MuJoCo takes the max of the two geoms' friction, so a ground at the lowest
        # sampled friction lets the robot's own per-env coefficient decide.
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            cone=cfg.cone,
            iterations=cfg.solver_iterations,
            ls_iterations=cfg.ls_iterations,
            njmax=cfg.njmax,
            nconmax=cfg.nconmax,
            use_mujoco_contacts=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.ref_state = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.ref_joint_q = wp.zeros_like(self.model.joint_q)
        self.ref_joint_qd = wp.zeros_like(self.model.joint_qd)
        self._fk_mask_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fk_mask = wp.from_torch(self._fk_mask_t, dtype=wp.bool)

        self.graph = None
        if self.cfg.use_cuda_graph and wp.get_device(self.device).is_cuda:
            self._simulate()  # warm up / compile outside the capture
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate()
            self.graph = capture.graph

    def _simulate(self):
        for _ in range(self.cfg.decimation):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.cfg.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _init_buffers(self):
        n, d, na = self.num_envs, self.device, self.NUM_ACTIONS
        B = self.bodies_per_world

        # zero-copy torch views of Newton arrays
        self.joint_q = wp.to_torch(self.state_0.joint_q).view(n, self.coords_per_world)
        self.joint_qd = wp.to_torch(self.state_0.joint_qd).view(n, self.dofs_per_world)
        self.body_q = wp.to_torch(self.state_0.body_q).view(n, B, 7)
        self.body_qd = wp.to_torch(self.state_0.body_qd).view(n, B, 6)
        self.ref_q = wp.to_torch(self.ref_joint_q).view(n, self.coords_per_world)
        self.ref_qd = wp.to_torch(self.ref_joint_qd).view(n, self.dofs_per_world)
        self.ref_body_q = wp.to_torch(self.ref_state.body_q).view(n, B, 7)
        self.ref_body_qd = wp.to_torch(self.ref_state.body_qd).view(n, B, 6)
        self.joint_target = wp.to_torch(self.control.joint_target_pos).view(n, self.dofs_per_world)
        self.body_com = wp.to_torch(self.model.body_com).view(n, B, 3)
        self.shape_mu = wp.to_torch(self.model.shape_material_mu)
        shape_world = wp.to_torch(self.model.shape_world).long()
        robot_shapes = (shape_world >= 0).nonzero(as_tuple=False).squeeze(-1)
        self._robot_shape_ids = robot_shapes
        self._robot_shape_world = shape_world[robot_shapes]

        self.root_idx = self.body_names.index(self.robot.root_body)
        self.anchor_idx = self.body_names.index(self.cfg.anchor_body)
        self.torso_idx = self.body_names.index(self.robot.torso_body)
        self.tracked_idx = torch.tensor([self.body_names.index(b) for b in self.tracked_bodies], device=d)
        self.nominal_torso_com = self.body_com[0, self.torso_idx].clone()
        # MuJoCo Warp's per-world body CoM, written directly on reset: notifying
        # BODY_INERTIAL_PROPERTIES re-runs set_const over every world (~50 ms at 4096
        # envs), and a few-cm CoM shift at fixed mass leaves those constants unchanged.
        self.mjw_body_ipos = wp.to_torch(self.solver.mjw_model.body_ipos)
        mjc_to_newton = wp.to_torch(self.solver.mjc_body_to_newton)[0]
        self.mjc_torso_idx = int((mjc_to_newton == self.torso_idx).nonzero()[0, 0])
        assert torch.allclose(self.mjw_body_ipos[0, self.mjc_torso_idx], self.nominal_torso_com, atol=1e-6)

        act = self.robot.actuator_params()
        self.action_scale = torch.tensor(act["action_scale"], device=d)
        self.default_joint_pos = torch.tensor(self.robot.default_joint_pos(), device=d)
        lower = wp.to_torch(self.model.joint_limit_lower).view(n, -1)[0, 6:]
        upper = wp.to_torch(self.model.joint_limit_upper).view(n, -1)[0, 6:]
        mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower) * self.cfg.soft_joint_limit_factor
        self.soft_lower, self.soft_upper = mid - half, mid + half

        self.frame = torch.zeros(n, dtype=torch.long, device=d)
        self.pinned_reset_frames = None
        self.clip_end = torch.zeros(n, dtype=torch.long, device=d)
        self.episode_step = torch.zeros(n, dtype=torch.long, device=d)
        self.actions = torch.zeros(n, na, device=d)
        self.prev_actions = torch.zeros(n, na, device=d)
        self.history = torch.zeros(n, self.cfg.history_len, self.PROPRIO_DIM, device=d)

        # sim context xi
        self.friction = torch.full((n,), self.cfg.nominal_friction, device=d)
        self.joint_offset = torch.zeros(n, na, device=d)
        self.torso_com_offset = torch.zeros(n, 3, device=d)

        self.noise_scale = torch.cat(
            (
                torch.full((na,), self.cfg.noise_joint_pos),
                torch.full((na,), self.cfg.noise_joint_vel),
                torch.full((3,), self.cfg.noise_ang_vel),
                torch.full((3,), self.cfg.noise_gravity),
                torch.zeros(na),
            )
        ).to(d)
        self._future = torch.arange(1, self.cfg.future_frames + 1, device=d)
        self._reset_pose_range = torch.tensor(self.cfg.reset_pose_range, device=d)
        self._reset_vel_range = torch.tensor(self.cfg.reset_vel_range, device=d)
        self._torso_com_range = torch.tensor(self.cfg.torso_com_range, device=d)

    # ------------------------------------------------------------------ API

    def reset(self, frames: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Reset every env; ``frames`` (flat library indices) pins the start frames, e.g. clip starts for evaluation."""
        self._reset_envs(torch.arange(self.num_envs, device=self.device), frames)
        return self._compute_obs(fill_history=torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def reset_envs(self, env_ids: torch.Tensor, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reset only ``env_ids`` to ``frames`` between steps, e.g. to switch an env to another clip.

        The other envs keep their observation history, so their next policy input is unchanged.
        """
        self._reset_envs(env_ids, frames)
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        history = self.history.clone()
        obs = self._compute_obs(fill_history=mask)
        self.history[~mask] = history[~mask]
        obs["policy"] = self.history.clone()
        return obs

    def pin_reset_frames(self, frames: torch.Tensor | None):
        """Make every later reset (including auto-resets in ``step``) start env i at ``frames[i]``; None unpins."""
        self.pinned_reset_frames = None if frames is None else frames.to(self.device, torch.long).clone()

    def step(self, actions: torch.Tensor):
        """Advance one 50 Hz control step.

        Returns ``(obs, reward, terminated, truncated, info)``. Finished envs are
        reset in place; ``obs`` is post-reset, while ``info["next_obs"]`` holds the
        observation each env actually reached (what a replay buffer should store).
        """
        self.actions = actions.detach().clone()
        self.joint_target[:, 6:] = self.default_joint_pos + self.joint_offset + self.action_scale * self.actions
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

        self.episode_step += 1
        self.frame = torch.minimum(self.frame + 1, self.clip_end)
        self._reference_fk(self.frame)

        reward, terms = self._compute_reward()
        terminated = self._compute_termination()
        truncated = ((self.frame >= self.clip_end) | (self.episode_step >= self.max_episode_steps)) & ~terminated

        self.prev_actions = self.actions
        next_obs = self._compute_obs()
        if self.cfg.adaptive_sampling:
            self.sampler.update(reward, terminated, truncated, self.frame)

        done = terminated | truncated
        obs = next_obs
        if done.any():
            self._reset_envs(done.nonzero(as_tuple=False).squeeze(-1))
            obs = self._compute_obs(fill_history=done)
        info = {"next_obs": next_obs, "reward_terms": terms}
        return obs, reward, terminated, truncated, info

    def render(self, viewer, sim_time: float):
        viewer.begin_frame(sim_time)
        viewer.log_state(self.state_0)
        viewer.end_frame()

    # ------------------------------------------------------------------ robot / reference quantities

    def _origin_lin_vel(self, body_q, body_qd, body_com):
        """Link-origin linear velocity from Newton's (v_com, omega) body twist."""
        com_w = quat_apply(body_q[..., 3:], body_com)
        return body_qd[..., :3] - torch.cross(body_qd[..., 3:], com_w, dim=-1)

    def _reference_fk(self, frames: torch.Tensor):
        m = self.motions
        root_quat = m.root_quat[frames]
        omega = m.root_ang_vel[frames]
        com_w = quat_apply(root_quat, self.body_com[:, self.root_idx])
        self.ref_q[:, :3] = m.root_pos[frames]
        self.ref_q[:, 3:7] = root_quat
        self.ref_q[:, 7:] = m.joint_pos[frames]
        self.ref_qd[:, :3] = m.root_lin_vel[frames] + torch.cross(omega, com_w, dim=-1)
        self.ref_qd[:, 3:6] = omega
        self.ref_qd[:, 6:] = m.joint_vel[frames]
        newton.eval_fk(self.model, self.ref_joint_q, self.ref_joint_qd, self.ref_state)

    def _bodies(self, body_q, body_qd):
        pos, quat = body_q[..., :3], body_q[..., 3:]
        return pos, quat, self._origin_lin_vel(body_q, body_qd, self.body_com), body_qd[..., 3:]

    # ------------------------------------------------------------------ reward / termination

    def _compute_reward(self):
        bodies = self._bodies(self.body_q, self.body_qd)
        ref_bodies = self._bodies(self.ref_body_q, self.ref_body_qd)
        errs = tracking_errors(bodies, ref_bodies, self.anchor_idx, self.tracked_idx)
        return tracking_reward(self.cfg.rewards, self.control_dt, errs, self.joint_q[:, 7:], self.actions,
                               self.prev_actions, self.soft_lower, self.soft_upper)

    def _compute_termination(self):
        root_pos, root_quat = self.body_q[:, self.root_idx, :3], self.body_q[:, self.root_idx, 3:]
        ref_pos, ref_quat = self.motions.root_pos[self.frame], self.motions.root_quat[self.frame]
        height_err, tilt_err = termination_errors(root_pos, root_quat, ref_pos, ref_quat)
        bad = ~torch.isfinite(self.joint_q).all(-1) | ~torch.isfinite(self.joint_qd).all(-1)
        return (height_err > self.cfg.max_root_height_error) | (tilt_err > self.cfg.max_tilt_error) | bad

    # ------------------------------------------------------------------ observations

    def _compute_obs(self, fill_history: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        root_quat = self.body_q[:, self.root_idx, 3:]
        q, qd = self.joint_q[:, 7:], self.joint_qd[:, 6:]
        proprio = proprio_obs(root_quat, self.body_qd[:, self.root_idx, 3:], q, qd, self.default_joint_pos,
                              self.joint_offset, self.prev_actions)
        noisy = proprio
        if self.cfg.add_noise:
            noisy = proprio + (2.0 * torch.rand_like(proprio) - 1.0) * self.noise_scale
        self.history = torch.roll(self.history, -1, dims=1)
        self.history[:, -1] = noisy
        if fill_history is not None:
            self.history[fill_history] = noisy[fill_history, None].expand(-1, self.cfg.history_len, -1)

        root_pos = self.body_q[:, self.root_idx, :3]
        r = self.root_idx
        root_lin_vel = self._origin_lin_vel(self.body_q[:, r], self.body_qd[:, r], self.body_com[:, r])
        state = torch.cat((root_pos, root_quat, root_lin_vel, self.body_qd[:, self.root_idx, 3:], q, qd), dim=-1)
        context = torch.cat((self.friction[:, None], self.joint_offset, self.torso_com_offset), dim=-1)
        return {
            "proprio": proprio,
            "policy": self.history.clone(),
            "goal": self._goal(root_pos, root_quat),
            "state": state,
            "context": context,
            "frame": self.frame.clone(),
        }

    def _goal(self, root_pos, root_quat):
        m = self.motions
        frames = torch.minimum(self.frame[:, None] + self._future, self.clip_end[:, None])  # (N, K)
        return goal_obs(m.root_pos[frames], m.root_quat[frames], m.joint_pos[frames], m.joint_vel[frames],
                        root_pos, root_quat, self.default_joint_pos)

    # ------------------------------------------------------------------ reset

    def _uniform(self, ranges: torch.Tensor, n: int) -> torch.Tensor:
        """Samples of shape (n, len(ranges)) from per-column (lo, hi) ``ranges``."""
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(n, len(ranges), device=self.device)

    def _reset_envs(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        n = env_ids.numel()
        cfg, m = self.cfg, self.motions
        if frames is None and self.pinned_reset_frames is not None:
            frames = self.pinned_reset_frames[env_ids]
        if frames is None:
            if cfg.adaptive_sampling:
                frames = self.sampler.sample_frames(n)
            else:
                frames = torch.randint(0, m.num_frames, (n,), device=self.device)
        self.frame[env_ids] = frames
        self.clip_end[env_ids] = m.clip_end[m.frame_clip[frames]]
        self.episode_step[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        if cfg.adaptive_sampling:
            self.sampler.begin(env_ids, frames)

        if cfg.randomize:
            self._randomize_context(env_ids)

        # robot state = reference frame + BeyondMimic-style perturbation
        pose = self._uniform(self._reset_pose_range, n)
        vel = self._uniform(self._reset_vel_range, n)
        root_quat = quat_normalize(quat_mul(quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), m.root_quat[frames]))
        omega = m.root_ang_vel[frames] + vel[:, 3:]
        joint_pos = m.joint_pos[frames] + cfg.reset_joint_pos_range * (2.0 * torch.rand(n, self.NUM_ACTIONS, device=self.device) - 1.0)
        joint_pos = torch.maximum(torch.minimum(joint_pos, self.soft_upper), self.soft_lower)
        com_w = quat_apply(root_quat, self.body_com[env_ids, self.root_idx])

        q = self.joint_q[env_ids]
        q[:, :3] = m.root_pos[frames] + pose[:, :3]
        q[:, 3:7] = root_quat
        q[:, 7:] = joint_pos
        qd = self.joint_qd[env_ids]
        qd[:, :3] = m.root_lin_vel[frames] + vel[:, :3] + torch.cross(omega, com_w, dim=-1)
        qd[:, 3:6] = omega
        qd[:, 6:] = m.joint_vel[frames]
        self.joint_q[env_ids] = q
        self.joint_qd[env_ids] = qd
        self.joint_target[env_ids, 6:] = self.default_joint_pos + self.joint_offset[env_ids]

        self._fk_mask_t.zero_()
        self._fk_mask_t[env_ids] = True
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, mask=self._fk_mask)

    def _randomize_context(self, env_ids: torch.Tensor):
        cfg, n = self.cfg, env_ids.numel()
        lo, hi = cfg.friction_range
        self.friction[env_ids] = lo + (hi - lo) * torch.rand(n, device=self.device)
        self.joint_offset[env_ids] = cfg.joint_offset_range * (2.0 * torch.rand(n, self.NUM_ACTIONS, device=self.device) - 1.0)
        self.torso_com_offset[env_ids] = self._uniform(self._torso_com_range, n)

        self.shape_mu[self._robot_shape_ids] = self.friction[self._robot_shape_world]
        torso_com = self.nominal_torso_com + self.torso_com_offset[env_ids]
        self.body_com[env_ids, self.torso_idx] = torso_com
        self.mjw_body_ipos[env_ids, self.mjc_torso_idx] = torso_com
        self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.SHAPE_PROPERTIES)
