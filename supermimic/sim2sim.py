"""Sim-to-sim evaluation: run a trained policy on stock CPU MuJoCo instead of Newton + MuJoCo Warp.

The robot comes straight from its MJCF, with the training PD gains, armature and effort limits and
implicitfast integration at the training substep. Everything Newton sets on its own stays at the
MJCF values: one floor plane, soft joint limits, and contact solver settings, with the solver run
to convergence. Observations, reward and termination are the shared terms of ``envs.tracking``, on
the benchmark settings of ``eval_env_cfg``, so scores read the same as ``scripts/eval_policy.py``.
"""

from __future__ import annotations

import multiprocessing

import mujoco
import numpy as np
import torch

from supermimic.envs import G1TrackingEnvCfg
from supermimic.envs.tracking import goal_obs, proprio_obs, termination_errors, tracking_errors, tracking_reward
from supermimic.motion.library import MotionLibrary
from supermimic.robots import g1


def build_model(robot: g1.Robot, sim_dt: float, iterations: int = 100, ls_iterations: int = 50) -> mujoco.MjModel:
    """The robot's MJCF with the training actuators; ctrl is the joint position target."""
    spec = mujoco.MjSpec.from_file(robot.mjcf_path())
    spec.option.timestep = sim_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = iterations
    spec.option.ls_iterations = ls_iterations
    act = robot.actuator_params()
    params = {n: i for i, n in enumerate(robot.joint_names)}
    for name, i in params.items():
        joint = spec.joint(name)
        joint.armature = act["armature"][i]
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = [-act["effort_limit"][i], act["effort_limit"][i]]
    for actuator in spec.actuators:
        i = params[actuator.target]
        actuator.set_to_position(kp=act["stiffness"][i], kv=act["damping"][i])
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec.compile()


class MujocoTracker:
    """One robot tracking one clip at a time; the observation dict matches a one-env ``G1TrackingEnv``."""

    def __init__(self, cfg: G1TrackingEnvCfg, motions: MotionLibrary, sim_dt: float | None = None):
        self.cfg, self.motions = cfg, motions
        self.robot = robot = g1.ROBOTS[cfg.robot]
        sim_dt = sim_dt or cfg.sim_dt
        self.substeps = int(round(cfg.sim_dt * cfg.decimation / sim_dt))
        self.model = m = build_model(robot, sim_dt)
        self.data, self.ref = mujoco.MjData(m), mujoco.MjData(m)

        joints = [m.joint(n) for n in robot.joint_names]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        joint_of_actuator = m.actuator_trnid[:, 0].tolist()
        self.ctrl = np.array([joint_of_actuator.index(j.id) for j in joints])
        free = next(i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
        self.root_q, self.root_d = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self.root = m.body(robot.root_body).id

        tracked = list(cfg.tracked_bodies or robot.tracked_bodies)
        self.tracked_ids = np.array([m.body(b).id for b in tracked])
        self.anchor_id = m.body(cfg.anchor_body).id
        # tracking_errors indexes bodies: anchor first, then the tracked set
        self.body_ids = np.concatenate(([self.anchor_id], self.tracked_ids))
        self.tracked_idx = torch.arange(1, len(tracked) + 1)

        act = robot.actuator_params()
        self.action_scale = torch.tensor(act["action_scale"])
        self.default_joint_pos = torch.tensor(robot.default_joint_pos())
        self.joint_offset = torch.zeros(1, robot.num_joints)
        lower, upper = torch.tensor(m.jnt_range[[j.id for j in joints]], dtype=torch.float32).T
        mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower) * cfg.soft_joint_limit_factor
        self.soft_lower, self.soft_upper = mid - half, mid + half
        self.future = torch.arange(1, cfg.future_frames + 1)

    # ------------------------------------------------------------------ state

    def _set_state(self, d: mujoco.MjData, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        """Writes a world-frame root state (xyzw quaternion, link-origin velocity) and joint state."""
        w = np.asarray(root_quat, dtype=np.float64)
        d.qpos[self.root_q : self.root_q + 3] = root_pos
        d.qpos[self.root_q + 3 : self.root_q + 7] = w[3], w[0], w[1], w[2]
        d.qpos[self.qadr] = joint_pos
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, d.qpos[self.root_q + 3 : self.root_q + 7])
        d.qvel[self.root_d : self.root_d + 3] = root_lin_vel
        d.qvel[self.root_d + 3 : self.root_d + 6] = rot.reshape(3, 3).T @ np.asarray(root_ang_vel, dtype=np.float64)
        d.qvel[self.dadr] = joint_vel

    def _kinematics(self, d: mujoco.MjData):
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_comVel(self.model, d)

    def _bodies(self, d: mujoco.MjData):
        """(pos, xyzw quat, lin vel, ang vel) of ``body_ids``, world frame at the link origins, each (1, B, ...)."""
        ids = self.body_ids
        pos, quat, cvel = d.xpos[ids], d.xquat[ids], d.cvel[ids]
        ang = cvel[:, :3]
        lin = cvel[:, 3:] + np.cross(ang, pos - d.subtree_com[self.model.body_rootid[ids]])
        as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)[None]
        return as_tensor(pos), as_tensor(quat[:, [1, 2, 3, 0]]), as_tensor(lin), as_tensor(ang)

    def _root(self):
        d = self.data
        as_tensor = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32)[None]
        pos = as_tensor(d.qpos[self.root_q : self.root_q + 3])
        quat = as_tensor(d.qpos[self.root_q + 3 : self.root_q + 7][[1, 2, 3, 0]])
        return pos, quat, as_tensor(d.cvel[self.root, :3])

    def _obs(self, fill_history: bool = False) -> dict[str, torch.Tensor]:
        d = self.data
        root_pos, root_quat, root_ang_vel = self._root()
        q = torch.as_tensor(d.qpos[self.qadr], dtype=torch.float32)[None]
        qd = torch.as_tensor(d.qvel[self.dadr], dtype=torch.float32)[None]
        proprio = proprio_obs(root_quat, root_ang_vel, q, qd, self.default_joint_pos, self.joint_offset, self.prev_actions)
        if fill_history:
            self.history = proprio[:, None].repeat(1, self.cfg.history_len, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], proprio[:, None]), dim=1)
        frames = (self.frame + self.future).clamp_max(self.last)
        m = self.clip
        goal = goal_obs(m["root_pos"][frames][None], m["root_quat"][frames][None], m["joint_pos"][frames][None],
                        m["joint_vel"][frames][None], root_pos, root_quat, self.default_joint_pos)
        return {"proprio": proprio, "policy": self.history.clone(), "goal": goal}

    # ------------------------------------------------------------------ episode

    def reset(self, clip: int) -> dict[str, torch.Tensor]:
        """Starts ``clip`` from its first frame: the benchmark's reset, with no perturbation."""
        m = self.motions
        frames = torch.arange(int(m.clip_start[clip]), int(m.clip_end[clip]) + 1)
        self.clip = {k: getattr(m, k)[frames] for k in
                     ("root_pos", "root_quat", "joint_pos", "root_lin_vel", "root_ang_vel", "joint_vel")}
        self.frame, self.last, self.diverged = 0, len(frames) - 1, False
        self.prev_actions = torch.zeros(1, self.robot.num_joints)

        mujoco.mj_resetData(self.model, self.data)
        c = self.clip
        joint_pos = torch.maximum(torch.minimum(c["joint_pos"][0], self.soft_upper), self.soft_lower)
        self._set_state(self.data, c["root_pos"][0], c["root_quat"][0], joint_pos, c["root_lin_vel"][0],
                        c["root_ang_vel"][0], c["joint_vel"][0])
        self.data.ctrl[self.ctrl] = self.default_joint_pos.numpy()
        self._kinematics(self.data)
        return self._obs(fill_history=True)

    def step(self, action: torch.Tensor):
        """Advances one control step; returns ``(obs, reward, terminated, truncated)``."""
        d, c = self.data, self.clip
        d.ctrl[self.ctrl] = (self.default_joint_pos + self.action_scale * action[0]).numpy()
        bad = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, d)
        self.diverged = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > bad or not np.isfinite(d.qpos).all()

        self.frame = min(self.frame + 1, self.last)
        f = self.frame
        self._set_state(self.ref, c["root_pos"][f], c["root_quat"][f], c["joint_pos"][f], c["root_lin_vel"][f],
                        c["root_ang_vel"][f], c["joint_vel"][f])
        self._kinematics(self.ref)
        self._kinematics(d)

        errs = tracking_errors(self._bodies(d), self._bodies(self.ref), 0, self.tracked_idx)
        q = torch.as_tensor(d.qpos[self.qadr], dtype=torch.float32)[None]
        reward, _ = tracking_reward(self.cfg.rewards, self.cfg.sim_dt * self.cfg.decimation, errs, q, action,
                                    self.prev_actions, self.soft_lower, self.soft_upper)
        root_pos, root_quat, _ = self._root()
        height, tilt = termination_errors(root_pos, root_quat, c["root_pos"][f:f + 1], c["root_quat"][f:f + 1])
        terminated = self.diverged or bool(height > self.cfg.max_root_height_error) or bool(tilt > self.cfg.max_tilt_error)
        self.prev_actions = action
        return self._obs(), float(reward), terminated, f >= self.last and not terminated


def rollout(tracker: MujocoTracker, policy, clip: int, on_step=None) -> dict:
    """One benchmark episode, scored as in ``scripts/eval_policy.py``; ``on_step(tracker)`` runs before each action."""
    obs, ret, steps, rate, prev, failed = tracker.reset(clip), 0.0, 0, 0.0, None, False
    done = False
    while not done:
        if on_step:
            on_step(tracker)
        with torch.no_grad():
            action = policy(obs)
        if prev is not None:
            rate += float((action - prev).square().sum())
        prev = action
        obs, reward, failed, truncated = tracker.step(action)
        ret += reward
        steps += 1
        done = failed or truncated
    return {"name": tracker.motions.names[clip], "failed": failed, "diverged": tracker.diverged,
            "normalized_return": ret / (0.1 * steps),
            "action_rate": rate / max(steps - 1, 1), "steps": steps}


_worker = None


def _init_worker(cfg, motions, policy, sim_dt):
    global _worker
    torch.set_num_threads(1)
    _worker = MujocoTracker(cfg, motions, sim_dt), policy


def _run(clip: int) -> dict:
    return rollout(*_worker, clip)


def evaluate(cfg, motions, policy, clips: list[int], workers: int, sim_dt: float | None = None, progress=None) -> list[dict]:
    """Per-clip results in the order of ``clips``; forked workers share the library and policy read-only."""
    with multiprocessing.get_context("fork").Pool(workers, _init_worker, (cfg, motions, policy, sim_dt)) as pool:
        out = []
        for result in pool.imap(_run, clips, chunksize=1):
            out.append(result)
            if progress:
                progress(out)
    return out
