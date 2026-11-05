"""The MuJoCo evaluator against the training env: same physics inputs, observations and scores."""

import os

import numpy as np
import pytest
import torch

from supermimic.envs import G1TrackingEnv
from supermimic.eval import eval_env_cfg
from supermimic.motion.library import MotionLibrary
from supermimic.sim2sim import MujocoTracker

MOTIONS = {"g1_29dof": "data/motions/test", "g1_23dof": "data/motions/test_23dof.npz"}


@pytest.fixture(scope="module", params=list(MOTIONS))
def pair(request):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if not os.path.exists(MOTIONS[request.param]):
        pytest.skip(f"{MOTIONS[request.param]} missing (scripts/pack_motions.py --limit 20)")
    cfg = eval_env_cfg(MOTIONS[request.param], 1, robot=request.param, use_cuda_graph=False)
    env = G1TrackingEnv(cfg)
    cpu = MotionLibrary(cfg.motion_files, fps=env.motions.fps, device="cpu", joint_names=env.robot.joint_names)
    return env, MujocoTracker(cfg, cpu)


def test_model_matches_training_actuators(pair):
    env, sim = pair
    m, act = sim.model, env.robot.actuator_params()
    dofs = sim.dadr
    assert np.allclose(m.dof_armature[dofs], act["armature"])
    assert np.allclose(m.actuator_gainprm[sim.ctrl, 0], act["stiffness"])
    assert np.allclose(-m.actuator_biasprm[sim.ctrl, 2], act["damping"])
    assert np.allclose(-m.actuator_biasprm[sim.ctrl, 1], act["stiffness"])
    joints = m.dof_jntid[dofs]
    assert np.allclose(m.jnt_actfrcrange[joints, 1], act["effort_limit"])


def test_reset_observation_matches_env(pair):
    env, sim = pair
    clip = 0
    ref = env.reset(frames=env.motions.clip_start[clip:clip + 1])
    obs = sim.reset(clip)
    for key in ("proprio", "policy", "goal"):
        assert torch.allclose(obs[key], ref[key].cpu(), atol=2e-3), key


def test_reward_matches_env_on_the_same_state(pair):
    env, sim = pair
    clip = 0
    frames = env.motions.clip_start[clip:clip + 1]
    env.reset(frames=frames)
    sim.reset(clip)
    action = torch.zeros(1, env.NUM_ACTIONS)
    # one step from the reference state: contact differs, the first-step tracking terms barely can
    _, r_env, *_ = env.step(action.to(env.device))
    _, r_sim, *_ = sim.step(action)
    assert abs(r_sim - float(r_env)) < 0.1 * env.control_dt


def test_body_kinematics_match_env(pair):
    """The scoring inputs on a moving state: world poses and link-origin twists of the tracked bodies."""
    env, sim = pair
    env.reset(frames=env.motions.clip_start[:1])
    torch.manual_seed(0)
    for _ in range(15):
        obs, *_ = env.step(0.3 * torch.randn(1, env.NUM_ACTIONS, device=env.device))
    s, j = obs["state"][0].cpu(), env.NUM_ACTIONS
    sim._set_state(sim.data, s[:3], s[3:7], s[13 : 13 + j], s[7:10], s[10:13], s[13 + j :])
    sim._kinematics(sim.data)
    pos, quat, lin, ang = sim._bodies(sim.data)
    ids = torch.cat((torch.tensor([env.anchor_idx], device=env.device), env.tracked_idx))
    rpos, rquat, rlin, rang = (x[:, ids].cpu() for x in env._bodies(env.body_q, env.body_qd))
    assert (pos - rpos).abs().max() < 1e-3
    assert torch.minimum((quat - rquat).abs().amax(-1), (quat + rquat).abs().amax(-1)).max() < 1e-3
    assert (ang - rang).abs().max() < 1e-2
    assert (lin - rlin).abs().max() < 1e-2
