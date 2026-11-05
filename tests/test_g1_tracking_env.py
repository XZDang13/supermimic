"""Consistency checks for the G1 tracking env. Run: pytest tests -s

Needs the synthetic clips from ``python scripts/make_test_motions.py``.
"""

import pytest
import torch
import warp as wp

from supermimic.envs import G1TrackingEnv, G1TrackingEnvCfg
from supermimic.utils.math import quat_conj, quat_error_magnitude, quat_mul, quat_to_axis_angle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

NO_PERTURB = ((0.0, 0.0),) * 6


def make_env(**kw):
    cfg = dict(
        motion_files="data/motions/test",
        num_envs=16,
        add_noise=False,
        reset_pose_range=NO_PERTURB,
        reset_vel_range=NO_PERTURB,
        reset_joint_pos_range=0.0,
    )
    cfg.update(kw)
    return G1TrackingEnv(G1TrackingEnvCfg(**cfg))


@pytest.fixture(scope="module")
def env():
    return make_env()


def test_reset_matches_reference(env):
    env.reset()
    env._reference_fk(env.frame)
    pos, quat, lin, ang = env._bodies(env.body_q, env.body_qd)
    rpos, rquat, rlin, rang = env._bodies(env.ref_body_q, env.ref_body_qd)
    assert torch.allclose(pos, rpos, atol=1e-5)
    assert quat_error_magnitude(quat, rquat).max() < 1e-4
    assert torch.allclose(lin, rlin, atol=1e-4)
    assert torch.allclose(ang, rang, atol=1e-4)
    reward, terms = env._compute_reward()
    tracking = sum(v for k, v in terms.items() if k not in ("action_rate", "joint_limit"))
    assert torch.allclose(tracking, torch.full_like(tracking, 0.1), atol=1e-5)


def test_reference_velocities_match_finite_differences(env):
    """FK twists of the reference agree with differentiating FK poses over frames."""
    m = env.motions
    clip = m.names.index("wave")
    f = torch.arange(int(m.clip_start[clip]) + 5, int(m.clip_start[clip]) + 5 + env.num_envs, device=env.device)
    poses = []
    for df in (-1, 0, 1):
        env._reference_fk(f + df)
        poses.append(env._bodies(env.ref_body_q.clone(), env.ref_body_qd.clone()))
    (p0, q0, _, _), (_, _, lin, ang), (p2, q2, _, _) = poses
    fd_lin = (p2 - p0) / (2 * m.dt)
    fd_ang = quat_to_axis_angle(quat_mul(q2, quat_conj(q0))) / (2 * m.dt)
    assert (fd_lin - lin).abs().max() < 0.05
    assert (fd_ang - ang).abs().max() < 0.1


def test_domain_randomization_reaches_mujoco():
    env = make_env(num_envs=8, randomize=True)
    env.reset()
    mjw = env.solver.mjw_model
    friction = wp.to_torch(mjw.geom_friction)[..., 0]  # (worlds, geoms)
    geom_world_max = friction.max(dim=1).values
    assert torch.allclose(geom_world_max, env.friction.clamp_min(env.cfg.friction_range[0]), atol=1e-5)
    ipos = env.mjw_body_ipos[:, env.mjc_torso_idx]
    assert torch.allclose(ipos, env.nominal_torso_com + env.torso_com_offset, atol=1e-6)
    assert env.joint_offset.abs().max() <= env.cfg.joint_offset_range


def test_graph_matches_eager():
    outs = []
    for graph in (True, False):
        torch.manual_seed(0)
        env = make_env(num_envs=4, use_cuda_graph=graph, randomize=False, adaptive_sampling=False, seed=0)
        env.reset()
        for _ in range(10):
            env.step(torch.zeros(env.num_envs, env.NUM_ACTIONS, device=env.device))
        outs.append(env.joint_q.clone())
    assert torch.allclose(outs[0], outs[1], atol=1e-4)
