"""World-model pieces against the simulator: torch FK vs Newton FK, analytic heads vs env."""

import pytest
import torch

from supermimic.envs import G1TrackingEnv, G1TrackingEnvCfg
from supermimic.wm.kinematics import TorchFK

import os

MOTIONS = {"g1_29dof": "data/motions/test", "g1_23dof": "data/motions/test_23dof.npz"}


@pytest.fixture(scope="module", params=list(MOTIONS))
def env(request):
    if not os.path.exists(MOTIONS[request.param]):
        pytest.skip(f"{MOTIONS[request.param]} missing (scripts/pack_motions.py --limit 20)")
    e = G1TrackingEnv(G1TrackingEnvCfg(motion_files=MOTIONS[request.param], num_envs=32, use_cuda_graph=False,
                                       robot=request.param))
    e.reset()
    for _ in range(10):  # moving, contact-rich states
        e.step(0.5 * torch.randn(e.num_envs, e.NUM_ACTIONS, device=e.device))
    return e


def test_fk_matches_newton(env):
    fk = TorchFK(env.model, env.device)
    s = env._compute_obs()["state"]
    from supermimic.wm.task import split_state

    pos, quat, lin, ang = fk(*split_state(s))
    rpos, rquat, rlin, rang = env._bodies(env.body_q, env.body_qd)
    assert (pos - rpos).abs().max() < 1e-4
    assert torch.minimum((quat - rquat).abs().amax(-1), (quat + rquat).abs().amax(-1)).max() < 1e-4
    assert (lin - rlin).abs().max() < 1e-3
    assert (ang - rang).abs().max() < 1e-3


def test_analytic_heads_match_env(env):
    from supermimic.wm.task import TrackingTask

    task = TrackingTask(env)
    a0 = 0.3 * torch.randn(env.num_envs, env.NUM_ACTIONS, device=env.device)
    env.step(a0)
    a1 = 0.3 * torch.randn(env.num_envs, env.NUM_ACTIONS, device=env.device)
    _, reward, terminated, truncated, info = env.step(a1)
    nxt = info["next_obs"]
    s, ctx, frames = nxt["state"], nxt["context"], nxt["frame"]
    ok = ~(terminated | truncated)
    r_hat, _ = task.reward(s, a1, a0, frames)
    assert (r_hat - reward)[ok].abs().max() < 1e-5
    assert (task.proprio(s, a1, ctx) - nxt["proprio"]).abs().max() < 1e-5
    assert (task.goal(s, frames, task.clip_end(frames)) - nxt["goal"]).abs().max() < 1e-5
    assert torch.equal(task.termination(s, frames, hard=True).bool(), terminated)
    d = task.termination(s, frames)
    assert ((d > 0.5) == terminated).float().mean() > 0.9


def test_replay_windows_and_history():
    from supermimic.wm.replay import TransitionBuffer

    n, cap = 3, 20
    buf = TransitionBuffer(n, cap, state_dim=1, action_dim=1, context_dim=1, device="cpu")
    for t in range(30):  # state = action = global step; env e resets every 4 + e steps
        done = torch.tensor([(t + 1) % (4 + e) == 0 for e in range(n)])
        x = torch.full((n, 1), float(t))
        buf.add(x, x, x, torch.zeros(n, dtype=torch.long), done)
    w = buf.sample_windows(256, 3)
    s, a = w["state"][w["valid"], :, 0], w["action"][w["valid"], :, 0]
    assert w["valid"].any() and (s[:, 1:] - s[:, :-1] == 1).all() and (a == s[:, :-1]).all()
    h = buf.sample_starts(256, 5)
    hs, hp, s0 = h["history_state"][..., 0], h["history_prev_action"][..., 0], h["state"][:, 0]
    assert (hs[:, -1] == s0).all() and (hs[:, 1:] >= hs[:, :-1]).all()
    moved = hs[:, 1:] > hs[:, :-1]
    assert (hs[:, 1:] - hs[:, :-1])[moved].eq(1).all()
    # previous action is the step before, or zero at the start of the history (episode start)
    assert ((hp == hs - 1) | (hp == 0)).all()
