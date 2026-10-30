"""Smoke test / throughput check for the G1 tracking env.

    python scripts/test_env.py --num-envs 4096 --policy ref
    python scripts/test_env.py --num-envs 4 --viewer          # watch it

Policies: ``zero`` holds the default pose; ``ref`` feeds the next reference joint
angles as PD setpoints (an open-loop, feed-forward-free tracker) and should
survive the gentle test clips.
"""

import argparse
import time

import torch

from supermimic.envs import G1TrackingEnv, G1TrackingEnvCfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--policy", choices=("zero", "ref", "random"), default="ref")
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--no-self-collisions", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()

    cfg = G1TrackingEnvCfg(
        motion_files=args.motions,
        num_envs=args.num_envs,
        randomize=not args.no_randomize,
        self_collisions=not args.no_self_collisions,
        use_cuda_graph=not args.no_graph,
    )
    t0 = time.time()
    env = G1TrackingEnv(cfg)
    print(env.motions)
    print(f"built {cfg.num_envs} envs in {time.time() - t0:.1f}s")
    obs = env.reset()
    for k, v in obs.items():
        print(f"  obs[{k!r}]: {tuple(v.shape)}")

    viewer = None
    if args.viewer:
        import newton.viewer

        viewer = newton.viewer.ViewerGL()
        viewer.set_model(env.model)

    def act():
        if args.policy == "zero":
            return torch.zeros(env.num_envs, env.NUM_ACTIONS, device=env.device)
        if args.policy == "random":
            return 0.5 * torch.randn(env.num_envs, env.NUM_ACTIONS, device=env.device)
        nxt = torch.minimum(env.frame + 1, env.clip_end)
        target = env.motions.joint_pos[nxt]
        return (target - env.default_joint_pos - env.joint_offset) / env.action_scale

    ret = torch.zeros(env.num_envs, device=env.device)
    ep_returns, ep_lengths, n_term, n_trunc = [], [], 0, 0
    length = torch.zeros(env.num_envs, device=env.device)
    term_sums = {}
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        obs, rew, term, trunc, info = env.step(act())
        ret += rew
        length += 1
        for k, v in info["reward_terms"].items():
            term_sums[k] = term_sums.get(k, 0.0) + float(v.mean())
        done = term | trunc
        n_term += int(term.sum())
        n_trunc += int(trunc.sum())
        if done.any():
            ep_returns.append(ret[done].cpu())
            ep_lengths.append(length[done].cpu())
            ret[done] = 0.0
            length[done] = 0.0
        if viewer is not None:
            if not viewer.is_running():
                break
            env.render(viewer, i * env.control_dt)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    steps = i + 1

    print(f"\n{steps} steps x {env.num_envs} envs in {elapsed:.1f}s -> {steps * env.num_envs / elapsed:,.0f} env-steps/s")
    print(f"terminations {n_term}, truncations {n_trunc}")
    if ep_lengths:
        L = torch.cat(ep_lengths)
        R = torch.cat(ep_returns)
        print(f"episodes {len(L)}: mean length {L.mean():.1f} steps, mean return {R.mean():.3f}, return/step {(R / L).mean():.4f} (max 0.1)")
    print("mean per-step reward terms:")
    for k, v in term_sums.items():
        print(f"  {k:14s} {v / steps:+.5f}")
    for k, v in env.sampler.stats().items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
