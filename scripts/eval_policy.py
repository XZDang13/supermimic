"""Benchmark a PPO-Sim or SuperMimic checkpoint the way SuperMimic's App. F does.

Each clip is played once from its first frame with deterministic actions, foot friction 1.0,
no domain randomization, no observation noise and no reset perturbation. A clip fails if the
rollout triggers the training termination before the clip ends.

    python scripts/eval_policy.py runs/ppo/model_final.pt --motions data/motions/test
    python scripts/eval_policy.py runs/supermimic/model_final.pt --motions data/motions/bones_seed

To watch a policy, use scripts/view_policy.py.
"""

import argparse
import json

import torch

from supermimic.envs import G1TrackingEnv
from supermimic.eval import eval_env_cfg, load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--out", default=None, help="write per-clip results as JSON (view_policy.py --results reads it)")
    args = p.parse_args()

    env_cfg = eval_env_cfg(args.motions, args.num_envs)
    env = G1TrackingEnv(env_cfg)
    policy = load_policy(args.checkpoint, env_cfg, env.device)
    m = env.motions
    print(m)

    failed = torch.zeros(m.num_clips, dtype=torch.bool)
    tracked = torch.zeros(m.num_clips)
    action_rate = torch.zeros(m.num_clips)
    for start in range(0, m.num_clips, env.num_envs):
        clips = torch.arange(start, min(start + env.num_envs, m.num_clips), device=env.device)
        # spare envs replay the last clip of the chunk and are ignored
        env_clip = torch.cat((clips, clips[-1:].expand(env.num_envs - len(clips))))
        obs = env.reset(frames=m.clip_start[env_clip])
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        ret = torch.zeros(env.num_envs, device=env.device)
        steps = torch.zeros(env.num_envs, device=env.device)
        rate = torch.zeros(env.num_envs, device=env.device)
        prev = None
        while active.any():
            with torch.no_grad():
                action = policy(obs)
            if prev is not None:
                rate += (action - prev).square().sum(-1) * active
            prev = action
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward * active
            steps += active
            fail = terminated & active
            failed[env_clip[fail].cpu()] = True
            active &= ~(terminated | truncated)
        n = len(clips)
        tracked[clips.cpu()] = (ret[:n] / (0.1 * steps[:n])).cpu()
        action_rate[clips.cpu()] = (rate[:n] / (steps[:n] - 1).clamp_min(1)).cpu()

    print(f"\nfailed {int(failed.sum())} / {m.num_clips} clips; mean normalized return {tracked.mean():.3f}; "
          f"mean action rate |a_t - a_t-1|^2 {action_rate.mean():.4f}")
    for i in failed.nonzero().squeeze(-1).tolist()[:50]:
        print(f"  FAIL {m.names[i]}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "clips": [
                {"name": m.names[i], "failed": bool(failed[i]), "normalized_return": float(tracked[i]),
                 "action_rate": float(action_rate[i])}
                for i in range(m.num_clips)]}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
