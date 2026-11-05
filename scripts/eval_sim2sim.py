"""Score a checkpoint on stock CPU MuJoCo, the sim-to-sim check for ``scripts/eval_policy.py``.

Plays each clip from its first frame with deterministic actions and no randomization, on a model
built from the robot's MJCF (see ``supermimic/sim2sim.py``). ``--results`` takes the JSON from
eval_policy.py and prints both simulators side by side on the same clips.

    python scripts/eval_sim2sim.py runs/supermimic/model_final.pt --motions data/motions/eval1000_23dof.npz
    python scripts/eval_sim2sim.py CKPT --motions data/motions/bones_seed_23dof.npz --num-clips 2000 \\
        --results results.json --out sim2sim.json
"""

import argparse
import json
import os
import time

import torch

from supermimic.eval import checkpoint_robot, eval_env_cfg, load_policy
from supermimic.motion.library import MotionLibrary
from supermimic.robots import g1
from supermimic.sim2sim import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--num-clips", type=int, default=None, help="score a random subset of this many clips")
    p.add_argument("--clip", default=None, help="keep clips whose name contains this text")
    p.add_argument("--seed", type=int, default=0, help="seed of the --num-clips subset")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--sim-dt", type=float, default=None, help="physics step [s], a whole fraction of 20 ms (default: training's)")
    p.add_argument("--results", default=None, help="eval_policy.py JSON to compare with")
    p.add_argument("--out", default=None, help="write per-clip results as JSON (view_policy.py --results reads it)")
    args = p.parse_args()

    torch.set_num_threads(1)  # workers fork after this, and a forked OpenMP pool can hang
    cfg = eval_env_cfg(args.motions, 1, robot=checkpoint_robot(args.checkpoint), device="cpu", use_cuda_graph=False)
    motions = MotionLibrary(args.motions, fps=1.0 / (cfg.sim_dt * cfg.decimation), device="cpu",
                            joint_names=g1.ROBOTS[cfg.robot].joint_names)
    policy = load_policy(args.checkpoint, cfg, "cpu")
    print(motions)

    clips = [i for i in range(motions.num_clips) if not args.clip or args.clip.lower() in motions.names[i].lower()]
    if not clips:
        raise SystemExit(f"no clip name contains {args.clip!r}")
    if args.num_clips is not None and args.num_clips < len(clips):
        chosen = torch.randperm(len(clips), generator=torch.Generator().manual_seed(args.seed))[: args.num_clips]
        clips = [clips[i] for i in sorted(chosen.tolist())]

    start = time.time()

    def progress(done):
        if len(done) % max(1, len(clips) // 20) == 0:
            failed = sum(r["failed"] for r in done)
            print(f"{len(done)}/{len(clips)} clips, {failed} failed, {time.time() - start:.0f} s", flush=True)

    results = evaluate(cfg, motions, policy, clips, args.workers, args.sim_dt, progress)
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    failed = [r for r in results if r["failed"]]
    diverged = sum(r["diverged"] for r in results)
    print(f"\nMuJoCo: failed {len(failed)} / {len(results)} clips ({diverged} from solver divergence, not falls); mean normalized return "
          f"{mean(results, 'normalized_return'):.3f}; mean action rate {mean(results, 'action_rate'):.4f}")
    for r in sorted(failed, key=lambda r: r["name"])[:50]:
        print(f"  FAIL {r['name']} at step {r['steps']}{' (solver diverged)' if r['diverged'] else ''}")

    if args.results:
        by_name = {c["name"]: c for c in json.load(open(args.results))["clips"]}
        pairs = [(r, by_name[r["name"]]) for r in results if r["name"] in by_name]
        ref = [b for _, b in pairs]
        both = sum(a["failed"] and b["failed"] for a, b in pairs)
        print(f"training sim on the same {len(pairs)} clips: failed {sum(b['failed'] for b in ref)}; "
              f"mean normalized return {mean(ref, 'normalized_return'):.3f}; failed in both {both}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "simulator": "mujoco", "clips": results}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
