"""Train SuperMimic: world model + BPTT actor-critic on the G1 tracking env.

    python scripts/train_supermimic.py --motions data/motions/bones_seed --num-envs 4096 --iterations 20000
    python scripts/train_supermimic.py --resume runs/supermimic/model_1000.pt
"""

import argparse
import os

# less fragmentation in torch's caching allocator, so the simulator's CUDA-graph launches still find memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from supermimic.envs import G1TrackingEnvCfg
from supermimic.wm.trainer import SuperMimicCfg, Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=320_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/supermimic")
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imagine-batch", type=int, default=None)
    p.add_argument("--wm-batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=None,
                   help="recompute imagined activations in backward to save memory (default: off)")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None,
                   help="bf16 autocast for the world-model fit and imagination (default: on)")
    args = p.parse_args()

    env_cfg = G1TrackingEnvCfg(motion_files=args.motions, num_envs=args.num_envs, device=args.device)
    cfg = SuperMimicCfg()
    if args.imagine_batch:
        cfg.imagine_batch = args.imagine_batch
    if args.wm_batch_size:
        cfg.wm_batch_size = args.wm_batch_size
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.checkpoint is not None:
        cfg.checkpoint_steps = args.checkpoint
    if args.bf16 is not None:
        cfg.bf16 = args.bf16

    trainer = Trainer(env_cfg, cfg, seed=args.seed, log_dir=args.log_dir, use_wandb=args.wandb)
    print(trainer.env.motions)
    if args.resume:
        trainer.load(args.resume)
    try:
        trainer.train(args.iterations, save_interval=args.save_interval, log_interval=args.log_interval)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
