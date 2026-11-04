"""Train the PPO-Sim baseline on the G1 tracking env.

    python scripts/train_ppo.py --motions data/motions/test --num-envs 4096 --iterations 30000
    python scripts/train_ppo.py --resume runs/ppo/model_500.pt
"""

import argparse

from supermimic.envs import G1TrackingEnvCfg
from supermimic.ppo.trainer import PPOCfg, Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--robot", default="g1_29dof", choices=["g1_29dof", "g1_23dof"])
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/ppo")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--small", action="store_true", help="BeyondMimic-sized MLPs (512, 256, 128) for quick runs")
    args = p.parse_args()

    env_cfg = G1TrackingEnvCfg(motion_files=args.motions, num_envs=args.num_envs, device=args.device,
                               robot=args.robot)
    cfg = PPOCfg()
    if args.rollout_steps:
        cfg.rollout_steps = args.rollout_steps
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.small:
        cfg.encoder_dims, cfg.decoder_dims, cfg.critic_dims = [512, 256], [512, 256, 128], [512, 256, 128]

    trainer = Trainer(env_cfg, cfg, seed=args.seed, log_dir=args.log_dir, use_wandb=args.wandb)
    print(trainer.env.motions)
    if args.resume:
        trainer.load(args.resume)
    try:
        trainer.train(args.iterations, save_interval=args.save_interval)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
