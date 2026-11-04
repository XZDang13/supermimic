"""PPO-Sim baseline (SuperMimic §4.1): PPO directly on the simulator.

Follows the RLDemos PPO trainer (rollout -> GAE -> clipped PPO epochs) on RLAlg losses,
made GPU-native for the batched Newton env. Hyperparameters default to BeyondMimic's PPO:
24-step rollouts, 5 epochs x 4 minibatches, KL-adaptive learning rate (target 0.01),
entropy 0.005, value weight 1.0, clip 0.2, gamma 0.99, lambda 0.95.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from RLAlg.alg.ppo import PPO
from RLAlg.buffer import ReplayBuffer, compute_gae
from RLAlg.logger import WandbLogger
from RLAlg.nn.layers import NormPosition
from RLAlg.scheduler import KLAdaptiveLR
from RLAlg.utils import set_seed_everywhere

from supermimic.envs import G1TrackingEnv, G1TrackingEnvCfg
from supermimic.ppo.model import Actor, Critic, critic_observation, critic_observation_dim


@dataclass
class PPOCfg:
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 1e-3
    desired_kl: float = 0.01
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_ratio: float = 0.2
    entropy_weight: float = 0.005
    value_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    init_std: float = 1.0
    encoder_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    decoder_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    critic_dims: list[int] = field(default_factory=lambda: [256, 128])
    latent_dim: int = 32
    norm_position: str = "POST"


class Trainer:
    algorithm_name = "PPO-Sim"

    def __init__(self, env_cfg: G1TrackingEnvCfg, cfg: PPOCfg, *, seed: int = 0, log_dir: str = "runs/ppo",
                 use_wandb: bool = False):
        self.cfg = cfg
        self.seed = seed
        set_seed_everywhere(seed)
        env_cfg.seed = seed
        self.env = G1TrackingEnv(env_cfg)
        self.device = torch.device(env_cfg.device)
        self.num_envs = self.env.num_envs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        H, K = env_cfg.history_len, env_cfg.future_frames
        P, G, A = self.env.PROPRIO_DIM, self.env.GOAL_FRAME_DIM, self.env.NUM_ACTIONS
        C = critic_observation_dim(self.env)
        norm = NormPosition[cfg.norm_position]
        self.actor = Actor(H, P, K, G, A, cfg.encoder_dims, cfg.decoder_dims, cfg.latent_dim, norm, cfg.init_std).to(self.device)
        self.critic = Critic(C, cfg.critic_dims, norm).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.learning_rate)
        self.scheduler = KLAdaptiveLR(self.optimizer, cfg.desired_kl, min_lr=cfg.min_lr, max_lr=cfg.max_lr)

        self.replay_buffer = ReplayBuffer(self.num_envs, cfg.rollout_steps, device=self.device)
        for key, shape, dtype in (
            ("policy", (H, P), torch.float32),
            ("goal", (K, G), torch.float32),
            ("critic", (C,), torch.float32),
            ("actions", (A,), torch.float32),
            ("log_probs", (), torch.float32),
            ("rewards", (), torch.float32),
            ("values", (), torch.float32),
            ("terminated", (), torch.bool),
            ("truncated", (), torch.bool),
        ):
            self.replay_buffer.create_storage_space(key, shape, dtype)
        self.batch_keys = ["policy", "goal", "critic", "actions", "log_probs", "values", "returns", "advantages"]
        self.batch_size = self.num_envs * cfg.rollout_steps // cfg.minibatches

        self.global_step = 0
        self.iteration = 0
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, device=self.device)
        self.use_wandb = use_wandb
        if use_wandb:
            WandbLogger.init_project("supermimic", f"{self.algorithm_name}-{seed}",
                                     config={"ppo": asdict(cfg), "env": asdict(env_cfg)})

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def get_action(self, obs):
        critic_obs = critic_observation(obs)
        actor_step = self.actor(obs)
        value = self.critic(critic_obs).value
        return actor_step.action, actor_step.log_prob, value, critic_obs

    @torch.no_grad()
    def timeout_rewards(self, rewards, next_obs, terminated, truncated):
        # Timeouts (clip end / episode cap) bootstrap from V(final_obs), as in RLDemos PPO.
        timeouts = truncated & ~terminated
        values = self.critic(critic_observation(next_obs)).value
        return rewards + self.cfg.gamma * values * timeouts

    @torch.no_grad()
    def rollout(self, metrics):
        self.replay_buffer.reset()
        obs = self.obs
        for _ in range(self.cfg.rollout_steps):
            action, log_prob, value, critic_obs = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.global_step += self.num_envs
            self.replay_buffer.add_records({
                "policy": obs["policy"],
                "goal": obs["goal"],
                "critic": critic_obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": self.timeout_rewards(reward, info["next_obs"], terminated, truncated),
                "values": value,
                "terminated": terminated,
                "truncated": truncated,
            })
            self.track_episodes(reward, terminated, truncated, info, metrics)
            obs = next_obs
        self.obs = obs
        self.finish_rollout(obs)

    @torch.no_grad()
    def finish_rollout(self, obs):
        last_values = self.critic(critic_observation(obs)).value
        data = self.replay_buffer.data
        returns, advantages = compute_gae(
            rewards=data["rewards"], values=data["values"], terminated=data["terminated"],
            last_values=last_values, gamma=self.cfg.gamma, lambda_=self.cfg.lambda_, truncated=data["truncated"],
            bootstrap_timeouts=False,
        )
        self.replay_buffer.add_storage("returns", returns)
        self.replay_buffer.add_storage("advantages", advantages)

    def track_episodes(self, reward, terminated, truncated, info, metrics):
        self.episode_return += reward
        self.episode_length += 1
        for key, value in info["reward_terms"].items():
            metrics[f"reward/{key}"].append(value.mean())
        metrics["rollout/reward"].append(reward.mean())
        done = terminated | truncated
        metrics["rollout/termination_rate"].append(terminated.float().mean())
        if done.any():
            metrics["episode/return"].append(self.episode_return[done].mean())
            metrics["episode/length"].append(self.episode_length[done].mean())
            self.episode_return[done] = 0.0
            self.episode_length[done] = 0.0

    # ------------------------------------------------------------------ update

    def update(self, metrics):
        cfg = self.cfg
        for _ in range(cfg.epochs):
            for batch in self.replay_buffer.sample_batchs(self.batch_keys, self.batch_size):
                observations = {"policy": batch["policy"], "goal": batch["goal"]}
                policy = PPO.compute_policy_loss(
                    self.actor,
                    log_probs_hat=batch["log_probs"],
                    observations=observations,
                    actions=batch["actions"],
                    advantages=batch["advantages"],
                    clip_ratio=cfg.clip_ratio,
                    entropy_coef=cfg.entropy_weight,
                )
                value = PPO.compute_clipped_value_loss(
                    self.critic,
                    observations=batch["critic"],
                    values_hat=batch["values"],
                    returns=batch["returns"],
                    clip_ratio=cfg.clip_ratio,
                )
                loss = policy["loss"] + cfg.value_loss_weight * value["loss"]
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                # KL-adaptive learning rate, adjusted per minibatch as in rsl_rl
                self.scheduler.set_kl(policy["kl_divergence"].item())
                self.scheduler.step()
                for prefix, result in (("policy", policy), ("value", value)):
                    for key, metric in result.items():
                        name = f"{prefix}_loss" if key == "loss" else key
                        metrics[f"update/{name}"].append(metric.detach())
        metrics["update/learning_rate"].append(self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def update_normalizers(self):
        # Stats stay fixed while a rollout is collected and optimized, so the stored
        # log-probs match the policy at the first minibatch; refresh them afterwards.
        data = self.replay_buffer.data
        self.actor.update_normalizers({"policy": data["policy"], "goal": data["goal"]})
        self.critic.update_normalizers(data["critic"])

    # ------------------------------------------------------------------ loop

    def log(self, metrics, elapsed):
        # one host sync per metric, after the iteration
        info = {k: float(torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in v]).mean())
                for k, v in metrics.items()}
        info["perf/env_steps_per_s"] = self.cfg.rollout_steps * self.num_envs / elapsed
        info.update(self.env.sampler.stats())
        if self.use_wandb:
            WandbLogger.log_metrics(info, self.global_step)
        keys = ("episode/return", "episode/length", "rollout/reward", "rollout/termination_rate",
                "update/kl_divergence", "update/value_explained_variance", "update/action_std", "update/learning_rate", "perf/env_steps_per_s")
        shown = " ".join(f"{k.split('/')[-1]}={info[k]:.4g}" for k in keys if k in info)
        print(f"[{self.iteration:6d}] steps={self.global_step:.3e} {shown}", flush=True)
        return info

    def train(self, iterations: int, save_interval: int = 500):
        self.obs = self.env.reset()
        for _ in range(iterations):
            metrics = defaultdict(list)
            t0 = time.time()
            self.rollout(metrics)
            self.update(metrics)
            self.update_normalizers()
            torch.cuda.synchronize()
            self.iteration += 1
            self.log(metrics, time.time() - t0)
            if self.iteration % save_interval == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        self.save(self.log_dir / "model_final.pt")

    # ------------------------------------------------------------------ checkpoints

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "robot": self.env.cfg.robot,
            "iteration": self.iteration,
            "global_step": self.global_step,
            "ppo_cfg": asdict(self.cfg),
            "sampler": {k: getattr(self.env.sampler, k) for k in ("bin_score", "bin_visits", "past_weight")},
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.iteration, self.global_step = ckpt["iteration"], ckpt["global_step"]
        for k, v in ckpt.get("sampler", {}).items():
            if getattr(self.env.sampler, k).shape == v.shape:
                getattr(self.env.sampler, k).copy_(v)

    def close(self):
        if self.use_wandb:
            WandbLogger.finish_project()
