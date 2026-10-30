"""SuperMimic training loop (Algorithm A.1, Table A.5).

Each iteration:
1. Collect: the deterministic policy plus action noise N(0, 0.1²) steps the simulator; the
   executed actions go into the replay buffer (1000 steps per env). With an update-to-data
   ratio of 0.4, an iteration collects 2.5 simulator steps per env on average.
2. Fit the world model: 10 batches of 2048 k=8 windows, autoregressive on logged actions (Eq. 1).
3. Improve the policy: 4096 imagined rollouts of H=32 steps from replay states through the
   frozen world model, sensor noise re-added to each predicted observation; maximize the
   survival-weighted return bootstrapped by the critic (Eq. 2) by BPTT.
4. Fit the critic to GAE(λ) targets of the imagined rollout (Eq. 3), 5 epochs x 4 minibatches.

Imagined rollouts also stop at the clip end, where the critic bootstraps the rest, as the
simulator truncates there.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from RLAlg.logger import WandbLogger
from RLAlg.nn.layers import NormPosition
from RLAlg.utils import set_seed_everywhere

from supermimic.envs import G1TrackingEnv, G1TrackingEnvCfg
from supermimic.ppo.model import Critic, critic_observation_dim
from supermimic.wm.dynamics import WorldModel
from supermimic.wm.policy import Policy
from supermimic.wm.replay import TransitionBuffer
from supermimic.wm.task import TrackingTask


@dataclass
class SuperMimicCfg:
    # world model
    window: int = 8
    learning_rate: float = 2e-4             # shared by world model, policy and critic
    wm_batches: int = 10
    wm_batch_size: int = 2048
    wm_dims: list[int] = field(default_factory=lambda: [2048, 2048, 2048])
    # imagination and policy
    horizon: int = 32
    imagine_batch: int = 4096
    policy_betas: tuple[float, float] = (0.9, 0.95)
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    max_grad_value: float = 0.1             # policy only, after the norm clip
    termination_temperature: tuple[float, float] = (0.02, 0.05)  # sigmoid widths for height (m) and tilt
    checkpoint_steps: bool = False          # recompute each imagined step in backward (~13 -> 5 GiB, +0.25 s)
    compile: bool = True                    # torch.compile the imagined step and the world-model step
    bf16: bool = True                       # bf16 autocast for the world-model fit and imagination (~2.6x matmul rate on Ampere)
    # critic
    lambda_: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    # networks
    encoder_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    decoder_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    critic_dims: list[int] = field(default_factory=lambda: [256, 128])
    latent_dim: int = 32
    norm_position: str = "POST"
    # data collection
    action_noise: float = 0.1
    buffer_steps: int = 1000
    warmup_steps: int = 64                  # simulator steps per env before the first update
    utd_ratio: float = 0.4                  # iterations per simulator step
    # schedule
    lr_decay_iterations: int = 250_000
    lr_final_ratio: float = 0.03


class Trainer:
    algorithm_name = "SuperMimic"
    # printed each log interval (all metrics go to wandb)
    log_keys = ("episode/return", "episode/length", "rollout/termination_rate", "wm/loss", "wm/k_step_root_pos_m",
                "wm/k_step_joint_pos_rad", "policy/objective", "imagine/survival_H", "critic/explained_variance",
                "perf/iteration_s", "perf/collect_s", "perf/world_model_s", "perf/policy_s", "perf/critic_s")

    def __init__(self, env_cfg: G1TrackingEnvCfg, cfg: SuperMimicCfg, *, seed: int = 0, log_dir: str = "runs/supermimic",
                 use_wandb: bool = False):
        self.cfg = cfg
        self.seed = seed
        set_seed_everywhere(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        env_cfg.seed = seed
        self.env_cfg = env_cfg
        self.env = G1TrackingEnv(env_cfg)
        self.task = TrackingTask(self.env, cfg.termination_temperature)
        self.device = torch.device(env_cfg.device)
        self.num_envs = self.env.num_envs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        H, K = env_cfg.history_len, env_cfg.future_frames
        P, G, A = G1TrackingEnv.PROPRIO_DIM, G1TrackingEnv.GOAL_FRAME_DIM, G1TrackingEnv.NUM_ACTIONS
        norm = NormPosition[cfg.norm_position]
        self.policy = Policy(H, P, K, G, A, cfg.encoder_dims, cfg.decoder_dims, cfg.latent_dim, norm).to(self.device)
        self.critic = Critic(critic_observation_dim(K), cfg.critic_dims, norm).to(self.device)
        self.world_model = WorldModel(hidden_dims=cfg.wm_dims, norm_position=norm).to(self.device)

        lr = cfg.learning_rate
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=cfg.policy_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.optimizers = (self.wm_optimizer, self.policy_optimizer, self.critic_optimizer)
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(o, self._lr_factor) for o in self.optimizers]

        self._step_fn = self._imagine_step
        if cfg.compile:
            # thousands of small FK / quaternion / reward ops per step: fusing them removes the launch overhead
            self._step_fn = torch.compile(self._imagine_step, dynamic=False)
            self.world_model.compile_step()
        self.buffer = TransitionBuffer(self.num_envs, cfg.buffer_steps, G1TrackingEnv.STATE_DIM, A,
                                       G1TrackingEnv.CONTEXT_DIM, self.device)
        self.global_step = 0
        self.iteration = 0
        self._collect_credit = 0.0
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, device=self.device)
        self.use_wandb = use_wandb
        if use_wandb:
            WandbLogger.init_project("supermimic", f"{self.algorithm_name}-{seed}",
                                     config={"supermimic": asdict(cfg), "env": asdict(env_cfg)})

    def _lr_factor(self, it):
        cfg = self.cfg
        if it >= cfg.lr_decay_iterations:
            return cfg.lr_final_ratio
        cos = 0.5 * (1.0 + math.cos(math.pi * it / cfg.lr_decay_iterations))
        return cfg.lr_final_ratio + (1.0 - cfg.lr_final_ratio) * cos

    # ------------------------------------------------------------------ 1. collect

    @torch.no_grad()
    def collect(self, steps: int, metrics):
        obs = self.obs
        for _ in range(steps):
            self.policy.update_normalizers(obs)
            action = self.policy(obs)
            action = action + self.cfg.action_noise * torch.randn_like(action)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            self.buffer.add(obs["state"], action, obs["context"], obs["frame"], done)
            self.global_step += self.num_envs
            self.track_episodes(reward, terminated, truncated, info, metrics)
            obs = next_obs
        self.obs = obs

    def track_episodes(self, reward, terminated, truncated, info, metrics):
        self.episode_return += reward
        self.episode_length += 1
        metrics["rollout/reward"].append(reward.mean())
        metrics["rollout/termination_rate"].append(terminated.float().mean())
        done = terminated | truncated
        if done.any():
            metrics["episode/return"].append(self.episode_return[done].mean())
            metrics["episode/length"].append(self.episode_length[done].mean())
            self.episode_return[done] = 0.0
            self.episode_length[done] = 0.0

    # ------------------------------------------------------------------ 2. world model

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16)

    def update_world_model(self, metrics):
        cfg, wm = self.cfg, self.world_model
        for _ in range(cfg.wm_batches):
            w = self.buffer.sample_windows(cfg.wm_batch_size, cfg.window)
            v = w["valid"]
            wm.update_normalizers(w["state"][v], w["action"][v], w["context"][v])
            with self._autocast():
                loss, err = wm.rollout_loss(w["state"], w["action"], w["context"], v)
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            metrics["wm/grad_norm"].append(torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.max_grad_norm))
            self.wm_optimizer.step()
            metrics["wm/loss"].append(loss.detach())
        # last batch: error after k steps, in normalized units and in metres / radians
        err_k = err[v, -1]
        scaled = err_k * wm._out_std()
        metrics["wm/k_step_error"].append(err_k.square().mean())
        metrics["wm/k_step_root_pos_m"].append(scaled[:, 0:3].norm(dim=-1).mean())
        metrics["wm/k_step_joint_pos_rad"].append(scaled[:, 12:41].abs().mean())

    # ------------------------------------------------------------------ 3. policy through the world model

    def _imagine_step(self, state, history, prev_action, goal, frames, clip_end, context):
        task = self.task
        action = self.policy({"policy": history, "goal": goal})
        next_state = self.world_model(state, action, context)
        next_frames = torch.minimum(frames + 1, clip_end)
        reward, _ = task.reward(next_state, action, prev_action, next_frames)
        done = task.termination(next_state, next_frames)
        proprio = task.proprio(next_state, action, context)
        next_goal = task.goal(next_state, next_frames, clip_end)
        history = torch.cat((history[:, 1:], task.add_noise(proprio)[:, None]), 1)
        critic_obs = task.critic_obs(proprio, next_state, next_goal, context)
        return next_state, history, action, next_goal, reward, done, critic_obs

    def imagination_start(self, batch):
        """Replay start points with the policy's noisy observation history and goal.

        Returns state, history, prev_action, goal, frames, clip_end, context and the start's noise-free proprio.
        """
        task = self.task
        start = self.buffer.sample_starts(batch, self.env_cfg.history_len)
        context, frames = start["context"], start["frame"]
        clip_end = task.clip_end(frames)
        hs, hp = start["history_state"], start["history_prev_action"]
        n_hist = hs.shape[1]
        ctx_h = context[:, None].expand(-1, n_hist, -1).flatten(0, 1)
        proprio_hist = task.proprio(hs.flatten(0, 1), hp.flatten(0, 1), ctx_h).view(batch, n_hist, -1)
        history = task.add_noise(proprio_hist)
        state, prev_action = start["state"], hp[:, -1]
        goal = task.goal(state, frames, clip_end)
        return state, history, prev_action, goal, frames, clip_end, context, proprio_hist[:, -1]

    def imagine(self, metrics):
        cfg, task = self.cfg, self.task
        B, T, gamma = cfg.imagine_batch, cfg.horizon, cfg.gamma
        state, history, prev_action, goal, frames, clip_end, context, proprio = self.imagination_start(B)

        survival = torch.ones(B, device=self.device)                # ĉ_t
        live = (frames < clip_end).float()                          # not yet truncated at the clip end
        objective = torch.zeros(B, device=self.device)
        rec = defaultdict(list)
        rec["critic_obs"].append(task.critic_obs(proprio, state, goal, context))

        self.world_model.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            with self._autocast():
                for t in range(T):
                    args = (state, history, prev_action, goal, frames, clip_end, context)
                    if cfg.checkpoint_steps:
                        out = checkpoint(self._step_fn, *args, use_reentrant=False)
                    else:
                        out = self._step_fn(*args)
                    state, history, prev_action, goal, reward, done, critic_obs = out
                    frames = torch.minimum(frames + 1, clip_end)
                    truncated = (frames >= clip_end).float()
                    discount = gamma**t
                    objective = objective + discount * survival * live * reward
                    rec["reward"].append(reward.detach())
                    rec["done"].append(done.detach())
                    rec["truncated"].append(truncated)
                    rec["live"].append(live)
                    rec["survival"].append(survival.detach())
                    rec["critic_obs"].append(critic_obs.detach())
                    survival = survival * (1.0 - done)
                    # clip end: the critic bootstraps the rest of the episode
                    bootstrap = truncated * live
                    if t == T - 1:
                        bootstrap = live
                    value = self.critic(critic_obs).value
                    objective = objective + gamma ** (t + 1) * survival * bootstrap * value
                    live = live * (1.0 - truncated)
            # backward while ω and ψ are frozen: checkpointed steps are recomputed here
            loss = -objective.mean() / T
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
        finally:
            self.world_model.requires_grad_(True)
            self.critic.requires_grad_(True)

        metrics["policy/grad_norm"].append(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm))
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.max_grad_value)
        self.policy_optimizer.step()

        metrics["policy/objective"].append(objective.detach().mean())
        metrics["imagine/reward"].append(torch.stack(rec["reward"]).mean())
        metrics["imagine/survival_H"].append(survival.detach().mean())
        metrics["imagine/action_abs"].append(prev_action.detach().abs().mean())
        return {k: torch.stack(v) for k, v in rec.items()}

    # ------------------------------------------------------------------ 4. critic

    def update_critic(self, rec, metrics):
        cfg = self.cfg
        obs = rec["critic_obs"]                                      # (T+1, B, C)
        T = obs.shape[0] - 1
        mask = rec["live"].bool()                                    # (T, B)
        self.critic.update_normalizers(obs[:-1][mask])
        with torch.no_grad():
            values = self.critic(obs.flatten(0, 1)).value.view(T + 1, -1)
            targets = torch.zeros_like(rec["reward"])
            next_return = values[T]
            for t in reversed(range(T)):
                # cut at the clip end: R_{t+1} = V_{t+1}
                next_return = torch.where(rec["truncated"][t].bool(), values[t + 1], next_return)
                targets[t] = rec["reward"][t] + cfg.gamma * (1.0 - rec["done"][t]) * (
                    (1.0 - cfg.lambda_) * values[t + 1] + cfg.lambda_ * next_return)
                next_return = targets[t]
        x, y = obs[:-1][mask], targets[mask]
        n = x.shape[0]
        size = max(1, n // cfg.critic_minibatches)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * size : (i + 1) * size]
                loss = (self.critic(x[idx]).value - y[idx]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()
                metrics["critic/loss"].append(loss.detach())
        with torch.no_grad():
            pred = self.critic(x).value
            metrics["critic/explained_variance"].append(1.0 - (y - pred).var() / y.var().clamp_min(1e-8))
            metrics["critic/value"].append(pred.mean())

    # ------------------------------------------------------------------ loop

    def log(self, metrics, timings):
        info = {k: float(torch.stack([torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in v]).mean())
                for k, v in metrics.items()}
        total = sum(timings.values())
        info.update({f"perf/{k}_s": v for k, v in timings.items()})
        info["perf/iteration_s"] = total
        info["update/learning_rate"] = self.policy_optimizer.param_groups[0]["lr"]
        info.update(self.env.sampler.stats())
        if self.use_wandb:
            WandbLogger.log_metrics(info, self.global_step)
        shown = " ".join(f"{k.split('/')[-1]}={info[k]:.4g}" for k in self.log_keys if k in info)
        print(f"[{self.iteration:6d}] steps={self.global_step:.3e} {shown}", flush=True)
        return info

    def train(self, iterations: int, save_interval: int = 1000, log_interval: int = 10):
        self.obs = self.env.reset()
        warmup = defaultdict(list)
        self.collect(self.cfg.warmup_steps, warmup)
        metrics, timings = defaultdict(list), defaultdict(float)
        for _ in range(iterations):
            self._collect_credit += 1.0 / self.cfg.utd_ratio
            steps = int(self._collect_credit)
            self._collect_credit -= steps
            for name, fn in (("collect", lambda: self.collect(steps, metrics)),
                             ("world_model", lambda: self.update_world_model(metrics)),
                             ("policy", lambda: self.imagine(metrics))):
                t0 = time.time()
                out = fn()
                torch.cuda.synchronize()
                timings[name] += time.time() - t0
            t0 = time.time()
            self.update_critic(out, metrics)
            torch.cuda.synchronize()
            timings["critic"] += time.time() - t0
            for s in self.schedulers:
                s.step()
            self.iteration += 1
            if self.iteration % log_interval == 0:
                self.log(metrics, {k: v / log_interval for k, v in timings.items()})
                metrics, timings = defaultdict(list), defaultdict(float)
            if self.iteration % save_interval == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        self.save(self.log_dir / "model_final.pt")

    # ------------------------------------------------------------------ checkpoints

    def save(self, path):
        torch.save({
            "algorithm": self.algorithm_name,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model": self.world_model.state_dict(),
            "optimizers": [o.state_dict() for o in self.optimizers],
            "iteration": self.iteration,
            "global_step": self.global_step,
            "cfg": asdict(self.cfg),
            "sampler": {k: getattr(self.env.sampler, k) for k in ("bin_score", "bin_visits", "past_weight")},
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.world_model.load_state_dict(ckpt["world_model"])
        for o, s in zip(self.optimizers, ckpt["optimizers"]):
            o.load_state_dict(s)
        self.iteration, self.global_step = ckpt["iteration"], ckpt["global_step"]
        for s in self.schedulers:
            s.last_epoch = self.iteration
        for k, v in ckpt.get("sampler", {}).items():
            if getattr(self.env.sampler, k).shape == v.shape:
                getattr(self.env.sampler, k).copy_(v)

    def close(self):
        if self.use_wandb:
            WandbLogger.finish_project()
