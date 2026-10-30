"""Benchmark setup shared by the evaluation script and the policy viewer (SuperMimic App. F)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from RLAlg.nn.layers import NormPosition

from supermimic.envs import G1TrackingEnv, G1TrackingEnvCfg
from supermimic.ppo.model import Actor
from supermimic.ppo.trainer import PPOCfg
from supermimic.wm.policy import Policy
from supermimic.wm.trainer import SuperMimicCfg


def eval_env_cfg(motions, num_envs: int, **overrides) -> G1TrackingEnvCfg:
    """Foot friction 1.0, no domain randomization, no observation noise, no reset perturbation."""
    zero = ((0.0, 0.0),) * 6
    cfg = dict(
        motion_files=motions, num_envs=num_envs, randomize=False, add_noise=False,
        reset_pose_range=zero, reset_vel_range=zero, reset_joint_pos_range=0.0,
        adaptive_sampling=False, episode_length_s=1e6,
    )
    cfg.update(overrides)
    return G1TrackingEnvCfg(**cfg)


def require_checkpoint(path):
    """Exit with the checkpoints that do exist when ``path`` is missing."""
    if Path(path).is_file():
        return
    found = sorted(Path("runs").glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)[-10:]
    listing = "\n".join(f"  {p}" for p in found) or "  (none under runs/)"
    raise SystemExit(f"checkpoint not found: {path}\nmost recent checkpoints:\n{listing}")


def load_policy(path, env_cfg, device):
    """Deterministic action function obs -> action for a PPO-Sim or SuperMimic checkpoint.

    Any checkpoint with a SuperMimic ``Policy`` (``"policy"`` key) loads; config fields SuperMimicCfg lacks are ignored.
    """
    require_checkpoint(path)
    ckpt = torch.load(path, map_location=device)
    dims = (env_cfg.history_len, G1TrackingEnv.PROPRIO_DIM, env_cfg.future_frames, G1TrackingEnv.GOAL_FRAME_DIM,
            G1TrackingEnv.NUM_ACTIONS)
    if "policy" in ckpt:
        known = {f.name for f in fields(SuperMimicCfg)}
        cfg = SuperMimicCfg(**{k: v for k, v in ckpt["cfg"].items() if k in known})
        policy = Policy(*dims, cfg.encoder_dims, cfg.decoder_dims, cfg.latent_dim, NormPosition[cfg.norm_position])
        policy.load_state_dict(ckpt["policy"])
        return policy.to(device).eval()
    cfg = PPOCfg(**ckpt["ppo_cfg"])
    actor = Actor(*dims, cfg.encoder_dims, cfg.decoder_dims, cfg.latent_dim, NormPosition[cfg.norm_position],
                  cfg.init_std).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return lambda obs: actor(obs).mean
