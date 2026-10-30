"""Reference-motion library.

A motion file is an ``.npz`` with

* ``fps``        scalar
* ``root_pos``   (T, 3)  pelvis position in world [m]
* ``root_quat``  (T, 4)  pelvis orientation, (x, y, z, w)
* ``joint_pos``  (T, J)  joint angles [rad]
* ``joint_names`` (J,)   optional; when present the columns are reordered to ``robots.g1.JOINT_NAMES``

Clips are resampled to the control rate and concatenated into flat frame
buffers. Only the generalized coordinates and their finite-difference
velocities are stored; body poses and twists of the reference are produced by
forward kinematics in the env, which keeps memory linear in 71 floats/frame
instead of 13 floats per body per frame.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

from supermimic.robots import g1
from supermimic.utils.math import (
    quat_conj,
    quat_continuous,
    quat_mul,
    quat_normalize,
    quat_slerp,
    quat_to_axis_angle,
)


def expand_motion_paths(paths: list[str] | str) -> list[str]:
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True)))
        else:
            out.extend(sorted(glob.glob(p)))
    if not out:
        raise FileNotFoundError(f"no motion files found in {paths}")
    return out


def _resample(t_src: np.ndarray, t_dst: np.ndarray, root_pos, root_quat, joint_pos):
    """Linear (slerp for rotations) resampling of one clip."""
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    alpha = ((t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx])).clip(0.0, 1.0)
    a = torch.from_numpy(alpha).float()
    lerp = lambda x: torch.from_numpy(x[idx]) * (1 - a[:, None]) + torch.from_numpy(x[idx + 1]) * a[:, None]
    rq = quat_slerp(torch.from_numpy(root_quat[idx]), torch.from_numpy(root_quat[idx + 1]), a)
    return lerp(root_pos), rq, lerp(joint_pos)


@dataclass
class Clip:
    name: str
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor


def load_clip(path: str, fps: float) -> Clip:
    data = np.load(path, allow_pickle=True)
    root_pos = data["root_pos"].astype(np.float32)
    root_quat = data["root_quat"].astype(np.float32)
    joint_pos = data["joint_pos"].astype(np.float32)
    if "joint_names" in data:
        names = [str(n) for n in data["joint_names"]]
        joint_pos = joint_pos[:, [names.index(n) for n in g1.JOINT_NAMES]]
    if joint_pos.shape[1] != len(g1.JOINT_NAMES):
        raise ValueError(f"{path}: expected {len(g1.JOINT_NAMES)} joints, got {joint_pos.shape[1]}")
    root_quat = quat_continuous(quat_normalize(torch.from_numpy(root_quat))).numpy()

    src_fps = float(data["fps"])
    if abs(src_fps - fps) > 1e-6:
        t_src = np.arange(len(root_pos)) / src_fps
        t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / fps)
        rp, rq, jp = _resample(t_src, t_dst, root_pos, root_quat, joint_pos)
    else:
        rp, rq, jp = (torch.from_numpy(x) for x in (root_pos, root_quat, joint_pos))
    return Clip(os.path.splitext(os.path.basename(path))[0], rp.float(), quat_continuous(rq).float(), jp.float())


def _finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    return torch.from_numpy(np.gradient(x.numpy(), dt, axis=0).astype(np.float32))


def _angular_velocity(q: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity of a (T, 4) quaternion track (central differences)."""
    if q.shape[0] < 2:
        return torch.zeros(q.shape[0], 3)
    q_next = torch.cat((q[1:], q[-1:]))
    q_prev = torch.cat((q[:1], q[:-1]))
    span = torch.full((q.shape[0], 1), 2.0 * dt)
    span[0] = span[-1] = dt
    return quat_to_axis_angle(quat_mul(q_next, quat_conj(q_prev))) / span


class MotionLibrary:
    """All clips concatenated into flat (F, ...) tensors on ``device``."""

    def __init__(self, paths: list[str] | str, fps: float, device: str, bin_seconds: float = 1.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.device = device
        files = expand_motion_paths(paths)

        root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel = [], [], [], [], [], []
        self.names, lengths = [], []
        for f in files:
            clip = load_clip(f, fps)
            if clip.root_pos.shape[0] < 2:
                continue
            self.names.append(clip.name)
            lengths.append(clip.root_pos.shape[0])
            root_pos.append(clip.root_pos)
            root_quat.append(clip.root_quat)
            joint_pos.append(clip.joint_pos)
            root_lin_vel.append(_finite_diff(clip.root_pos, self.dt))
            root_ang_vel.append(_angular_velocity(clip.root_quat, self.dt))
            joint_vel.append(_finite_diff(clip.joint_pos, self.dt))

        cat = lambda xs: torch.cat(xs).to(device)
        self.root_pos = cat(root_pos)
        self.root_quat = cat(root_quat)
        self.joint_pos = cat(joint_pos)
        self.root_lin_vel = cat(root_lin_vel)
        self.root_ang_vel = cat(root_ang_vel)
        self.joint_vel = cat(joint_vel)

        self.num_clips = len(lengths)
        self.clip_length = torch.tensor(lengths, device=device, dtype=torch.long)
        self.clip_start = torch.cumsum(self.clip_length, 0) - self.clip_length
        self.clip_end = self.clip_start + self.clip_length - 1  # last valid frame, inclusive
        self.num_frames = int(self.clip_length.sum())
        # clip id of every flat frame
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), self.clip_length)

        # Fixed-duration bins over each clip's timeline, for return-based reset sampling.
        frames_per_bin = max(1, round(bin_seconds * fps))
        bins_per_clip = (self.clip_length + frames_per_bin - 1) // frames_per_bin
        self.num_bins = int(bins_per_clip.sum())
        bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), bins_per_clip)
        first_bin = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        local_bin = torch.arange(self.num_bins, device=device) - first_bin[bin_clip]
        self.bin_start = self.clip_start[bin_clip] + local_bin * frames_per_bin
        self.bin_end = torch.minimum(self.bin_start + frames_per_bin - 1, self.clip_end[bin_clip])
        local_frame = torch.arange(self.num_frames, device=device) - self.clip_start[self.frame_clip]
        self.frame_bin = first_bin[self.frame_clip] + local_frame // frames_per_bin

    @property
    def duration_hours(self) -> float:
        return self.num_frames * self.dt / 3600.0

    def __repr__(self) -> str:
        return (
            f"MotionLibrary({self.num_clips} clips, {self.num_frames} frames, "
            f"{self.duration_hours:.2f} h @ {self.fps:g} Hz, {self.num_bins} bins)"
        )
