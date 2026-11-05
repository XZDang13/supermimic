"""Reference-motion library.

A motion file is an ``.npz`` with

* ``fps``        scalar
* ``root_pos``   (T, 3)  pelvis position in world [m]
* ``root_quat``  (T, 4)  pelvis orientation, (x, y, z, w)
* ``joint_pos``  (T, J)  joint angles [rad]
* ``joint_names`` (J,)   optional; when present the columns are reordered to the robot's joint order

A packed file (``scripts/pack_motions.py``) holds many clips at once: the same keys over all
frames, concatenated, plus ``clip_length`` (C,) and ``clip_names`` (C,).

Clips are resampled to the control rate and concatenated into flat frame
buffers. Only the generalized coordinates and their finite-difference
velocities are stored; body poses and twists of the reference are produced by
forward kinematics in the env, which keeps memory linear in 13 + 2J floats/frame
instead of 13 floats per body per frame. Large libraries keep joint angles and all
velocities in float16 (root pose stays float32), about half the memory.
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


def _select_joints(path: str, data, joint_pos: np.ndarray, joint_names) -> np.ndarray:
    if "joint_names" in data:
        names = [str(n) for n in data["joint_names"]]
        missing = [n for n in joint_names if n not in names]
        if missing:
            raise ValueError(f"{path}: no columns for joints {missing}")
        joint_pos = joint_pos[:, [names.index(n) for n in joint_names]]
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(f"{path}: expected {len(joint_names)} joints, got {joint_pos.shape[1]}")
    return joint_pos


def load_clip(path: str, fps: float, joint_names=g1.JOINT_NAMES) -> Clip:
    data = np.load(path, allow_pickle=True)
    root_pos = data["root_pos"].astype(np.float32)
    root_quat = data["root_quat"].astype(np.float32)
    joint_pos = _select_joints(path, data, data["joint_pos"].astype(np.float32), joint_names)
    root_quat = quat_continuous(quat_normalize(torch.from_numpy(root_quat))).numpy()

    src_fps = float(data["fps"])
    if abs(src_fps - fps) > 1e-6:
        t_src = np.arange(len(root_pos)) / src_fps
        t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / fps)
        rp, rq, jp = _resample(t_src, t_dst, root_pos, root_quat, joint_pos)
    else:
        rp, rq, jp = (torch.from_numpy(x) for x in (root_pos, root_quat, joint_pos))
    return Clip(os.path.splitext(os.path.basename(path))[0], rp.float(), quat_continuous(rq).float(), jp.float())


def load_packed(path: str, fps: float, joint_names=g1.JOINT_NAMES) -> list[Clip]:
    """Clips of a packed file (all at one frame rate), split from its flat arrays."""
    data = np.load(path, allow_pickle=True)
    if abs(float(data["fps"]) - fps) > 1e-6:
        raise ValueError(f"{path}: packed at {float(data['fps'])} Hz, the library runs at {fps} Hz")
    lengths = data["clip_length"].astype(np.int64)
    split = lambda x: np.split(x, np.cumsum(lengths)[:-1])
    joint_pos = _select_joints(path, data, data["joint_pos"].astype(np.float32), joint_names)
    root_quat = quat_normalize(torch.from_numpy(data["root_quat"].astype(np.float32)))
    clips = []
    for name, rp, rq, jp in zip(data["clip_names"], split(data["root_pos"].astype(np.float32)),
                                split(root_quat.numpy()), split(joint_pos)):
        clips.append(Clip(str(name), torch.from_numpy(rp), quat_continuous(torch.from_numpy(rq)), torch.from_numpy(jp)))
    return clips


def _neighbors(clip_start: torch.Tensor, clip_end: torch.Tensor, frame_clip: torch.Tensor):
    """Previous and next frame of every flat frame, clamped to its clip."""
    f = torch.arange(frame_clip.shape[0])
    return torch.maximum(f - 1, clip_start[frame_clip]), torch.minimum(f + 1, clip_end[frame_clip])


def _finite_diff(x: torch.Tensor, prev: torch.Tensor, nxt: torch.Tensor, dt: float) -> torch.Tensor:
    """Central differences inside each clip, one-sided at its ends (as ``np.gradient``)."""
    span = ((nxt - prev).clamp_min(1) * dt).unsqueeze(-1)
    return (x[nxt] - x[prev]) / span


def _angular_velocity(q: torch.Tensor, prev: torch.Tensor, nxt: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity of a flat (F, 4) quaternion track (central differences per clip)."""
    span = ((nxt - prev).clamp_min(1) * dt).unsqueeze(-1)
    return quat_to_axis_angle(quat_mul(q[nxt], quat_conj(q[prev]))) / span


class FrameBuffer:
    """A flat (F, ...) frame tensor, possibly stored in float16; indexing returns float32."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx].float()

    @property
    def shape(self):
        return self.data.shape

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()


class MotionLibrary:
    """All clips concatenated into flat (F, ...) tensors on ``device``."""

    # above this many frames, joint angles and velocities are stored in float16
    HALF_PRECISION_FRAMES = 5_000_000

    def __init__(self, paths: list[str] | str, fps: float, device: str, bin_seconds: float = 1.0,
                 joint_names=g1.JOINT_NAMES, half: bool | None = None):
        self.fps = fps
        self.dt = 1.0 / fps
        self.device = device

        clips = []
        for f in expand_motion_paths(paths):
            with np.load(f, allow_pickle=True) as data:
                packed = "clip_length" in data
            clips.extend(load_packed(f, fps, joint_names) if packed else [load_clip(f, fps, joint_names)])
        clips = [c for c in clips if c.root_pos.shape[0] >= 2]
        self.names = [c.name for c in clips]
        lengths = torch.tensor([c.root_pos.shape[0] for c in clips], dtype=torch.long)
        root_pos = torch.cat([c.root_pos for c in clips])
        root_quat = torch.cat([c.root_quat for c in clips])
        joint_pos = torch.cat([c.joint_pos for c in clips])
        del clips

        # velocities on the host, before the float16 cast
        clip_start = torch.cumsum(lengths, 0) - lengths
        frame_clip = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        prev, nxt = _neighbors(clip_start, clip_start + lengths - 1, frame_clip)
        root_lin_vel = _finite_diff(root_pos, prev, nxt, self.dt)
        root_ang_vel = _angular_velocity(root_quat, prev, nxt, self.dt)
        joint_vel = _finite_diff(joint_pos, prev, nxt, self.dt)
        del prev, nxt, frame_clip

        if half is None:
            half = root_pos.shape[0] > self.HALF_PRECISION_FRAMES
        low = torch.float16 if half else torch.float32
        self.half = half
        self.root_pos = FrameBuffer(root_pos.to(device))
        self.root_quat = FrameBuffer(root_quat.to(device))
        self.joint_pos = FrameBuffer(joint_pos.to(device, low))
        self.root_lin_vel = FrameBuffer(root_lin_vel.to(device, low))
        self.root_ang_vel = FrameBuffer(root_ang_vel.to(device, low))
        self.joint_vel = FrameBuffer(joint_vel.to(device, low))

        self.num_clips = len(lengths)
        self.clip_length = lengths.to(device)
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

    @property
    def nbytes(self) -> int:
        buffers = (self.root_pos, self.root_quat, self.joint_pos, self.root_lin_vel, self.root_ang_vel, self.joint_vel)
        return sum(b.nbytes for b in buffers) + 2 * self.num_frames * 8  # + frame_clip, frame_bin

    def __repr__(self) -> str:
        return (
            f"MotionLibrary({self.num_clips} clips, {self.num_frames} frames, "
            f"{self.duration_hours:.2f} h @ {self.fps:g} Hz, {self.num_bins} bins, "
            f"{self.nbytes / 2**30:.2f} GiB{', float16' if self.half else ''})"
        )
