"""Batched torch rotation helpers.

Quaternions use Newton/Warp's (x, y, z, w) layout throughout the package.
"""

from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` by unit quaternions ``q``."""
    xyz, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conj(q), v)


def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle) of ``q``, taking the short way round."""
    q = torch.where(q[..., 3:] < 0.0, -q, q)
    xyz, w = q[..., :3], q[..., 3]
    s = xyz.norm(dim=-1)
    angle = 2.0 * torch.atan2(s, w)
    # angle / sin(angle/2) -> 2 as s -> 0
    scale = torch.where(s > 1e-6, angle / s.clamp_min(1e-9), 2.0 + angle**2 / 12.0)
    return xyz * scale.unsqueeze(-1)


def axis_angle_to_quat(v: torch.Tensor) -> torch.Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(angle > 1e-6, torch.sin(half) / angle.clamp_min(1e-9), 0.5 - angle**2 / 48.0)
    return torch.cat((v * scale, torch.cos(half)), dim=-1)


def quat_error_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotations ``a`` and ``b`` (= ||log(a b^-1)||)."""
    return quat_to_axis_angle(quat_mul(a, quat_conj(b))).norm(dim=-1)


def quat_error_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle, without the norm whose gradient is undefined at zero error."""
    return quat_to_axis_angle(quat_mul(a, quat_conj(b))).square().sum(-1)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Heading-only part of ``q`` (rotation about world z)."""
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return torch.stack((zeros, zeros, torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)), dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    )


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix (the 6D rotation representation)."""
    x_axis = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand_as(q[..., :3]))
    y_axis = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=q.device).expand_as(q[..., :3]))
    return torch.cat((x_axis, y_axis), dim=-1)


def quat_slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0.0, -b, b)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    t = t.unsqueeze(-1) if t.dim() < a.dim() else t
    wa = torch.where(sin_theta > 1e-6, torch.sin((1.0 - t) * theta) / sin_theta.clamp_min(1e-9), 1.0 - t)
    wb = torch.where(sin_theta > 1e-6, torch.sin(t * theta) / sin_theta.clamp_min(1e-9), t)
    return quat_normalize(wa * a + wb * b)


def quat_continuous(q: torch.Tensor) -> torch.Tensor:
    """Flip signs along a (T, 4) sequence so neighbouring quaternions share a hemisphere."""
    flips = torch.ones(q.shape[0], device=q.device, dtype=q.dtype)
    flips[1:] = torch.where((q[1:] * q[:-1]).sum(-1) < 0.0, -1.0, 1.0)
    return q * torch.cumprod(flips, dim=0).unsqueeze(-1)
