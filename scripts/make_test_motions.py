"""Write a few synthetic G1 reference clips for smoke-testing the env.

Each clip perturbs the default pose with sinusoids; the root height is set per
frame by forward kinematics so the lowest foot contact sphere touches the floor.

    python scripts/make_test_motions.py --out data/motions/test
"""

import argparse
import math
import os

import newton
import numpy as np
import warp as wp

from supermimic.robots import g1

FOOT_SPHERES = np.array([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
FOOT_RADIUS = 0.005


def quat_apply_np(q, v):
    xyz, w = q[:3], q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def build_fk():
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(g1.mjcf_path(), floating=True)
    model = b.finalize(device="cpu")
    names = [lbl.rsplit("/", 1)[-1] for lbl in b.body_label]
    feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
    return model, model.state(), feet


def ground_root(model, state, feet, root_xy, root_quat, joint_pos):
    q = np.concatenate(([*root_xy, 0.0], root_quat, joint_pos)).astype(np.float32)
    newton.eval_fk(model, wp.array(q, dtype=wp.float32, device="cpu"), model.joint_qd, state)
    body_q = state.body_q.numpy()
    zmin = min(
        (body_q[f, :3] + quat_apply_np(body_q[f, 3:], s))[2] for f in feet for s in FOOT_SPHERES
    )
    return FOOT_RADIUS - zmin


def make_clip(kind: str, seconds: float, fps: float, model, state, feet):
    t = np.arange(0.0, seconds, 1.0 / fps)
    q0 = np.array(g1.default_joint_pos())
    j = {n: i for i, n in enumerate(g1.JOINT_NAMES)}
    joint_pos = np.tile(q0, (len(t), 1))
    yaw = np.zeros_like(t)
    if kind == "stand":
        pass
    elif kind == "wave":
        w = 2 * math.pi * 0.5
        joint_pos[:, j["left_shoulder_pitch_joint"]] += -1.2 * (1 - np.cos(w * t)) / 2
        joint_pos[:, j["left_shoulder_roll_joint"]] += 0.6 * (1 - np.cos(w * t)) / 2
        joint_pos[:, j["right_shoulder_pitch_joint"]] += -0.8 * np.sin(w * t)
        joint_pos[:, j["right_elbow_joint"]] += 0.5 * np.sin(2 * w * t)
        joint_pos[:, j["waist_yaw_joint"]] += 0.4 * np.sin(w * t)
    elif kind == "squat":
        s = 0.5 * (1 - np.cos(2 * math.pi * 0.4 * t))
        for side in ("left", "right"):
            joint_pos[:, j[f"{side}_hip_pitch_joint"]] += -0.6 * s
            joint_pos[:, j[f"{side}_knee_joint"]] += 1.2 * s
            joint_pos[:, j[f"{side}_ankle_pitch_joint"]] += -0.6 * s
        joint_pos[:, j["waist_pitch_joint"]] += 0.2 * s
    elif kind == "turn":
        yaw = 0.8 * np.sin(2 * math.pi * 0.25 * t)
    else:
        raise ValueError(kind)

    root_quat = np.stack((np.zeros_like(t), np.zeros_like(t), np.sin(yaw / 2), np.cos(yaw / 2)), -1)
    root_pos = np.zeros((len(t), 3))
    for i in range(len(t)):
        root_pos[i, 2] = ground_root(model, state, feet, (0.0, 0.0), root_quat[i], joint_pos[i])
    return root_pos, root_quat, joint_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/motions/test")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--seconds", type=float, default=8.0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model, state, feet = build_fk()
    for kind in ("stand", "wave", "squat", "turn"):
        rp, rq, jp = make_clip(kind, args.seconds, args.fps, model, state, feet)
        path = os.path.join(args.out, f"{kind}.npz")
        np.savez(path, fps=args.fps, root_pos=rp, root_quat=rq, joint_pos=jp, joint_names=np.array(g1.JOINT_NAMES))
        print(f"wrote {path}: {len(rp)} frames, root z {rp[:, 2].min():.3f}..{rp[:, 2].max():.3f}")


if __name__ == "__main__":
    main()
