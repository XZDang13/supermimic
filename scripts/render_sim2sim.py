"""Render stock-MuJoCo rollouts of a checkpoint to MP4: the robot in grey, the reference as an orange ghost.

Uses the same rollout as ``scripts/eval_sim2sim.py``. ``--clip`` takes exact clip names. Rendering is off-screen
(EGL by default), so no display is needed. Requires ``imageio`` with ``imageio-ffmpeg``.

    python scripts/render_sim2sim.py runs/supermimic/model_final.pt --motions data/motions/eval1000_23dof.npz \\
        --clip dance_basic_slide_180_R_001__A308_M --out runs/sim2sim_replays
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import torch

from supermimic.eval import checkpoint_robot, eval_env_cfg, load_policy
from supermimic.motion.library import MotionLibrary
from supermimic.robots import g1
from supermimic.sim2sim import MujocoTracker, rollout

GHOST_RGBA = (0.95, 0.55, 0.15, 0.35)
MAX_SECONDS = 12.0
VIDEO_FPS = 25


def display_model(robot: g1.Robot, width: int, height: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(robot.mjcf_path())
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], castshadow=False)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.86, 0.90, 0.95], rgb2=[0.55, 0.62, 0.72], width=64, height=64)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    return model


def render_clip(tracker: MujocoTracker, policy, clip: int, model, renderer) -> tuple[list, dict]:
    """Rolls ``clip`` out and renders its last ``MAX_SECONDS`` if it fell, otherwise its first."""
    qpos, frames = [], []

    def record(t):
        qpos.append(t.data.qpos.copy())
        frames.append(t.frame)

    result = rollout(tracker, policy, clip, record)
    fps = round(tracker.motions.fps)
    span = int(MAX_SECONDS * fps)
    first = max(0, len(qpos) - span) if result["failed"] else 0
    last = len(qpos) if result["failed"] else min(len(qpos), span)

    robot, ghost = mujoco.MjData(model), mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.8, 135, -14
    camera.lookat[:] = qpos[first][:3]
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    c, images = tracker.clip, []
    for i in range(first, last, fps // VIDEO_FPS):
        robot.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, robot)
        f = frames[i]
        tracker._set_state(ghost, c["root_pos"][f], c["root_quat"][f], c["joint_pos"][f], c["root_lin_vel"][f],
                           c["root_ang_vel"][f], c["joint_vel"][f])
        mujoco.mj_kinematics(model, ghost)
        camera.lookat[:2] += 0.1 * (qpos[i][:2] - camera.lookat[:2])
        camera.lookat[2] = 0.75
        renderer.update_scene(robot, camera=camera, scene_option=option)
        start = renderer.scene.ngeom
        mujoco.mjv_addGeoms(model, ghost, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
        for g in range(start, renderer.scene.ngeom):
            renderer.scene.geoms[g].rgba[:] = GHOST_RGBA
        images.append(renderer.render().copy())
    return images, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--clip", nargs="+", required=True, help="exact clip names")
    p.add_argument("--out", default="runs/sim2sim_replays")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    torch.set_num_threads(1)
    cfg = eval_env_cfg(args.motions, 1, robot=checkpoint_robot(args.checkpoint), device="cpu", use_cuda_graph=False)
    robot = g1.ROBOTS[cfg.robot]
    motions = MotionLibrary(args.motions, fps=1.0 / (cfg.sim_dt * cfg.decimation), device="cpu",
                            joint_names=robot.joint_names)
    missing = [c for c in args.clip if c not in motions.names]
    if missing:
        raise SystemExit(f"no such clips: {missing}")
    policy = load_policy(args.checkpoint, cfg, "cpu")
    tracker = MujocoTracker(cfg, motions)
    model = display_model(robot, args.width, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width, max_geom=4000)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    os.makedirs(args.out, exist_ok=True)
    for name in args.clip:
        images, r = render_clip(tracker, policy, motions.names.index(name), model, renderer)
        path = os.path.join(args.out, f"{name}.mp4")
        imageio.mimsave(path, images, fps=VIDEO_FPS, codec="libx264", quality=7, macro_block_size=1)
        print(f"{path}: {'FAILED' if r['failed'] else 'ok'} at step {r['steps']}, score {r['normalized_return']:.3f}", flush=True)


if __name__ == "__main__":
    main()
