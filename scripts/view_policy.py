"""Watch a PPO-Sim or SuperMimic policy track motions in the Newton viewer, next to its reference.

Each slot plays one clip from its first frame under the benchmark settings of eval_policy.py
(friction 1, no domain randomization or noise, deterministic actions). The reference is drawn
as an orange ghost beside the robot. When a clip ends or the robot falls (a failure), the slot
moves on to the next clip in the list, wrapping around at its end; --loop replays the same clips.

    python scripts/view_policy.py runs/supermimic_bones/model_3500.pt --motions data/motions/bones_seed
    python scripts/view_policy.py CKPT --motions data/motions/bones_seed --clip dance --grid 4
    python scripts/eval_policy.py CKPT --motions data/motions/bones_seed --out results.json
    python scripts/view_policy.py CKPT --motions data/motions/bones_seed --results results.json  # failed, then worst first

Keys: space pause, . step while paused, N / P next / previous clips, R restart them, C reset the camera,
F frame all, H hide the UI, Esc quit.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import newton
import newton.viewer
import numpy as np
import torch
import warp as wp

from supermimic.envs import G1TrackingEnv
from supermimic.eval import checkpoint_robot, eval_env_cfg, load_policy, require_checkpoint

GHOST_COLOR = (0.95, 0.55, 0.15)


def build_display_model(mjcf: str, num_slots: int, ghost: bool, device):
    """Kinematic-only scene: per slot the simulated robot, then (optionally) the reference ghost."""
    robot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    robot.add_mjcf(mjcf, floating=True, enable_self_collisions=False)
    slot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(slot)
    for _ in range(2 if ghost else 1):
        slot.add_builder(robot)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(slot, num_slots)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    if ghost:
        bodies = robot.body_count
        shape_body = model.shape_body.numpy()
        colors = model.shape_color.numpy()
        is_ghost = (shape_body >= 0) & ((shape_body % (2 * bodies)) >= bodies)
        colors[is_ghost] = GHOST_COLOR
        model.shape_color.assign(colors)
    return model, robot.joint_coord_count


def clip_order(motions, pattern, results):
    names = motions.names
    order = [i for i in range(motions.num_clips) if not pattern or pattern.lower() in names[i].lower()]
    if not order:
        raise SystemExit(f"no clip name contains {pattern!r}")
    if results:
        by_name = {c["name"]: c for c in json.load(open(results))["clips"]}
        key = lambda i: (not by_name.get(names[i], {}).get("failed", False),
                         by_name.get(names[i], {}).get("normalized_return", 1.0))
        order.sort(key=key)
    return order


class PolicyViewer:
    def __init__(self, args):
        self.args = args
        self.env_cfg = eval_env_cfg(args.motions, args.grid, robot=checkpoint_robot(args.checkpoint))
        self.env = env = G1TrackingEnv(self.env_cfg)
        self.motions = m = env.motions
        self.policy = load_policy(args.checkpoint, self.env_cfg, env.device)
        self.order = clip_order(m, args.clip, args.results)
        self.results = None
        if args.results:
            self.results = {c["name"]: c for c in json.load(open(args.results))["clips"]}
        self.loop = args.loop
        self.speed = 1.0
        self.follow = True
        self.ghost_offset = torch.tensor(args.ghost_offset, device=env.device)
        self.watched = defaultdict(lambda: {"plays": 0, "fails": 0, "last": ""})

        self.model, self.coords = build_display_model(env.robot.mjcf_path(), args.grid, not args.no_ghost, env.device)
        self.state = self.model.state()
        self.display_q = wp.to_torch(self.model.joint_q).view(args.grid, -1)

        self.viewer = newton.viewer.ViewerGL(headless=args.headless)
        self.viewer.set_model(self.model)
        # slots in a row along y, each robot with its ghost beside it
        self.viewer.set_world_offsets((0.0, args.spacing, 0.0))
        self.viewer.register_ui_callback(self.gui, position="panel")
        self._keys = {}
        self.load_clips(0)
        self.reset_camera()

    # ------------------------------------------------------------------ playback

    def clip_at(self, pos):
        return self.order[pos % len(self.order)]

    def load_clips(self, start):
        """Fill the slots with clips ``start, start+1, ...`` of the order and restart them."""
        self.slot_pos = [start + i for i in range(self.args.grid)]
        self.next_pos = start + self.args.grid
        self.clips = torch.tensor([self.clip_at(p) for p in self.slot_pos], device=self.env.device)
        self.restart()

    def restart(self):
        starts = self.motions.clip_start[self.clips]
        self.env.pin_reset_frames(starts)
        self.obs = self.env.reset(frames=starts)
        self.ret = torch.zeros(self.args.grid, device=self.env.device)
        self.steps = torch.zeros(self.args.grid, device=self.env.device)
        self._follow_target = None

    def step(self):
        with torch.no_grad():
            action = self.policy(self.obs)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.ret += reward
        self.steps += 1
        for key, value in info["reward_terms"].items():
            self.viewer.log_scalar(f"reward/{key}", float(value[0]))
        self.viewer.log_scalar("reward/total", float(reward[0]))
        done = terminated | truncated
        if not done.any():
            return
        ids = done.nonzero().squeeze(-1)
        for i in ids.tolist():
            name = self.motions.names[int(self.clips[i])]
            w = self.watched[name]
            w["plays"] += 1
            w["fails"] += int(terminated[i])
            norm = float(self.ret[i] / (0.1 * self.steps[i]))
            w["last"] = (f"FELL at {int(self.steps[i])} / {self.clip_length(i)}" if terminated[i]
                         else f"done, return {norm:.2f}")
            if not self.loop:
                self.slot_pos[i] = self.next_pos
                self.next_pos += 1
                self.clips[i] = self.clip_at(self.slot_pos[i])
        self.ret[done] = 0.0
        self.steps[done] = 0.0
        if not self.loop:
            # the env already restarted these slots at their old clip; switch them to the new one
            starts = self.motions.clip_start[self.clips]
            self.env.pin_reset_frames(starts)
            self.obs = self.env.reset_envs(ids, starts[ids])

    def clip_length(self, i):
        c = int(self.clips[i])
        return int(self.motions.clip_end[c] - self.motions.clip_start[c])

    def render(self, sim_time):
        env, m, q = self.env, self.motions, self.display_q
        n = self.coords
        q[:, :n] = env.joint_q
        if not self.args.no_ghost:
            f = env.frame
            q[:, n : n + 3] = m.root_pos[f] + self.ghost_offset
            q[:, n + 3 : n + 7] = m.root_quat[f]
            q[:, n + 7 : 2 * n] = m.joint_pos[f]
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        if self.follow:
            self.follow_camera()
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def slot_origin(self, i):
        """Ground point under slot i's robot, in viewer coordinates (world offsets applied)."""
        target = self.env.joint_q[i, :3].cpu().numpy().copy()
        target[2] = 0.0
        if self.viewer.world_offsets is not None:
            target += self.viewer.world_offsets.numpy()[i]
        return target

    def reset_camera(self):
        """Look at all slots from in front of the robots (+x), robot and ghost side by side."""
        points = np.stack([self.slot_origin(i) for i in range(self.args.grid)])
        center = points.mean(0) + 0.5 * np.array(self.args.ghost_offset) * (not self.args.no_ghost)
        width = np.ptp(points[:, 1]) + abs(self.args.ghost_offset[1]) + 1.0
        distance = max(3.5, 1.3 * width)
        self.viewer.camera.pos = self.viewer.camera._as_vec3((center + [distance, 0.0, 1.2]).tolist())
        self.viewer.camera.look_at((center + [0.0, 0.0, 0.7]).tolist())
        self._follow_target = None

    def follow_camera(self):
        # keep the camera's offset to slot 0's robot, so resets and walking stay in view
        target = self.slot_origin(0)
        if self._follow_target is not None:
            self.viewer.camera.translate(tuple(float(x) for x in target - self._follow_target))
        self._follow_target = target

    def handle_keys(self):
        for key, fn in (("n", lambda: self.move(1)), ("p", lambda: self.move(-1)), ("r", self.restart),
                        ("c", self.reset_camera)):
            down = self.viewer.is_key_down(key)
            if down and not self._keys.get(key):
                fn()
            self._keys[key] = down

    def move(self, direction):
        """Skip to the next unplayed clips, or back to the ones before the current slots."""
        if direction > 0:
            self.load_clips(self.next_pos)
        else:
            self.load_clips(max(0, min(self.slot_pos) - self.args.grid))

    # ------------------------------------------------------------------ UI

    def gui(self, imgui):
        if not imgui.collapsing_header("Policy", flags=imgui.TreeNodeFlags_.default_open):
            return
        imgui.text(Path(self.args.checkpoint).name)
        n = len(self.order)
        played = sum(1 for w in self.watched.values() if w["plays"])
        fell = sum(1 for w in self.watched.values() if w["fails"])
        imgui.text(f"{n} clips; next is #{self.next_pos % n + 1} (pass {self.next_pos // n + 1})")
        imgui.text(f"watched {played}, fell on {fell}")
        if imgui.button("< Prev"):
            self.move(-1)
        imgui.same_line()
        if imgui.button("Next >"):
            self.move(1)
        imgui.same_line()
        if imgui.button("Restart"):
            self.restart()
        _, self.speed = imgui.slider_float("Speed", self.speed, 0.1, 2.0, "%.2fx")
        _, self.loop = imgui.checkbox("Loop current clips", self.loop)
        _, self.follow = imgui.checkbox("Follow slot 0", self.follow)
        imgui.same_line()
        if imgui.button("Reset camera"):
            self.reset_camera()
        imgui.separator()
        for i in range(self.args.grid):
            name = self.motions.names[int(self.clips[i])]
            imgui.text(f"[{i}] #{self.slot_pos[i] % n + 1} {name}")
            steps = max(1.0, float(self.steps[i]))
            imgui.text(f"    frame {int(self.steps[i])} / {self.clip_length(i)}   return so far "
                       f"{float(self.ret[i]) / (0.1 * steps):.2f}")
            w = self.watched.get(name)
            if w:
                imgui.text(f"    played {w['plays']}, fell {w['fails']}; last: {w['last']}")
            if self.results and name in self.results:
                r = self.results[name]
                imgui.text(f"    benchmark: {'FAILED' if r['failed'] else 'ok'}, return {r['normalized_return']:.2f}")

    # ------------------------------------------------------------------ loop

    def run(self):
        dt = self.env.control_dt
        sim_time, frames = 0.0, 0
        while self.viewer.is_running():
            t0 = time.time()
            self.handle_keys()
            if self.viewer.should_step():
                self.step()
                sim_time += dt
            self.render(sim_time)
            frames += 1
            if self.args.max_frames and frames >= self.args.max_frames:
                break
            if not self.args.headless:
                time.sleep(max(0.0, dt / self.speed - (time.time() - t0)))
        return frames


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--motions", default="data/motions/test")
    p.add_argument("--clip", default=None, help="only clips whose name contains this text")
    p.add_argument("--results", default=None, help="eval_policy.py --out JSON: show failed, then worst clips first")
    p.add_argument("--grid", type=int, default=1, help="clips shown at once")
    p.add_argument("--spacing", type=float, default=2.5, help="distance between slots [m]")
    p.add_argument("--ghost-offset", type=float, nargs=3, default=(0.0, 1.0, 0.0), help="reference ghost offset [m]")
    p.add_argument("--no-ghost", action="store_true")
    p.add_argument("--loop", action="store_true", help="replay the shown clips instead of moving through the list")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0: run until closed)")
    args = p.parse_args()
    # fail before building the env and the viewer
    require_checkpoint(args.checkpoint)
    if args.results and not Path(args.results).is_file():
        raise SystemExit(f"results file not found: {args.results}\n"
                         f"make it with: python scripts/eval_policy.py {args.checkpoint} --motions {args.motions} "
                         f"--out {args.results}")
    PolicyViewer(args).run()


if __name__ == "__main__":
    main()
