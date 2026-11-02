# SuperMimic on Newton

Reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
(arXiv 2610.28480) on [Newton](https://github.com/newton-physics/newton). Physics goes through
`newton.solvers.SolverMuJoCo`, which runs MuJoCo Warp, the simulator the paper uses.

Done so far: the simulator side (a vectorized G1 motion-tracking env), the PPO-Sim baseline (built on
[RLAlg](https://github.com/XZDang13/RLAlg) losses, following the [RLDemos](https://github.com/XZDang13/RLDemos) PPO trainer), and SuperMimic itself: replay
buffer, learned dynamics f_ω fit on k=8 windows, and the H=32 BPTT actor-critic.

```bash
conda activate isaaclab
pip install -e . --no-deps
python scripts/make_test_motions.py                     # 4 synthetic clips -> data/motions/test
python scripts/test_env.py --num-envs 4096 --policy ref # throughput + reward sanity
python scripts/test_env.py --num-envs 4 --viewer        # watch
pytest tests -s

# PPO-Sim baseline
python scripts/train_ppo.py --motions data/motions/test --num-envs 4096 --iterations 30000 --log-dir runs/ppo
python scripts/eval_policy.py runs/ppo/model_final.pt --motions data/motions/test   # App. F protocol

# SuperMimic (world model + BPTT actor-critic)
python scripts/train_supermimic.py --motions data/motions/bones_seed --num-envs 4096 --iterations 320000 --log-dir runs/supermimic
python scripts/eval_policy.py runs/supermimic/model_final.pt --motions data/motions/bones_seed --out results.json

# watch a policy next to its reference (orange ghost); --results shows failed, then worst clips first
python scripts/view_policy.py runs/supermimic/model_final.pt --motions data/motions/bones_seed --results results.json --grid 4

# BONES-SEED subset: select from metadata, retarget with ../retargeter (offline solver + validator), convert
python scripts/prepare_bones_seed.py --num-clips 300 --out data/motions/bones_seed
```

## Layout

| path | what |
|---|---|
| `supermimic/robots/g1.py` | G1 29-DoF MJCF, BeyondMimic actuator gains/armature/action scale, default pose, tracked bodies |
| `supermimic/motion/library.py` | clip loading, resampling to 50 Hz, finite-difference velocities, flat GPU buffers, 1 s bins |
| `supermimic/motion/sampler.py` | return-based reset sampling (App. D.1) |
| `supermimic/envs/g1_tracking_env.py` | the env |
| `supermimic/ppo/model.py` | Table A.4 actor (motion encoder -> tanh z(32) -> decoder over z + 10 proprio frames) and privileged critic, with running normalizers |
| `supermimic/ppo/trainer.py` | PPO-Sim trainer: RLAlg `PPO` losses, `ReplayBuffer`, `compute_gae`, `KLAdaptiveLR`; BeyondMimic hyperparameters |
| `supermimic/envs/tracking.py` | observation, goal, reward and termination as pure torch functions, shared by the env and the world model |
| `supermimic/wm/kinematics.py` | differentiable torch FK (link poses and origin twists) read from the Newton model |
| `supermimic/wm/task.py` | analytic world-model heads: ō, goal, r̂, soft d̂, critic input; precomputed reference link states |
| `supermimic/wm/replay.py` | replay buffer (1000 steps x envs): k-step windows within an episode, start states with their 10-frame history |
| `supermimic/wm/dynamics.py` | f_ω(ŝ, a, ξ): MLP 2048 x 3 predicting the one-step change in the root frame; k-step loss (Eq. 1) |
| `supermimic/wm/policy.py` | deterministic policy: motion encoder -> tanh z(32) -> decoder over z + 10 noisy proprio frames |
| `supermimic/wm/trainer.py` | Algorithm A.1 with Table A.5 hyperparameters |
| `scripts/train_ppo.py`, `scripts/train_supermimic.py` | training entry points |
| `supermimic/eval.py` | benchmark env settings and checkpoint loading (PPO-Sim or SuperMimic), shared by the two scripts below |
| `scripts/eval_policy.py` | benchmark for either checkpoint: plays each clip from frame 0 (friction 1, no DR or noise, deterministic); `--out` writes per-clip results |
| `scripts/view_policy.py` | interactive viewer: robot and reference ghost per clip, clip browsing, speed, follow camera, reward-term plots |
| `scripts/prepare_bones_seed.py` | BONES-SEED selection from `~/Downloads/seed_metadata_v004.csv`, `retarget offline` on the SOMA BVHs, accepted clips to library `.npz` |
| `scripts/convert_csv_motion.py` | retargeted G1 CSV (`xyz, quat, 29 joints`) to library `.npz` |

## Env spec (paper Appendix A)

- **Sim**: 50 Hz control, 4 × 5 ms MuJoCo Warp substeps in one CUDA graph; Newton solver, implicitfast, flat ground.
- **Action**: `q_sp = q0 + Δq0 + α·a`, PD with BeyondMimic's low gains (10 Hz, ζ = 2 about the armature), `α = 0.25·τmax/kp`.
- **Observation** `proprio` (93): `q − q0 − Δq0`, `q̇`, IMU angular velocity, projected gravity, previous action.
  `policy` holds the last 10 noisy frames (Table A.1 noise). `proprio` itself is noise-free (ō), because the world model predicts ō and adds noise outside.
- **Goal** (10 × 67): the next K=10 reference frames: joint pos (relative to q0), joint vel, root position and 6D orientation in the robot root frame.
- **Reward** (Table A.2): BeyondMimic tracking terms, with no self-collision penalty and joint-limit weight −2. Every term is multiplied by Δt = 0.02, so the max is 0.1 per step.
- **Termination**: root height error > 0.25 m or tilt error > 0.8. There is no end-effector term. Truncation at clip end or 10 s.
- **Sim context ξ** (33, Table A.3): friction U[0.3, 1.2], joint calibration offsets U[±0.01], torso CoM offset. Resampled every episode. There are no root-velocity pushes.
- **State** (71): root pose and twist plus joint q, q̇. This is the full generalized state; the world-model state is built from it.
- `step()` auto-resets. `info["next_obs"]` is the pre-reset observation, which is what goes into the replay buffer.

## Decisions / deviations to revisit

- **Anchor body = pelvis.** The paper says "root (pelvis)". BeyondMimic used `torso_link`. Configurable via `anchor_body`.
- **Reset perturbation**: BeyondMimic's pose, velocity and joint ranges are kept. The paper doesn't say whether it keeps them.
- **Return-based sampler**: G_b is tracked over 250-step windows. Timeouts are normalised by the attainable return for their length; terminations are normalised by the full window. The "past softmax weight" is the accumulated softmax mass. The paper doesn't give these details.
- **Torso CoM DR** writes MuJoCo Warp's `body_ipos` directly. Using `notify_model_changed(BODY_INERTIAL_PROPERTIES)` re-runs `set_const` for every world (~50 ms at 4096 envs). The shift is a few cm at fixed mass, so the constants it would recompute are unchanged in practice.
- **Friction**: MuJoCo combines geom friction with `max`, so the ground is set to 0.3 and each env's robot coefficient decides.
- **`njmax` = 600** (150 contacts x 4 pyramidal rows). At 300, MuJoCo Warp printed `nefc overflow` and dropped
  constraint rows whenever robots fell into contact-heavy poses (6,903 warnings over the PPO BONES-SEED run). 600
  removes them for about 4% more step time. The PPO runs above predate the change.
- **Torch runs on Warp's CUDA stream** (set process-wide by the env), because buffers are shared zero-copy.
- **Self-collisions** use MJCF convex hulls (on). The paper only removes the *penalty*.
- The 29 `<motor>` actuators from the MJCF are imported but stay at ctrl 0. The PD loop is Newton's position actuators.

## Notes

- **Without a policy the robot falls.** BeyondMimic's gains are a torque proxy. Ankle stiffness (2 × 28.5 Nm/rad) is far below the toppling stiffness m·g·h ≈ 240 Nm/rad, so zero or reference-PD actions fall within about 1 s. With 10× gains it stands still, which confirms the physics.
- **Throughput** (RTX 3090, 4096 envs, DR on): about 128k env-steps/s. Physics is about 47 ms per step and env overhead about 20 ms.
- **Memory at full scale**: the library stores 71 floats per frame, and the reference body states come from FK every step. 264 h at 50 Hz is about 13.5 GB fp32. The full BONES-SEED set will need fp16 or CPU-pinned storage.

## PPO-Sim baseline

- **Settings**: BeyondMimic's PPO: 24-step rollouts, 5 epochs × 4 minibatches, KL-adaptive learning rate (target 0.01, adjusted per minibatch), entropy 0.005, value weight 1, clip 0.2 (policy and value), γ 0.99, λ 0.95. Networks use SuperMimic's sizes, as the paper's PPO-Sim does.
- **Timeouts** bootstrap from `V(info["next_obs"])`, as in RLDemos.
- **Normalizers** stay frozen while a rollout is collected and optimized, and are refreshed from it afterwards, so the stored log-probs match at the first minibatch. The std is floored at 0.01, as in rsl_rl.
- **Learning rate** settles around 1e-6 to 4e-5, not 1e-3. A single Adam step on the 2048-wide nets already reaches the 0.01 KL target. `--small` switches to BeyondMimic's 512-256-128 MLPs.
- **Sanity run** (4 synthetic clips, 4096 envs, 400 iterations ≈ 39M steps, ~20 min on a 3090): episode length went from 12 to 195 steps and termination rate from 4% to 0.07% per step. Evaluation fails 0 of 4 clips, with normalized return 0.94. That checkpoint predates the std floor.
- **Throughput**: 20–40k env-steps/s including the update.

## BONES-SEED data

- **Selection** (`prepare_bones_seed.py`): original takes only (no `_M` mirrors), body position "standing", no props,
  no object/environment interaction or stunts, 2-30 s. Clips are drawn round-robin over `content_type_of_movement`
  so walking and jogging do not dominate. The selection is saved to `data/bones_seed_retarget/selection.json`.
- **Retargeting**: [retargeter](https://github.com/XZDang13/retargeter) (`retarget offline`, checked out at `../retargeter`) from the SOMA-uniform BVHs. Only clips its validator accepts
  (`Configs/reject/g1.json`) are converted. Root height is taken relative to the solver's ground estimate, and
  quaternions go from wxyz to xyzw.
- **First subset** (seed 0): 300 selected, 233 accepted (0.48 h), about 1 min of retargeting on the 3090. Rejections
  are mostly joint acceleration, self-collision and dynamics residual. In the library, feet rest at the same
  height as the synthetic standing clip, and no frame exceeds hard joint limits.
- **PPO-Sim on it** (`runs/ppo_bones`, 4096 envs, 1500 iterations ≈ 147M steps, ~65 min): learns more slowly than on the
  synthetic clips. By iteration 1250, episode length was 177 steps and termination 0.13% per step.
  The benchmark fails 14 / 233 clips (6%), with mean normalized return 0.70. Failures are mostly
  fast turns, spins and jog starts.
- BONES-SEED data and anything derived from it is licensed; keep `data/` out of any shared repo.

## SuperMimic

Algorithm A.1 as one loop per iteration: collect simulator steps into the replay buffer, fit f_ω on 10 batches
of 2048 k=8 windows, run 4096 imagined H=32 rollouts through the frozen f_ω and step the policy by BPTT, and
fit the critic to GAE(λ=0.95) targets (5 epochs x 4 minibatches). Hyperparameters are Table A.5's.

- **Only the dynamics is learned.** ō, the goal, r̂ and d̂ are analytic heads computed from ŝ by the same functions
  as the env (`envs/tracking.py`), with a differentiable torch FK for the link states. §3.2 allows either kind of
  head. Analytic heads have no loss term, so Eq. 1 reduces to the state term. Tests check that FK matches Newton's
  and that the heads reproduce the env's observation, goal, reward and termination on sim states.
- **ŝ is the 71-D generalized state** (root pose, root link-origin twist, q, q̇). f_ω sees a heading-invariant view
  (root height, projected gravity, root-frame velocities, q, q̇) with a and ξ, and predicts the change over one
  step in the root frame. The state loss weighs each dimension by its one-step-change std (running stats).
- **Soft termination**: d̂ = 1 − σ((0.25 − e_h)/0.02)·σ((0.8 − e_tilt)/0.05), the env's thresholds as a sigmoid,
  so the survival weight ĉ stays differentiable. The widths are my choice; the paper learns d̂.
- **Clip ends**: an imagined rollout that reaches its clip's last frame stops collecting reward there, and the
  critic bootstraps (as the simulator truncates there). The GAE targets are cut the same way.
- **Update-to-data ratio 0.4** is read as iterations per simulator step, so an iteration collects 2.5 steps per
  env on average. Warm-up: 64 steps before the first update.
- **Policy**: deterministic, same encoder/decoder sizes as PPO-Sim, linear action output initialized near zero.
  Its observation normalizers update from the simulator observations during collection.
- **Speed (3090, 4096 envs)**: about 1.5 s per iteration. That is about 0.1 s simulator, 0.5 s world-model fit,
  0.85 s imagination with backward, and 0.1 s critic. The imagined step and the f_ω step are `torch.compile`d, which
  is 4x faster than eager imagination (compile takes ~2.5 min on the first iteration). Both phases run under bf16
  autocast and imagined activations are kept for backward: an iteration takes about 0.9 s (world-model fit 0.33 s,
  imagination 0.37 s) with peak memory about 13 GiB, so the paper's 320k iterations would take about 3.3 days here.
  `--no-bf16 --checkpoint` restores TF32 with per-step activation checkpointing (1.65 s per iteration, about 5 GiB).
  In a 1000-iteration A/B the bf16 setup learned as well or slightly better (k=8 loss 0.18 vs 0.25; 8 vs 9 failed
  clips, normalized return 0.68 vs 0.65 at iteration 1000).
- **Overhead pass**: about 0.73 s per iteration (collect 0.11 s, world-model fit 0.21 s, imagination 0.34-0.36 s,
  critic 0.065 s), so 320k iterations take about 2.7 days. Changes: the whole k=8 world-model loss is one compiled graph
  (before, only f_ω was compiled, so `state_error` and the 8-way gradient sums ran as ~48k eager ops). Normalizer
  updates from masked rows go through `masked_update`, with no boolean index (host sync) and no float64 copy of the
  105M-element critic batch. Critic minibatches have a fixed size and weight out clip-end rows, so their loss compiles.
  Adam is fused. The sampler and episode tracking no longer sync per step. GEMMs are autotuned
  (`max-autotune-no-cudagraphs`, ~5%; the first compile takes ~6-10 min and is cached, `--no-max-autotune` turns it off).
  Learning matched the earlier run at the same seed over 300 iterations. Every phase is now GPU-bound. Collect is the
  MuJoCo Warp graph (31 of 34.5 ms per env step). The rest of imagination is small FK / quaternion / LayerNorm kernels.

### First run on BONES-SEED (233 clips, 0.48 h)

`runs/supermimic_bones`, 4096 envs, default settings. The run was meant for 5000 iterations but stopped at about
3800 (the shell running it ended); the last checkpoint is `model_3500.pt`. Benchmark (`eval_policy.py`, App. F protocol):

| | iterations | sim steps | wall-clock | failed clips | normalized return |
|---|---|---|---|---|---|
| PPO-Sim (final) | 1500 | 147M | 65 min | 14 / 233 | 0.70 |
| SuperMimic | 500 | 5.4M | 14 min | 43 / 233 | 0.54 |
| SuperMimic | 1000 | 10.5M | 30 min | 5 / 233 | 0.68 |
| SuperMimic | 1500 | 15.6M | 46 min | 2 / 233 | 0.77 |
| SuperMimic | 2500 | 25.9M | 74 min | 1 / 233 | 0.89 |
| SuperMimic | 3500 | 36.1M | 103 min | 0 / 233 | 0.90 |

- The world model's 8-step root-position error fell from 1.8 cm to 2.7 mm, and the joint error from 0.024 to 0.006 rad.
  Imagined survival over H=32 reached 0.999, and the critic's explained variance 0.997.
- SuperMimic was still improving when the run stopped. The PPO-Sim run predates `njmax` = 600.

## Policy viewer

`scripts/view_policy.py` runs the benchmark settings in Newton's ViewerGL. Each slot plays one clip from its first
frame. When the clip ends or the robot falls (a failure), that slot moves on to the next clip in the list
(`G1TrackingEnv.reset_envs` switches it between steps), wrapping around at the end. `--loop` (or the panel checkbox)
replays the shown clips instead. The reference is drawn as an orange ghost 1 m to the robot's left
(`--ghost-offset`, `--no-ghost`) in a separate kinematic-only display model, so the simulation is unchanged.

- `--clip TEXT` keeps clips whose name contains TEXT. `--results` (from `eval_policy.py --out`) orders failed clips
  first, then by ascending return, and shows each clip's benchmark result in the panel. `--grid N` shows N clips in a row.
- The panel shows how many clips were watched and fell, and per slot the clip's list position, frame, return so far,
  plays and falls, and the last outcome. It has Prev / Next (a page of clips) and Restart buttons, a loop toggle, a speed slider, a follow camera and a camera reset. The plots window shows slot 0's reward terms.
- Keys: space pause, `.` single step, `N` / `P` next / previous clips, `R` restart, `C` reset camera, `F` frame all,
  `H` hide UI, Esc quit.

## Next

1. More BONES-SEED data (the App. E name and feasibility filters, mirrors) and LAFAN1-G1.
2. Longer SuperMimic runs on more data; periodic benchmark during training.
