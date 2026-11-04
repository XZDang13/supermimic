"""Pack retargeter output (accepted clips) into one library file, for libraries too large to load clip by clip.

Each clip is converted as in ``prepare_bones_seed.py``: root height relative to the solver's ground
estimate, root quaternion wxyz -> xyzw. All clips must share one frame rate (the library's 50 Hz).

    python scripts/pack_motions.py ~/Desktop/bones_seed/g1_23dof/accepted --out data/motions/bones_seed_23dof.npz
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np


def read(path):
    d = np.load(path, allow_pickle=True)
    root_pos = d["root_pos"].astype(np.float32)
    root_pos[:, 2] -= float(np.median(d["ground_height"]))
    return (os.path.splitext(os.path.basename(path))[0], float(d["fps"]), root_pos,
            d["root_quat"][:, [1, 2, 3, 0]].astype(np.float32),  # wxyz -> xyzw
            d["dof_pos"].astype(np.float32), tuple(str(n) for n in d["joint_names"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="+", help="directories (searched recursively for .npz) or files")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int, default=None, help="first N clips only (sorted by path)")
    args = p.parse_args()

    files = []
    for s in args.src:
        files += sorted(glob.glob(os.path.join(s, "**", "*.npz"), recursive=True)) if os.path.isdir(s) else [s]
    files = files[: args.limit]
    print(f"packing {len(files)} clips with {args.workers} workers", flush=True)

    names, lengths, root_pos, root_quat, joint_pos = [], [], [], [], []
    fps = joint_names = None
    with Pool(args.workers) as pool:
        for i, (name, f, rp, rq, jp, jn) in enumerate(pool.imap(read, files, chunksize=64)):
            fps = f if fps is None else fps
            joint_names = jn if joint_names is None else joint_names
            if abs(f - fps) > 1e-6 or jn != joint_names:
                raise ValueError(f"{files[i]}: {f} Hz / {len(jn)} joints differ from the first clip")
            names.append(name)
            lengths.append(len(rp))
            root_pos.append(rp)
            root_quat.append(rq)
            joint_pos.append(jp)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1} / {len(files)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, fps=fps, joint_names=np.array(joint_names), clip_names=np.array(names),
             clip_length=np.array(lengths, dtype=np.int64), root_pos=np.concatenate(root_pos),
             root_quat=np.concatenate(root_quat), joint_pos=np.concatenate(joint_pos))
    frames = sum(lengths)
    print(f"wrote {args.out}: {len(names)} clips, {frames} frames, {frames / fps / 3600:.1f} h @ {fps:g} Hz")


if __name__ == "__main__":
    main()
