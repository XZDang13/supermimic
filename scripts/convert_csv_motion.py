"""Convert retargeted G1 CSV clips to the motion-library ``.npz`` format.

Expected CSV row layout (Unitree / LAFAN1-G1 retarget convention, one row per frame):
``x, y, z, qx, qy, qz, qw, <29 joint angles in robots.g1.JOINT_NAMES order>``

    python scripts/convert_csv_motion.py path/to/clips/*.csv --fps 30 --out data/motions/lafan
"""

import argparse
import os

import numpy as np

from supermimic.robots import g1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--fps", type=float, required=True, help="frame rate of the CSV files")
    p.add_argument("--out", required=True)
    p.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw")
    p.add_argument("--skiprows", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nj = len(g1.JOINT_NAMES)
    for path in args.csv:
        data = np.loadtxt(path, delimiter=",", skiprows=args.skiprows, dtype=np.float32)
        if data.shape[1] != 7 + nj:
            raise ValueError(f"{path}: expected {7 + nj} columns, got {data.shape[1]}")
        quat = data[:, 3:7]
        if args.quat_order == "wxyz":
            quat = quat[:, [1, 2, 3, 0]]
        name = os.path.splitext(os.path.basename(path))[0]
        np.savez(
            os.path.join(args.out, f"{name}.npz"),
            fps=args.fps,
            root_pos=data[:, :3],
            root_quat=quat,
            joint_pos=data[:, 7:],
            joint_names=np.array(g1.JOINT_NAMES),
        )
        print(f"{name}: {len(data)} frames")


if __name__ == "__main__":
    main()
