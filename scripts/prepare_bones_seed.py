"""Select BONES-SEED clips, retarget them to the G1 with ../retargeter, and convert them for the library.

1. Selection from the metadata CSV: original takes only (no mirrors), standing, no props, no object
   or environment interaction, no stunts, 2-30 s. Clips are spread evenly over movement types.
2. `retarget offline` (whole-trajectory solver: stance locking, ground contact, self-collision,
   centroidal dynamics). Its validator splits clips into accepted/ and rejected/.
3. Accepted clips become library .npz files (xyzw root quaternion, root height relative to the
   solver's ground estimate).

    python scripts/prepare_bones_seed.py --num-clips 300 --out data/motions/bones_seed
"""

import argparse
import csv
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

EXCLUDED_CATEGORIES = {"Object Manipulation", "Object Interaction", "Environments", "Stunts"}


def select(metadata: str, num_clips: int, seed: int, min_s: float, max_s: float) -> list[dict]:
    groups = defaultdict(list)
    with open(metadata) as f:
        for row in csv.DictReader(f):
            seconds = int(row["move_duration_frames"]) / 120.0
            if (row["is_mirror"] == "True" or row["content_props"] != "0"
                    or row["content_body_position"].strip() != "standing"
                    or row["category"] in EXCLUDED_CATEGORIES or not min_s <= seconds <= max_s):
                continue
            groups[row["content_type_of_movement"].strip()].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    # round-robin over movement types so frequent ones (walking, jogging) do not dominate
    selected, order = [], sorted(groups, key=lambda k: -len(groups[k]))
    while len(selected) < num_clips and any(groups.values()):
        for k in order:
            if groups[k] and len(selected) < num_clips:
                selected.append(groups[k].pop())
    return selected


def convert(src: Path, dst: Path):
    d = np.load(src, allow_pickle=True)
    root_pos = d["root_pos"].astype(np.float32).copy()
    root_pos[:, 2] -= float(np.median(d["ground_height"]))
    np.savez(
        dst,
        fps=float(d["fps"]),
        root_pos=root_pos,
        root_quat=d["root_quat"][:, [1, 2, 3, 0]].astype(np.float32),  # wxyz -> xyzw
        joint_pos=d["dof_pos"].astype(np.float32),
        joint_names=d["joint_names"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-root", default=os.path.expanduser("~/Downloads"),
                   help="folder holding soma_uniform/ and the metadata CSV")
    p.add_argument("--metadata", default=None)
    p.add_argument("--retargeter", default=os.path.expanduser("~/Desktop/retargeter"))
    p.add_argument("--num-clips", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--work", default="data/bones_seed_retarget", help="raw retargeter output")
    p.add_argument("--out", default="data/motions/bones_seed")
    args = p.parse_args()

    root = Path(args.seed_root)
    metadata = args.metadata or str(root / "seed_metadata_v004.csv")
    rows = select(metadata, args.num_clips, args.seed, args.min_seconds, args.max_seconds)
    bvhs = [str(root / r["move_soma_uniform_path"]) for r in rows]
    print(f"selected {len(bvhs)} clips, {sum(int(r['move_duration_frames']) for r in rows) / 120 / 60:.1f} min")

    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(work / "selection.json", "w") as f:
        json.dump([{k: r[k] for k in ("filename", "category", "content_type_of_movement",
                                      "content_short_description")} for r in rows], f, indent=1)

    exe = Path(args.retargeter) / "build/linux/x86_64/release/retarget"
    subprocess.run([str(exe), "offline", *bvhs, "--robot", "g1_29dof", "--out", str(work.resolve())],
                   cwd=args.retargeter, check=True)

    accepted = sorted(work.rglob("accepted/*.npz"))
    for src in accepted:
        convert(src, out / src.name)
    rejected = [json.loads(l) for l in open(work / "manifest.jsonl") if '"accepted"' not in l]
    print(f"converted {len(accepted)} accepted clips to {out}; {len(rejected)} rejected or failed")
    reasons = defaultdict(int)
    for r in rejected:
        for reason in r.get("reasons") or [r.get("status")]:
            reasons[" ".join(str(reason).split()[:2])] += 1  # "robot: joint_acc_ratio_max"
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
