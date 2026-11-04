"""Unitree G1 as used by BeyondMimic / SuperMimic: the 29-DoF robot and the 23-DoF variant
(waist roll/pitch and wrist pitch/yaw locked, rubber hands).

Actuator gains follow BeyondMimic: each joint is a PD loop with a 10 Hz natural
frequency and damping ratio 2 about its reflected rotor armature, so the gains
are low and the setpoint behaves like a torque proxy. The action scale is
0.25 * effort_limit / stiffness.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import newton.utils

MJCF_FILE = "g1_29dof_rev_1_0.xml"


def mjcf_path(file: str = MJCF_FILE) -> str:
    return str(newton.utils.download_asset("unitree_g1") / "mjcf" / file)


# Actuated joints in MJCF (= Newton dof) order.
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ROOT_BODY = "pelvis"

# BeyondMimic's tracked body set T.
TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

TORSO_BODY = "torso_link"

DEFAULT_ROOT_HEIGHT = 0.76
DEFAULT_JOINT_POS = {
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r".*_elbow_joint": 0.6,
    r"left_shoulder_roll_joint": 0.2,
    r"left_shoulder_pitch_joint": 0.2,
    r"right_shoulder_roll_joint": -0.2,
    r"right_shoulder_pitch_joint": 0.2,
}

# Reflected rotor inertia of the G1 motor families [kg m^2].
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

# (regex, effort limit [Nm], armature) per joint group.
_ACTUATOR_GROUPS = [
    (r".*_hip_pitch_joint", 88.0, ARMATURE_7520_14),
    (r".*_hip_roll_joint", 139.0, ARMATURE_7520_22),
    (r".*_hip_yaw_joint", 88.0, ARMATURE_7520_14),
    (r".*_knee_joint", 139.0, ARMATURE_7520_22),
    (r".*_ankle_(pitch|roll)_joint", 50.0, 2.0 * ARMATURE_5020),
    (r"waist_yaw_joint", 88.0, ARMATURE_7520_14),
    (r"waist_(roll|pitch)_joint", 50.0, 2.0 * ARMATURE_5020),
    (r".*_shoulder_(pitch|roll|yaw)_joint", 25.0, ARMATURE_5020),
    (r".*_elbow_joint", 25.0, ARMATURE_5020),
    (r".*_wrist_roll_joint", 25.0, ARMATURE_5020),
    (r".*_wrist_(pitch|yaw)_joint", 5.0, ARMATURE_4010),
]


def _match(name: str, table):
    for pattern, *values in table:
        if re.fullmatch(pattern, name):
            return values
    raise KeyError(f"no actuator group for joint {name}")


def actuator_params(joint_names=JOINT_NAMES) -> dict[str, list[float]]:
    """Per-joint stiffness, damping, armature, effort limit and action scale (``joint_names`` order)."""
    out = {"stiffness": [], "damping": [], "armature": [], "effort_limit": [], "action_scale": []}
    for name in joint_names:
        effort, armature = _match(name, _ACTUATOR_GROUPS)
        kp = armature * NATURAL_FREQ**2
        kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        out["stiffness"].append(kp)
        out["damping"].append(kd)
        out["armature"].append(armature)
        out["effort_limit"].append(effort)
        out["action_scale"].append(0.25 * effort / kp)
    return out


def default_joint_pos(joint_names=JOINT_NAMES) -> list[float]:
    q = []
    for name in joint_names:
        value = 0.0
        for pattern, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                value = v
                break
        q.append(value)
    return q


# ---------------------------------------------------------------------- variants

JOINT_NAMES_23DOF = [n for n in JOINT_NAMES
                     if n not in ("waist_roll_joint", "waist_pitch_joint", "left_wrist_pitch_joint",
                                  "left_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
TRACKED_BODIES_23DOF = [{"left_wrist_yaw_link": "left_wrist_roll_rubber_hand",
                         "right_wrist_yaw_link": "right_wrist_roll_rubber_hand"}.get(b, b) for b in TRACKED_BODIES]


@dataclass(frozen=True)
class Robot:
    """One G1 variant: model file, actuated joints in MJCF order and the tracked bodies."""

    name: str
    mjcf_file: str
    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    root_body: str = ROOT_BODY
    torso_body: str = TORSO_BODY
    default_root_height: float = DEFAULT_ROOT_HEIGHT

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def mjcf_path(self) -> str:
        return mjcf_path(self.mjcf_file)

    def actuator_params(self) -> dict[str, list[float]]:
        return actuator_params(self.joint_names)

    def default_joint_pos(self) -> list[float]:
        return default_joint_pos(self.joint_names)


ROBOTS = {
    "g1_29dof": Robot("g1_29dof", MJCF_FILE, tuple(JOINT_NAMES), tuple(TRACKED_BODIES)),
    "g1_23dof": Robot("g1_23dof", "g1_23dof_rev_1_0.xml", tuple(JOINT_NAMES_23DOF), tuple(TRACKED_BODIES_23DOF)),
}
