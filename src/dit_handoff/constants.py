"""Project-wide constants for the state26 -> absjoint18 handoff pipeline."""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("DIT_WORKSPACE_ROOT", "/home/ubuntu/Workspace/dit")).expanduser()
DATA_ROOT = Path(os.environ.get("DIT_DATA_ROOT", str(WORKSPACE_ROOT / "datasets" / "dit"))).expanduser()
CONDA_ROOT = Path(os.environ.get("DIT_CONDA_ROOT", "/home/ubuntu/miniconda3")).expanduser()
ISAACLAB_ROOT = Path(os.environ.get("DIT_ISAACLAB_ROOT", "/home/ubuntu/Workspace/IsaacLab")).expanduser()

RAW_ROOT = DATA_ROOT / "raw"
LEROBOT_ROOT = DATA_ROOT / "lerobot"
RUNS_ROOT = DATA_ROOT / "runs"
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"
EVAL_ROOT = DATA_ROOT / "eval"
REPORT_ROOT = DATA_ROOT / "reports"
VIDEO_ROOT = DATA_ROOT / "videos"
CACHE_ROOT = DATA_ROOT / "cache"

RAW_SCHEMA_VERSION = "dit_raw_handoff_v1"
LEROBOT_CONTRACT_VERSION = "state26_absjoint18_v1"
LEROBOT_RELEE_POSE14_CONTRACT_VERSION = "state26_relee_pose14_v1"

IK_TASK_ID = "Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0"
JOINT_POS_TASK_ID = "Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-Joint-Pos-Visuomotor-v0"

LANGUAGE_INSTRUCTION = (
    "First place the blue cube on the yellow middle handoff area, "
    "then place it on the red target area."
)

CAMERA_OBS_FEATURES = ("wrist_rgb", "observer_wrist_rgb", "global_rgb")
CAMERA_SENSOR_NAMES = ("wrist_cam", "observer_wrist_cam", "global_cam")
CAMERA_SENSOR_TO_FEATURE = dict(zip(CAMERA_SENSOR_NAMES, CAMERA_OBS_FEATURES, strict=True))
IMAGE_SIZE = (256, 256)

LEFT_ARM_ASSET = "robot"
RIGHT_ARM_ASSET = "observer_robot"
LEFT_ARM_NAME = "left_actor"
RIGHT_ARM_NAME = "right_observer"

PANDA_JOINT_POS_ACTION_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)

LEFT_JOINT_NAMES = tuple(f"left/{name}" for name in PANDA_JOINT_POS_ACTION_NAMES)
RIGHT_JOINT_NAMES = tuple(f"right/{name}" for name in PANDA_JOINT_POS_ACTION_NAMES)
JOINT_NAMES_18 = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES
ACTION_NAMES_18 = tuple(f"target/{name}" for name in JOINT_NAMES_18)

STATE26_LAYOUT = (
    tuple(f"left.joint_pos.{name}" for name in PANDA_JOINT_POS_ACTION_NAMES)
    + ("left.tcp_pos_w.x", "left.tcp_pos_w.y", "left.tcp_pos_w.z", "left.gripper_opening")
    + tuple(f"right.joint_pos.{name}" for name in PANDA_JOINT_POS_ACTION_NAMES)
    + ("right.tcp_pos_w.x", "right.tcp_pos_w.y", "right.tcp_pos_w.z", "right.gripper_opening")
)

ACTION18_LAYOUT = ACTION_NAMES_18

POSE14_LAYOUT = (
    "left.ee_delta_root.dx",
    "left.ee_delta_root.dy",
    "left.ee_delta_root.dz",
    "left.ee_delta_root.axis_angle.x",
    "left.ee_delta_root.axis_angle.y",
    "left.ee_delta_root.axis_angle.z",
    "left.gripper_sign",
    "right.ee_delta_root.dx",
    "right.ee_delta_root.dy",
    "right.ee_delta_root.dz",
    "right.ee_delta_root.axis_angle.x",
    "right.ee_delta_root.axis_angle.y",
    "right.ee_delta_root.axis_angle.z",
    "right.gripper_sign",
)

STATE_DIM = 26
ACTION_DIM = 18
POSE14_ACTION_DIM = 14

ACTION_INTERFACE = "Joint-Pos absolute target 18D"
ACTION_REPRESENTATION = "absolute_joint_position_target"
ACTION_COORDINATE_FRAME = "Franka joint position targets in each arm asset joint order"

RELEE_POSE14_ACTION_INTERFACE = "Dual Franka IK relative EE pose 14D"
RELEE_POSE14_ACTION_REPRESENTATION = "anchored_relative_end_effector_pose_axis_angle"
RELEE_POSE14_COORDINATE_FRAME = "Each arm TCP pose delta in its own robot root frame"

QUATERNION_CONVENTION = "wxyz"
GRIPPER_OPENING_DEFINITION = "sum(abs(panda_finger_joint1), abs(panda_finger_joint2))"
GRIPPER_OPENING_SIGN_THRESHOLD = 0.02

ROBOT_ROOT_POSE_W = {
    "left": {"pos": (0.0, 0.30, 0.0), "quat": (1.0, 0.0, 0.0, 0.0)},
    "right": {"pos": (0.0, -0.30, 0.0), "quat": (1.0, 0.0, 0.0, 0.0)},
}

DEFAULT_FPS = 50.0
DEFAULT_HORIZON_SECONDS = 1.0
DEFAULT_N_ACTION_SECONDS = 0.8
DEFAULT_N_OBS_STEPS = 2

FORBIDDEN_RAW_DERIVED_FIELDS = (
    "subtask",
    "phase",
    "active_arm",
    "tcp_cube_distance",
    "first_close_flag",
    "grasp_candidate",
    "area_success_heuristic",
    "relative_pose",
    "progress",
    "red_stage",
    "yellow_stage",
)
