"""Schema helpers for raw handoff data."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from dit_handoff.constants import (
    ACTION18_LAYOUT,
    ACTION_DIM,
    ACTION_INTERFACE,
    ACTION_NAMES_18,
    ACTION_REPRESENTATION,
    CAMERA_OBS_FEATURES,
    DATA_ROOT,
    DEFAULT_FPS,
    FORBIDDEN_RAW_DERIVED_FIELDS,
    GRIPPER_OPENING_DEFINITION,
    IMAGE_SIZE,
    JOINT_NAMES_18,
    JOINT_POS_TASK_ID,
    LANGUAGE_INSTRUCTION,
    QUATERNION_CONVENTION,
    RAW_SCHEMA_VERSION,
    STATE26_LAYOUT,
    STATE_DIM,
)


def git_commit(workspace: Path | None = None) -> str | None:
    workspace = workspace or Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def collection_command(argv: list[str] | None = None) -> str:
    argv = argv or sys.argv
    return " ".join(shlex.quote(part) for part in argv)


def default_randomization(enabled: bool, profile: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "profile": profile,
        "cube_pose": "enabled_if_env_configured" if enabled else "disabled",
        "camera_image_augmentation": "disabled",
        "material_randomization": "not_implemented",
    }


def raw_dataset_manifest(
    dataset_name: str,
    *,
    task_id: str = JOINT_POS_TASK_ID,
    fps: float = DEFAULT_FPS,
    seed: int = 2000,
    randomization: dict[str, Any] | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": RAW_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "task_name": "cube_handoff_yellow_red_dual_franka",
        "env_id": task_id,
        "language_instruction": LANGUAGE_INSTRUCTION,
        "data_root": str(DATA_ROOT),
        "action_interface": ACTION_INTERFACE,
        "action_representation": ACTION_REPRESENTATION,
        "fps": float(fps),
        "control_hz": float(fps),
        "sim_dt": None,
        "decimation": None,
        "camera_names": list(CAMERA_OBS_FEATURES),
        "image_size": {"height": int(image_size[0]), "width": int(image_size[1])},
        "joint_names": list(JOINT_NAMES_18),
        "action_names": list(ACTION_NAMES_18),
        "action_layout": {
            "left_actor": [0, 9],
            "right_observer": [9, 18],
            "names": list(ACTION18_LAYOUT),
        },
        "state26_layout": list(STATE26_LAYOUT),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "seed": int(seed),
        "randomization_enabled": bool((randomization or {}).get("enabled", False)),
        "randomization_config": randomization or default_randomization(False, "none"),
        "collection_command": collection_command(),
        "code_version": {
            "git_commit": git_commit(Path(__file__).resolve().parents[3]),
            "python": sys.version.split()[0],
        },
        "success_definition": {
            "source": "env_info_or_episode_meta",
            "description": "Environment-provided success if available; collector does not infer phase success.",
        },
        "quaternion_convention": QUATERNION_CONVENTION,
        "rotation_training_representation": "axis_angle_preferred_when_needed; state26 uses tcp_pos only",
        "gripper_opening_definition": GRIPPER_OPENING_DEFINITION,
        "forbidden_raw_derived_fields": list(FORBIDDEN_RAW_DERIVED_FIELDS),
        "train_val_split_policy": "episode_split_only",
        "large_file_policy": f"All raw images, datasets, checkpoints, videos, reports, and caches stay under {DATA_ROOT}.",
    }
    if extra:
        manifest.update(extra)
    return manifest


def episode_meta(
    *,
    dataset_name: str,
    episode_id: int,
    seed: int,
    max_steps: int,
    randomization: dict[str, Any],
    task_id: str = JOINT_POS_TASK_ID,
) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "episode_id": int(episode_id),
        "task_name": "cube_handoff_yellow_red_dual_franka",
        "env_id": task_id,
        "action_interface": ACTION_INTERFACE,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "randomization": randomization,
        "language_instruction": LANGUAGE_INSTRUCTION,
        "episode_length": None,
        "episode_success": None,
        "success_source": "env_info_if_available",
    }


def env_var_snapshot(names: tuple[str, ...]) -> dict[str, str]:
    return {name: os.environ[name] for name in names if name in os.environ}
