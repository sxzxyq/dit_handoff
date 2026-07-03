"""Validate generic raw handoff datasets before conversion."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from dit_handoff.constants import (
    ACTION_DIM,
    ACTION_INTERFACE,
    CAMERA_OBS_FEATURES,
    FORBIDDEN_RAW_DERIVED_FIELDS,
    RAW_SCHEMA_VERSION,
)
from dit_handoff.utils.io import iter_jsonl, read_json, write_json


class ValidationIssue:
    def __init__(self, severity: str, path: str, message: str):
        self.severity = severity
        self.path = path
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "path": self.path, "message": self.message}


def _is_sequence(value: Any, length: int | None = None) -> bool:
    ok = isinstance(value, list | tuple)
    if ok and length is not None:
        ok = len(value) == length
    return ok


def _almost_equal_list(a: Any, b: Any, tol: float = 1.0e-8) -> bool:
    if not (_is_sequence(a) and _is_sequence(b)) or len(a) != len(b):
        return False
    try:
        return all(math.isclose(float(x), float(y), abs_tol=tol, rel_tol=tol) for x, y in zip(a, b, strict=True))
    except Exception:
        return False


def _arm_joint_concat(obs: dict[str, Any]) -> list[float] | None:
    arms = obs.get("arms", {})
    left = arms.get("left", {}).get("joint_pos")
    right = arms.get("right", {}).get("joint_pos")
    if _is_sequence(left, 9) and _is_sequence(right, 9):
        return list(left) + list(right)
    return None


def _check_forbidden_fields(row: dict[str, Any], row_path: str, issues: list[ValidationIssue]) -> None:
    for field in FORBIDDEN_RAW_DERIVED_FIELDS:
        if field in row:
            issues.append(ValidationIssue("error", row_path, f"raw step contains derived field {field!r}"))


def _check_observation(obs: Any, row_path: str, label: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(obs, dict):
        issues.append(ValidationIssue("error", row_path, f"{label} must be an object"))
        return
    arms = obs.get("arms")
    if not isinstance(arms, dict):
        issues.append(ValidationIssue("error", row_path, f"{label}.arms missing"))
        return
    for arm_name in ("left", "right"):
        arm = arms.get(arm_name)
        if not isinstance(arm, dict):
            issues.append(ValidationIssue("error", row_path, f"{label}.arms.{arm_name} missing"))
            continue
        for key, length in (("joint_pos", 9), ("joint_vel", 9), ("tcp_pos_w", 3)):
            if key in arm and not _is_sequence(arm[key], length):
                issues.append(ValidationIssue("error", row_path, f"{label}.arms.{arm_name}.{key} must be {length}D"))
        if "gripper_opening" in arm and not isinstance(arm["gripper_opening"], int | float):
            issues.append(ValidationIssue("error", row_path, f"{label}.arms.{arm_name}.gripper_opening must be scalar"))


def _check_images(episode_dir: Path, row: dict[str, Any], row_path: str, issues: list[ValidationIssue]) -> None:
    images = row.get("pre_observation", {}).get("images", {})
    if not isinstance(images, dict):
        issues.append(ValidationIssue("error", row_path, "pre_observation.images missing"))
        return
    for camera in CAMERA_OBS_FEATURES:
        rel = images.get(camera)
        if not rel:
            issues.append(ValidationIssue("error", row_path, f"missing image path for {camera}"))
            continue
        if not (episode_dir / rel).exists():
            issues.append(ValidationIssue("error", row_path, f"image path does not exist for {camera}: {rel}"))


def _check_action(row: dict[str, Any], row_path: str, issues: list[ValidationIssue]) -> None:
    action = row.get("action")
    if not isinstance(action, dict):
        issues.append(ValidationIssue("error", row_path, "action object missing"))
        return
    if action.get("action_interface") == ACTION_INTERFACE or row.get("action_interface") == ACTION_INTERFACE:
        target = action.get("joint_target_18_commanded")
        if not _is_sequence(target, ACTION_DIM):
            issues.append(ValidationIssue("error", row_path, "joint_target_18_commanded must exist and be 18D"))
        source = action.get("joint_target_18_commanded_source")
        if source != "commanded_env_action":
            issues.append(
                ValidationIssue(
                    "error",
                    row_path,
                    "joint_target_18_commanded_source must be commanded_env_action, not post observation",
                )
            )
        post_joints = _arm_joint_concat(row.get("post_observation", {}))
        if target is not None and post_joints is not None and _almost_equal_list(target, post_joints):
            issues.append(
                ValidationIssue(
                    "warning",
                    row_path,
                    "joint_target_18_commanded equals post joint_pos exactly; verify this is commanded target, not backfilled label",
                )
            )
    raw_action = action.get("raw_env_action")
    if raw_action is not None and not _is_sequence(raw_action, ACTION_DIM):
        issues.append(ValidationIssue("error", row_path, "raw_env_action must be 18D for this pipeline"))


def _validate_episode(dataset_dir: Path, episode_dir: Path, manifest: dict[str, Any], issues: list[ValidationIssue]) -> int:
    meta_path = episode_dir / "episode_meta.json"
    steps_path = episode_dir / "steps.jsonl"
    if not meta_path.exists():
        issues.append(ValidationIssue("error", str(meta_path), "episode_meta.json missing"))
    if not steps_path.exists():
        issues.append(ValidationIssue("error", str(steps_path), "steps.jsonl missing"))
        return 0

    previous_step = -1
    previous_sim_time: float | None = None
    rows = 0
    for rows, row in enumerate(iter_jsonl(steps_path), start=1):
        row_path = f"{steps_path}:{rows}"
        _check_forbidden_fields(row, row_path, issues)
        step_index = row.get("step_index")
        if step_index != previous_step + 1:
            issues.append(ValidationIssue("error", row_path, f"step_index must be consecutive, got {step_index}"))
        previous_step = int(step_index) if isinstance(step_index, int) else previous_step
        sim_time = row.get("sim_time")
        if isinstance(sim_time, int | float):
            if previous_sim_time is not None and sim_time < previous_sim_time:
                issues.append(ValidationIssue("error", row_path, "sim_time is not monotonic"))
            previous_sim_time = float(sim_time)
        _check_observation(row.get("pre_observation"), row_path, "pre_observation", issues)
        _check_observation(row.get("post_observation"), row_path, "post_observation", issues)
        _check_action(row, row_path, issues)
        _check_images(episode_dir, row, row_path, issues)
        for key in ("reward", "terminated", "truncated", "env_info"):
            if key not in row:
                issues.append(ValidationIssue("error", row_path, f"{key} missing"))

    if rows == 0:
        issues.append(ValidationIssue("error", str(steps_path), "episode has no steps"))
    return rows


def validate_raw_dataset(dataset_dir: str | Path, *, write_report_file: bool = True) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    issues: list[ValidationIssue] = []
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        issues.append(ValidationIssue("error", str(manifest_path), "dataset_manifest.json missing"))
        manifest: dict[str, Any] = {}
    else:
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != RAW_SCHEMA_VERSION:
            issues.append(ValidationIssue("error", str(manifest_path), "unexpected raw schema_version"))
        if manifest.get("action_interface") != ACTION_INTERFACE:
            issues.append(ValidationIssue("error", str(manifest_path), "manifest action_interface is not Joint-Pos 18D"))
        if manifest.get("action_dim") != ACTION_DIM:
            issues.append(ValidationIssue("error", str(manifest_path), "manifest action_dim must be 18"))

    episodes_root = dataset_dir / "episodes"
    if not episodes_root.exists():
        issues.append(ValidationIssue("error", str(episodes_root), "episodes directory missing"))
        episodes = []
    else:
        episodes = sorted(path for path in episodes_root.iterdir() if path.is_dir() and path.name.startswith("episode_"))
        if not episodes:
            issues.append(ValidationIssue("error", str(episodes_root), "no episode directories found"))

    row_counts = {}
    for episode_dir in episodes:
        row_counts[episode_dir.name] = _validate_episode(dataset_dir, episode_dir, manifest, issues)

    split_path = dataset_dir / "train_val_split.json"
    if split_path.exists():
        split = read_json(split_path)
        train = set(split.get("train_episodes", []))
        val = set(split.get("val_episodes", []))
        if train & val:
            issues.append(ValidationIssue("error", str(split_path), "train/val split overlaps episodes"))

    report = {
        "dataset_dir": str(dataset_dir),
        "valid": not any(issue.severity == "error" for issue in issues),
        "issue_count": len(issues),
        "error_count": sum(issue.severity == "error" for issue in issues),
        "warning_count": sum(issue.severity == "warning" for issue in issues),
        "episodes": len(episodes),
        "row_counts": row_counts,
        "issues": [issue.to_dict() for issue in issues],
    }
    if write_report_file:
        write_json(dataset_dir / "validation_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a DIT raw handoff dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args(argv)
    report = validate_raw_dataset(args.dataset_dir, write_report_file=not args.no_write_report)
    print(f"valid={report['valid']} errors={report['error_count']} warnings={report['warning_count']}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

