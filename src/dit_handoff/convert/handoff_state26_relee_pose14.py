"""Convert raw handoff data to state26 -> anchored relative EE pose14."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dit_handoff.constants import (
    CAMERA_OBS_FEATURES,
    GRIPPER_OPENING_SIGN_THRESHOLD,
    IMAGE_SIZE,
    LANGUAGE_INSTRUCTION,
    LEROBOT_RELEE_POSE14_CONTRACT_VERSION,
    LEROBOT_ROOT,
    POSE14_ACTION_DIM,
    POSE14_LAYOUT,
    RELEE_POSE14_ACTION_INTERFACE,
    RELEE_POSE14_ACTION_REPRESENTATION,
    RELEE_POSE14_COORDINATE_FRAME,
    ROBOT_ROOT_POSE_W,
    STATE26_LAYOUT,
    STATE_DIM,
)
from dit_handoff.convert.handoff_state26_absjoint18 import _as_vector, _episode_dirs, _feature_stats, _split_episodes, state26_from_raw_step
from dit_handoff.data.raw_validator import validate_raw_dataset
from dit_handoff.utils.io import ensure_dir, iter_jsonl, read_json, write_json
from dit_handoff.utils.pose_math import pose_delta_axis_angle, pose_world_to_root


def gripper_opening_to_sign(opening: Any, threshold: float = GRIPPER_OPENING_SIGN_THRESHOLD) -> float:
    return 1.0 if float(opening) > float(threshold) else -1.0


def _arm_root_pose(snapshot: dict[str, Any], side: str) -> tuple[list[float], list[float]]:
    arm = snapshot.get("arms", {}).get(side, {})
    root = arm.get("root_pose_w") or arm.get("root") or ROBOT_ROOT_POSE_W[side]
    return _as_vector(root.get("pos", root.get("pos_w", ROBOT_ROOT_POSE_W[side]["pos"])), 3, f"{side} root pos"), _as_vector(
        root.get("quat", root.get("quat_w", ROBOT_ROOT_POSE_W[side]["quat"])), 4, f"{side} root quat"
    )


def arm_tcp_pose_root(snapshot: dict[str, Any], side: str) -> tuple[list[float], list[float]]:
    arm = snapshot["arms"][side]
    root_pos, root_quat = _arm_root_pose(snapshot, side)
    return pose_world_to_root(
        _as_vector(arm["tcp_pos_w"], 3, f"{side} tcp_pos_w"),
        _as_vector(arm["tcp_quat_w"], 4, f"{side} tcp_quat_w"),
        root_pos,
        root_quat,
    )


def _gripper_sign_from_row(row: dict[str, Any], side: str, target_snapshot: dict[str, Any]) -> float:
    commanded = row.get("action", {}).get("joint_target_18_commanded")
    if isinstance(commanded, list | tuple) and len(commanded) == 18:
        start = 0 if side == "left" else 9
        opening = abs(float(commanded[start + 7])) + abs(float(commanded[start + 8]))
        return gripper_opening_to_sign(opening)
    return gripper_opening_to_sign(target_snapshot["arms"][side]["gripper_opening"])


def pose14_between(
    anchor_snapshot: dict[str, Any],
    target_snapshot: dict[str, Any],
    target_row: dict[str, Any],
) -> list[float]:
    action: list[float] = []
    for side in ("left", "right"):
        anchor_pos, anchor_quat = arm_tcp_pose_root(anchor_snapshot, side)
        target_pos, target_quat = arm_tcp_pose_root(target_snapshot, side)
        action.extend(pose_delta_axis_angle(anchor_pos, anchor_quat, target_pos, target_quat))
        action.append(_gripper_sign_from_row(target_row, side, target_snapshot))
    if len(action) != POSE14_ACTION_DIM:
        raise ValueError(f"pose14 action built {len(action)} values")
    return action


def action14_from_raw_step(row: dict[str, Any]) -> list[float]:
    return pose14_between(row["pre_observation"], row["post_observation"], row)


def _np_action_stats(values: Any) -> dict[str, list[float]]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": [], "std": [], "min": [], "max": [], "count": [0]}
    if arr.ndim != 2 or arr.shape[1] != POSE14_ACTION_DIM:
        raise ValueError(f"expected stats array [N,{POSE14_ACTION_DIM}], got {arr.shape}")
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _write_action_chunk_sidecar(
    raw_dir: Path,
    output_dir: Path,
    episodes: list[Path],
    *,
    horizon: int,
    n_obs_steps: int,
) -> dict[str, Any]:
    import numpy as np

    rows_by_episode = [list(iter_jsonl(ep_dir / "steps.jsonl")) for ep_dir in episodes]
    total_frames = sum(len(rows) for rows in rows_by_episode)
    actions = np.zeros((total_frames, horizon, POSE14_ACTION_DIM), dtype=np.float32)
    is_pad = np.ones((total_frames, horizon), dtype=np.bool_)
    valid_actions: list[list[float]] = []
    global_index = 0
    first_action_slot = n_obs_steps - 1

    for rows in rows_by_episode:
        for local_index, row in enumerate(rows):
            anchor = row["pre_observation"]
            for slot in range(horizon):
                offset = slot - first_action_slot
                target_index = local_index + offset
                if target_index < local_index or target_index < 0 or target_index >= len(rows):
                    continue
                target_row = rows[target_index]
                action = pose14_between(anchor, target_row["post_observation"], target_row)
                actions[global_index, slot] = np.asarray(action, dtype=np.float32)
                is_pad[global_index, slot] = False
                valid_actions.append(action)
            global_index += 1

    sidecar_dir = ensure_dir(output_dir / "action_chunks")
    actions_path = sidecar_dir / "actions.npy"
    is_pad_path = sidecar_dir / "action_is_pad.npy"
    np.save(actions_path, actions)
    np.save(is_pad_path, is_pad)
    metadata = {
        "contract": LEROBOT_RELEE_POSE14_CONTRACT_VERSION,
        "raw_dataset_path": str(raw_dir),
        "actions_path": str(actions_path.relative_to(output_dir)),
        "action_is_pad_path": str(is_pad_path.relative_to(output_dir)),
        "shape": list(actions.shape),
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "first_executed_slot": first_action_slot,
        "slot_offset_rule": "target_raw_index = anchor_raw_index + slot - (n_obs_steps - 1)",
        "padding_rule": "slots before the anchor and slots beyond episode end are masked",
        "action_layout": list(POSE14_LAYOUT),
        "valid_action_count": len(valid_actions),
        "action_stats": _np_action_stats(valid_actions),
    }
    write_json(sidecar_dir / "metadata.json", metadata)
    return metadata


def _overwrite_action_stats(output_dir: Path, action_stats: dict[str, Any]) -> None:
    stats_path = output_dir / "meta" / "stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    stats["action"] = action_stats
    write_json(stats_path, stats)


def _try_official_lerobot(raw_dir: Path, output_dir: Path, episodes: list[Path], fps: float, repo_id: str) -> tuple[bool, str]:
    try:
        import numpy as np
        from PIL import Image
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        return False, f"official LeRobot import unavailable: {exc}"

    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": list(STATE26_LAYOUT)},
        "action": {"dtype": "float32", "shape": (POSE14_ACTION_DIM,), "names": list(POSE14_LAYOUT)},
    }
    for camera in CAMERA_OBS_FEATURES:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            "names": ["height", "width", "channel"],
        }

    try:
        video_codec = os.environ.get("DIT_LEROBOT_VIDEO_CODEC", "h264")
        video_crf = float(os.environ.get("DIT_LEROBOT_VIDEO_CRF", "23"))
        preset_env = os.environ.get("DIT_LEROBOT_VIDEO_PRESET")
        if preset_env is None:
            video_preset = "veryfast" if video_codec == "h264" else None
        elif preset_env.lower() in {"", "none", "null"}:
            video_preset = None
        else:
            video_preset = int(preset_env) if preset_env.isdigit() else preset_env
        rgb_encoder = RGBEncoderConfig(
            vcodec=video_codec, pix_fmt="yuv420p", g=2, crf=video_crf, preset=video_preset
        )
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_dir,
            fps=int(round(fps)),
            features=features,
            robot_type="dual_franka_handoff",
            use_videos=True,
            batch_encoding_size=1,
            rgb_encoder=rgb_encoder,
            encoder_threads=2,
            image_writer_processes=0,
            image_writer_threads=8,
        )
        total = len(episodes)
        for ep_i, ep_dir in enumerate(episodes, start=1):
            frame_count = 0
            for row in iter_jsonl(ep_dir / "steps.jsonl"):
                frame: dict[str, Any] = {
                    "observation.state": np.asarray(state26_from_raw_step(row), dtype=np.float32),
                    "action": np.asarray(action14_from_raw_step(row), dtype=np.float32),
                    "task": LANGUAGE_INSTRUCTION,
                }
                images = row["pre_observation"].get("images", {})
                for camera in CAMERA_OBS_FEATURES:
                    image_path = (ep_dir / images[camera]).resolve()
                    with Image.open(image_path) as img:
                        frame[f"observation.images.{camera}"] = img.convert("RGB").copy()
                dataset.add_frame(frame)
                frame_count += 1
            dataset.save_episode(parallel_encoding=True)
            print(f"[convert-relee-pose14] saved episode {ep_i}/{total}: {ep_dir.name}, frames={frame_count}", flush=True)
        finalize = getattr(dataset, "finalize", None)
        if finalize is not None:
            finalize()
        return True, "official LeRobotDataset video writer succeeded for state26_relee_pose14"
    except Exception as exc:
        return False, f"official LeRobotDataset writer failed: {exc}"


def convert(
    raw_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    train_split: float = 0.9,
    repo_id: str | None = None,
    horizon: int = 50,
    n_obs_steps: int = 2,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    manifest = read_json(raw_dir / "dataset_manifest.json")
    report = validate_raw_dataset(raw_dir)
    if not report["valid"]:
        raise RuntimeError(f"raw validation failed: {raw_dir / 'validation_report.json'}")
    dataset_name = manifest["dataset_name"]
    suffix = f"state26_relee_pose14_h{horizon}_obs{n_obs_steps}"
    output_dir = Path(output_dir) if output_dir else LEROBOT_ROOT / f"{dataset_name}_{suffix}"
    ensure_dir(output_dir.parent)
    episodes = _episode_dirs(raw_dir)
    split = _split_episodes(episodes, train_split)
    write_json(raw_dir / "train_val_split_state26_relee_pose14.json", split)
    fps = float(manifest.get("fps") or manifest.get("control_hz") or 50.0)
    repo_id = repo_id or f"local/{dataset_name}_{suffix}"

    official_ok, official_message = _try_official_lerobot(raw_dir, output_dir, episodes, fps, repo_id)
    if not official_ok:
        raise RuntimeError(official_message)

    sidecar = _write_action_chunk_sidecar(raw_dir, output_dir, episodes, horizon=horizon, n_obs_steps=n_obs_steps)
    _overwrite_action_stats(output_dir, sidecar["action_stats"])

    converted_states = []
    converted_actions = []
    for ep_dir in episodes:
        for row in iter_jsonl(ep_dir / "steps.jsonl"):
            converted_states.append(state26_from_raw_step(row))
            converted_actions.append(action14_from_raw_step(row))

    dataset_manifest = {
        "contract": LEROBOT_RELEE_POSE14_CONTRACT_VERSION,
        "writer": "official_lerobot",
        "writer_message": official_message,
        "raw_dataset_path": str(raw_dir),
        "output_dir": str(output_dir),
        "repo_id": repo_id,
        "fps": fps,
        "control_hz": fps,
        "state_dim": STATE_DIM,
        "action_dim": POSE14_ACTION_DIM,
        "state_layout": list(STATE26_LAYOUT),
        "action_layout": list(POSE14_LAYOUT),
        "image_features": list(CAMERA_OBS_FEATURES),
        "language_instruction": LANGUAGE_INSTRUCTION,
        "train_val_split": split,
        "stats_source": "anchored action chunk valid slots",
        "action_source": "relative TCP pose from anchor pre_observation to future post_observation",
        "action_interface": RELEE_POSE14_ACTION_INTERFACE,
        "action_representation": RELEE_POSE14_ACTION_REPRESENTATION,
        "action_coordinate_frame": RELEE_POSE14_COORDINATE_FRAME,
        "gripper_sign": {"open": 1.0, "close": -1.0, "opening_threshold": GRIPPER_OPENING_SIGN_THRESHOLD},
        "root_pose_w_source": "raw arm root_pose_w when present, otherwise fixed env cfg defaults",
        "robot_root_pose_w_default": ROBOT_ROOT_POSE_W,
        "action_chunk_sidecar": {
            "metadata": "action_chunks/metadata.json",
            "actions": sidecar["actions_path"],
            "action_is_pad": sidecar["action_is_pad_path"],
            "shape": sidecar["shape"],
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
            "first_executed_slot": sidecar["first_executed_slot"],
        },
    }
    write_json(output_dir / "manifest.json", dataset_manifest)
    summary = {
        **dataset_manifest,
        "num_episodes": len(episodes),
        "num_frames": len(converted_states),
        "state_stats": _feature_stats(converted_states),
        "single_step_action_stats": _feature_stats(converted_actions),
        "chunk_action_stats": sidecar["action_stats"],
    }
    write_json(output_dir / "conversion_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert raw handoff data to state26 -> anchored relative EE pose14.")
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    args = parser.parse_args(argv)
    summary = convert(
        args.raw_dir,
        args.output_dir,
        train_split=args.train_split,
        repo_id=args.repo_id,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
    )
    print(json.dumps({"output_dir": summary["output_dir"], "writer": summary["writer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
