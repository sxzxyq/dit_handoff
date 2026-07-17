"""Convert raw rectangular-bar handoff data to the state26 -> absjoint18 LeRobot contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from dit_handoff.constants import (
    ACTION18_LAYOUT,
    ACTION_DIM,
    CAMERA_OBS_FEATURES,
    BAR_HANDOFF_LANGUAGE_INSTRUCTION,
    BAR_HANDOFF_LEROBOT_CONTRACT_VERSION,
    LEROBOT_ROOT,
    STATE26_LAYOUT,
    STATE_DIM,
)
from dit_handoff.data.raw_validator import validate_raw_dataset
from dit_handoff.utils.io import ensure_dir, iter_jsonl, read_json, write_json


def _as_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{label} must be {length}D")
    return [float(v) for v in value]


def state26_from_raw_step(row: dict[str, Any]) -> list[float]:
    arms = row["pre_observation"]["arms"]
    left = arms["left"]
    right = arms["right"]
    state = (
        _as_vector(left["joint_pos"], 9, "left joint_pos")
        + _as_vector(left["tcp_pos_w"], 3, "left tcp_pos_w")
        + [float(left["gripper_opening"])]
        + _as_vector(right["joint_pos"], 9, "right joint_pos")
        + _as_vector(right["tcp_pos_w"], 3, "right tcp_pos_w")
        + [float(right["gripper_opening"])]
    )
    if len(state) != STATE_DIM:
        raise ValueError(f"state26 built {len(state)} values")
    return state


def action18_from_raw_step(row: dict[str, Any]) -> list[float]:
    action = row["action"]
    source = action.get("joint_target_18_commanded_source")
    if source != "commanded_env_action":
        raise ValueError(f"action source must be commanded_env_action, got {source!r}")
    return _as_vector(action["joint_target_18_commanded"], ACTION_DIM, "joint_target_18_commanded")


def _episode_dirs(raw_dir: Path) -> list[Path]:
    return sorted((raw_dir / "episodes").glob("episode_*"))


def _split_episodes(episodes: list[Path], train_ratio: float) -> dict[str, list[str]]:
    if not episodes:
        return {"train_episodes": [], "val_episodes": []}
    if len(episodes) == 1:
        return {"train_episodes": [episodes[0].name], "val_episodes": []}
    train_count = max(1, min(len(episodes) - 1, round(len(episodes) * train_ratio)))
    return {
        "train_episodes": [ep.name for ep in episodes[:train_count]],
        "val_episodes": [ep.name for ep in episodes[train_count:]],
    }


def _feature_stats(vectors: list[list[float]]) -> dict[str, list[float]]:
    if not vectors:
        return {"mean": [], "std": [], "min": [], "max": []}
    cols = list(zip(*vectors, strict=True))
    return {
        "mean": [mean(col) for col in cols],
        "std": [pstdev(col) if len(col) > 1 else 0.0 for col in cols],
        "min": [min(col) for col in cols],
        "max": [max(col) for col in cols],
    }


def _iter_frames(raw_dir: Path, episodes: list[Path]):
    for ep_index, ep_dir in enumerate(episodes):
        steps_path = ep_dir / "steps.jsonl"
        for frame_index, row in enumerate(iter_jsonl(steps_path)):
            state = state26_from_raw_step(row)
            action = action18_from_raw_step(row)
            images = row["pre_observation"].get("images", {})
            yield ep_index, ep_dir, frame_index, row, state, action, images


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
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ACTION18_LAYOUT)},
    }
    for camera in CAMERA_OBS_FEATURES:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (256, 256, 3),
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
        fps_i = int(round(fps))
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_dir,
            fps=fps_i,
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
                    "action": np.asarray(action18_from_raw_step(row), dtype=np.float32),
                    "task": BAR_HANDOFF_LANGUAGE_INSTRUCTION,
                }
                images = row["pre_observation"].get("images", {})
                for camera in CAMERA_OBS_FEATURES:
                    image_path = (ep_dir / images[camera]).resolve()
                    with Image.open(image_path) as img:
                        frame[f"observation.images.{camera}"] = img.convert("RGB").copy()
                dataset.add_frame(frame)
                frame_count += 1
            dataset.save_episode(parallel_encoding=True)
            print(f"[convert-video] saved episode {ep_i}/{total}: {ep_dir.name}, frames={frame_count}", flush=True)
        finalize = getattr(dataset, "finalize", None)
        if finalize is not None:
            finalize()
        return True, "official LeRobotDataset video writer succeeded via add_frame h264 gop2"
    except Exception as exc:
        return False, f"official LeRobotDataset writer failed: {exc}"

def _write_portable_fallback(raw_dir: Path, output_dir: Path, episodes: list[Path], fps: float, split: dict[str, list[str]]) -> None:
    ensure_dir(output_dir)
    meta_dir = ensure_dir(output_dir / "meta")
    data_dir = ensure_dir(output_dir / "data")
    image_root = ensure_dir(output_dir / "images")
    frames_path = data_dir / "frames.jsonl"
    if frames_path.exists():
        frames_path.unlink()

    states: list[list[float]] = []
    actions: list[list[float]] = []
    for ep_index, ep_dir, frame_index, row, state, action, images in _iter_frames(raw_dir, episodes):
        copied_images = {}
        for camera in CAMERA_OBS_FEATURES:
            src = ep_dir / images[camera]
            dst = image_root / ep_dir.name / camera / src.name
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)
            copied_images[camera] = str(dst.relative_to(output_dir))
        frame = {
            "episode_index": ep_index,
            "episode_name": ep_dir.name,
            "frame_index": frame_index,
            "timestamp": row.get("sim_time"),
            "observation.state": state,
            "action": action,
            "task": BAR_HANDOFF_LANGUAGE_INSTRUCTION,
            "images": copied_images,
        }
        with frames_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(frame) + "\n")
        states.append(state)
        actions.append(action)

    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": list(STATE26_LAYOUT)},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": list(ACTION18_LAYOUT)},
    }
    for camera in CAMERA_OBS_FEATURES:
        features[f"observation.images.{camera}"] = {"dtype": "image", "shape": [256, 256, 3]}

    write_json(
        meta_dir / "info.json",
        {
            "codebase_version": "portable_fallback_not_official_lerobot",
            "fps": fps,
            "features": features,
            "task": BAR_HANDOFF_LANGUAGE_INSTRUCTION,
        },
    )
    write_json(meta_dir / "stats.json", {"observation.state": _feature_stats(states), "action": _feature_stats(actions)})
    (meta_dir / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": BAR_HANDOFF_LANGUAGE_INSTRUCTION}) + "\n", encoding="utf-8")
    write_json(meta_dir / "episodes.json", {"episodes": [ep.name for ep in episodes], "split": split})


def convert(raw_dir: str | Path, output_dir: str | Path | None = None, *, train_split: float = 0.9, repo_id: str | None = None, allow_portable_fallback: bool = True) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    manifest = read_json(raw_dir / "dataset_manifest.json")
    report = validate_raw_dataset(raw_dir)
    if not report["valid"]:
        raise RuntimeError(f"raw validation failed: {raw_dir / 'validation_report.json'}")
    dataset_name = manifest["dataset_name"]
    output_dir = Path(output_dir) if output_dir else LEROBOT_ROOT / f"{dataset_name}_bar_handoff_state26_absjoint18"
    ensure_dir(output_dir.parent)
    episodes = _episode_dirs(raw_dir)
    split = _split_episodes(episodes, train_split)
    write_json(raw_dir / "train_val_split.json", split)
    fps = float(manifest.get("fps") or manifest.get("control_hz") or 50.0)
    repo_id = repo_id or f"local/{dataset_name}_bar_handoff_state26_absjoint18"

    official_ok, official_message = _try_official_lerobot(raw_dir, output_dir, episodes, fps, repo_id)
    writer = "official_lerobot" if official_ok else "portable_fallback"
    if not official_ok:
        if not allow_portable_fallback:
            raise RuntimeError(official_message)
        _write_portable_fallback(raw_dir, output_dir, episodes, fps, split)

    converted_states = []
    converted_actions = []
    for _, _, _, _, state, action, _ in _iter_frames(raw_dir, episodes):
        converted_states.append(state)
        converted_actions.append(action)

    dataset_manifest = {
        "contract": BAR_HANDOFF_LEROBOT_CONTRACT_VERSION,
        "writer": writer,
        "writer_message": official_message,
        "raw_dataset_path": str(raw_dir),
        "output_dir": str(output_dir),
        "repo_id": repo_id,
        "fps": fps,
        "control_hz": fps,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": list(STATE26_LAYOUT),
        "action_layout": list(ACTION18_LAYOUT),
        "image_features": list(CAMERA_OBS_FEATURES),
        "language_instruction": BAR_HANDOFF_LANGUAGE_INSTRUCTION,
        "train_val_split": split,
        "stats_source": "converted frames",
        "action_source": "raw action.joint_target_18_commanded",
    }
    write_json(output_dir / "manifest.json", dataset_manifest)
    summary = {
        **dataset_manifest,
        "num_episodes": len(episodes),
        "num_frames": len(converted_states),
        "state_stats": _feature_stats(converted_states),
        "action_stats": _feature_stats(converted_actions),
    }
    write_json(output_dir / "conversion_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert raw rectangular-bar handoff data to state26 -> absjoint18 LeRobot dataset.")
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--no-portable-fallback", action="store_true")
    args = parser.parse_args(argv)
    summary = convert(
        args.raw_dir,
        args.output_dir,
        train_split=args.train_split,
        repo_id=args.repo_id,
        allow_portable_fallback=not args.no_portable_fallback,
    )
    print(json.dumps({"output_dir": summary["output_dir"], "writer": summary["writer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

