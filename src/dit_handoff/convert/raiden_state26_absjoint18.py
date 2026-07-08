"""Convert Raiden fake-DIT recordings to DIT raw and LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dit_handoff.constants import (
    ACTION_INTERFACE,
    CAMERA_OBS_FEATURES,
    DEFAULT_FPS,
    JOINT_POS_TASK_ID,
    LANGUAGE_INSTRUCTION,
    RAW_ROOT,
)
from dit_handoff.convert.handoff_state26_absjoint18 import convert as convert_lerobot
from dit_handoff.data.raw_validator import validate_raw_dataset
from dit_handoff.data.schema import default_randomization, episode_meta, raw_dataset_manifest
from dit_handoff.utils.io import ensure_dir, iter_jsonl, read_json, write_json

CAMERA_NAME_TO_FEATURE = {
    "left_wrist_camera": "wrist_rgb",
    "right_wrist_camera": "observer_wrist_rgb",
    "scene_camera": "global_rgb",
}


def _load_state_samples(recording_dir: Path) -> list[dict[str, Any]]:
    state_path = recording_dir / "state_data.jsonl"
    if not state_path.exists():
        raise FileNotFoundError(f"missing Raiden fake state file: {state_path}")
    samples = list(iter_jsonl(state_path))
    if not samples:
        raise ValueError(f"no state samples in {state_path}")
    return samples


def _available_extracted_cameras(recording_dir: Path) -> dict[str, Path]:
    rgb_root = recording_dir / "0000" / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(
            f"missing extracted RGB directory: {rgb_root}; run Raiden SVO extraction first"
        )
    return {path.name: path for path in rgb_root.iterdir() if path.is_dir()}


def _select_camera_sources(
    recording_dir: Path,
    *,
    allow_single_camera_duplicate: bool,
) -> dict[str, Path]:
    cameras = _available_extracted_cameras(recording_dir)
    mapped: dict[str, Path] = {}
    for camera_name, feature in CAMERA_NAME_TO_FEATURE.items():
        if camera_name in cameras:
            mapped[feature] = cameras[camera_name]

    missing = [feature for feature in CAMERA_OBS_FEATURES if feature not in mapped]
    if not missing:
        return mapped

    if allow_single_camera_duplicate and len(cameras) == 1:
        only_camera = next(iter(cameras.values()))
        return {feature: only_camera for feature in CAMERA_OBS_FEATURES}

    available = ", ".join(sorted(cameras)) or "<none>"
    raise ValueError(
        f"missing camera features {missing}; available extracted cameras: {available}"
    )


def _frame_paths(camera_sources: dict[str, Path]) -> list[Path]:
    first_source = camera_sources[CAMERA_OBS_FEATURES[0]]
    paths = sorted(first_source.glob("*.png"))
    if not paths:
        raise ValueError(f"no PNG frames found in {first_source}")
    counts = {
        feature: len(list(source.glob("*.png")))
        for feature, source in camera_sources.items()
    }
    n_frames = min(counts.values())
    if n_frames <= 0:
        raise ValueError(f"invalid camera frame counts: {counts}")
    return paths[:n_frames]


def _load_reference_timestamps(camera_source: Path, n_frames: int) -> np.ndarray | None:
    ts_path = camera_source / "timestamps.npy"
    if not ts_path.exists():
        return None
    timestamps = np.load(ts_path)
    if len(timestamps) < n_frames:
        return None
    return timestamps[:n_frames].astype(np.int64)


def _nearest_state_indices(
    samples: list[dict[str, Any]],
    *,
    frame_timestamps: np.ndarray | None,
    n_frames: int,
    fps: float,
) -> list[int]:
    if frame_timestamps is not None:
        sample_ts = np.asarray([int(sample["timestamp_ns"]) for sample in samples])
        indices = np.searchsorted(sample_ts, frame_timestamps)
        indices = np.clip(indices, 0, len(samples) - 1)
        prev = np.clip(indices - 1, 0, len(samples) - 1)
        choose_prev = np.abs(sample_ts[prev] - frame_timestamps) <= np.abs(
            sample_ts[indices] - frame_timestamps
        )
        indices[choose_prev] = prev[choose_prev]
        return indices.astype(int).tolist()

    sample_elapsed = np.asarray([float(sample["elapsed_s"]) for sample in samples])
    frame_elapsed = np.arange(n_frames, dtype=np.float64) / float(fps)
    indices = np.searchsorted(sample_elapsed, frame_elapsed)
    indices = np.clip(indices, 0, len(samples) - 1)
    prev = np.clip(indices - 1, 0, len(samples) - 1)
    choose_prev = np.abs(sample_elapsed[prev] - frame_elapsed) <= np.abs(
        sample_elapsed[indices] - frame_elapsed
    )
    indices[choose_prev] = prev[choose_prev]
    return indices.astype(int).tolist()


def _copy_resized_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    with Image.open(src) as image:
        image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        image.save(dst, compress_level=1)


def _observation_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    obs = dict(sample["pre_observation"])
    obs.pop("images", None)
    return obs


def convert(
    recording_dir: str | Path,
    *,
    dataset_name: str,
    output_raw_dir: str | Path | None = None,
    output_lerobot_dir: str | Path | None = None,
    fps: float | None = None,
    seed: int = 2000,
    allow_single_camera_duplicate: bool = True,
    skip_lerobot: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    recording_dir = Path(recording_dir)
    metadata = read_json(recording_dir / "metadata.json")
    samples = _load_state_samples(recording_dir)
    camera_sources = _select_camera_sources(
        recording_dir,
        allow_single_camera_duplicate=allow_single_camera_duplicate,
    )
    frame_paths = _frame_paths(camera_sources)
    n_frames = len(frame_paths)
    fps = float(fps or metadata.get("camera_fps") or DEFAULT_FPS)

    dataset_dir = Path(output_raw_dir) if output_raw_dir else RAW_ROOT / dataset_name
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_dir} already exists; pass --overwrite")
        shutil.rmtree(dataset_dir)

    episode_dir = ensure_dir(dataset_dir / "episodes" / "episode_000000")
    for feature in CAMERA_OBS_FEATURES:
        ensure_dir(episode_dir / "images" / feature)

    frame_timestamps = _load_reference_timestamps(
        camera_sources[CAMERA_OBS_FEATURES[0]],
        n_frames,
    )
    state_indices = _nearest_state_indices(
        samples,
        frame_timestamps=frame_timestamps,
        n_frames=n_frames,
        fps=fps,
    )

    randomization = default_randomization(False, "none")
    manifest = raw_dataset_manifest(
        dataset_name,
        task_id=JOINT_POS_TASK_ID,
        fps=fps,
        seed=seed,
        randomization=randomization,
        extra={
            "source": "raiden.recorder_dit_fake",
            "source_recording_dir": str(recording_dir),
            "source_state": metadata.get("state_source", "fake_dit_absjoint18"),
            "camera_feature_sources": {
                feature: source.name for feature, source in camera_sources.items()
            },
            "single_camera_duplicate": len(set(camera_sources.values())) == 1,
        },
    )
    write_json(dataset_dir / "dataset_manifest.json", manifest)

    meta = episode_meta(
        dataset_name=dataset_name,
        episode_id=0,
        seed=seed,
        max_steps=n_frames,
        randomization=randomization,
        task_id=JOINT_POS_TASK_ID,
    )
    meta.update(
        {
            "episode_length": n_frames,
            "episode_success": None,
            "success_source": "not_applicable_fake_raiden_smoke",
            "source_recording_dir": str(recording_dir),
        }
    )
    write_json(episode_dir / "episode_meta.json", meta)

    steps_path = episode_dir / "steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as steps_file:
        for frame_index in range(n_frames):
            image_paths: dict[str, str] = {}
            for feature, source in camera_sources.items():
                src = source / f"{frame_index:010d}.png"
                dst = episode_dir / "images" / feature / f"{frame_index:06d}.png"
                _copy_resized_image(src, dst)
                image_paths[feature] = str(dst.relative_to(episode_dir))

            state_index = state_indices[frame_index]
            next_state_index = state_indices[min(frame_index + 1, n_frames - 1)]
            sample = samples[state_index]
            next_sample = samples[next_state_index]
            pre = _observation_from_sample(sample)
            pre["images"] = image_paths
            post = _observation_from_sample(next_sample)
            row = {
                "schema_version": manifest["schema_version"],
                "dataset_name": dataset_name,
                "episode_id": 0,
                "step_index": frame_index,
                "sim_time": float(sample.get("elapsed_s", frame_index / fps)),
                "wall_time": None,
                "pre_observation": pre,
                "action": sample["action"],
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "env_info": {
                    "source": "raiden_fake_dit_smoke",
                    "state_sample_index": int(state_index),
                    "language_instruction": LANGUAGE_INSTRUCTION,
                    "action_interface": ACTION_INTERFACE,
                },
                "post_observation": post,
            }
            steps_file.write(json.dumps(row, sort_keys=False) + "\n")

    report = validate_raw_dataset(dataset_dir)
    if not report["valid"]:
        raise RuntimeError(f"raw validation failed: {dataset_dir / 'validation_report.json'}")

    summary: dict[str, Any] = {
        "raw_dataset_dir": str(dataset_dir),
        "recording_dir": str(recording_dir),
        "num_frames": n_frames,
        "camera_feature_sources": {
            feature: source.name for feature, source in camera_sources.items()
        },
        "validation_report": report,
    }

    if not skip_lerobot:
        lerobot_summary = convert_lerobot(
            dataset_dir,
            output_lerobot_dir,
            repo_id=f"local/{dataset_name}_state26_absjoint18",
        )
        summary["lerobot"] = lerobot_summary

    write_json(dataset_dir / "raiden_conversion_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Raiden fake-DIT recording to DIT raw/LeRobot."
    )
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-raw-dir", type=Path, default=None)
    parser.add_argument("--output-lerobot-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--no-single-camera-duplicate", action="store_true")
    parser.add_argument("--skip-lerobot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = convert(
        args.recording_dir,
        dataset_name=args.dataset_name,
        output_raw_dir=args.output_raw_dir,
        output_lerobot_dir=args.output_lerobot_dir,
        fps=args.fps,
        seed=args.seed,
        allow_single_camera_duplicate=not args.no_single_camera_duplicate,
        skip_lerobot=args.skip_lerobot,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
