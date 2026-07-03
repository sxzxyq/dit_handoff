"""Merge successful raw collection shards into one raw dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dit_handoff.utils.io import ensure_dir, iter_jsonl, read_json, write_json


def _success_bool(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return bool(value[0]) if value else False
    return bool(value)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _attach_images(src_images: Path, dst_images: Path, link_mode: str) -> None:
    if dst_images.exists() or dst_images.is_symlink():
        raise FileExistsError(dst_images)
    if link_mode == "symlink":
        dst_images.symlink_to(src_images.resolve(), target_is_directory=True)
    elif link_mode == "hardlink":
        shutil.copytree(src_images, dst_images, copy_function=_link_or_copy)
    elif link_mode == "copy":
        shutil.copytree(src_images, dst_images)
    else:
        raise ValueError(f"unknown link mode: {link_mode}")


def _rewrite_steps(src_steps: Path, dst_steps: Path, dataset_name: str, episode_id: int) -> int:
    rows = 0
    with dst_steps.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(src_steps):
            row["dataset_name"] = dataset_name
            row["episode_id"] = episode_id
            f.write(json.dumps(row, sort_keys=False) + "\n")
            rows += 1
    return rows


def merge_raw_shards(
    shard_dirs: list[Path],
    output_dir: Path,
    *,
    dataset_name: str,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a new DIT_RUN_STAMP/DIT_RAW_NAME")
    if not shard_dirs:
        raise ValueError("no shard directories provided")

    ensure_dir(output_dir / "episodes")
    first_manifest = read_json(shard_dirs[0] / "dataset_manifest.json")
    manifest = dict(first_manifest)
    manifest["dataset_name"] = dataset_name
    manifest["merged_from"] = [str(path) for path in shard_dirs]
    manifest["merge_link_mode"] = link_mode
    write_json(output_dir / "dataset_manifest.json", manifest)

    episode_count = 0
    row_counts: dict[str, int] = {}
    sources: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        shard_manifest = read_json(shard_dir / "dataset_manifest.json")
        shard_name = shard_manifest.get("dataset_name", shard_dir.name)
        for src_ep in sorted((shard_dir / "episodes").glob("episode_*")):
            meta = read_json(src_ep / "episode_meta.json")
            if not _success_bool(meta.get("episode_success")):
                raise RuntimeError(f"non-success episode in shard: {src_ep}")
            dst_ep = output_dir / "episodes" / f"episode_{episode_count:06d}"
            ensure_dir(dst_ep)
            _attach_images(src_ep / "images", dst_ep / "images", link_mode)
            row_count = _rewrite_steps(src_ep / "steps.jsonl", dst_ep / "steps.jsonl", dataset_name, episode_count)

            original_episode_id = meta.get("episode_id")
            meta["dataset_name"] = dataset_name
            meta["episode_id"] = episode_count
            meta["source_dataset_name"] = shard_name
            meta["source_episode_name"] = src_ep.name
            meta["source_episode_id"] = original_episode_id
            write_json(dst_ep / "episode_meta.json", meta)

            row_counts[dst_ep.name] = row_count
            sources.append(
                {
                    "episode": dst_ep.name,
                    "source_dataset_name": shard_name,
                    "source_episode_name": src_ep.name,
                    "rows": row_count,
                }
            )
            episode_count += 1

    summary = {
        "output_dir": str(output_dir),
        "dataset_name": dataset_name,
        "num_shards": len(shard_dirs),
        "num_episodes": episode_count,
        "num_frames": sum(row_counts.values()),
        "link_mode": link_mode,
        "row_counts": row_counts,
        "sources": sources,
    }
    write_json(output_dir / "merge_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge raw DIT collection shards.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    summary = merge_raw_shards(
        args.shard_dirs,
        args.output_dir,
        dataset_name=args.dataset_name,
        link_mode=args.link_mode,
    )
    print(json.dumps({k: summary[k] for k in ("output_dir", "num_shards", "num_episodes", "num_frames", "link_mode")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
