"""Replay ground-truth anchored relative EE pose14 chunks in the IK handoff env."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dit_handoff.collect.raw_collector import _observation_snapshot, _to_serializable
from dit_handoff.constants import EVAL_ROOT, IK_TASK_ID, LEROBOT_ROOT, POSE14_ACTION_DIM
from dit_handoff.convert.handoff_state26_absjoint18 import _episode_dirs
from dit_handoff.env import register_tasks
from dit_handoff.eval.adapters import pose14_anchor_from_live_snapshot, relee_pose14_to_ik_env_action
from dit_handoff.eval.closed_loop_handoff_state26_relee_pose14 import _encode_saved_videos
from dit_handoff.utils.io import ensure_dir, iter_jsonl, read_json, write_json
from dit_handoff.utils.pose_math import apply_pose_delta_axis_angle, pose_delta_axis_angle


def _episode_lengths(raw_dir: Path) -> tuple[list[Path], list[int], list[int]]:
    episodes = _episode_dirs(raw_dir)
    lengths = [sum(1 for _ in iter_jsonl(ep / "steps.jsonl")) for ep in episodes]
    starts: list[int] = []
    total = 0
    for length in lengths:
        starts.append(total)
        total += length
    return episodes, lengths, starts


def _resolve_episode(episodes: list[Path], episode: str) -> int:
    if episode.isdigit():
        index = int(episode)
        if index < 0 or index >= len(episodes):
            raise IndexError(f"episode index {index} outside 0..{len(episodes) - 1}")
        return index
    for idx, ep in enumerate(episodes):
        if ep.name == episode:
            return idx
    raise ValueError(f"episode {episode!r} not found in raw dataset")


def _load_sidecar(dataset_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    metadata = read_json(dataset_root / "action_chunks" / "metadata.json")
    actions = np.load(dataset_root / metadata["actions_path"], mmap_mode="r")
    is_pad = np.load(dataset_root / metadata["action_is_pad_path"], mmap_mode="r")
    if actions.ndim != 3 or actions.shape[-1] != POSE14_ACTION_DIM:
        raise ValueError(f"expected actions [N,H,{POSE14_ACTION_DIM}], got {actions.shape}")
    if is_pad.shape != actions.shape[:2]:
        raise ValueError(f"pad mask shape {is_pad.shape} does not match actions {actions.shape[:2]}")
    return actions, is_pad, metadata


def _pose14_target_debug(
    action: list[float],
    anchor_pose: dict[str, tuple[list[float], list[float]]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    from dit_handoff.convert.handoff_state26_relee_pose14 import arm_tcp_pose_root

    out: dict[str, Any] = {}
    for side, start in (("left", 0), ("right", 7)):
        target_pos, target_quat = apply_pose_delta_axis_angle(
            anchor_pose[side][0], anchor_pose[side][1], action[start : start + 6]
        )
        current_pos, current_quat = arm_tcp_pose_root(snapshot, side)
        out[side] = {
            "current_tcp_root": {"pos": current_pos, "quat": current_quat},
            "target_tcp_root": {"pos": target_pos, "quat": target_quat},
            "target_error_before_step": pose_delta_axis_angle(current_pos, current_quat, target_pos, target_quat),
            "gripper_sign": action[start + 6],
        }
    return out


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    import gymnasium as gym
    import torch

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab_tasks.utils import parse_env_cfg

    register_tasks()
    dataset_root = Path(args.dataset_root)
    dataset_manifest = read_json(dataset_root / "manifest.json")
    raw_dir = Path(args.raw_dir or dataset_manifest["raw_dataset_path"])
    actions, is_pad, sidecar_meta = _load_sidecar(dataset_root)
    episodes, lengths, starts = _episode_lengths(raw_dir)
    ep_index = _resolve_episode(episodes, args.episode)
    ep_dir = episodes[ep_index]
    ep_meta = read_json(ep_dir / "episode_meta.json")
    ep_rows = list(iter_jsonl(ep_dir / "steps.jsonl"))
    seed = int(args.seed if args.seed is not None else ep_meta.get("seed", 2000))
    first_slot = int(args.first_executed_slot if args.first_executed_slot is not None else sidecar_meta["first_executed_slot"])
    n_action_steps = int(args.n_action_steps)
    episode_start = starts[ep_index]
    episode_length = lengths[ep_index]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else EVAL_ROOT / f"gt_replay_relee_pose14_{ep_dir.name}_{timestamp}"
    ensure_dir(out_dir)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    policy_calls_path = out_dir / "policy_calls.jsonl"
    steps_path = out_dir / "steps.jsonl"
    step_index = -1
    env_step_count = 0
    policy_call_count = 0
    last_info: dict[str, Any] = {}
    video_error: str | None = None
    videos: dict[str, str] = {}

    try:
        reset_out = env.reset(seed=seed)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        if args.camera_warmup_steps > 0:
            warmup_action = torch.zeros((args.num_envs, POSE14_ACTION_DIM), dtype=torch.float32, device=env.unwrapped.device)
            warmup_action[:, 6] = 1.0
            warmup_action[:, 13] = 1.0
            for _ in range(int(args.camera_warmup_steps)):
                obs, _, terminated, truncated, _ = env.step(warmup_action)
                if bool(terminated[0].detach().cpu().item()) or bool(truncated[0].detach().cpu().item()):
                    reset_out = env.reset(seed=seed)
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        action_queue: list[dict[str, Any]] = []
        terminated = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        reward = torch.zeros(args.num_envs, device=env.unwrapped.device)
        for step_index in range(args.max_steps):
            if not simulation_app.is_running():
                break
            if step_index >= episode_length and not action_queue:
                break
            snapshot = _observation_snapshot(env, obs, out_dir, step_index, float(args.fps), save_images=args.save_video)
            if not action_queue:
                local_anchor_index = min(step_index, episode_length - 1)
                global_anchor_index = episode_start + local_anchor_index
                if args.anchor_source == "raw":
                    from dit_handoff.convert.handoff_state26_relee_pose14 import arm_tcp_pose_root

                    raw_anchor_snapshot = ep_rows[local_anchor_index]["pre_observation"]
                    anchor_pose = {side: arm_tcp_pose_root(raw_anchor_snapshot, side) for side in ("left", "right")}
                else:
                    anchor_pose = pose14_anchor_from_live_snapshot(snapshot)
                chunk_actions: list[list[float]] = []
                chunk_slots: list[int] = []
                for slot in range(first_slot, min(actions.shape[1], first_slot + n_action_steps)):
                    if bool(is_pad[global_anchor_index, slot]):
                        continue
                    chunk_actions.append([float(v) for v in actions[global_anchor_index, slot].tolist()])
                    chunk_slots.append(int(slot))
                if not chunk_actions:
                    break
                action_queue = [
                    {"model_action14": act, "anchor_pose": anchor_pose, "slot": slot}
                    for act, slot in zip(chunk_actions, chunk_slots, strict=True)
                ]
                with policy_calls_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "policy_call_index": policy_call_count,
                                "env_step_index": step_index,
                                "episode": ep_dir.name,
                                "episode_local_anchor_index": local_anchor_index,
                                "global_anchor_index": global_anchor_index,
                                "anchor_source": args.anchor_source,
                                "executed_steps_from_chunk": len(action_queue),
                                "first_executed_slot": first_slot,
                                "n_action_steps": n_action_steps,
                                "sidecar_slots": chunk_slots,
                                "anchor_pose": _to_serializable(anchor_pose),
                                "action_chunk": [item["model_action14"] for item in action_queue],
                            }
                        )
                        + "\n"
                    )
                policy_call_count += 1

            queued = action_queue.pop(0)
            action = queued["model_action14"]
            target_debug = _pose14_target_debug(action, queued["anchor_pose"], snapshot)
            env_action = relee_pose14_to_ik_env_action(
                action, queued["anchor_pose"], snapshot, ik_action_scale=float(args.ik_action_scale)
            ).to(env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(env_action)
            env_step_count += 1
            last_info = _to_serializable(info) if isinstance(info, dict) else {"repr": repr(info)}
            with steps_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step_index": step_index,
                            "episode": ep_dir.name,
                            "sidecar_slot": queued["slot"],
                            "sent_model_action14": action,
                            "sent_env_action_ik_relative14": env_action.detach().cpu().tolist()[0],
                            "target_debug": _to_serializable(target_debug),
                            "live_snapshot": _to_serializable(
                                {
                                    "sim_time": snapshot.get("sim_time"),
                                    "cube": snapshot.get("cube"),
                                    "left_tcp_pos_w": snapshot.get("arms", {}).get("left", {}).get("tcp_pos_w"),
                                    "left_tcp_quat_w": snapshot.get("arms", {}).get("left", {}).get("tcp_quat_w"),
                                    "left_gripper_opening": snapshot.get("arms", {}).get("left", {}).get("gripper_opening"),
                                    "right_tcp_pos_w": snapshot.get("arms", {}).get("right", {}).get("tcp_pos_w"),
                                    "right_tcp_quat_w": snapshot.get("arms", {}).get("right", {}).get("tcp_quat_w"),
                                    "right_gripper_opening": snapshot.get("arms", {}).get("right", {}).get("gripper_opening"),
                                }
                            ),
                            "ik_action_scale": float(args.ik_action_scale),
                            "reward": _to_serializable(reward),
                            "terminated": bool(terminated[0].detach().cpu().item()),
                            "truncated": bool(truncated[0].detach().cpu().item()),
                            "env_info": last_info,
                        }
                    )
                    + "\n"
                )
            if bool(terminated[0].detach().cpu().item()) or bool(truncated[0].detach().cpu().item()):
                break
    finally:
        if args.save_video:
            try:
                videos = _encode_saved_videos(out_dir, float(args.fps))
            except Exception as exc:
                video_error = repr(exc)
        summary = {
            "task": args.task,
            "contract": "state26_relee_pose14_v1",
            "mode": "ground_truth_sidecar_replay",
            "anchor_source": args.anchor_source,
            "dataset_root": str(dataset_root),
            "raw_dir": str(raw_dir),
            "episode": ep_dir.name,
            "episode_index": ep_index,
            "episode_seed": seed,
            "episode_length": episode_length,
            "steps_requested": args.max_steps,
            "steps_executed": max(0, step_index + 1),
            "policy_calls": policy_call_count,
            "first_executed_slot": first_slot,
            "n_action_steps": n_action_steps,
            "last_info": last_info,
            "save_video": args.save_video,
            "ik_action_scale": float(args.ik_action_scale),
            "video_error": video_error,
            "outputs": {"steps": str(steps_path), "policy_calls": str(policy_calls_path), "videos": videos},
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        env.close()
        simulation_app.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay GT sidecar chunks for handoff state26 -> relative EE pose14.")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default=IK_TASK_ID)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=LEROBOT_ROOT / "handoff_jointpos_scripted_success120_0701_225643_state26_relee_pose14_h50_obs2",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--episode", default="0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=2600)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--first-executed-slot", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=40)
    parser.add_argument("--anchor-source", choices=["raw", "live"], default="raw", help="Use raw sidecar anchor for GT replay, or live current pose to mimic policy eval anchoring.")
    parser.add_argument("--camera-warmup-steps", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--ik-action-scale", type=float, default=0.5)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
