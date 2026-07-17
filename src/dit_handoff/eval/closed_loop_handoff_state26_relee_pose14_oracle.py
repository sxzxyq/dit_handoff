"""Closed-loop oracle replay for state26 -> anchored relative EE pose14 sidecar labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dit_handoff.collect.raw_collector import _observation_snapshot, _to_serializable
from dit_handoff.constants import EVAL_ROOT, IK_TASK_ID, POSE14_ACTION_DIM
from dit_handoff.env import register_tasks
from dit_handoff.eval.adapters import pose14_anchor_from_live_snapshot, relee_pose14_to_ik_env_action
from dit_handoff.eval.closed_loop_handoff_state26_relee_pose14 import _encode_saved_videos
from dit_handoff.utils.io import ensure_dir, read_json, write_json
from dit_handoff.utils.pose_math import apply_pose_delta_axis_angle, pose_delta_axis_angle


def _episode_dir(raw_dir: Path, episode_id: int) -> Path:
    ep_dir = raw_dir / "episodes" / f"episode_{episode_id:06d}"
    if not ep_dir.exists():
        raise FileNotFoundError(f"episode directory not found: {ep_dir}")
    return ep_dir


def _count_steps(ep_dir: Path) -> int:
    with (ep_dir / "steps.jsonl").open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _episode_global_offset(raw_dir: Path, episode_id: int) -> int:
    offset = 0
    for ep in range(episode_id):
        offset += _count_steps(_episode_dir(raw_dir, ep))
    return offset


def _load_sidecar(dataset_dir: Path):
    import numpy as np

    manifest = read_json(dataset_dir / "manifest.json")
    sidecar = manifest.get("action_chunk_sidecar") or {}
    metadata = read_json(dataset_dir / sidecar.get("metadata", "action_chunks/metadata.json"))
    actions_rel = sidecar.get("actions") or metadata["actions_path"]
    mask_rel = sidecar.get("action_is_pad") or metadata["action_is_pad_path"]
    actions = np.load(dataset_dir / actions_rel, mmap_mode="r")
    is_pad = np.load(dataset_dir / mask_rel, mmap_mode="r")
    if actions.ndim != 3 or actions.shape[-1] != POSE14_ACTION_DIM:
        raise ValueError(f"expected sidecar [N,H,{POSE14_ACTION_DIM}], got {actions.shape}")
    if is_pad.shape != actions.shape[:2]:
        raise ValueError(f"mask shape {is_pad.shape} does not match actions {actions.shape}")
    return manifest, metadata, actions, is_pad


def _oracle_chunk(actions: Any, is_pad: Any, global_index: int, first_slot: int, n_action_steps: int):
    import torch

    end_slot = min(first_slot + n_action_steps, actions.shape[1])
    chunk_values = []
    chunk_pad = []
    for slot in range(first_slot, end_slot):
        chunk_values.append([float(v) for v in actions[global_index, slot].tolist()])
        chunk_pad.append(bool(is_pad[global_index, slot]))
    valid_values = [act for act, pad in zip(chunk_values, chunk_pad, strict=True) if not pad]
    if not valid_values:
        return torch.zeros((1, 0, POSE14_ACTION_DIM), dtype=torch.float32), chunk_pad
    return torch.tensor([valid_values], dtype=torch.float32), chunk_pad


def _pose14_targets(
    action: list[float],
    anchor_pose: dict[str, tuple[list[float], list[float]]],
) -> dict[str, tuple[list[float], list[float]]]:
    targets = {}
    for side, start in (("left", 0), ("right", 7)):
        targets[side] = apply_pose_delta_axis_angle(anchor_pose[side][0], anchor_pose[side][1], action[start : start + 6])
    return targets


def _pose_error_summary(
    target_pose: dict[str, tuple[list[float], list[float]]],
    actual_snapshot: dict[str, Any],
) -> dict[str, dict[str, float]]:
    from dit_handoff.convert.handoff_state26_relee_pose14 import arm_tcp_pose_root

    summary = {}
    for side in ("left", "right"):
        actual_pos, actual_quat = arm_tcp_pose_root(actual_snapshot, side)
        target_pos, target_quat = target_pose[side]
        delta = pose_delta_axis_angle(actual_pos, actual_quat, target_pos, target_quat)
        pos_err = sum(v * v for v in delta[:3]) ** 0.5
        rot_err = sum(v * v for v in delta[3:6]) ** 0.5
        summary[side] = {"pos_err_m": float(pos_err), "rot_err_rad": float(rot_err)}
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    import gymnasium as gym
    import torch

    raw_dir = Path(args.raw_dir)
    dataset_dir = Path(args.dataset_dir)
    ep_dir = _episode_dir(raw_dir, int(args.episode_id))
    ep_meta = read_json(ep_dir / "episode_meta.json")
    episode_length = int(ep_meta.get("episode_length") or _count_steps(ep_dir))
    seed = int(args.seed if args.seed is not None else ep_meta.get("seed", 2000 + int(args.episode_id)))
    global_offset = _episode_global_offset(raw_dir, int(args.episode_id))

    manifest, sidecar_meta, actions, is_pad = _load_sidecar(dataset_dir)
    first_slot = int(args.first_slot if args.first_slot is not None else sidecar_meta.get("first_executed_slot", 1))
    n_action_steps = int(args.n_action_steps)
    max_steps = int(args.max_steps if args.max_steps is not None else episode_length)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab_tasks.utils import parse_env_cfg

    register_tasks()
    out_dir = Path(args.output_dir) if args.output_dir else EVAL_ROOT / f"closed_loop_relee_pose14_oracle_ep{int(args.episode_id):06d}_{int(time.time())}"
    ensure_dir(out_dir)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)

    policy_calls_path = out_dir / "policy_calls.jsonl"
    steps_path = out_dir / "steps.jsonl"
    step_index = -1
    policy_call_count = 0
    last_info: dict[str, Any] = {}
    videos: dict[str, str] = {}
    video_error: str | None = None

    try:
        reset_out = env.reset(seed=seed)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        for _ in range(max(0, int(args.warmup_steps))):
            warmup_action = torch.zeros((1, POSE14_ACTION_DIM), dtype=torch.float32, device=env.unwrapped.device)
            warmup_action[:, 6] = 1.0
            warmup_action[:, 13] = 1.0
            obs, _, _, _, _ = env.step(warmup_action)

        action_queue: list[dict[str, Any]] = []
        terminated = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        reward = torch.zeros(args.num_envs, device=env.unwrapped.device)

        for step_index in range(max_steps):
            if not simulation_app.is_running():
                break
            if step_index >= episode_length:
                break
            snapshot = _observation_snapshot(env, obs, out_dir, step_index, float(args.fps), save_images=args.save_video)
            if not action_queue:
                anchor_pose = pose14_anchor_from_live_snapshot(snapshot)
                global_index = global_offset + step_index
                chunk, chunk_pad = _oracle_chunk(actions, is_pad, global_index, first_slot, n_action_steps)
                chunk_list = chunk.tolist()[0]
                execute_steps = len(chunk_list)
                if args.override_n_action_steps is not None:
                    execute_steps = min(execute_steps, int(args.override_n_action_steps))
                action_queue = [
                    {"model_action14": list(map(float, act)), "anchor_pose": anchor_pose}
                    for act in chunk_list[:execute_steps]
                ]
                call_row = {
                    "policy_call_index": policy_call_count,
                    "env_step_index": step_index,
                    "raw_episode_id": int(args.episode_id),
                    "raw_local_index": step_index,
                    "sidecar_global_index": global_index,
                    "sidecar_slots": [first_slot, first_slot + n_action_steps],
                    "action_chunk_shape": list(chunk.shape),
                    "executed_steps_from_chunk": execute_steps,
                    "pad_flags_in_requested_slots": chunk_pad,
                    "anchor_pose": _to_serializable(anchor_pose),
                    "action_chunk": [item["model_action14"] for item in action_queue],
                }
                with policy_calls_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(call_row) + "\n")
                policy_call_count += 1
                if not action_queue:
                    break

            queued = action_queue.pop(0)
            action = queued["model_action14"]
            target_pose = _pose14_targets(action, queued["anchor_pose"])
            env_action = relee_pose14_to_ik_env_action(
                action, queued["anchor_pose"], snapshot, ik_action_scale=float(args.ik_action_scale)
            ).to(env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(env_action)
            post_snapshot = _observation_snapshot(env, obs, out_dir, step_index, float(args.fps), save_images=False)
            tracking_error = _pose_error_summary(target_pose, post_snapshot)
            last_info = _to_serializable(info) if isinstance(info, dict) else {"repr": repr(info)}
            step_row = {
                "step_index": step_index,
                "raw_episode_id": int(args.episode_id),
                "raw_local_index": step_index,
                "sent_model_action14": action,
                "sent_env_action_ik_relative14": env_action.detach().cpu().tolist()[0],
                "ik_action_scale": float(args.ik_action_scale),
                "target_pose": _to_serializable(target_pose),
                "post_step_tracking_error": tracking_error,
                "reward": _to_serializable(reward),
                "terminated": bool(terminated[0].detach().cpu().item()),
                "truncated": bool(truncated[0].detach().cpu().item()),
                "env_info": last_info,
            }
            with steps_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(step_row) + "\n")
            if step_row["terminated"] or step_row["truncated"]:
                break
    finally:
        if args.save_video:
            try:
                videos = _encode_saved_videos(out_dir, float(args.fps))
            except Exception as exc:
                video_error = repr(exc)
        summary = {
            "task": args.task,
            "contract": "state26_relee_pose14_v1_oracle_sidecar",
            "raw_dir": str(raw_dir),
            "dataset_dir": str(dataset_dir),
            "raw_episode_id": int(args.episode_id),
            "raw_episode_length": episode_length,
            "raw_episode_success": bool(ep_meta.get("episode_success", False)),
            "seed": seed,
            "global_offset": global_offset,
            "first_slot": first_slot,
            "n_action_steps": n_action_steps,
            "steps_requested": max_steps,
            "steps_executed": max(0, step_index + 1),
            "policy_calls": policy_call_count,
            "last_info": last_info,
            "save_video": args.save_video,
            "ik_action_scale": float(args.ik_action_scale),
            "warmup_steps": int(args.warmup_steps),
            "video_error": video_error,
            "outputs": {"steps": str(steps_path), "policy_calls": str(policy_calls_path), "videos": videos},
            "source_manifest_repo_id": manifest.get("repo_id"),
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        env.close()
        simulation_app.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop oracle replay for pose14 sidecar labels.")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default=IK_TASK_ID)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--n-action-steps", type=int, default=40)
    parser.add_argument("--first-slot", type=int, default=None)
    parser.add_argument("--override-n-action-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
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
