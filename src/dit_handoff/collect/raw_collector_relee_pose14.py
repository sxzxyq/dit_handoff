"""Raw collector for handoff state26 -> commanded IK-relative EE pose14 data."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from dit_handoff.collect.raw_collector import _observation_snapshot, _success_bool, _to_serializable
from dit_handoff.constants import (
    CAMERA_OBS_FEATURES,
    DATA_ROOT,
    DEFAULT_FPS,
    IK_TASK_ID,
    LANGUAGE_INSTRUCTION,
    POSE14_ACTION_DIM,
    POSE14_LAYOUT,
    RAW_ROOT,
    RELEE_POSE14_ACTION_INTERFACE,
    RELEE_POSE14_ACTION_REPRESENTATION,
)
from dit_handoff.data.schema import default_randomization, episode_meta, raw_dataset_manifest
from dit_handoff.env import register_tasks
from dit_handoff.utils.io import ensure_dir, write_json

DEFAULT_IK_ACTION_SCALE = 0.5


def _load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dotted_callable(spec: str):
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Callable must be formatted as module:attribute, got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def _format_context_progress(context: dict[str, Any]) -> str:
    state = context.get("_scripted_pose14_state")
    if not isinstance(state, dict):
        return ""
    parts = []
    if "phase_idx" in state:
        parts.append(f"phase_idx={state['phase_idx']}")
    if "phase_steps" in state:
        parts.append(f"phase_steps={state['phase_steps']}")
    if "yellow_achieved" in state:
        parts.append(f"yellow={bool(state['yellow_achieved'])}")
    if "red_achieved" in state:
        parts.append(f"red={bool(state['red_achieved'])}")
    return " " + " ".join(parts) if parts else ""


def _pose14_noop_action(env: Any):
    import torch

    action = torch.zeros((env.unwrapped.num_envs, POSE14_ACTION_DIM), device=env.unwrapped.device)
    action[:, 6] = 1.0
    action[:, 13] = 1.0
    return action


def _warmup_cameras(env: Any, obs: Any, warmup_steps: int) -> Any:
    for _ in range(max(0, int(warmup_steps))):
        action = _pose14_noop_action(env)
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated[0].detach().cpu().item()) or bool(truncated[0].detach().cpu().item()):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    return obs


def _flatten_action(action: Any) -> list[float]:
    action_list = _to_serializable(action)
    if isinstance(action_list, list) and action_list and isinstance(action_list[0], list):
        action_list = action_list[0]
    if not isinstance(action_list, list) or len(action_list) != POSE14_ACTION_DIM:
        raise ValueError(f"expected 14D pose14 action, got {action_list!r}")
    return [float(v) for v in action_list]


def _commanded_physical_delta(action_list: list[float], ik_action_scale: float) -> list[float]:
    commanded = list(action_list)
    for start in (0, 7):
        for offset in range(6):
            commanded[start + offset] = float(action_list[start + offset]) * float(ik_action_scale)
        commanded[start + 6] = 1.0 if float(action_list[start + 6]) >= 0.0 else -1.0
    return commanded


def _action_row(action: Any, ik_action_scale: float) -> dict[str, Any]:
    action_list = _flatten_action(action)
    commanded = _commanded_physical_delta(action_list, ik_action_scale)
    return {
        "raw_env_action": action_list,
        "action_interface": RELEE_POSE14_ACTION_INTERFACE,
        "action_representation": RELEE_POSE14_ACTION_REPRESENTATION,
        "action_layout": {"left_actor": [0, 7], "right_observer": [7, 14], "names": list(POSE14_LAYOUT)},
        "ik_action_scale": float(ik_action_scale),
        "pose14_delta_commanded": commanded,
        "pose14_delta_commanded_source": "commanded_expert_action",
    }


def _pose14_manifest(args: argparse.Namespace, randomization: dict[str, Any]) -> dict[str, Any]:
    return raw_dataset_manifest(
        args.dataset_name,
        task_id=args.task,
        fps=args.fps,
        seed=args.seed,
        randomization=randomization,
        extra={
            "action_interface": RELEE_POSE14_ACTION_INTERFACE,
            "action_representation": RELEE_POSE14_ACTION_REPRESENTATION,
            "action_dim": POSE14_ACTION_DIM,
            "action_names": list(POSE14_LAYOUT),
            "action_layout": {"left_actor": [0, 7], "right_observer": [7, 14], "names": list(POSE14_LAYOUT)},
            "expert": args.expert,
            "ik_action_scale": float(args.ik_action_scale),
            "camera_warmup_steps": int(args.camera_warmup_steps),
            "camera_warmup_policy": "14D IK-relative no-op with both grippers open; warmup frames are not recorded",
        },
    )


def _pose14_episode_meta(args: argparse.Namespace, episode_id: int, seed: int, randomization: dict[str, Any]) -> dict[str, Any]:
    meta = episode_meta(
        dataset_name=args.dataset_name,
        episode_id=episode_id,
        seed=seed,
        max_steps=args.max_steps,
        randomization=randomization,
        task_id=args.task,
    )
    meta["action_interface"] = RELEE_POSE14_ACTION_INTERFACE
    meta["action_dim"] = POSE14_ACTION_DIM
    meta["ik_action_scale"] = float(args.ik_action_scale)
    meta["camera_warmup_policy"] = "14D IK-relative no-op with both grippers open; warmup frames are not recorded"
    return meta


def collect(args: argparse.Namespace) -> Path:
    from isaaclab.app import AppLauncher

    import gymnasium as gym
    import torch

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab_tasks.utils import parse_env_cfg

    register_tasks()
    expert = _dotted_callable(args.expert)

    dataset_dir = RAW_ROOT / args.dataset_name
    if dataset_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{dataset_dir} already exists; pass --overwrite to replace/append intentionally")
    episodes_dir = dataset_dir / "episodes"
    ensure_dir(episodes_dir)
    randomization = default_randomization(args.enable_randomization, args.randomization_profile)
    manifest = _pose14_manifest(args, randomization)
    write_json(dataset_dir / "dataset_manifest.json", manifest)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device

    try:
        kept_episodes = 0
        attempts = 0
        collection_started_at = time.time()
        while kept_episodes < args.episodes:
            if args.max_attempts > 0 and attempts >= args.max_attempts:
                raise RuntimeError(
                    f"reached max attempts {args.max_attempts} with {kept_episodes}/{args.episodes} successful episodes"
                )
            if not simulation_app.is_running():
                raise RuntimeError(f"simulation app stopped with {kept_episodes}/{args.episodes} successful episodes")
            attempt_id = attempts
            episode_id = kept_episodes if args.require_success else attempts
            attempts += 1
            episode_dir = episodes_dir / f"episode_{episode_id:06d}"
            if episode_dir.exists() and args.overwrite:
                shutil.rmtree(episode_dir)
            ensure_dir(episode_dir / "images")
            for camera in CAMERA_OBS_FEATURES:
                ensure_dir(episode_dir / "images" / camera)

            seed = args.seed + attempt_id
            obs_out = env.reset(seed=seed)
            obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
            obs = _warmup_cameras(env, obs, args.camera_warmup_steps)
            meta = _pose14_episode_meta(args, episode_id, seed, randomization)
            meta["source_attempt_id"] = attempt_id
            meta["camera_warmup_steps"] = int(args.camera_warmup_steps)
            write_json(episode_dir / "episode_meta.json", meta)

            steps_path = episode_dir / "steps.jsonl"
            if steps_path.exists() and args.overwrite:
                steps_path.unlink()
            elif steps_path.exists():
                raise FileExistsError(f"{steps_path} already exists; pass --overwrite")
            terminated = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
            truncated = torch.zeros_like(terminated)
            reward = torch.zeros(args.num_envs, device=device)
            last_info: dict[str, Any] = {}
            episode_context: dict[str, Any] = {
                "dataset_name": args.dataset_name,
                "language_instruction": LANGUAGE_INSTRUCTION,
                "episode_id": episode_id,
                "seed": seed,
                "source_attempt_id": attempt_id,
                "scripted_pose14_cfg": {"ik_action_scale": float(args.ik_action_scale)},
            }

            episode_started_at = time.time()
            print(
                f"[pose14-collector] start episode_{episode_id:06d} attempt={attempt_id} "
                f"seed={seed} kept={kept_episodes}/{args.episodes}",
                flush=True,
            )

            step_index = -1
            for step_index in range(args.max_steps):
                if not simulation_app.is_running():
                    break
                pre = _observation_snapshot(env, obs, episode_dir, step_index, args.fps, save_images=True)
                with torch.no_grad():
                    action = expert(env, obs, step_index, episode_id, episode_context)
                    if action.shape[-1] != POSE14_ACTION_DIM:
                        raise RuntimeError(f"expert returned action dim {action.shape[-1]}, expected {POSE14_ACTION_DIM}")
                    obs, reward, terminated, truncated, info = env.step(action.to(device))
                post = _observation_snapshot(env, obs, episode_dir, step_index + 1, args.fps, save_images=False)
                last_info = _to_serializable(info) if isinstance(info, dict) else {"repr": repr(info)}
                row = {
                    "schema_version": manifest["schema_version"],
                    "dataset_name": args.dataset_name,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "sim_time": pre.get("sim_time"),
                    "wall_time": time.time(),
                    "pre_observation": pre,
                    "action": _action_row(action, float(args.ik_action_scale)),
                    "reward": _to_serializable(reward),
                    "terminated": bool(terminated[0].detach().cpu().item()),
                    "truncated": bool(truncated[0].detach().cpu().item()),
                    "env_info": last_info,
                    "post_observation": post,
                }
                with steps_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=False) + "\n")
                if args.progress_log_every > 0 and (step_index + 1) % args.progress_log_every == 0:
                    episode_elapsed = time.time() - episode_started_at
                    total_elapsed = time.time() - collection_started_at
                    phase = _format_context_progress(episode_context)
                    print(
                        f"[pose14-collector] progress episode_{episode_id:06d} "
                        f"step={step_index + 1}/{args.max_steps} attempt={attempt_id} "
                        f"kept={kept_episodes}/{args.episodes} "
                        f"episode_s={episode_elapsed:.1f} total_min={total_elapsed / 60.0:.1f}"
                        f"{phase}",
                        flush=True,
                    )
                if (
                    bool(terminated[0].detach().cpu().item())
                    or bool(truncated[0].detach().cpu().item())
                    or bool(episode_context.get("expert_done", False))
                ):
                    break

            meta["episode_length"] = step_index + 1 if step_index >= 0 else 0
            env_success = _to_serializable(last_info).get("success") if isinstance(last_info, dict) else None
            expert_success = episode_context.get("expert_success")
            meta["episode_success"] = env_success if env_success is not None else expert_success
            meta["success_source"] = "env_info" if env_success is not None else "expert_context_final"
            meta["expert_done"] = bool(episode_context.get("expert_done", False))
            write_json(episode_dir / "episode_meta.json", meta)
            success = _success_bool(meta.get("episode_success"))
            if args.require_success and not success:
                rejected = {
                    "attempt_id": attempt_id,
                    "episode_slot": episode_id,
                    "seed": seed,
                    "episode_length": meta["episode_length"],
                    "episode_success": meta.get("episode_success"),
                    "success_source": meta.get("success_source"),
                    "expert_done": meta.get("expert_done"),
                }
                with (dataset_dir / "rejected_episodes.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rejected, sort_keys=False) + "\n")
                shutil.rmtree(episode_dir, ignore_errors=True)
                print(
                    f"[pose14-collector] rejected attempt {attempt_id} for episode_{episode_id:06d}; "
                    f"kept {kept_episodes}/{args.episodes}",
                    flush=True,
                )
                continue
            kept_episodes += 1
            print(
                f"[pose14-collector] kept episode_{episode_id:06d} "
                f"({kept_episodes}/{args.episodes}, attempt {attempt_id}, success={success})",
                flush=True,
            )
    finally:
        env.close()
        simulation_app.close()

    return dataset_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect raw handoff data with commanded IK-relative EE pose14 actions.")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-name", default="handoff_relee_pose14_raw_smoke")
    parser.add_argument("--task", default=IK_TASK_ID)
    parser.add_argument("--expert", default="dit_handoff.collect.experts:scripted_handoff_relee_pose14_action")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--record-image-every", type=int, default=1)
    parser.add_argument("--camera-warmup-steps", type=int, default=1)
    parser.add_argument("--ik-action-scale", type=float, default=DEFAULT_IK_ACTION_SCALE)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--progress-log-every", type=int, default=100)
    parser.add_argument("--enable-randomization", action="store_true")
    parser.add_argument("--randomization-profile", default="none")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args, _ = parser.parse_known_args(raw_argv)
    config = _load_json_config(args.config)
    if config:
        valid_dests = {action.dest for action in parser._actions}
        translated = []
        for key, value in config.items():
            dest = "task" if key == "task_id" else key
            if dest not in valid_dests:
                continue
            option = dest.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    translated.append(f"--{option}")
            else:
                translated.extend([f"--{option}", str(value)])
        args = parser.parse_args(translated + raw_argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(DATA_ROOT).startswith("/data/shared_folder/datasets/dit"):
        raise RuntimeError(f"DATA_ROOT must stay fixed under /data/shared_folder/datasets/dit, got {DATA_ROOT}")
    dataset_dir = collect(args)
    print(dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
