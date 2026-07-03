"""Generic raw handoff collector.

The collector records env-provided observations, commanded actions, post-step observations,
raw env info, rewards, done flags, and camera images. It deliberately does not store manual
phase/subtask/progress labels.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from dit_handoff.constants import (
    ACTION_DIM,
    ACTION_INTERFACE,
    CAMERA_OBS_FEATURES,
    DATA_ROOT,
    DEFAULT_FPS,
    JOINT_POS_TASK_ID,
    LEFT_ARM_ASSET,
    LANGUAGE_INSTRUCTION,
    RAW_ROOT,
    RIGHT_ARM_ASSET,
)
from dit_handoff.data.schema import default_randomization, episode_meta, raw_dataset_manifest
from dit_handoff.env import register_tasks
from dit_handoff.utils.io import ensure_dir, write_json

TCP_OFFSET = (0.0, 0.0, 0.107)


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


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _success_bool(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return bool(value[0]) if value else False
    return bool(value)


def _format_context_progress(context: dict[str, Any]) -> str:
    state = context.get("_scripted_jointpos_state")
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


def _policy_obs(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict) and "policy" in obs and isinstance(obs["policy"], dict):
        return obs["policy"]
    if isinstance(obs, dict):
        return obs
    return {}


def _tensor_row(tensor: Any, env_id: int = 0) -> Any:
    value = tensor[env_id].detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _asset(env: Any, name: str) -> Any:
    return env.unwrapped.scene[name]


def _body_id(env: Any, arm_name: str, body_name: str = "panda_hand") -> int | None:
    try:
        body_ids, _ = _asset(env, arm_name).find_bodies(body_name)
    except Exception:
        return None
    if len(body_ids) != 1:
        return None
    return int(body_ids[0])


def _gripper_joint_ids(env: Any, arm_name: str) -> list[int]:
    try:
        joint_ids, _ = _asset(env, arm_name).find_joints(["panda_finger.*"])
    except Exception:
        return []
    return [int(idx) for idx in joint_ids]


def _tcp_pos_w(env: Any, arm_name: str):
    import torch
    from isaaclab.utils import math as math_utils

    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    if body_idx is None:
        return None
    hand_pos = robot.data.body_pos_w[:, body_idx, :]
    hand_quat = robot.data.body_quat_w[:, body_idx, :]
    offset = torch.tensor(TCP_OFFSET, device=env.unwrapped.device, dtype=hand_pos.dtype).repeat(env.unwrapped.num_envs, 1)
    return hand_pos + math_utils.quat_apply(hand_quat, offset)


def _tcp_quat_w(env: Any, arm_name: str):
    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    if body_idx is None:
        return None
    return robot.data.body_quat_w[:, body_idx, :]


def _arm_snapshot(env: Any, arm_name: str) -> dict[str, Any]:
    robot = _asset(env, arm_name)
    gripper_ids = _gripper_joint_ids(env, arm_name)
    gripper_opening = None
    if gripper_ids:
        import torch

        gripper_opening = torch.sum(torch.abs(robot.data.joint_pos[:, gripper_ids]), dim=1)
    tcp_pos = _tcp_pos_w(env, arm_name)
    tcp_quat = _tcp_quat_w(env, arm_name)
    snapshot = {
        "joint_pos": _tensor_row(robot.data.joint_pos[:, :9]),
        "joint_vel": _tensor_row(robot.data.joint_vel[:, :9]),
        "tcp_pos_w": _tensor_row(tcp_pos) if tcp_pos is not None else None,
        "tcp_quat_w": _tensor_row(tcp_quat) if tcp_quat is not None else None,
        "gripper_opening": _tensor_row(gripper_opening) if gripper_opening is not None else None,
    }
    return snapshot


def _static_asset_snapshot(env: Any, asset: Any, scene_name: str) -> dict[str, Any] | None:
    cfg = getattr(asset, "cfg", None)
    if cfg is None:
        try:
            cfg = getattr(env.unwrapped.cfg.scene, scene_name)
        except Exception:
            cfg = None
    init_state = getattr(cfg, "init_state", None) if cfg is not None else None
    if init_state is None:
        return None

    out: dict[str, Any] = {}
    pos = getattr(init_state, "pos", None)
    rot = getattr(init_state, "rot", None)
    if pos is not None:
        out["pos_w"] = [float(v) for v in pos]
        object_center_z = getattr(getattr(env.unwrapped, "cfg", None), "object_center_z", None)
        if object_center_z is not None and len(pos) >= 2:
            out["target_cube_center_w"] = [float(pos[0]), float(pos[1]), float(object_center_z)]
    if rot is not None:
        out["quat_w"] = [float(v) for v in rot]
    return out or None


def _object_snapshot(env: Any, scene_name: str) -> dict[str, Any] | None:
    try:
        asset = env.unwrapped.scene[scene_name]
    except Exception:
        return None
    data = getattr(asset, "data", None)
    out: dict[str, Any] = {}
    if data is not None:
        for attr, key in (("root_pos_w", "pos_w"), ("root_quat_w", "quat_w"), ("root_lin_vel_w", "lin_vel_w")):
            value = getattr(data, attr, None)
            if value is not None:
                out[key] = _tensor_row(value)
    if out:
        return out
    return _static_asset_snapshot(env, asset, scene_name)


def _sim_time(env: Any, fallback_step: int, fps: float) -> float:
    sim = getattr(env.unwrapped, "sim", None)
    for attr in ("current_time", "time", "sim_time"):
        value = getattr(sim, attr, None) if sim is not None else None
        if value is not None:
            try:
                return float(value)
            except TypeError:
                pass
    return float(fallback_step) / float(fps)


def _numeric_policy_obs(policy_obs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in policy_obs.items():
        if key in CAMERA_OBS_FEATURES:
            continue
        if hasattr(value, "detach") or isinstance(value, int | float | bool | str | list | tuple | dict):
            out[key] = _to_serializable(value)
    return out


def _save_rgb_image(image: Any, image_path: Path) -> None:
    from PIL import Image

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if str(image.dtype) != "torch.uint8":
        image = image.clamp(0, 255).to(dtype=__import__("torch").uint8)
    Image.fromarray(image.numpy()).save(image_path, compress_level=1)


def _save_images(policy_obs: dict[str, Any], episode_dir: Path, step_index: int) -> dict[str, str]:
    paths: dict[str, str] = {}
    for camera in CAMERA_OBS_FEATURES:
        image = policy_obs.get(camera)
        if image is None:
            continue
        image_path = episode_dir / "images" / camera / f"{step_index:06d}.png"
        _save_rgb_image(image, image_path)
        paths[camera] = str(image_path.relative_to(episode_dir))
    return paths


def _observation_snapshot(env: Any, obs: Any, episode_dir: Path, step_index: int, fps: float, *, save_images: bool) -> dict[str, Any]:
    policy_obs = _policy_obs(obs)
    snapshot = {
        "sim_time": _sim_time(env, step_index, fps),
        "arms": {
            "left": _arm_snapshot(env, LEFT_ARM_ASSET),
            "right": _arm_snapshot(env, RIGHT_ARM_ASSET),
        },
        "cube": _object_snapshot(env, "object"),
        "target": _object_snapshot(env, "target_area"),
        "yellow_handoff_area": _object_snapshot(env, "yellow_area"),
        "env_policy_observation": _numeric_policy_obs(policy_obs),
    }
    if save_images:
        snapshot["images"] = _save_images(policy_obs, episode_dir, step_index)
    return snapshot


def _action_row(action: Any) -> dict[str, Any]:
    action_list = _to_serializable(action)
    if isinstance(action_list, list) and action_list and isinstance(action_list[0], list):
        action_list = action_list[0]
    return {
        "raw_env_action": action_list,
        "action_interface": ACTION_INTERFACE,
        "action_layout": {"left_actor": [0, 9], "right_observer": [9, 18]},
        "joint_target_18_commanded": action_list,
        "joint_target_18_commanded_source": "commanded_env_action",
    }


def _current_jointpos_action(env: Any):
    import torch

    left = _asset(env, LEFT_ARM_ASSET).data.joint_pos[:, :9]
    right = _asset(env, RIGHT_ARM_ASSET).data.joint_pos[:, :9]
    return torch.cat([left, right], dim=-1).detach().clone()


def _warmup_cameras(env: Any, obs: Any, warmup_steps: int, device: Any) -> Any:
    """Advance no-op frames after reset so camera buffers contain current images."""
    for _ in range(max(0, int(warmup_steps))):
        action = _current_jointpos_action(env).to(device)
        obs, _, terminated, truncated, _ = env.step(action)
        if bool(terminated[0].detach().cpu().item()) or bool(truncated[0].detach().cpu().item()):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    return obs


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
    manifest = raw_dataset_manifest(
        args.dataset_name,
        task_id=args.task,
        fps=args.fps,
        seed=args.seed,
        randomization=randomization,
        extra={
            "expert": args.expert,
            "camera_warmup_steps": int(args.camera_warmup_steps),
            "camera_warmup_policy": "no-op current Joint-Pos targets; warmup frames are not recorded",
        },
    )
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
                    f"reached max attempts {args.max_attempts} with "
                    f"{kept_episodes}/{args.episodes} successful episodes"
                )
            if not simulation_app.is_running():
                raise RuntimeError(
                    f"simulation app stopped with {kept_episodes}/{args.episodes} successful episodes"
                )
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
            obs = _warmup_cameras(env, obs, args.camera_warmup_steps, device)
            meta = episode_meta(
                dataset_name=args.dataset_name,
                episode_id=episode_id,
                seed=seed,
                max_steps=args.max_steps,
                randomization=randomization,
                task_id=args.task,
            )
            meta["source_attempt_id"] = attempt_id
            meta["camera_warmup_steps"] = int(args.camera_warmup_steps)
            meta["camera_warmup_policy"] = "no-op current Joint-Pos targets; warmup frames are not recorded"
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
            }

            episode_started_at = time.time()
            print(
                f"[collector] start episode_{episode_id:06d} attempt={attempt_id} "
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
                    if action.shape[-1] != ACTION_DIM:
                        raise RuntimeError(f"expert returned action dim {action.shape[-1]}, expected {ACTION_DIM}")
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
                    "action": _action_row(action),
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
                        f"[collector] progress episode_{episode_id:06d} "
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
                    f"[collector] rejected attempt {attempt_id} for episode_{episode_id:06d}; "
                    f"kept {kept_episodes}/{args.episodes}",
                    flush=True,
                )
                continue
            kept_episodes += 1
            print(
                f"[collector] kept episode_{episode_id:06d} "
                f"({kept_episodes}/{args.episodes}, attempt {attempt_id}, success={success})",
                flush=True,
            )
    finally:
        env.close()
        simulation_app.close()

    return dataset_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect generic raw handoff data.")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-name", default="handoff_jointpos_raw_smoke")
    parser.add_argument("--task", default=JOINT_POS_TASK_ID)
    parser.add_argument("--expert", default="dit_handoff.collect.experts:hold_current_jointpos_action")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--record-image-every", type=int, default=1)
    parser.add_argument("--camera-warmup-steps", type=int, default=1)
    parser.add_argument("--require-success", action="store_true", help="Keep collecting until --episodes successful episodes are retained.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Maximum attempts when --require-success is set; 0 means unlimited.")
    parser.add_argument("--progress-log-every", type=int, default=100, help="Print collector progress every N recorded steps; 0 disables step progress logs.")
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

