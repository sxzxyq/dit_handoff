"""Closed-loop state26 -> anchored relative EE pose14 evaluation in the IK handoff env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from dit_handoff.collect.raw_collector import _observation_snapshot, _policy_obs, _to_serializable
from dit_handoff.constants import CAMERA_OBS_FEATURES, EVAL_ROOT, IK_TASK_ID, LANGUAGE_INSTRUCTION, POSE14_ACTION_DIM
from dit_handoff.env import register_tasks
from dit_handoff.eval.adapters import pose14_anchor_from_live_snapshot, relee_pose14_to_ik_env_action, state26_from_live_snapshot
from dit_handoff.utils.io import ensure_dir, read_json, write_json


def _encode_video_from_frames(frame_dir: Path, video_path: Path, fps: float) -> str | None:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return None
    import imageio.v2 as imageio

    ensure_dir(video_path.parent)
    with imageio.get_writer(video_path, fps=float(fps), codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    return str(video_path)


def _encode_saved_videos(out_dir: Path, fps: float) -> dict[str, str]:
    videos: dict[str, str] = {}
    for camera in CAMERA_OBS_FEATURES:
        frame_dir = out_dir / "images" / camera
        encoded = _encode_video_from_frames(frame_dir, out_dir / "videos" / f"{camera}.mp4", fps)
        if encoded is not None:
            videos[camera] = encoded
    return videos


def _resolve_policy_dir(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    if (checkpoint / "model.safetensors").exists() and (checkpoint / "train_config.json").exists():
        return checkpoint
    candidate = checkpoint / "pretrained_model"
    if (candidate / "model.safetensors").exists() and (candidate / "train_config.json").exists():
        return candidate
    raise FileNotFoundError(f"cannot find LeRobot pretrained_model under {checkpoint}")


def _load_policy_stack(checkpoint: Path, device: str):
    try:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies import make_policy, make_pre_post_processors
    except Exception as exc:
        raise RuntimeError("LeRobot 0.5.x policy/dataset APIs unavailable; run setup in the LeRobot env") from exc

    policy_dir = _resolve_policy_dir(checkpoint)
    cfg = TrainPipelineConfig.from_pretrained(str(policy_dir / "train_config.json"))
    cfg.policy.pretrained_path = policy_dir
    cfg.policy.device = device
    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=policy_dir,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return {
        "cfg": cfg,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "policy_dir": policy_dir,
    }


def _checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    for rel in ("dit_checkpoint_config.json", "dit_train_config.json", "train_config.json", "pretrained_model/train_config.json"):
        path = checkpoint / rel
        if path.exists():
            try:
                return read_json(path)
            except Exception:
                return {"config_path": str(path)}
    return {}


def _live_image_tensor(image: Any):
    import torch

    tensor = image.detach() if hasattr(image, "detach") else torch.as_tensor(image)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"expected camera tensor with 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32) / 255.0
    else:
        tensor = tensor.to(dtype=torch.float32)
        if float(tensor.max().detach().cpu()) > 2.0:
            tensor = tensor / 255.0
    return tensor.unsqueeze(0)


def _repeat_obs_window(batch: dict[str, Any], n_obs_steps: int) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key.startswith("observation."):
            out[key] = value.unsqueeze(1).repeat(1, n_obs_steps, *([1] * (value.ndim - 1)))
        else:
            out[key] = value
    return out


class _PolicyWorkerClient:
    def __init__(self, checkpoint: Path, policy_python: str, device: str, out_dir: Path):
        self.input_dir = ensure_dir(out_dir / "policy_worker_inputs")
        self.request_index = 0
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/qsh/dit/src" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen(
            [
                policy_python,
                "-m",
                "dit_handoff.eval.policy_worker_handoff_state26_relee_pose14",
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        ready = self._read_json()
        if not ready.get("ready"):
            raise RuntimeError(f"policy worker did not become ready: {ready}")
        self.policy_dir = ready.get("policy_dir")
        self.n_action_steps = int(ready.get("n_action_steps") or 0)

    def _read_json(self) -> dict[str, Any]:
        if self.proc.stdout is None:
            raise RuntimeError("policy worker stdout is closed")
        while True:
            line = self.proc.stdout.readline()
            if line == "" and self.proc.poll() is not None:
                raise RuntimeError(f"policy worker exited with code {self.proc.returncode}")
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                print(f"[policy-worker] {line.rstrip()}", flush=True)

    def action_chunk(self, snapshot: dict[str, Any], obs: Any, language: str) -> Any:
        import numpy as np
        import torch

        request_id = self.request_index
        self.request_index += 1
        input_path = self.input_dir / f"request_{request_id:06d}.npz"
        payload: dict[str, Any] = {
            "state": np.asarray(state26_from_live_snapshot(snapshot), dtype=np.float32),
        }
        policy_obs = _policy_obs(obs)
        for camera in CAMERA_OBS_FEATURES:
            image = policy_obs.get(camera)
            if image is not None:
                payload[camera] = _live_image_tensor(image).squeeze(0).detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(input_path, **payload)
        if self.proc.stdin is None:
            raise RuntimeError("policy worker stdin is closed")
        self.proc.stdin.write(json.dumps({"request_id": request_id, "input_npz": str(input_path), "language": language}) + "\n")
        self.proc.stdin.flush()
        response = self._read_json()
        if not response.get("ok"):
            raise RuntimeError(f"policy worker failed: {response}")
        return torch.tensor(response["action_chunk"], dtype=torch.float32)

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def _policy_action_chunk(policy_stack: dict[str, Any], snapshot: dict[str, Any], obs: Any, language: str, device: str):
    import torch

    if isinstance(policy_stack, _PolicyWorkerClient):
        return policy_stack.action_chunk(snapshot, obs, language)

    cfg = policy_stack["cfg"]
    policy = policy_stack["policy"]
    preprocessor = policy_stack["preprocessor"]
    postprocessor = policy_stack["postprocessor"]
    n_obs_steps = int(cfg.policy.n_obs_steps)

    state = state26_from_live_snapshot(snapshot)
    batch: dict[str, Any] = {
        "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
        "task": [language],
    }
    policy_obs = _policy_obs(obs)
    for camera in CAMERA_OBS_FEATURES:
        image = policy_obs.get(camera)
        if image is not None:
            batch[f"observation.images.{camera}"] = _live_image_tensor(image)
    batch = _repeat_obs_window(batch, n_obs_steps)
    batch = preprocessor(batch)
    with torch.no_grad():
        prepared = policy._prepare_batch(batch)
        normalized_chunk = policy._generate_actions(prepared)
        chunk = postprocessor(normalized_chunk)
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(1)
    return chunk.detach().cpu()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    import gymnasium as gym
    import torch

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    from isaaclab_tasks.utils import parse_env_cfg

    register_tasks()
    out_dir = Path(args.output_dir) if args.output_dir else EVAL_ROOT / f"closed_loop_{int(time.time())}"
    ensure_dir(out_dir)

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    ckpt_cfg = _checkpoint_config(checkpoint) if checkpoint else {}
    checkpoint_n_action_steps = ckpt_cfg.get("n_action_steps")
    checkpoint_n_action_seconds = ckpt_cfg.get("n_action_seconds")
    override = args.override_n_action_steps
    if override is not None and checkpoint_n_action_steps is not None:
        print(
            f"WARNING: overriding checkpoint n_action_steps={checkpoint_n_action_steps} with {override}",
            flush=True,
        )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    policy_stack = None
    policy_device = args.policy_device or args.device
    if not args.smoke_hold_current:
        if args.policy_python:
            policy_stack = _PolicyWorkerClient(checkpoint, args.policy_python, policy_device, out_dir)
        else:
            policy_stack = _load_policy_stack(checkpoint, policy_device)
    if policy_stack is not None:
        if isinstance(policy_stack, _PolicyWorkerClient):
            checkpoint_n_action_steps = policy_stack.n_action_steps
        else:
            checkpoint_n_action_steps = int(policy_stack["cfg"].policy.n_action_steps)
        checkpoint_n_action_seconds = float(checkpoint_n_action_steps) / float(args.fps)
    policy_calls_path = out_dir / "policy_calls.jsonl"
    steps_path = out_dir / "steps.jsonl"
    step_index = -1
    policy_call_count = 0
    last_info: dict[str, Any] = {}
    videos: dict[str, str] = {}
    video_error: str | None = None

    try:
        reset_out = env.reset(seed=args.seed)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        action_queue: list[dict[str, Any]] = []
        terminated = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        reward = torch.zeros(args.num_envs, device=env.unwrapped.device)
        for step_index in range(args.max_steps):
            if not simulation_app.is_running():
                break
            snapshot = _observation_snapshot(env, obs, out_dir, step_index, float(args.fps), save_images=args.save_video)
            if not action_queue:
                anchor_pose = pose14_anchor_from_live_snapshot(snapshot)
                if args.smoke_hold_current:
                    smoke_steps = int(override or 1)
                    chunk = torch.zeros((1, smoke_steps, POSE14_ACTION_DIM), dtype=torch.float32)
                    chunk[..., 6] = 1.0
                    chunk[..., 13] = 1.0
                else:
                    chunk = _policy_action_chunk(policy_stack, snapshot, obs, LANGUAGE_INSTRUCTION, args.device)
                chunk_list = chunk.tolist()[0]
                execute_steps = len(chunk_list)
                if override is not None:
                    execute_steps = min(execute_steps, int(override))
                action_queue = [
                    {"model_action14": list(map(float, act)), "anchor_pose": anchor_pose}
                    for act in chunk_list[:execute_steps]
                ]
                call_row = {
                    "policy_call_index": policy_call_count,
                    "env_step_index": step_index,
                    "action_chunk_shape": list(chunk.shape),
                    "executed_steps_from_chunk": execute_steps,
                    "checkpoint_n_action_steps": checkpoint_n_action_steps,
                    "override_n_action_steps": override,
                    "checkpoint_n_action_seconds": checkpoint_n_action_seconds,
                    "override_n_action_seconds": (float(override) / float(args.fps)) if override else None,
                    "anchor_pose": _to_serializable(anchor_pose),
                    "action_chunk": [item["model_action14"] for item in action_queue],
                }
                with policy_calls_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(call_row) + "\n")
                policy_call_count += 1
            queued = action_queue.pop(0)
            action = queued["model_action14"]
            env_action = relee_pose14_to_ik_env_action(
                action, queued["anchor_pose"], snapshot, ik_action_scale=float(args.ik_action_scale)
            ).to(env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(env_action)
            last_info = _to_serializable(info) if isinstance(info, dict) else {"repr": repr(info)}
            step_row = {
                "step_index": step_index,
                "sent_model_action14": action,
                "sent_env_action_ik_relative14": env_action.detach().cpu().tolist()[0],
                "ik_action_scale": float(args.ik_action_scale),
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
            "contract": "state26_relee_pose14_v1",
            "checkpoint": str(checkpoint) if checkpoint else None,
            "smoke_hold_current": args.smoke_hold_current,
            "steps_requested": args.max_steps,
            "steps_executed": max(0, step_index + 1),
            "policy_calls": policy_call_count,
            "checkpoint_n_action_steps": checkpoint_n_action_steps,
            "override_n_action_steps": override,
            "checkpoint_n_action_seconds": checkpoint_n_action_seconds,
            "override_n_action_seconds": (float(override) / float(args.fps)) if override else None,
            "last_info": last_info,
            "policy_python": args.policy_python,
            "policy_device": policy_device if not args.smoke_hold_current else None,
            "save_video": args.save_video,
            "ik_action_scale": float(args.ik_action_scale),
            "video_error": video_error,
            "outputs": {"steps": str(steps_path), "policy_calls": str(policy_calls_path), "videos": videos},
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if isinstance(policy_stack, _PolicyWorkerClient):
            policy_stack.close()
        env.close()
        simulation_app.close()

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop eval for state26 -> anchored relative EE pose14.")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default=IK_TASK_ID)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--override-n-action-steps", type=int, default=None)
    parser.add_argument("--policy-python", default=None)
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--smoke-hold-current", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Save camera frames and encode one MP4 per camera.")
    parser.add_argument("--ik-action-scale", type=float, default=0.5, help="Scale used by IsaacLab IK action terms; model deltas are divided by this before env.step.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

