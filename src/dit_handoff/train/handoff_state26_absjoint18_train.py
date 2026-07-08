"""Thin wrapper around official LeRobot MultiTask DiT training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dit_handoff.constants import (
    ACTION_DIM,
    CAMERA_OBS_FEATURES,
    CHECKPOINT_ROOT,
    DEFAULT_HORIZON_SECONDS,
    DEFAULT_N_ACTION_SECONDS,
    DEFAULT_N_OBS_STEPS,
    LEROBOT_CONTRACT_VERSION,
    RUNS_ROOT,
    STATE_DIM,
)
from dit_handoff.train.common import (
    _build_training_subprocess_env,
    _load_profile,
    _resolve_env_executable,
    _round_steps,
)
from dit_handoff.utils.io import ensure_dir, read_json, write_json



def build_training_config(dataset_dir: Path, profile: dict[str, Any], *, run_name: str, smoke: bool) -> dict[str, Any]:
    manifest = read_json(dataset_dir / "manifest.json")
    if manifest.get("contract") != LEROBOT_CONTRACT_VERSION:
        raise ValueError(f"dataset contract must be {LEROBOT_CONTRACT_VERSION}")
    fps = float(manifest["fps"])
    horizon_seconds = float(profile.get("horizon_seconds", DEFAULT_HORIZON_SECONDS))
    n_action_seconds = profile.get("n_action_seconds")
    horizon = int(profile.get("horizon") or _round_steps(fps, horizon_seconds))
    if "n_action_steps" in profile:
        n_action_steps = int(profile["n_action_steps"])
        n_action_seconds = float(n_action_steps) / fps
    else:
        n_action_seconds = float(n_action_seconds if n_action_seconds is not None else DEFAULT_N_ACTION_SECONDS)
        n_action_steps = _round_steps(fps, n_action_seconds)
    n_obs_steps = int(profile.get("n_obs_steps", DEFAULT_N_OBS_STEPS))
    physical_batch_size = int(profile.get("physical_batch_size", profile.get("batch_size", 32)))
    grad_accum = int(profile.get("gradient_accumulation_steps", 1))
    effective_batch_size = physical_batch_size * grad_accum
    steps = int(profile.get("smoke_steps" if smoke else "steps", 10 if smoke else 30000))
    output_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    warnings = []
    if effective_batch_size < 32:
        warnings.append("single-process effective batch size is below 32; check the actual DDP effective batch before a long run")
    if grad_accum > 1:
        warnings.append("Gradient accumulation is enabled through local DIT_GRADIENT_ACCUMULATION_STEPS support in lerobot-train")
    warnings.append("LeRobot 0.5.2 train CLI has no dataset.push_to_hub flag; only policy/checkpoint hub push is disabled explicitly")
    if "n_action_steps" in profile and profile.get("profile", "").startswith("diagnostic"):
        warnings.append("short action chunk profile is diagnostic only and not the formal baseline default")
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_repo_id": manifest.get("repo_id") or str(dataset_dir),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "policy_type": "multi_task_dit",
        "policy_repo_id": profile.get("policy_repo_id", f"local/{run_name}"),
        "policy_push_to_hub": bool(profile.get("policy_push_to_hub", False)),
        "dataset_push_to_hub": bool(profile.get("dataset_push_to_hub", False)),
        "save_checkpoint_to_hub": bool(profile.get("save_checkpoint_to_hub", False)),
        "fps": fps,
        "control_hz": fps,
        "horizon": horizon,
        "horizon_seconds": horizon_seconds,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "n_action_seconds": n_action_seconds,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "image_features": list(CAMERA_OBS_FEATURES),
        "action_representation": "absolute_joint_position_target",
        "objective": profile.get("objective", "diffusion"),
        "noise_scheduler_type": profile.get("noise_scheduler_type", "DDPM"),
        "num_train_timesteps": profile.get("num_train_timesteps", 100),
        "num_inference_steps": profile.get("num_inference_steps", 10),
        "use_amp": bool(profile.get("use_amp", False)),
        "mixed_precision": profile.get("mixed_precision"),
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_accumulation_supported_by_cli": False,
        "gradient_accumulation_runtime_env": "DIT_GRADIENT_ACCUMULATION_STEPS",
        "effective_batch_size": effective_batch_size,
        "learning_rate": profile.get("optimizer_lr", 2e-5),
        "vision_encoder_lr_multiplier": profile.get("vision_encoder_lr_multiplier", 0.1),
        "steps": steps,
        "num_workers": int(profile.get("num_workers", 4)),
        "prefetch_factor": int(profile.get("prefetch_factor", 4)),
        "persistent_workers": bool(profile.get("persistent_workers", True)),
        "save_checkpoint": bool(profile.get("save_checkpoint", True)),
        "save_freq": profile.get("save_freq", 1000),
        "log_freq": profile.get("log_freq", 100),
        "eval_freq": profile.get("eval_freq", 1000),
        "eval_steps": profile.get("eval_steps", 0),
        "seed": profile.get("seed", 2000),
        "transform_config": {
            "image_crop_shape": profile.get("image_crop_shape", [224, 224]),
            "image_crop_is_random": profile.get("image_crop_is_random", True),
            "train_random_augmentation_only": True,
            "eval_deterministic_preprocessing": True,
        },
        "model": {
            "num_layers": profile.get("num_layers", 6),
            "hidden_dim": profile.get("hidden_dim", 512),
            "num_heads": profile.get("num_heads", 8),
            "use_rope": profile.get("use_rope", True),
            "separate_rgb_encoder_per_camera": profile.get("separate_rgb_encoder_per_camera", False),
        },
        "warnings": warnings,
    }



def _resolve_lerobot_train_executable() -> str | None:
    return _resolve_env_executable("lerobot-train")


def _resolve_accelerate_executable() -> str | None:
    return _resolve_env_executable("accelerate")



def build_lerobot_train_command(
    config: dict[str, Any],
    *,
    device: str,
    wandb: bool,
    executable: str = "lerobot-train",
    accelerate_executable: str | None = None,
) -> list[str]:
    command = [
        executable,
        f"--dataset.repo_id={config['dataset_repo_id']}",
        f"--dataset.root={config['dataset_dir']}",
        f"--output_dir={config['output_dir']}",
        f"--save_checkpoint_to_hub={str(config['save_checkpoint_to_hub']).lower()}",
        f"--batch_size={config['physical_batch_size']}",
        f"--steps={config['steps']}",
        f"--num_workers={config['num_workers']}",
        f"--prefetch_factor={config['prefetch_factor']}",
        f"--persistent_workers={str(config['persistent_workers']).lower()}",
        f"--save_checkpoint={str(config['save_checkpoint']).lower()}",
        f"--save_freq={config['save_freq']}",
        f"--log_freq={config['log_freq']}",
        f"--env_eval_freq={config['eval_freq']}",
        f"--eval_steps={config['eval_steps']}",
        f"--seed={config['seed']}",
        "--policy.type=multi_task_dit",
        f"--policy.repo_id={config['policy_repo_id']}",
        f"--policy.push_to_hub={str(config['policy_push_to_hub']).lower()}",
        f"--policy.device={device}",
        f"--policy.horizon={config['horizon']}",
        f"--policy.n_obs_steps={config['n_obs_steps']}",
        f"--policy.n_action_steps={config['n_action_steps']}",
        f"--policy.objective={config['objective']}",
        f"--policy.noise_scheduler_type={config['noise_scheduler_type']}",
        f"--policy.num_train_timesteps={config['num_train_timesteps']}",
        f"--policy.num_inference_steps={config['num_inference_steps']}",
        f"--policy.use_amp={str(config['use_amp']).lower()}",
        f"--policy.num_layers={config['model']['num_layers']}",
        f"--policy.hidden_dim={config['model']['hidden_dim']}",
        f"--policy.num_heads={config['model']['num_heads']}",
        f"--policy.use_rope={str(config['model']['use_rope']).lower()}",
        f"--policy.use_separate_rgb_encoder_per_camera={str(config['model']['separate_rgb_encoder_per_camera']).lower()}",
        f"--policy.optimizer_lr={config['learning_rate']}",
        f"--policy.vision_encoder_lr_multiplier={config['vision_encoder_lr_multiplier']}",
        f"--policy.image_crop_shape={config['transform_config']['image_crop_shape']}",
        f"--policy.image_crop_is_random={str(config['transform_config']['image_crop_is_random']).lower()}",
        f"--wandb.enable={str(bool(wandb)).lower()}",
    ]
    num_processes = int(config.get("num_processes", 1))
    if num_processes > 1:
        if accelerate_executable is None:
            raise RuntimeError("accelerate is required for multi-GPU training")
        launch = [
            accelerate_executable,
            "launch",
            "--multi_gpu",
            f"--num_processes={num_processes}",
        ]
        if config.get("mixed_precision"):
            launch.append(f"--mixed_precision={config['mixed_precision']}")
        if config.get("gpu_ids"):
            launch.append(f"--gpu_ids={config['gpu_ids']}")
        if config.get("main_process_port"):
            launch.append(f"--main_process_port={config['main_process_port']}")
        command = launch + command
    return command


def train(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    profile = _load_profile(args.profile)
    config = build_training_config(dataset_dir, profile, run_name=args.run_name, smoke=args.smoke)
    subprocess_env, cache_env = _build_training_subprocess_env()
    config["cache_env"] = cache_env
    config["num_processes"] = int(args.num_processes)
    config["gpu_ids"] = args.gpu_ids
    config["main_process_port"] = args.main_process_port
    config["single_process_effective_batch_size"] = config["effective_batch_size"]
    config["actual_effective_batch_size"] = (
        config["physical_batch_size"] * config["num_processes"] * config["gradient_accumulation_steps"]
    )
    config["effective_batch_size"] = config["actual_effective_batch_size"]
    if config["gradient_accumulation_steps"] > 1:
        subprocess_env["DIT_GRADIENT_ACCUMULATION_STEPS"] = str(config["gradient_accumulation_steps"])
    if config["num_processes"] > 1:
        config["warnings"].append(
            "Multi-GPU uses Accelerate DDP; actual effective batch is physical_batch_size * num_processes * gradient_accumulation_steps"
        )
    output_dir = Path(config["output_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    ensure_dir(output_dir.parent)
    ensure_dir(checkpoint_dir)

    train_executable = _resolve_lerobot_train_executable()
    accelerate_executable = _resolve_accelerate_executable() if config["num_processes"] > 1 else None
    command = build_lerobot_train_command(
        config,
        device=args.device,
        wandb=args.wandb,
        executable=train_executable or "lerobot-train",
        accelerate_executable=accelerate_executable,
    )

    if args.dry_run:
        config["warnings"].append("dry-run metadata is written to checkpoint_dir only so the real output_dir remains unused")
    preflight = {"command": command, "dry_run": args.dry_run}
    write_json(checkpoint_dir / "dit_checkpoint_config.json", config)
    write_json(checkpoint_dir / "lerobot_train_command.json", preflight)

    if train_executable is None:
        config["warnings"].append("lerobot-train not found in the active Python environment or on PATH; run setup first")
        if not args.dry_run:
            raise RuntimeError("lerobot-train not found in the active Python environment or on PATH")
    if config["num_processes"] > 1 and accelerate_executable is None:
        config["warnings"].append("accelerate not found in the active Python environment or on PATH; multi-GPU training cannot start")
        if not args.dry_run:
            raise RuntimeError("accelerate not found in the active Python environment or on PATH")
    if not args.dry_run:
        subprocess.run(command, check=True, env=subprocess_env)
        ensure_dir(output_dir)
        write_json(output_dir / "dit_train_config.json", config)
        write_json(output_dir / "lerobot_train_command.json", preflight)
    return {"config": config, "command": command}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run official LeRobot MultiTask DiT training wrapper.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--run-name", default="handoff_state26_absjoint18_official_time")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids for Accelerate multi-GPU launch, e.g. 0,1.")
    parser.add_argument("--main-process-port", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    result = train(args)
    print(json.dumps({"command": result["command"], "warnings": result["config"]["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
