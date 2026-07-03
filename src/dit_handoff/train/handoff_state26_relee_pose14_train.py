"""Task-specific MultiTask DiT training for state26 -> anchored relative EE pose14."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dit_handoff.constants import (
    CAMERA_OBS_FEATURES,
    CHECKPOINT_ROOT,
    DEFAULT_HORIZON_SECONDS,
    DEFAULT_N_ACTION_SECONDS,
    DEFAULT_N_OBS_STEPS,
    LEROBOT_RELEE_POSE14_CONTRACT_VERSION,
    POSE14_ACTION_DIM,
    RUNS_ROOT,
    STATE_DIM,
)
from dit_handoff.train.common import _build_training_subprocess_env, _load_profile, _resolve_env_executable, _round_steps
from dit_handoff.utils.io import ensure_dir, read_json, write_json



def build_training_config(dataset_dir: Path, profile: dict[str, Any], *, run_name: str, smoke: bool) -> dict[str, Any]:
    manifest = read_json(dataset_dir / "manifest.json")
    if manifest.get("contract") != LEROBOT_RELEE_POSE14_CONTRACT_VERSION:
        raise ValueError(f"dataset contract must be {LEROBOT_RELEE_POSE14_CONTRACT_VERSION}")
    sidecar = manifest.get("action_chunk_sidecar") or {}
    sidecar_meta = read_json(dataset_dir / sidecar.get("metadata", "action_chunks/metadata.json"))
    fps = float(manifest["fps"])
    horizon_seconds = float(profile.get("horizon_seconds", DEFAULT_HORIZON_SECONDS))
    horizon = int(profile.get("horizon") or _round_steps(fps, horizon_seconds))
    n_obs_steps = int(profile.get("n_obs_steps", DEFAULT_N_OBS_STEPS))
    if int(sidecar_meta["horizon"]) != horizon:
        raise ValueError(f"sidecar horizon={sidecar_meta['horizon']} does not match training horizon={horizon}")
    if int(sidecar_meta["n_obs_steps"]) != n_obs_steps:
        raise ValueError(f"sidecar n_obs_steps={sidecar_meta['n_obs_steps']} does not match training n_obs_steps={n_obs_steps}")
    if "n_action_steps" in profile:
        n_action_steps = int(profile["n_action_steps"])
        n_action_seconds = float(n_action_steps) / fps
    else:
        n_action_seconds = float(profile.get("n_action_seconds", DEFAULT_N_ACTION_SECONDS))
        n_action_steps = _round_steps(fps, n_action_seconds)
    physical_batch_size = int(profile.get("physical_batch_size", profile.get("batch_size", 16)))
    grad_accum = int(profile.get("gradient_accumulation_steps", 1))
    steps = int(profile.get("smoke_steps" if smoke else "steps", 30 if smoke else 50000))
    output_dir = RUNS_ROOT / run_name
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    warnings = [
        "This task-specific launcher patches LeRobot make_train_eval_datasets to inject anchored pose14 chunks.",
        "Do not use plain lerobot-train for this contract; it will use default future-row action chunks.",
    ]
    if grad_accum > 1:
        warnings.append("Gradient accumulation uses DIT_GRADIENT_ACCUMULATION_STEPS in the local LeRobot train entrypoint")
    return {
        "contract": LEROBOT_RELEE_POSE14_CONTRACT_VERSION,
        "dataset_dir": str(dataset_dir),
        "dataset_repo_id": manifest.get("repo_id") or str(dataset_dir),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "policy_type": "multi_task_dit",
        "policy_repo_id": profile.get("policy_repo_id", f"local/{run_name}"),
        "policy_push_to_hub": bool(profile.get("policy_push_to_hub", False)),
        "save_checkpoint_to_hub": bool(profile.get("save_checkpoint_to_hub", False)),
        "fps": fps,
        "horizon": horizon,
        "horizon_seconds": horizon_seconds,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "n_action_seconds": n_action_seconds,
        "state_dim": STATE_DIM,
        "action_dim": POSE14_ACTION_DIM,
        "image_features": list(CAMERA_OBS_FEATURES),
        "action_representation": "anchored_relative_end_effector_pose_axis_angle",
        "objective": profile.get("objective", "diffusion"),
        "noise_scheduler_type": profile.get("noise_scheduler_type", "DDPM"),
        "num_train_timesteps": profile.get("num_train_timesteps", 100),
        "num_inference_steps": profile.get("num_inference_steps", 10),
        "use_amp": bool(profile.get("use_amp", False)),
        "mixed_precision": profile.get("mixed_precision"),
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": physical_batch_size * grad_accum,
        "learning_rate": profile.get("optimizer_lr", 2e-5),
        "vision_encoder_lr_multiplier": profile.get("vision_encoder_lr_multiplier", 0.1),
        "steps": steps,
        "num_workers": int(profile.get("num_workers", 4)),
        "prefetch_factor": int(profile.get("prefetch_factor", 4)),
        "persistent_workers": bool(profile.get("persistent_workers", True)),
        "save_checkpoint": bool(profile.get("save_checkpoint", True)),
        "save_freq": profile.get("save_freq", 1000),
        "log_freq": profile.get("log_freq", 100),
        "eval_freq": profile.get("eval_freq", 0),
        "eval_steps": profile.get("eval_steps", 0),
        "seed": profile.get("seed", 2000),
        "do_mask_loss_for_padding": bool(profile.get("do_mask_loss_for_padding", True)),
        "transform_config": {
            "image_crop_shape": profile.get("image_crop_shape", [224, 224]),
            "image_crop_is_random": profile.get("image_crop_is_random", True),
        },
        "model": {
            "num_layers": profile.get("num_layers", 6),
            "hidden_dim": profile.get("hidden_dim", 512),
            "num_heads": profile.get("num_heads", 8),
            "use_rope": profile.get("use_rope", True),
            "separate_rgb_encoder_per_camera": profile.get("separate_rgb_encoder_per_camera", False),
        },
        "action_chunk_sidecar": sidecar,
        "warnings": warnings,
    }


def _official_train_args(config: dict[str, Any], *, device: str, wandb: bool) -> list[str]:
    return [
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
        f"--policy.do_mask_loss_for_padding={str(config['do_mask_loss_for_padding']).lower()}",
        f"--policy.num_layers={config['model']['num_layers']}",
        f"--policy.hidden_dim={config['model']['hidden_dim']}",
        f"--policy.num_heads={config['model']['num_heads']}",
        f"--policy.use_rope={str(config['model']['use_rope']).lower()}",
        f"--policy.optimizer_lr={config['learning_rate']}",
        f"--policy.vision_encoder_lr_multiplier={config['vision_encoder_lr_multiplier']}",
        f"--policy.image_crop_shape={config['transform_config']['image_crop_shape']}",
        f"--policy.image_crop_is_random={str(config['transform_config']['image_crop_is_random']).lower()}",
        f"--wandb.enable={str(bool(wandb)).lower()}",
    ]


def build_command(config: dict[str, Any], *, device: str, wandb: bool) -> list[str]:
    python = sys.executable
    command = [python, "-m", "dit_handoff.train.handoff_state26_relee_pose14_train", "--inner"] + _official_train_args(
        config, device=device, wandb=wandb
    )
    num_processes = int(config.get("num_processes", 1))
    if num_processes > 1:
        accelerate = _resolve_env_executable("accelerate")
        if accelerate is None:
            raise RuntimeError("accelerate is required for multi-GPU training")
        training_script = str(Path(__file__).resolve())
        command = [training_script, "--inner"] + _official_train_args(config, device=device, wandb=wandb)
        launch = [accelerate, "launch", "--multi_gpu", f"--num_processes={num_processes}"]
        if config.get("mixed_precision"):
            launch.append(f"--mixed_precision={config['mixed_precision']}")
        if config.get("gpu_ids"):
            launch.append(f"--gpu_ids={config['gpu_ids']}")
        if config.get("main_process_port"):
            launch.append(f"--main_process_port={config['main_process_port']}")
        command = launch + command
    return command


def run_inner(official_args: list[str]) -> int:
    from dit_handoff.lerobot_bridge.anchored_relee_dataset import wrap_train_eval_datasets
    import lerobot.scripts.lerobot_train as official_train

    original_make_train_eval = official_train.make_train_eval_datasets

    def make_train_eval_with_pose14(cfg):
        train_dataset, eval_dataset = original_make_train_eval(cfg)
        return wrap_train_eval_datasets(train_dataset, eval_dataset, dataset_root=cfg.dataset.root)

    official_train.make_train_eval_datasets = make_train_eval_with_pose14
    sys.argv = [sys.argv[0], *official_args]
    official_train.main()
    return 0


def train(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    profile = _load_profile(args.profile)
    config = build_training_config(dataset_dir, profile, run_name=args.run_name, smoke=args.smoke)
    env, cache_env = _build_training_subprocess_env()
    env["PYTHONPATH"] = "/home/qsh/dit/src" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    config["cache_env"] = cache_env
    config["num_processes"] = int(args.num_processes)
    config["gpu_ids"] = args.gpu_ids
    config["main_process_port"] = args.main_process_port
    config["effective_batch_size"] = (
        config["physical_batch_size"] * config["num_processes"] * config["gradient_accumulation_steps"]
    )
    if config["gradient_accumulation_steps"] > 1:
        env["DIT_GRADIENT_ACCUMULATION_STEPS"] = str(config["gradient_accumulation_steps"])
    ensure_dir(Path(config["output_dir"]).parent)
    ensure_dir(Path(config["checkpoint_dir"]))
    command = build_command(config, device=args.device, wandb=args.wandb)
    if args.dry_run:
        config["warnings"].append("dry-run metadata is written to checkpoint_dir only so the real output_dir remains unused")
    preflight = {"command": command, "dry_run": args.dry_run}
    write_json(Path(config["checkpoint_dir"]) / "dit_checkpoint_config.json", config)
    write_json(Path(config["checkpoint_dir"]) / "lerobot_train_command.json", preflight)
    if args.dry_run:
        pass
    else:
        subprocess.run(command, check=True, env=env)
        ensure_dir(config["output_dir"])
        write_json(Path(config["output_dir"]) / "dit_train_config.json", config)
        write_json(Path(config["output_dir"]) / "lerobot_train_command.json", preflight)
    return {"config": config, "command": command}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train state26 -> anchored relative EE pose14 MultiTask DiT.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--profile", type=Path, default=Path("/home/qsh/dit/configs/train/handoff_state26_relee_pose14_2gpu_bs16_accum4_50k.json"))
    parser.add_argument("--run-name", default="handoff_state26_relee_pose14_mtdp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--main-process-port", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--inner":
        return run_inner(argv[1:])
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps({"command": result["command"], "warnings": result["config"]["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
