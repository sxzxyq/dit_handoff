"""Open-loop state26 -> absjoint18 action regression evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dit_handoff.constants import ACTION_DIM, REPORT_ROOT
from dit_handoff.convert.handoff_state26_absjoint18 import action18_from_raw_step, state26_from_raw_step
from dit_handoff.utils.io import ensure_dir, iter_jsonl, read_json, write_json


def _mae_values(a, b) -> float:
    return float((a - b).abs().mean().item())


def _mse_values(a, b) -> float:
    diff = a - b
    return float((diff * diff).mean().item())


def _mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def _mse(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a)


def _episode_indices_from_manifest(manifest: dict[str, Any]) -> list[int] | None:
    split = manifest.get("train_val_split", {})
    episode_names = split.get("val_episodes") or split.get("train_episodes") or []
    indices = []
    for name in episode_names:
        try:
            indices.append(int(str(name).rsplit("_", 1)[1]))
        except Exception:
            return None
    return indices or None


def _resolve_policy_dir(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    if (checkpoint / "model.safetensors").exists() and (checkpoint / "train_config.json").exists():
        return checkpoint
    candidate = checkpoint / "pretrained_model"
    if (candidate / "model.safetensors").exists() and (candidate / "train_config.json").exists():
        return candidate
    raise FileNotFoundError(f"cannot find LeRobot pretrained_model under {checkpoint}")


def _build_official_eval_stack(dataset_dir: Path, checkpoint: Path, device: str, manifest: dict[str, Any]):
    try:
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets import make_dataset
        from lerobot.policies import make_policy, make_pre_post_processors
    except Exception as exc:
        raise RuntimeError("LeRobot 0.5.x policy/dataset APIs unavailable; run setup in the LeRobot env") from exc

    policy_dir = _resolve_policy_dir(checkpoint)
    cfg = TrainPipelineConfig.from_pretrained(str(policy_dir / "train_config.json"))
    cfg.dataset.root = str(dataset_dir)
    cfg.dataset.repo_id = manifest.get("repo_id") or cfg.dataset.repo_id
    cfg.dataset.episodes = _episode_indices_from_manifest(manifest)
    cfg.policy.pretrained_path = policy_dir
    cfg.policy.device = device

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=policy_dir,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return cfg, dataset, policy, preprocessor, postprocessor, policy_dir


def _sample_to_batch(sample: dict[str, Any], camera_keys: list[str]) -> tuple[dict[str, Any], Any, Any]:
    import torch

    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if key.startswith("observation.") or key == "action" or key.endswith("_is_pad"):
                batch[key] = value.unsqueeze(0)
        elif key == "task":
            batch[key] = [value]
    for camera_key in camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0
    target_action = batch["action"].detach().clone().cpu()
    action_is_pad = batch.get("action_is_pad")
    action_is_pad = action_is_pad.detach().clone().cpu() if action_is_pad is not None else None
    return batch, target_action, action_is_pad


def _predict_chunk(policy: Any, preprocessor: Any, postprocessor: Any, batch: dict[str, Any]):
    import torch

    batch = preprocessor(batch)
    with torch.no_grad():
        prepared = policy._prepare_batch(batch)
        normalized_chunk = policy._generate_actions(prepared)
        chunk = postprocessor(normalized_chunk)
    return chunk.detach().cpu()


def _evaluate_checkpoint(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    cfg, dataset, policy, preprocessor, postprocessor, policy_dir = _build_official_eval_stack(
        dataset_dir, Path(args.checkpoint), args.device, manifest
    )
    start = int(cfg.policy.n_obs_steps) - 1
    end = start + int(cfg.policy.n_action_steps)

    rows = []
    chunk_mae_sum = 0.0
    chunk_mse_sum = 0.0
    first_mae_sum = 0.0
    first_mse_sum = 0.0
    count = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        batch, target_action, action_is_pad = _sample_to_batch(sample, dataset.meta.camera_keys)
        target_chunk = target_action[:, start:end]
        pred_chunk = _predict_chunk(policy, preprocessor, postprocessor, batch)
        valid = None
        if action_is_pad is not None:
            valid = ~action_is_pad[:, start:end].bool()
            if not bool(valid.any()):
                continue
        if valid is not None:
            pred_eval = pred_chunk[valid]
            target_eval = target_chunk[valid]
        else:
            pred_eval = pred_chunk.reshape(-1, ACTION_DIM)
            target_eval = target_chunk.reshape(-1, ACTION_DIM)
        first_pred = pred_chunk[:, 0]
        first_target = target_chunk[:, 0]
        row = {
            "dataset_index": int(index),
            "episode_index": int(sample.get("episode_index", -1)),
            "frame_index": int(sample.get("frame_index", -1)),
            "first_action_mae": _mae_values(first_pred, first_target),
            "first_action_mse": _mse_values(first_pred, first_target),
            "chunk_mae": _mae_values(pred_eval, target_eval),
            "chunk_mse": _mse_values(pred_eval, target_eval),
            "valid_chunk_steps": int(pred_eval.shape[0]),
        }
        rows.append(row)
        first_mae_sum += row["first_action_mae"]
        first_mse_sum += row["first_action_mse"]
        chunk_mae_sum += row["chunk_mae"]
        chunk_mse_sum += row["chunk_mse"]
        count += 1
        if args.max_samples and count >= args.max_samples:
            break

    summary = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(policy_dir),
        "target_only": False,
        "samples": count,
        "first_action_mae": first_mae_sum / count if count else None,
        "first_action_mse": first_mse_sum / count if count else None,
        "chunk_mae": chunk_mae_sum / count if count else None,
        "chunk_mse": chunk_mse_sum / count if count else None,
        "action_dim": ACTION_DIM,
        "n_obs_steps": int(cfg.policy.n_obs_steps),
        "horizon": int(cfg.policy.horizon),
        "n_action_steps": int(cfg.policy.n_action_steps),
        "eval_episodes": cfg.dataset.episodes,
        "action_chunk_dim_check": "LeRobot temporal dataset target[:, n_obs_steps-1:n_obs_steps-1+n_action_steps]",
    }
    out_dir = Path(args.output_dir) if args.output_dir else REPORT_ROOT / "open_loop_handoff_state26_absjoint18"
    ensure_dir(out_dir)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "per_sample.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return summary


def _evaluate_target_only(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    raw_dir = Path(manifest["raw_dataset_path"])
    split = manifest.get("train_val_split", {})
    episode_names = split.get("val_episodes") or split.get("train_episodes") or []
    if not episode_names:
        raise RuntimeError("dataset manifest has no episodes for open-loop eval")

    rows = []
    per_dim_abs = [0.0 for _ in range(ACTION_DIM)]
    count = 0
    for ep_name in episode_names:
        ep_dir = raw_dir / "episodes" / ep_name
        for row in iter_jsonl(ep_dir / "steps.jsonl"):
            target = action18_from_raw_step(row)
            state26_from_raw_step(row)
            pred = target
            rows.append({"episode": ep_name, "step_index": row["step_index"], "mae": _mae(pred, target), "mse": _mse(pred, target)})
            for i, (x, y) in enumerate(zip(pred, target, strict=True)):
                per_dim_abs[i] += abs(x - y)
            count += 1
            if args.max_samples and count >= args.max_samples:
                break
        if args.max_samples and count >= args.max_samples:
            break

    summary = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": None,
        "target_only": True,
        "samples": count,
        "mae": sum(row["mae"] for row in rows) / count if count else None,
        "mse": sum(row["mse"] for row in rows) / count if count else None,
        "per_dim_mae": [value / count for value in per_dim_abs] if count else [],
        "action_dim": ACTION_DIM,
        "action_chunk_dim_check": "single-step target self-comparison",
    }
    out_dir = Path(args.output_dir) if args.output_dir else REPORT_ROOT / "open_loop_handoff_state26_absjoint18"
    ensure_dir(out_dir)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "per_sample.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    manifest = read_json(dataset_dir / "manifest.json")
    if args.target_only:
        return _evaluate_target_only(args, manifest)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --target-only is set")
    return _evaluate_checkpoint(args, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open-loop eval for state26 -> absjoint18.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--target-only", action="store_true", help="Smoke mode: compare targets to themselves without loading a policy.")
    args = parser.parse_args(argv)
    summary = evaluate(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
