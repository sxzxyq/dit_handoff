"""Shared helpers for task-specific LeRobot train launchers."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from dit_handoff.constants import CACHE_ROOT, WORKSPACE_ROOT
from dit_handoff.utils.io import ensure_dir, read_json


def _load_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = WORKSPACE_ROOT / "configs/train/handoff_state26_absjoint18_official_time_profile.json"
    return read_json(path)


def _round_steps(fps: float, seconds: float) -> int:
    return max(1, int(round(float(fps) * float(seconds))))


def _resolve_env_executable(name: str) -> str | None:
    env_local = Path(sys.executable).resolve().parent / name
    if env_local.is_file():
        return str(env_local)
    return shutil.which(name)


def _build_training_subprocess_env() -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    cache_defaults = {
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": CACHE_ROOT / "huggingface" / "transformers",
        "TORCH_HOME": CACHE_ROOT / "torch",
        "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
        "WANDB_DIR": CACHE_ROOT / "wandb",
        "WANDB_CACHE_DIR": CACHE_ROOT / "wandb",
        "WANDB_CONFIG_DIR": CACHE_ROOT / "wandb_config",
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "CONDA_PKGS_DIRS": CACHE_ROOT / "conda_pkgs",
    }
    for key, path in cache_defaults.items():
        ensure_dir(path)
        env.setdefault(key, str(path))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env, {key: env[key] for key in cache_defaults}
