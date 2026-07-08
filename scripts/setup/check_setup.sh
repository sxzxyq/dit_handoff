#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DIT_DATA_ROOT:-${REPO_ROOT}/datasets/dit}"
CACHE_ROOT="${DIT_CACHE_ROOT:-${DATA_ROOT}/cache}"
mkdir -p "${CACHE_ROOT}/huggingface/hub"   "${CACHE_ROOT}/huggingface/transformers"   "${CACHE_ROOT}/torch"   "${CACHE_ROOT}/xdg"   "${CACHE_ROOT}/wandb"   "${CACHE_ROOT}/wandb_config"   "${CACHE_ROOT}/pip"   "${CACHE_ROOT}/conda_pkgs"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_ROOT}/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/huggingface/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${CACHE_ROOT}/wandb_config}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CACHE_ROOT}/conda_pkgs}"


LEROBOT_ENV="${DIT_LEROBOT_ENV:-dit_lerobot_main}"
CONDA_ROOT="${DIT_CONDA_ROOT:-/home/ubuntu/miniconda3}"
ISAACLAB_ROOT="${DIT_ISAACLAB_ROOT:-/home/ubuntu/Workspace/IsaacLab}"
LEROBOT_PYTHON="${DIT_LEROBOT_PYTHON:-${CONDA_ROOT}/envs/${LEROBOT_ENV}/bin/python}"
ISAACLAB_PYTHON="${DIT_ISAACLAB_PYTHON:-${CONDA_ROOT}/envs/env_isaaclab/bin/python}"

export DIT_WORKSPACE_ROOT="${DIT_WORKSPACE_ROOT:-${REPO_ROOT}}"
export DIT_DATA_ROOT="${DATA_ROOT}"
export DIT_ISAACLAB_ROOT="${ISAACLAB_ROOT}"
export DIT_CONDA_ROOT="${CONDA_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${ISAACLAB_ROOT}/source:${PYTHONPATH:-}"

RUNNER_PYTHON="${LEROBOT_PYTHON}"
NEXT_IS_MODE=0
for arg in "$@"; do
  if (( NEXT_IS_MODE )); then
    if [[ "${arg}" == "isaaclab" ]]; then
      RUNNER_PYTHON="${ISAACLAB_PYTHON}"
    elif [[ "${arg}" == "lerobot" ]]; then
      RUNNER_PYTHON="${LEROBOT_PYTHON}"
    fi
    NEXT_IS_MODE=0
  elif [[ "${arg}" == "--mode" ]]; then
    NEXT_IS_MODE=1
  elif [[ "${arg}" == "--mode=isaaclab" ]]; then
    RUNNER_PYTHON="${ISAACLAB_PYTHON}"
  elif [[ "${arg}" == "--mode=lerobot" ]]; then
    RUNNER_PYTHON="${LEROBOT_PYTHON}"
  fi
done

exec "${RUNNER_PYTHON}" -m dit_handoff.utils.setup_check \
  --lerobot-python "${LEROBOT_PYTHON}" \
  --isaaclab-python "${ISAACLAB_PYTHON}" \
  "$@"
