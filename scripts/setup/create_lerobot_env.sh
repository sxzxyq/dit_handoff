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


ENV_NAME="${DIT_LEROBOT_ENV:-dit_lerobot_main}"
PYTHON_VERSION="${DIT_LEROBOT_PYTHON_VERSION:-3.12}"
INSTALL_SOURCE="${DIT_LEROBOT_INSTALL_SOURCE:-1}"
CONDA_ROOT="${DIT_CONDA_ROOT:-/home/ubuntu/miniconda3}"
CONDA_EXE="${DIT_CONDA_EXE:-${CONDA_ROOT}/bin/conda}"

export DIT_WORKSPACE_ROOT="${DIT_WORKSPACE_ROOT:-${REPO_ROOT}}"
export DIT_DATA_ROOT="${DATA_ROOT}"
export DIT_CONDA_ROOT="${CONDA_ROOT}"

if "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists; reusing it."
else
  "${CONDA_EXE}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
if [[ "${INSTALL_SOURCE}" == "1" ]]; then
  "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install "lerobot[training,multi-task-dit] @ git+https://github.com/huggingface/lerobot.git"
else
  "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install "lerobot[multi_task_dit]"
fi

echo "Created ${ENV_NAME}. Activate with: conda activate ${ENV_NAME}"
