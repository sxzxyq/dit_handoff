#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${DIT_CACHE_ROOT:-/data/shared_folder/datasets/dit/cache}"
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

ISAACLAB_PYTHON="${DIT_ISAACLAB_PYTHON:-/home/qsh/miniconda3/envs/env_isaaclab/bin/python}"
export PYTHONPATH="/home/qsh/dit/src:/home/qsh/IsaacLab/source:${PYTHONPATH:-}"
"${ISAACLAB_PYTHON}" -m dit_handoff.eval.closed_loop_handoff_state26_absjoint18_handoff_state26_relee_pose14 "$@"
