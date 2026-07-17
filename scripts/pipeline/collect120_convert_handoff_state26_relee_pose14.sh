#!/usr/bin/env bash
set -euo pipefail

EPISODES="${DIT_EPISODES:-120}"
STAMP="${DIT_RUN_STAMP:-$(date +%m%d_%H%M%S)}"
RAW_NAME="${DIT_RAW_NAME:-handoff_relee_pose14_scripted_success${EPISODES}_${STAMP}}"
RAW_DIR="/data/shared_folder/datasets/dit/raw/${RAW_NAME}"
LEROBOT_DIR="${DIT_LEROBOT_DIR:-/data/shared_folder/datasets/dit/lerobot/${RAW_NAME}_state26_relee_pose14_h50_obs2}"
COLLECT_CONFIG="${DIT_COLLECT_CONFIG:-/home/qsh/dit/configs/collect/raw_handoff_state26_relee_pose14_default.json}"
COLLECT_GPUS="${DIT_COLLECT_GPUS:-0}"
MAX_ATTEMPTS="${DIT_MAX_ATTEMPTS:-0}"
MAX_STEPS="${DIT_MAX_STEPS:-2600}"
PROGRESS_LOG_EVERY="${DIT_PROGRESS_LOG_EVERY:-100}"
MERGE_LINK_MODE="${DIT_MERGE_LINK_MODE:-symlink}"
HORIZON="${DIT_HORIZON:-50}"
N_OBS_STEPS="${DIT_N_OBS_STEPS:-2}"
TRAIN_SPLIT="${DIT_TRAIN_SPLIT:-0.9}"
LOG_DIR="/data/shared_folder/datasets/dit/logs/${RAW_NAME}"
REPO_ID="${DIT_REPO_ID:-local/${RAW_NAME}_state26_relee_pose14_h${HORIZON}_obs${N_OBS_STEPS}}"

export PYTHONPATH="/home/qsh/dit/src${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a COLLECT_GPU_ARRAY <<< "${COLLECT_GPUS}"
if (( ${#COLLECT_GPU_ARRAY[@]} == 0 )); then
  echo "[pipeline-pose14] DIT_COLLECT_GPUS is empty" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[pipeline-pose14] raw=${RAW_DIR}"
echo "[pipeline-pose14] lerobot=${LEROBOT_DIR}"
echo "[pipeline-pose14] repo_id=${REPO_ID}"
echo "[pipeline-pose14] collect_gpus=${COLLECT_GPUS}"
echo "[pipeline-pose14] logs=${LOG_DIR}"

SHARD_DIRS=()
PIDS=()
SHARD_GPU_IDS=()
SHARD_INDEXES=()
ACTIVE_SHARDS=0
for i in "${!COLLECT_GPU_ARRAY[@]}"; do
  GPU_ID="${COLLECT_GPU_ARRAY[$i]}"
  COUNT=$(( EPISODES / ${#COLLECT_GPU_ARRAY[@]} ))
  if (( i < EPISODES % ${#COLLECT_GPU_ARRAY[@]} )); then
    COUNT=$(( COUNT + 1 ))
  fi
  if (( COUNT == 0 )); then
    continue
  fi
  SHARD_NAME="${RAW_NAME}_shard$(printf '%02d' "${i}")"
  SHARD_DIR="/data/shared_folder/datasets/dit/raw/${SHARD_NAME}"
  SHARD_LOG="${LOG_DIR}/collect_shard$(printf '%02d' "${i}")_gpu${GPU_ID}.log"
  SHARD_DIRS+=("${SHARD_DIR}")
  SHARD_GPU_IDS+=("${GPU_ID}")
  SHARD_INDEXES+=("${i}")
  ACTIVE_SHARDS=$(( ACTIVE_SHARDS + 1 ))
  echo "[pipeline-pose14] start shard ${i}: gpu=${GPU_ID} episodes=${COUNT} raw=${SHARD_DIR} log=${SHARD_LOG}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    bash /home/qsh/dit/scripts/collect/collect_raw_handoff_state26_relee_pose14.sh \
      --config "${COLLECT_CONFIG}" \
      --dataset-name "${SHARD_NAME}" \
      --episodes "${COUNT}" \
      --device cuda:0 \
      --seed "$(( 2000 + i * 100000 ))" \
      --require-success \
      --max-attempts "${MAX_ATTEMPTS}" \
      --max-steps "${MAX_STEPS}" \
      --progress-log-every "${PROGRESS_LOG_EVERY}" \
      --headless
  ) > "${SHARD_LOG}" 2>&1 &
  PIDS+=("$!")
done

if (( ACTIVE_SHARDS == 0 )); then
  echo "[pipeline-pose14] no active shards; DIT_EPISODES=${EPISODES}" >&2
  exit 1
fi

STATUS=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    GPU_ID="${SHARD_GPU_IDS[$i]}"
    SHARD_INDEX="${SHARD_INDEXES[$i]}"
    SHARD_LOG="${LOG_DIR}/collect_shard$(printf '%02d' "${SHARD_INDEX}")_gpu${GPU_ID}.log"
    echo "[pipeline-pose14] shard ${SHARD_INDEX} failed; tail of ${SHARD_LOG}:" >&2
    tail -n 80 "${SHARD_LOG}" >&2 || true
    STATUS=1
  fi
done
if (( STATUS != 0 )); then
  exit "${STATUS}"
fi

echo "[pipeline-pose14] all ${ACTIVE_SHARDS} collection shards finished; merging raw shards"
/home/qsh/miniconda3/envs/dit_lerobot_main/bin/python -m dit_handoff.data.merge_raw_shards \
  --dataset-name "${RAW_NAME}" \
  --output-dir "${RAW_DIR}" \
  --link-mode "${MERGE_LINK_MODE}" \
  "${SHARD_DIRS[@]}"

/home/qsh/miniconda3/envs/dit_lerobot_main/bin/python - "${RAW_DIR}" "${EPISODES}" <<'__DIT_POSE14_SUCCESS_CHECK__'
import json
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
manifest = json.loads((raw_dir / "dataset_manifest.json").read_text())
metas = sorted((raw_dir / "episodes").glob("episode_*/episode_meta.json"))

def success_bool(value):
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)

success = 0
failed = []
bad_action = []
for meta_path in metas:
    meta = json.loads(meta_path.read_text())
    if success_bool(meta.get("episode_success")):
        success += 1
    else:
        failed.append(meta_path.parent.name)
    steps_path = meta_path.parent / "steps.jsonl"
    first = json.loads(steps_path.read_text().splitlines()[0])
    action = first.get("action", {})
    if (
        len(action.get("raw_env_action", [])) != 14
        or len(action.get("pose14_delta_commanded", [])) != 14
        or action.get("pose14_delta_commanded_source") != "commanded_expert_action"
    ):
        bad_action.append(meta_path.parent.name)

print(f"[pipeline-pose14] success episodes: {success}/{expected}")
print(f"[pipeline-pose14] action_interface: {manifest.get('action_interface')}")
if manifest.get("action_interface") != "Dual Franka IK relative EE pose 14D":
    raise SystemExit(f"unexpected action_interface={manifest.get('action_interface')!r}")
if int(manifest.get("action_dim", -1)) != 14:
    raise SystemExit(f"unexpected action_dim={manifest.get('action_dim')!r}")
if len(metas) != expected or success != expected:
    raise SystemExit(
        f"expected {expected} successful episodes, got {success}/{len(metas)}; "
        f"failed={failed[:20]}"
    )
if bad_action:
    raise SystemExit(f"episodes missing commanded pose14 action fields: {bad_action[:20]}")
__DIT_POSE14_SUCCESS_CHECK__

bash /home/qsh/dit/scripts/convert/convert_handoff_state26_relee_pose14.sh \
  "${RAW_DIR}" \
  --output-dir "${LEROBOT_DIR}" \
  --repo-id "${REPO_ID}" \
  --train-split "${TRAIN_SPLIT}" \
  --horizon "${HORIZON}" \
  --n-obs-steps "${N_OBS_STEPS}"

echo "[pipeline-pose14] done"
echo "[pipeline-pose14] raw=${RAW_DIR}"
echo "[pipeline-pose14] lerobot=${LEROBOT_DIR}"
