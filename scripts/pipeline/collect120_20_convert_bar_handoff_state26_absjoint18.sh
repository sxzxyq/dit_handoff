#!/usr/bin/env bash
set -euo pipefail

TRAIN_EPISODES="${DIT_TRAIN_EPISODES:-120}"
VAL_EPISODES="${DIT_VAL_EPISODES:-20}"
EPISODES=$(( TRAIN_EPISODES + VAL_EPISODES ))
STAMP="${DIT_RUN_STAMP:-$(date +%m%d_%H%M%S)}"
RAW_NAME="${DIT_RAW_NAME:-bar_handoff_jointpos_scripted_train${TRAIN_EPISODES}_val${VAL_EPISODES}_${STAMP}}"
RAW_DIR="/data/shared_folder/datasets/dit/raw/${RAW_NAME}"
LEROBOT_DIR="${DIT_LEROBOT_DIR:-/data/shared_folder/datasets/dit/lerobot/${RAW_NAME}_bar_handoff_state26_absjoint18}"
COLLECT_CONFIG="${DIT_COLLECT_CONFIG:-/home/qsh/dit/configs/collect/raw_bar_handoff_state26_absjoint18_default.json}"
COLLECT_GPUS="${DIT_COLLECT_GPUS:-0,1,2,3}"
MAX_ATTEMPTS="${DIT_MAX_ATTEMPTS:-0}"
MAX_STEPS="${DIT_MAX_STEPS:-4200}"
PROGRESS_LOG_EVERY="${DIT_PROGRESS_LOG_EVERY:-100}"
MERGE_LINK_MODE="${DIT_MERGE_LINK_MODE:-symlink}"
TRAIN_SPLIT="${DIT_TRAIN_SPLIT:-$(python3 - <<PY
train=${TRAIN_EPISODES}
total=${EPISODES}
print(train / total)
PY
)}"
LOG_DIR="/data/shared_folder/datasets/dit/logs/${RAW_NAME}"

export PYTHONPATH="/home/qsh/dit/src${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -r -a COLLECT_GPU_ARRAY <<< "${COLLECT_GPUS}"
if (( ${#COLLECT_GPU_ARRAY[@]} == 0 )); then
  echo "[pipeline-bar] DIT_COLLECT_GPUS is empty" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[pipeline-bar] raw=${RAW_DIR}"
echo "[pipeline-bar] lerobot=${LEROBOT_DIR}"
echo "[pipeline-bar] episodes=${EPISODES} train=${TRAIN_EPISODES} val=${VAL_EPISODES} train_split=${TRAIN_SPLIT}"
echo "[pipeline-bar] collect_gpus=${COLLECT_GPUS}"
echo "[pipeline-bar] logs=${LOG_DIR}"

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
  echo "[pipeline-bar] start shard ${i}: gpu=${GPU_ID} episodes=${COUNT} raw=${SHARD_DIR} log=${SHARD_LOG}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    bash /home/qsh/dit/scripts/collect/collect_raw_bar_handoff_state26_absjoint18.sh \
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
  echo "[pipeline-bar] no active shards; EPISODES=${EPISODES}" >&2
  exit 1
fi

STATUS=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    GPU_ID="${SHARD_GPU_IDS[$i]}"
    SHARD_INDEX="${SHARD_INDEXES[$i]}"
    SHARD_LOG="${LOG_DIR}/collect_shard$(printf '%02d' "${SHARD_INDEX}")_gpu${GPU_ID}.log"
    echo "[pipeline-bar] shard ${SHARD_INDEX} failed; tail of ${SHARD_LOG}:" >&2
    tail -n 80 "${SHARD_LOG}" >&2 || true
    STATUS=1
  fi
done
if (( STATUS != 0 )); then
  exit "${STATUS}"
fi

echo "[pipeline-bar] all ${ACTIVE_SHARDS} collection shards finished; merging raw shards"
/home/qsh/miniconda3/envs/dit_lerobot_main/bin/python -m dit_handoff.data.merge_raw_shards \
  --dataset-name "${RAW_NAME}" \
  --output-dir "${RAW_DIR}" \
  --link-mode "${MERGE_LINK_MODE}" \
  "${SHARD_DIRS[@]}"

/home/qsh/miniconda3/envs/dit_lerobot_main/bin/python - "${RAW_DIR}" "${EPISODES}" <<'__DIT_SUCCESS_CHECK__'
import json
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
metas = sorted((raw_dir / "episodes").glob("episode_*/episode_meta.json"))

def success_bool(value):
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)

success = 0
failed = []
for meta_path in metas:
    meta = json.loads(meta_path.read_text())
    if success_bool(meta.get("episode_success")):
        success += 1
    else:
        failed.append(meta_path.parent.name)
print(f"[pipeline-bar] success episodes: {success}/{expected}")
if len(metas) != expected or success != expected:
    raise SystemExit(
        f"expected {expected} successful episodes, got {success}/{len(metas)}; "
        f"failed={failed[:20]}"
    )
__DIT_SUCCESS_CHECK__

bash /home/qsh/dit/scripts/convert/convert_bar_handoff_state26_absjoint18.sh \
  "${RAW_DIR}" \
  --output-dir "${LEROBOT_DIR}" \
  --repo-id "local/${RAW_NAME}_bar_handoff_state26_absjoint18" \
  --train-split "${TRAIN_SPLIT}"

echo "[pipeline-bar] done"
echo "[pipeline-bar] raw=${RAW_DIR}"
echo "[pipeline-bar] lerobot=${LEROBOT_DIR}"

