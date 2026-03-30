#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
if [[ -n "${OPCDIMAGE_PROXY:-}" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-${OPCDIMAGE_PROXY}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${OPCDIMAGE_PROXY}}"
  export ALL_PROXY="${ALL_PROXY:-${OPCDIMAGE_PROXY}}"
fi

DATA_DIR="${PROJECT_DIR}/data/opcdimage_qwen3vl4b"
SOURCE_CSV="${PROJECT_DIR}/../ZwZ-RL-VQA-mini/train_crop_clean.csv"
SOURCE_ROOT="${PROJECT_DIR}/../ZwZ-RL-VQA-mini"
HF_DATASET_REPO_ID="${OPCDIMAGE_HF_DATASET_REPO_ID:-muyuho/opcdmini}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"

if [[ -f "${TRAIN_FILE}" && -f "${VAL_FILE}" ]]; then
  echo "Prepared dataset already exists at ${DATA_DIR}"
else
  if [[ -f "${SOURCE_CSV}" && -d "${SOURCE_ROOT}" ]]; then
    python3 "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" prepare \
      --input "${SOURCE_CSV}" \
      --dataset-root "${SOURCE_ROOT}" \
      --output-dir "${DATA_DIR}" \
      "$@"
  else
    python3 \
      "${PROJECT_DIR}/opcdimage_recipe/hf_data_tools.py" download \
      --output-dir "${DATA_DIR}" \
      --repo-id "${HF_DATASET_REPO_ID}" \
      "$@"
  fi
fi

python3 "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" validate \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}"
