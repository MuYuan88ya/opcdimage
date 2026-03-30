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
HF_DATASET_REPO_ID="${OPCDIMAGE_HF_DATASET_REPO_ID:-muyuho/opcdmini}"

python3 \
  "${PROJECT_DIR}/opcdimage_recipe/hf_data_tools.py" download \
  --output-dir "${DATA_DIR}" \
  --repo-id "${HF_DATASET_REPO_ID}" \
  "$@"

python3 "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" validate \
  --train-file "${DATA_DIR}/prepared/train.parquet" \
  --val-file "${DATA_DIR}/prepared/val.parquet"
