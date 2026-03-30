#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

dataset_dir_from_repo_id() {
  local repo_id="$1"
  printf '%s/data/%s\n' "${PROJECT_DIR}" "${repo_id}"
}

if [[ -n "${OPCDIMAGE_PROXY:-}" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-${OPCDIMAGE_PROXY}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${OPCDIMAGE_PROXY}}"
  export ALL_PROXY="${ALL_PROXY:-${OPCDIMAGE_PROXY}}"
fi

HF_DATASET_REPO_ID="${OPCDIMAGE_HF_DATASET_REPO_ID:-muyuho/opcdmini}"
DATA_DIR="$(dataset_dir_from_repo_id "${HF_DATASET_REPO_ID}")"

python3 \
  "${PROJECT_DIR}/opcdimage_recipe/hf_data_tools.py" download \
  --output-dir "${DATA_DIR}" \
  --repo-id "${HF_DATASET_REPO_ID}" \
  "$@"

python3 "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" validate \
  --train-file "${DATA_DIR}/prepared/train.parquet" \
  --val-file "${DATA_DIR}/prepared/val.parquet"
