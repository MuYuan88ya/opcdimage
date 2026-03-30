#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${OPCDIMAGE_HF_IMAGE_DATASET_REPO_ID:-muyuho/opcdmini}"
LOCAL_DIR="${OPCDIMAGE_HF_LOCAL_DIR:-hf_dataset/opcdimage_mini/archives}"
MODE="${OPCDIMAGE_HF_UPLOAD_MODE:-large-folder}"
NUM_WORKERS="${OPCDIMAGE_HF_UPLOAD_WORKERS:-8}"
ONLY_CROP="${OPCDIMAGE_HF_ONLY_CROP:-false}"
SKIP_EXISTING="${OPCDIMAGE_HF_SKIP_EXISTING:-true}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi

if [[ -n "${OPCDIMAGE_PROXY:-}" ]]; then
  export HTTP_PROXY="${HTTP_PROXY:-$OPCDIMAGE_PROXY}"
  export HTTPS_PROXY="${HTTPS_PROXY:-$OPCDIMAGE_PROXY}"
  export ALL_PROXY="${ALL_PROXY:-$OPCDIMAGE_PROXY}"
fi

CMD=(
  uv run --no-project --with huggingface_hub
  python opcdimage_recipe/upload_hf_dataset.py
  --local-dir "$LOCAL_DIR"
  --repo-id "$REPO_ID"
  --exist-ok
  --mode "$MODE"
  --num-workers "$NUM_WORKERS"
)

if [[ "$SKIP_EXISTING" == "true" ]]; then
  CMD+=(--skip-existing)
fi

if [[ "$ONLY_CROP" == "true" ]]; then
  CMD+=(--allow-pattern "crop_images.tar.gz")
fi

"${CMD[@]}"
