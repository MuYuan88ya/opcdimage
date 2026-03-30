#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/opcdimage_consolidate.sh" \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --ref_model_path "Qwen/Qwen3-VL-4B-Instruct" \
  "$@"
