#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/on_policy_distillation_trainer/run_qwen3_vl_4b_opcdimage.sh [options] [hydra_overrides...]

This is a thin wrapper around `opcdimage_consolidate.sh` that fixes:
  --model Qwen/Qwen3-VL-4B-Instruct
  --ref_model_path Qwen/Qwen3-VL-4B-Instruct

Everything else is forwarded as-is.
EOF
}

main() {
  local script_dir
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    echo
    exec "${script_dir}/opcdimage_consolidate.sh" --help
  fi

  exec "${script_dir}/opcdimage_consolidate.sh" \
    --model "Qwen/Qwen3-VL-4B-Instruct" \
    --ref_model_path "Qwen/Qwen3-VL-4B-Instruct" \
    "$@"
}

main "$@"
