#!/usr/bin/env bash
set -xeuo pipefail

# 用法说明：这个脚本是 opcdimage 当前主训练入口。
# 它负责三件事：
# 1. 解析实验参数
# 2. 自动准备 / 校验 prepared dataset
# 3. 启动 consolidate 训练
usage() {
  cat <<'EOF'
Usage:
  bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh [options] [hydra_overrides...]

Common options:
  --model PATH
  --ref_model_path PATH
  --exp_name NAME
  --project_name NAME
  --data_dir DIR
  --hf_dataset_repo_id REPO
  --train_batch_size N
  --max_prompt_length N
  --max_response_length N
  --actor_lr FLOAT
  --total_epochs N
  --total_training_steps N
  --save_freq N
  --test_freq N
  --n_gpus_per_node N
  --nnodes N
  --kl_loss_type TYPE
  --kl_topk N
  --kl_renorm_topk BOOL
  --use_fused_kernels BOOL
  --enforce_eager BOOL
  --processor_max_pixels N
  --processor_max_image_tokens N
  -h, --help

Examples:
  bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh

  bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
    --exp_name opcdimage-smoke \
    --train_batch_size 4 \
    --max_prompt_length 1024 \
    --max_response_length 128 \
    --total_training_steps 20 \
    trainer.logger="['console']"
EOF
}

setup_runtime_env() {
  # 这里是训练运行时的公共环境变量，尽量保持轻量和稳定。
  export NCCL_TIMEOUT=36000
  export TOKENIZERS_PARALLELISM=true
  export WANDB_INIT_TIMEOUT=600
  export WANDB_RESUME=never
  export HYDRA_FULL_ERROR=1

  if [[ -n "${OPCDIMAGE_PROXY:-}" ]]; then
    export HTTP_PROXY="${HTTP_PROXY:-${OPCDIMAGE_PROXY}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${OPCDIMAGE_PROXY}}"
    export ALL_PROXY="${ALL_PROXY:-${OPCDIMAGE_PROXY}}"
  fi
}

dataset_dir_from_repo_id() {
  local repo_id="$1"
  printf '%s/data/%s\n' "${PROJECT_DIR}" "${repo_id}"
}

init_defaults() {
  PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
  export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

  # Model and experiment identity.
  # MODEL_PATH: student / actor 使用的模型。
  # REF_MODEL_PATH: privileged reference 使用的模型；默认回退到 student 同模型。
  MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
  REF_MODEL_PATH=""
  EXP_NAME="opcdimage-qwen3-vl-4b-consolidate"
  PROJECT_NAME="opcdimage_qwen3vl4b"

  # Data locations.
  # DATA_DIR: 下载后的 HF 数据集目录。
  # 训练会直接读取这个目录下的 prepared/*.parquet，而不是额外拷贝出的文件。
  # HF_DATASET_REPO_ID: 当前统一数据集 repo，包含 prepared/* 和图像压缩包。
  HF_DATASET_REPO_ID="muyuho/opcdmini"
  DATA_DIR="$(dataset_dir_from_repo_id "${HF_DATASET_REPO_ID}")"
  DATA_DIR_EXPLICITLY_SET=0

  # Training scale.
  # TRAIN_BATCH_SIZE: 全局 train batch size。
  # MAX_PROMPT_LENGTH: prompt 长度上限，直接影响显存和保留上下文长度。
  # MAX_RESPONSE_LENGTH: response 长度上限，直接影响 rollout 开销。
  # ACTOR_LR: actor 学习率，是最值得优先扫的超参之一。
  # TOTAL_TRAINING_STEPS: 如果设置，则优先按 step 停止，适合做对比实验。
  TRAIN_BATCH_SIZE=64
  MAX_PROMPT_LENGTH=10000
  MAX_RESPONSE_LENGTH=512
  ACTOR_LR=1e-6
  TOTAL_EPOCHS=10
  TOTAL_TRAINING_STEPS=100
  SAVE_FREQ=100
  TEST_FREQ=10

  # Distributed settings.
  # N_GPUS_PER_NODE / NNODES: 分布式训练规模。
  N_GPUS_PER_NODE=1
  NNODES=1

  # KL / rollout settings.
  # 这里控制 privileged distillation 的关键形式：
  # - ROLLOUT_NAME: 当前默认 vllm
  # - KL_LOSS_TYPE: 默认 full KL
  # - KL_TOPK: full KL 的 top-k 截断
  # - ENFORCE_EAGER: rollout 是否强制 eager，常用于排查问题
  ROLLOUT_NAME="vllm"
  KL_LOSS_TYPE="full"
  KL_TOPK=256
  KL_RENORM_TOPK=True
  USE_FUSED_KERNELS=False
  ENFORCE_EAGER=False

  # Processor / vision budget settings.
  # PROCESSOR_MAX_IMAGE_TOKENS: 视觉 token 预算上限，脚本会自动换算成 max_pixels。
  # PROCESSOR_MAX_PIXELS: 如果你想直接按像素上限控制，可以显式覆盖它。
  # 注意：这里约束的是图像侧 token，不是“文本+图像”的总 prompt token。
  PROCESSOR_MAX_IMAGE_TOKENS=10000
  PROCESSOR_MAX_PIXELS=""

  # Extra Hydra overrides passed through to main_ppo.
  HYDRA_ARGS=()
}

parse_args() {
  # 已知参数在这里消费；其余参数原样透传给 Hydra。
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL_PATH="$2"; shift 2 ;;
      --ref_model_path) REF_MODEL_PATH="$2"; shift 2 ;;
      --exp_name) EXP_NAME="$2"; shift 2 ;;
      --project_name) PROJECT_NAME="$2"; shift 2 ;;
      --data_dir) DATA_DIR="$2"; DATA_DIR_EXPLICITLY_SET=1; shift 2 ;;
      --hf_dataset_repo_id) HF_DATASET_REPO_ID="$2"; shift 2 ;;
      --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
      --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
      --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
      --actor_lr) ACTOR_LR="$2"; shift 2 ;;
      --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
      --total_training_steps) TOTAL_TRAINING_STEPS="$2"; shift 2 ;;
      --save_freq) SAVE_FREQ="$2"; shift 2 ;;
      --test_freq) TEST_FREQ="$2"; shift 2 ;;
      --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2 ;;
      --nnodes) NNODES="$2"; shift 2 ;;
      --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
      --kl_topk) KL_TOPK="$2"; shift 2 ;;
      --kl_renorm_topk) KL_RENORM_TOPK="$2"; shift 2 ;;
      --use_fused_kernels) USE_FUSED_KERNELS="$2"; shift 2 ;;
      --enforce_eager) ENFORCE_EAGER="$2"; shift 2 ;;
      --processor_max_pixels) PROCESSOR_MAX_PIXELS="$2"; shift 2 ;;
      --processor_max_image_tokens) PROCESSOR_MAX_IMAGE_TOKENS="$2"; shift 2 ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        HYDRA_ARGS+=("$@")
        break
        ;;
      *)
        HYDRA_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

finalize_config() {
  if [[ "${DATA_DIR_EXPLICITLY_SET}" -eq 0 ]]; then
    DATA_DIR="$(dataset_dir_from_repo_id "${HF_DATASET_REPO_ID}")"
  fi

  REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_PATH}

  if [[ "${KL_LOSS_TYPE}" == "full" && "${KL_TOPK}" -gt 0 && "${USE_FUSED_KERNELS}" == "True" ]]; then
    echo "opcdimage reverse-KL recipe does not support use_fused_kernels=True with full top-k KL." >&2
    exit 1
  fi

  TRAIN_FILE="${DATA_DIR}/prepared/train.parquet"
  VAL_FILE="${DATA_DIR}/prepared/val.parquet"

  # 当前脚本把 actor / log_prob 的 micro batch 固定为 1，
  # 再通过 token 上限控制单卡负载，优先保证训练能稳定起。
  MICRO_BATCH_SIZE_PER_GPU=1
  MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
  MAX_TOKEN_LEN_PER_GPU=$(( MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) ))

  # 两种训练长度控制方式：
  # 1. 默认按 total_epochs
  # 2. 如果显式传 total_training_steps，则改为按 step 停止
  TRAINING_LENGTH_ARGS=(
    "trainer.total_epochs=${TOTAL_EPOCHS}"
  )
  if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    TRAINING_LENGTH_ARGS=(
      "trainer.total_epochs=1000000000"
      "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    )
  fi

  # 给 hf_processor 统一入口传视觉分辨率预算。
  # 优先级：
  # 1. 显式给 --processor_max_pixels
  # 2. 否则按 --processor_max_image_tokens 推导
  export VERL_PROCESSOR_MAX_IMAGE_TOKENS="${PROCESSOR_MAX_IMAGE_TOKENS}"
  if [[ -n "${PROCESSOR_MAX_PIXELS}" ]]; then
    export VERL_PROCESSOR_MAX_PIXELS="${PROCESSOR_MAX_PIXELS}"
  else
    unset VERL_PROCESSOR_MAX_PIXELS 2>/dev/null || true
  fi
}

prepare_dataset_if_needed() {
  # 只保留一条数据准备路径：
  # 从 opcdmini 下载并自动解压图像压缩包。
  if [[ -f "${TRAIN_FILE}" && -f "${VAL_FILE}" && -f "${DATA_DIR}/.hf_dataset_source.json" ]]; then
    return
  fi

  python3 \
    "${PROJECT_DIR}/opcdimage_recipe/hf_data_tools.py" download \
    --output-dir "${DATA_DIR}" \
    --repo-id "${HF_DATASET_REPO_ID}"
}

validate_dataset() {
  # 这里的校验很重要，会检查：
  # - 图像路径是否存在
  # - crop / original 是否对齐
  # - reward_model.ground_truth 是否和 answer 一致
  # - train / val 是否发生原图泄漏
  python3 \
    "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" validate \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}"
}

maybe_login_wandb() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "${WANDB_API_KEY}"
  fi
}

launch_training() {
  # 这里把 Hydra 参数按 Data / Model / Actor / Rollout / Reward / Trainer 分块，
  # 后续做实验时优先在对应块里改，避免在整条长命令里来回找。
  local -a cmd=(
    python3 -m opcdimage_recipe.main_ppo
    --config-name=ppo_trainer.yaml

    # Data.
    # data.custom_cls 指向自定义多模态数据集类；
    # data.image_key=original_images 表示 student rollout 看原图。
    "data.train_files=['${TRAIN_FILE}']"
    "data.val_files=['${VAL_FILE}']"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.shuffle=False"
    "data.prompt_key=problem"
    "data.image_key=original_images"
    "data.custom_cls.path=${PROJECT_DIR}/opcdimage_recipe/paired_vqa_dataset.py"
    "data.custom_cls.name=OPCDImagePairedVQADataset"

    # Model.
    # ref_model_path 是 privileged reference；
    # 当前默认 student / ref 同模型起步。
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.ref.model.path=${REF_MODEL_PATH}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}"

    # Actor.
    # actor.use_kl_loss=True 表示当前训练核心就是 KL 对齐；
    # ppo_mini_batch_size 默认跟全局 train_batch_size 对齐。
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.use_torch_compile=True"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
    "actor_rollout_ref.actor.kl_topk=${KL_TOPK}"
    "actor_rollout_ref.actor.kl_renorm_topk=${KL_RENORM_TOPK}"
    "actor_rollout_ref.actor.profile_kl=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"

    # Rollout / log-prob.
    # rollout.n=1 表示每个 prompt 默认只采样 1 条 response；
    # gpu_memory_utilization 可以作为显存紧张时的第一批调节项之一。
    "actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.8"
    "actor_rollout_ref.rollout.calculate_log_probs=False"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False"
    "actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_TOKENS}"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}"

    # Reward.
    # compute_score 是当前图像多选题 reward：
    # 预测选项 == ground_truth 记 1，否则记 0。
    "reward.custom_reward_function.path=${PROJECT_DIR}/opcdimage_recipe/reward_fn.py"
    "reward.custom_reward_function.name=compute_score"

    # Recipe-specific options.
    # 这些配置只对 opcdimage recipe 生效，不再污染通用 trainer config。
    # student 在原图上生成，ref 在 crop 条件下给同一条 response 打分。
    "++opcdimage.privileged_mode=crop"
    "++opcdimage.on_policy_merge=True"

    # Trainer.
    "trainer.val_before_train=False"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.use_legacy_worker_impl=enable"
    "trainer.resume_mode=disable"
  )

  cmd+=("${TRAINING_LENGTH_ARGS[@]}")
  cmd+=("${HYDRA_ARGS[@]}")

  "${cmd[@]}"
}

main() {
  setup_runtime_env
  init_defaults
  parse_args "$@"
  finalize_config
  prepare_dataset_if_needed
  validate_dataset
  maybe_login_wandb
  launch_training
}

main "$@"
