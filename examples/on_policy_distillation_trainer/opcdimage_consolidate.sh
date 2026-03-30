#!/usr/bin/env bash
set -xeuo pipefail

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

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
REF_MODEL_PATH=""
TRAIN_BATCH_SIZE=64
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=512
ACTOR_LR=1e-6
TOTAL_EPOCHS=10
TOTAL_TRAINING_STEPS=""
SAVE_FREQ=100
TEST_FREQ=10
N_GPUS_PER_NODE=4
NNODES=1
ROLLOUT_NAME="vllm"
EXP_NAME="opcdimage-qwen3-vl-4b-consolidate"
PROJECT_NAME="opcdimage_qwen3vl4b"
DATA_DIR="${PROJECT_DIR}/data/opcdimage_qwen3vl4b"
HF_DATASET_REPO_ID="muyuho/opcdimage_mini"
SOURCE_CSV="${PROJECT_DIR}/../ZwZ-RL-VQA-mini/train_crop_clean.csv"
SOURCE_ROOT="${PROJECT_DIR}/../ZwZ-RL-VQA-mini"
KL_LOSS_TYPE="full"
KL_TOPK=256
KL_RENORM_TOPK=False
ENFORCE_EAGER=False

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --ref_model_path) REF_MODEL_PATH="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --project_name) PROJECT_NAME="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
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
    --enforce_eager) ENFORCE_EAGER="$2"; shift 2 ;;
    *) break ;;
  esac
done

REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_PATH}

TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/val.parquet"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  if [[ -f "${SOURCE_CSV}" && -d "${SOURCE_ROOT}" ]]; then
    python3 \
      "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" prepare \
      --input "${SOURCE_CSV}" \
      --dataset-root "${SOURCE_ROOT}" \
      --output-dir "${DATA_DIR}"
  else
    python3 \
      "${PROJECT_DIR}/opcdimage_recipe/hf_data_tools.py" download \
      --output-dir "${DATA_DIR}" \
      --repo-id "${HF_DATASET_REPO_ID}"
  fi
fi

python3 \
  "${PROJECT_DIR}/opcdimage_recipe/data_tools.py" validate \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}"

MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
MICRO_BATCH_SIZE_PER_GPU=1
MAX_TOKEN_LEN_PER_GPU=$(( MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) ))

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login "${WANDB_API_KEY}"
fi

TRAINING_LENGTH_ARGS=(
  trainer.total_epochs=${TOTAL_EPOCHS}
)
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  TRAINING_LENGTH_ARGS=(
    trainer.total_epochs=1000000000
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
  )
fi

python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name='ppo_trainer.yaml' \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE}']" \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=False \
  data.prompt_key=problem \
  data.image_key=original_images \
  data.custom_cls.path="${PROJECT_DIR}/opcdimage_recipe/paired_vqa_dataset.py" \
  data.custom_cls.name=OPCDImagePairedVQADataset \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.ref_model_path="${REF_MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.actor.use_torch_compile=True \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
  actor_rollout_ref.actor.kl_topk=${KL_TOPK} \
  actor_rollout_ref.actor.kl_renorm_topk=${KL_RENORM_TOPK} \
  actor_rollout_ref.actor.profile_kl=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS} \
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_TOKENS} \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
  reward.custom_reward_function.path="${PROJECT_DIR}/opcdimage_recipe/reward_fn.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.stage=consolidate \
  trainer.privileged_mode=crop \
  trainer.on_policy_merge=True \
  trainer.generate_off_policy=False \
  trainer.experience_path='' \
  trainer.val_before_train=False \
  trainer.test_freq=${TEST_FREQ} \
  trainer.logger="['console','wandb']" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.use_legacy_worker_impl=enable \
  trainer.resume_mode=disable \
  "${TRAINING_LENGTH_ARGS[@]}" \
  "$@"
