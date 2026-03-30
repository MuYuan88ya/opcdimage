# opcdimage 训练启动说明

这份文档对应当前仓库中的 Vision-OPCD 训练配方，训练入口是：

- `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`

它会串起下面这条链路：

1. 检查 `data/opcdimage_qwen3vl4b/train.parquet` 和 `val.parquet` 是否存在
2. 如果不存在，优先尝试从本地 `../ZwZ-RL-VQA-mini` 生成 prepared dataset
3. 如果本地原始数据不存在，则从 Hugging Face dataset 下载最小可训练子集
4. 对 prepared dataset 执行 `validate`
5. 以 `trainer.stage=consolidate` 和 `trainer.privileged_mode=crop` 启动训练

## 1. 环境准备

建议在 Linux + CUDA 环境中执行，并先完成项目依赖安装。

最小要求：

- Python 3.10+
- 已安装仓库依赖
- 能访问模型权重与数据集
- 如果要记录实验，准备好 `WANDB_API_KEY`

示例：

```bash
cd /path/to/opcdimage
python -m pip install -U pip
python -m pip install -e .
python -m pip install pandas pyarrow pillow huggingface_hub wandb
```

如果网络需要代理，可以设置：

```bash
export OPCDIMAGE_PROXY=http://127.0.0.1:7890
```

## 2. 数据准备方式

### 方式 A：本地已有原始 ZwZ-RL-VQA-mini

脚本会自动查找：

- `../ZwZ-RL-VQA-mini/train_crop_clean.csv`
- `../ZwZ-RL-VQA-mini/`

也可以单独先准备数据：

```bash
bash examples/on_policy_distillation_trainer/prepare_opcdimage_data.sh
```

### 方式 B：本地没有原始数据，自动从 HF 下载

默认数据集仓库：

```bash
export OPCDIMAGE_HF_DATASET_REPO_ID=muyuho/opcdimage_mini
```

训练脚本在发现本地 parquet 不存在时，会自动调用：

```bash
python opcdimage_recipe/hf_data_tools.py download --output-dir data/opcdimage_qwen3vl4b
```

## 3. 如何启动训练

最简启动：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

使用包装脚本固定 Qwen3-VL-4B：

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_vl_4b_opcdimage.sh
```

一个更常见的自定义示例：

```bash
WANDB_API_KEY=xxx \
OPCDIMAGE_HF_DATASET_REPO_ID=muyuho/opcdimage_mini \
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --exp_name opcdimage-qwen3-vl-4b-debug \
  --project_name opcdimage_qwen3vl4b \
  --train_batch_size 32 \
  --max_prompt_length 4096 \
  --max_response_length 512 \
  --actor_lr 1e-6 \
  --total_epochs 3 \
  --n_gpus_per_node 4 \
  --nnodes 1 \
  trainer.logger="['console','wandb']"
```

说明：

- 脚本内置数据准备与校验，不需要手动先跑 parquet 生成
- 末尾追加的 `trainer.xxx`、`data.xxx`、`actor_rollout_ref.xxx` 都会原样传给 `verl.trainer.main_ppo`
- 当前默认训练模式是 full image student rollout + crop privileged reference

## 4. 当前默认关键配置

脚本中默认启用的关键项包括：

- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `trainer.generate_off_policy=False`
- `data.custom_cls.name=OPCDImagePairedVQADataset`
- `reward.custom_reward_function.name=compute_score`
- `actor_rollout_ref.actor.use_kl_loss=True`

这表示当前配方走的是 trainer-native consolidate 流程，而不是额外包一层独立 distillation 插件。

## 5. 支持调节的常用参数

### 5.1 脚本显式支持的参数

可以直接通过 `opcdimage_consolidate.sh` 传入：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-VL-4B-Instruct` | student / actor 模型路径 |
| `--ref_model_path` | 空，随后回退到 `model` | reference model 路径 |
| `--exp_name` | `opcdimage-qwen3-vl-4b-consolidate` | 实验名 |
| `--project_name` | `opcdimage_qwen3vl4b` | 项目名 |
| `--data_dir` | `data/opcdimage_qwen3vl4b` | prepared dataset 目录 |
| `--hf_dataset_repo_id` | `muyuho/opcdimage_mini` | 自动下载时使用的 HF dataset repo |
| `--train_batch_size` | `64` | 训练 prompt batch size |
| `--max_prompt_length` | `4096` | 最大 prompt 长度 |
| `--max_response_length` | `512` | 最大 response 长度 |
| `--actor_lr` | `1e-6` | actor 学习率 |
| `--total_epochs` | `10` | 训练 epoch 数 |
| `--total_training_steps` | 空 | 显式指定总步数，设置后会覆盖 epoch 驱动 |
| `--save_freq` | `100` | checkpoint 保存频率 |
| `--test_freq` | `10` | 验证频率 |
| `--n_gpus_per_node` | `4` | 单机 GPU 数 |
| `--nnodes` | `1` | 节点数 |
| `--kl_loss_type` | `full` | KL loss 类型 |
| `--kl_topk` | `256` | KL top-k 截断 |
| `--kl_renorm_topk` | `False` | 是否对 top-k KL 重归一化 |
| `--enforce_eager` | `False` | rollout 侧是否强制 eager 模式 |

### 5.2 通过 Hydra 直接透传的常用参数

除了上面这些脚本参数，还可以把任意 Hydra override 直接追加到命令末尾。

常见可调项：

| 覆盖参数 | 作用 |
| --- | --- |
| `trainer.default_local_dir=...` | 修改 checkpoint 输出目录 |
| `trainer.logger="['console']"` | 关闭 wandb，只保留控制台日志 |
| `trainer.resume_mode=auto` | 从已有 checkpoint 自动恢复 |
| `trainer.resume_from_path=...` | 指定恢复路径 |
| `data.shuffle=True` | 打开数据打乱 |
| `actor_rollout_ref.rollout.name=sglang` | 将 rollout 后端切到 sglang |
| `actor_rollout_ref.rollout.gpu_memory_utilization=0.6` | 调整 rollout 显存占用比例 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=...` | 调整每卡 actor micro batch |
| `actor_rollout_ref.actor.fsdp_config.param_offload=False` | 关闭参数 offload |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload=False` | 关闭优化器 offload |
| `actor_rollout_ref.model.enable_gradient_checkpointing=False` | 关闭梯度检查点 |
| `actor_rollout_ref.rollout.n=...` | 修改每个 prompt 生成条数 |

示例：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --train_batch_size 16 \
  --total_training_steps 200 \
  trainer.default_local_dir=checkpoints/opcdimage/debug \
  trainer.logger="['console']" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

## 6. 输出与校验

数据准备完成后，默认目录下通常会看到：

- `data/opcdimage_qwen3vl4b/train.parquet`
- `data/opcdimage_qwen3vl4b/val.parquet`
- `data/opcdimage_qwen3vl4b/train.jsonl`
- `data/opcdimage_qwen3vl4b/val.jsonl`
- `data/opcdimage_qwen3vl4b/summary.json`

训练输出默认在：

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

如果要离线评估预测结果，可以运行：

```bash
python opcdimage_recipe/evaluate_predictions.py \
  --dataset-file data/opcdimage_qwen3vl4b/val.parquet \
  --main-predictions <main_predictions.csv> \
  --baseline-predictions <full_baseline_predictions.csv> \
  --upper-bound-predictions <crop_upper_predictions.csv>
```

## 7. 这次检查确认到的事项

已确认的训练流程：

- `prepare/download -> validate -> main_ppo`
- dataset 会产出 `raw_prompt`、`crop_images`、`extra_info.crop_image`
- `consolidate` 分支会把 full-image prompt 改写成 crop-image privileged prompt
- reward 走 `opcdimage_recipe/reward_fn.py` 的选项匹配逻辑

这次同步修正了一个启动问题：

- 删除了训练脚本里额外传入的 `--config-path=config`
- 原因是 `verl.trainer.main_ppo` 自身已经内置 `config_path="config"`，脚本再显式传一个相对仓库根目录的 `config/` 会让入口更容易在项目根目录启动时报错

## 8. 当前未在本机完整实跑的原因

本机是 Windows 环境，而且当前 `.venv` 缺少最基本的运行依赖，例如：

- `packaging`
- `pandas`

所以这次无法在本机直接把训练入口完整跑到 `main_ppo`。现阶段完成的是：

- 代码链路核对
- 脚本参数核对
- 已发现启动问题的修正

建议在目标 Linux 训练环境中先补齐依赖后，再按本文命令实际启动。
