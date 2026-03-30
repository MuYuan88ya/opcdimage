# opcdimage 实验指南

这份文档只关注实验本身，不展开环境安装细节。默认你已经有一台可运行 `Qwen/Qwen3-VL-4B-Instruct` 的 Linux 训练机器，并且能正常执行仓库脚本。

训练入口：

- `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`

固定模型包装脚本：

- `examples/on_policy_distillation_trainer/run_qwen3_vl_4b_opcdimage.sh`

## 1. 当前实验在做什么

当前 `opcdimage` 不是文本版 OPcD 的 experience 注入，而是图像版的 crop-privileged consolidate：

- student / actor 看原图 `original_images`
- privileged reference 看 crop 图 `crop_images`
- student 先在原图条件下生成 response
- ref 再在 crop 条件下对同一条 response 计算 `exp_log_probs`
- actor 用 KL 把 student 分布往 privileged ref 分布拉近

当前脚本固定使用的关键配置是：

- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `trainer.generate_off_policy=False`
- `data.custom_cls.name=OPCDImagePairedVQADataset`
- `reward.custom_reward_function.name=compute_score`

reward 的定义很简单：

- 从模型输出里抽取选项
- 与 `ground_truth` 比较
- 正确记 `1.0`，错误记 `0.0`

## 2. 数据输入要求

训练脚本最终依赖的是 prepared dataset。默认位置是：

- `data/opcdimage_qwen3vl4b/train.parquet`
- `data/opcdimage_qwen3vl4b/val.parquet`

每条样本至少应包含这些字段：

- `problem`
- `original_images`
- `crop_images`
- `bbox`
- `answer`
- `reward_model`
- `extra_info`

其中最关键的是：

- `original_images`：student 看到的原图
- `crop_images`：privileged ref 看到的 crop 图
- `reward_model.ground_truth`：训练 reward 对照答案
- `extra_info.crop_image`：必须与 `crop_images[0]` 一致
- `extra_info.original_image`：必须与 `original_images[0]` 一致

现在这些图片路径默认存成相对路径，例如：

- `images/original_images/...`
- `images/crop/...`

训练运行时会以 `train.parquet` / `val.parquet` 所在目录为根，把这些相对路径解析成真实本地路径，所以不需要在下载后再改写 manifest。

训练前建议先跑一次：

```bash
python opcdimage_recipe/data_tools.py validate \
  --train-file data/opcdimage_qwen3vl4b/train.parquet \
  --val-file data/opcdimage_qwen3vl4b/val.parquet
```

这个校验会确认：

- 列存在
- 图像路径存在
- crop / original 对齐
- `reward_model.ground_truth` 与 `answer` 一致
- train / val 没有原图泄漏
- prompt 中不残留显式 bbox 特权提示

## 3. 数据从哪里来

现在默认只保留一条数据准备路径：从 Hugging Face 直接下载并解压。

默认 repo 是：

- `muyuho/opcdmini`

这个 repo 同时包含：

- `prepared/train.parquet`
- `prepared/val.parquet`
- `original_images.tar.gz`
- `crop_images.tar.gz`

训练脚本会先检查：

- `data/opcdimage_qwen3vl4b/train.parquet`
- `data/opcdimage_qwen3vl4b/val.parquet`

如果这两个文件不存在，就自动执行下载和解压。下载后 parquet 仍然保留相对路径，运行时再解析。

你也可以手动执行：

```bash
python opcdimage_recipe/hf_data_tools.py download \
  --output-dir data/opcdimage_qwen3vl4b \
  --repo-id muyuho/opcdmini
```

## 4. 最常用的实验命令

### 4.1 最简启动

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

这条命令会自动做：

1. 检查数据是否存在
2. 必要时下载并解压数据
3. 校验数据
4. 启动 `verl.trainer.main_ppo`

### 4.2 固定 Qwen3-VL-4B 的启动方式

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_vl_4b_opcdimage.sh
```

它本质上等价于：

- `--model Qwen/Qwen3-VL-4B-Instruct`
- `--ref_model_path Qwen/Qwen3-VL-4B-Instruct`

### 4.3 推荐的第一轮冒烟实验

第一轮不要直接跑大配置，先确认链路稳定：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --exp_name opcdimage-smoke \
  --project_name opcdimage_qwen3vl4b \
  --train_batch_size 4 \
  --max_prompt_length 1024 \
  --max_response_length 128 \
  --processor_max_image_tokens 10000 \
  --total_training_steps 20 \
  --save_freq 20 \
  --test_freq 20 \
  --n_gpus_per_node 1 \
  --nnodes 1 \
  trainer.logger="['console']"
```

这个实验的目的不是出结果，而是确认：

- 数据可读
- consolidate 流程可跑
- rollout / ref / actor 链路可跑
- checkpoint 可写

### 4.4 推荐的第一轮正式实验

当冒烟稳定后，再切到更接近正式训练的设置：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --ref_model_path Qwen/Qwen3-VL-4B-Instruct \
  --exp_name opcdimage-run1 \
  --project_name opcdimage_qwen3vl4b \
  --data_dir data/opcdimage_qwen3vl4b \
  --train_batch_size 16 \
  --max_prompt_length 2048 \
  --max_response_length 256 \
  --processor_max_image_tokens 10000 \
  --actor_lr 1e-6 \
  --total_training_steps 200 \
  --save_freq 50 \
  --test_freq 50 \
  --n_gpus_per_node 1 \
  --nnodes 1 \
  trainer.logger="['console','wandb']" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

如果这组参数能稳定跑完，再逐步放大：

- `train_batch_size`
- `max_prompt_length`
- `max_response_length`
- `n_gpus_per_node`
- `total_training_steps`

### 4.5 单机多卡实验示例

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --ref_model_path Qwen/Qwen3-VL-4B-Instruct \
  --exp_name opcdimage-qwen3vl4b-4gpu \
  --project_name opcdimage_qwen3vl4b \
  --train_batch_size 64 \
  --max_prompt_length 4096 \
  --max_response_length 512 \
  --processor_max_image_tokens 10000 \
  --actor_lr 1e-6 \
  --total_epochs 10 \
  --save_freq 100 \
  --test_freq 10 \
  --n_gpus_per_node 4 \
  --nnodes 1
```

## 5. 优先应该调哪些参数

下面这些参数最直接决定实验形态。

| 参数 | 默认值 | 影响 |
| --- | --- | --- |
| `--train_batch_size` | `64` | 每轮训练的全局 batch 大小 |
| `--hf_dataset_repo_id` | `muyuho/opcdmini` | 当前统一 HF 数据集 repo |
| `--max_prompt_length` | `10000` | prompt 上限，影响显存和可保留上下文 |
| `--max_response_length` | `512` | response 上限，影响生成长度和显存 |
| `--processor_max_image_tokens` | `10000` | 图像侧视觉 token 预算上限，脚本会自动换算成 `max_pixels` |
| `--processor_max_pixels` | 空 | 直接按像素上限限制 processor；若设置，则优先于 `--processor_max_image_tokens` |
| `--actor_lr` | `1e-6` | actor 学习率 |
| `--total_epochs` | `10` | 按 epoch 控制训练长度 |
| `--total_training_steps` | `100` | 按 step 控制训练长度，设置后会覆盖 epoch 驱动 |
| `--save_freq` | `100` | checkpoint 保存频率 |
| `--test_freq` | `10` | 验证频率 |
| `--n_gpus_per_node` | `1` | 单机 GPU 数 |
| `--nnodes` | `1` | 节点数 |
| `--kl_loss_type` | `full` | KL 形式 |
| `--kl_topk` | `256` | full KL 的 top-k |
| `--kl_renorm_topk` | `True` | top-k 后是否重归一化 |
| `--enforce_eager` | `False` | rollout 是否强制 eager |

其中最重要的几项可以这样理解：

- `train_batch_size`：先决定训练稳定性和吞吐
- `max_prompt_length`：先决定视觉问答上下文保留多少
- `max_response_length`：先决定答案空间和 rollout 代价
- `processor_max_image_tokens`：先决定图像分辨率上限，避免视觉 token 过多
- `total_training_steps`：最适合前期做可控对比实验
- `actor_lr`：最适合做学习率敏感性实验

这里有一个容易混淆的点：

- `max_prompt_length` 约束的是最终 prompt token 长度
- `processor_max_image_tokens` 约束的是图像侧视觉 token 预算

也就是说，`processor_max_image_tokens=10000` 并不等于“文本 token + 图像 token 总和一定不超过 10000”，它只负责在 processor 侧把图像分辨率压到不超过这部分预算。

## 6. 推荐的实验推进顺序

建议按这个顺序开展，而不是一上来就堆大配置。

### 6.1 阶段 A：链路验证

目标：

- 确认训练能启动
- 确认不会在前几十步崩掉

建议设置：

- `train_batch_size=4`
- `max_prompt_length=1024`
- `max_response_length=128`
- `total_training_steps=20`

### 6.2 阶段 B：基线跑通

目标：

- 获得一条可复现的 crop-privileged consolidate 基线

建议设置：

- `train_batch_size=16`
- `max_prompt_length=2048`
- `max_response_length=256`
- `total_training_steps=200` 或 `500`

### 6.3 阶段 C：长度和 batch 扩展

目标：

- 观察更长 prompt 和更大 batch 是否提升效果

优先对比：

1. `max_prompt_length: 1024 vs 2048 vs 4096`
2. `max_response_length: 128 vs 256 vs 512`
3. `train_batch_size: 8 vs 16 vs 32`

### 6.4 阶段 D：KL 相关对比

目标：

- 观察 privileged 对齐强度对训练的影响

优先对比：

1. `kl_topk: 128 vs 256 vs 512`
2. `kl_renorm_topk: False vs True`
3. `actor_lr: 5e-7 vs 1e-6 vs 2e-6`

## 7. 可以直接追加的 Hydra 覆盖项

除了 shell 参数，你还可以在命令末尾直接追加 Hydra override。

常用项如下：

| 覆盖参数 | 作用 |
| --- | --- |
| `trainer.default_local_dir=/path/to/checkpoints` | 修改输出目录 |
| `trainer.logger="['console']"` | 只打控制台日志 |
| `trainer.logger="['console','wandb']"` | 控制台 + wandb |
| `trainer.resume_mode=auto` | 自动恢复训练 |
| `trainer.resume_mode=resume_path` | 指定路径恢复 |
| `trainer.resume_from_path=/path/to/ckpt` | 恢复路径 |
| `data.shuffle=True` | 打乱训练数据 |
| `actor_rollout_ref.rollout.gpu_memory_utilization=0.6` | 调整 rollout 显存利用率 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=...` | 调整 actor micro batch |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=...` | 调整 actor token 上限 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=...` | 调整 ref/log_prob micro batch |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=...` | 调整 ref/log_prob token 上限 |
| `actor_rollout_ref.actor.fsdp_config.param_offload=False` | 关闭参数 offload |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload=False` | 关闭优化器 offload |
| `actor_rollout_ref.model.enable_gradient_checkpointing=False` | 关闭梯度检查点 |
| `actor_rollout_ref.rollout.n=...` | 每个 prompt 生成多条 response |
| `actor_rollout_ref.model.use_fused_kernels=False` | 关闭 fused kernels |
| `actor_rollout_ref.actor.use_torch_compile=False` | 关闭 torch compile |

示例：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --exp_name opcdimage-debug \
  --train_batch_size 16 \
  --total_training_steps 200 \
  trainer.default_local_dir=/data/checkpoints/opcdimage-debug \
  trainer.logger="['console']" \
  data.shuffle=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.actor.use_torch_compile=False
```

## 8. 输出、恢复和结果管理

默认输出目录是：

```text
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

也就是通常会落到：

```text
checkpoints/opcdimage_qwen3vl4b/<exp_name>
```

如果你要恢复实验：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --exp_name opcdimage-resume \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/path/to/checkpoints/opcdimage_qwen3vl4b/old_run
```

如果你想让同目录自动续跑：

```bash
trainer.resume_mode=auto
```

建议你在实验记录里固定保留这几项：

- `exp_name`
- 完整启动命令
- `train_batch_size`
- `max_prompt_length`
- `max_response_length`
- `actor_lr`
- `kl_topk`
- `kl_renorm_topk`
- `total_training_steps`
- 最终 checkpoint 路径

## 9. 当前脚本默认行为

这部分是“你不显式覆盖时，实验默认会怎么跑”。

### 9.0 视觉 token 预算相关

- 默认 `processor_max_image_tokens=10000`
- 如果同时设置 `processor_max_pixels`，则优先使用像素上限

当前实现方式是：

- 脚本把这两个值写成环境变量
- `hf_processor()` 在统一入口读取它们
- 然后把限制写入 Hugging Face `image_processor`

对于 Qwen3-VL 这类常见设置，`processor_max_image_tokens=10000` 会被换算成：

- `max_pixels=10240000`

这能保证图像侧视觉 token 不超过这部分预算。

### 9.1 数据相关

- `data.prompt_key=problem`
- `data.image_key=original_images`
- `data.custom_cls.path=.../opcdimage_recipe/paired_vqa_dataset.py`
- `data.custom_cls.name=OPCDImagePairedVQADataset`
- `data.filter_overlong_prompts=True`
- `data.truncation='error'`
- `data.shuffle=False`

含义：

- 训练时超长 prompt 会被过滤
- 默认不打乱数据

### 9.2 模型和 rollout 相关

- `actor_rollout_ref.model.enable_gradient_checkpointing=True`
- `actor_rollout_ref.model.use_remove_padding=True`
- `actor_rollout_ref.model.use_fused_kernels=True`
- `actor_rollout_ref.actor.use_torch_compile=True`
- `actor_rollout_ref.actor.use_dynamic_bsz=True`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.fsdp_config.param_offload=True`
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
- `actor_rollout_ref.rollout.n=1`

### 9.3 训练范式相关

- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `trainer.generate_off_policy=False`
- `trainer.experience_path=''`
- `trainer.val_before_train=False`
- `trainer.resume_mode=disable`

## 10. 你最先该做的三组实验

如果你现在要开始正式做实验，我建议优先做这三组，而不是同时改很多参数。

### 10.1 组一：链路稳定性

固定：

- `actor_lr=1e-6`
- `kl_topk=256`

对比：

- `max_prompt_length=1024`
- `max_prompt_length=2048`

### 10.2 组二：生成长度

固定：

- `train_batch_size=16`
- `max_prompt_length=2048`

对比：

- `max_response_length=128`
- `max_response_length=256`
- `max_response_length=512`

### 10.3 组三：KL 强度

固定：

- `train_batch_size=16`
- `max_prompt_length=2048`
- `max_response_length=256`

对比：

- `kl_topk=128`
- `kl_topk=256`
- `kl_topk=512`

## 11. 当前实现和实验含义的对应关系

这份指南对应的是当前已经修正后的实现，关键点有两个：

1. `crop privileged` 分支会同步替换 `multi_modal_inputs`，不是只改文本 prompt
2. `re_tokenize()` 会按 `max_prompt_length` 统一处理，使 batch 内样本可堆叠

所以当前实验的真实含义是：

- student 在 full image 上生成
- ref 在 crop image 条件下打分
- actor 学的是“如何在只看原图时，逼近 crop 条件下的 privileged 分布”
