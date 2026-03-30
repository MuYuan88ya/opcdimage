# opcdimage 实验指南

这份文档只回答三件事：

1. 这个实验的核心思想是什么
2. 当前训练方法具体在做什么
3. 如何直接启动训练

不展开环境配置。

## 1. 实验核心思想

`opcdimage` 延续的是 OPcD 的核心思想，但把“文本特权上下文”替换成了“图像特权上下文”。

文本版 OPcD 的做法是：

- student 在普通上下文下生成
- privileged reference 在更强的上下文下对同一条 response 打分
- actor 再用 KL 去逼近 privileged reference 的分布

`opcdimage` 对应地改成：

- student / actor 看原图 `original_images`
- privileged reference 看 crop 图 `crop_images`
- student 先在原图条件下生成 response
- reference 再在 crop 条件下，对这同一条 response 计算 `exp_log_probs`
- actor 用 KL 把自己在原图条件下的分布，往 crop 条件下的 privileged 分布拉近

所以这套实验想解决的问题是：

- 模型训练时可以利用 crop 这种更强视觉信息
- 但测试时 student 仍然只看原图
- 目标是让 student 学会在只看原图时，尽量逼近“如果看到了 crop 会怎么回答”

---

## 2. 当前训练方法

当前默认训练范式是：

- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `trainer.generate_off_policy=False`

可以把一次训练 step 理解成下面这条链路：

1. 从数据集中取一批样本
2. student 用原图 prompt 生成 response
3. privileged 分支把同一条 prompt 里的图像替换成 crop 图
4. reference 在 crop 条件下对 student 的 response 计算 `exp_log_probs`
5. actor 用 KL 把自己的分布往这个 privileged reference 分布拉近

对应关系是：

- student prompt：`original_images`
- privileged prompt：`crop_images`
- reward：答案和 `ground_truth` 匹配记 `1.0`，否则 `0.0`

这意味着当前训练不是普通 SFT，也不是单纯 PPO，而是：

- student 在自己的轨迹上采样
- privileged reference 不重新生成答案
- 只在“同一条 student response”上做对齐

这正是 OPcD 风格的核心。

---

## 3. 数据形式

当前默认数据源是 Hugging Face dataset：

- `muyuho/opcdmini`

当前训练数据整体都来自这个 repo。

repo 中包含训练所需的完整内容：

- `prepared/train.parquet`
- `prepared/val.parquet`
- `original_images.tar.gz`
- `crop_images.tar.gz`

也就是说，当前实验不依赖额外单独维护的数据文件源，也不依赖另一套本地原始数据准备流程。

训练启动时，脚本只是把 `muyuho/opcdmini` 里的内容同步到本地工作目录，保留 `prepared/` 和 `images/` 这套目录结构，作为运行时缓存和解压落地点；数据来源本身仍然是 `muyuho/opcdmini`。

默认本地落地点会按 repo id 命名，也就是：

- `data/muyuho/opcdmini`

每条样本至少包含：

- `problem`
- `original_images`
- `crop_images`
- `bbox`
- `answer`
- `reward_model`
- `extra_info`

关键字段含义：

- `original_images`：student 看到的原图
- `crop_images`：privileged reference 看到的 crop 图
- `reward_model.ground_truth`：reward 对照答案
- `extra_info.original_image`：对应原图路径
- `extra_info.crop_image`：对应 crop 路径

现在 parquet 里保存的是相对路径，例如：

- `images/original_images/...`
- `images/crop/...`

训练运行时会自动按数据目录把它们解析成真实本地路径。

---

## 4. 当前默认数据读取方式

现在默认只保留一条数据准备路径：

1. 从 Hugging Face dataset `muyuho/opcdmini` 下载
2. 解压图片压缩包
3. 在本地生成运行时可直接读取的缓存目录
4. 跑数据校验
5. 再启动训练

---

## 5. 如何启动训练

### 5.1 最简启动

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

这条命令会自动完成：

1. 检查本地是否已经有 `muyuho/opcdmini` 的已下载缓存
2. 如果没有，则从 `muyuho/opcdmini` 下载并解压
3. 跑 `validate`
4. 启动 `verl.trainer.main_ppo`

### 5.2 先单独准备数据，再训练

```bash
bash examples/on_policy_distillation_trainer/prepare_opcdimage_data.sh
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

### 5.3 固定 Qwen3-VL-4B 的入口

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_vl_4b_opcdimage.sh
```

---

## 6. 最常用实验命令

### 6.1 冒烟实验

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --exp_name opcdimage-smoke \
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

这个实验的目标只是确认：

- 数据链路正常
- consolidate 流程正常
- rollout / ref / actor 能顺利跑通

### 6.2 第一轮正式实验

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --ref_model_path Qwen/Qwen3-VL-4B-Instruct \
  --exp_name opcdimage-run1 \
  --project_name opcdimage_qwen3vl4b \
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

### 6.3 单机多卡实验

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

---

## 7. 最重要的实验参数

如果只抓最关键的参数，优先关注这些：

| 参数 | 含义 |
| --- | --- |
| `--train_batch_size` | 全局训练 batch 大小 |
| `--max_prompt_length` | prompt 长度上限 |
| `--max_response_length` | response 长度上限 |
| `--processor_max_image_tokens` | 图像侧视觉 token 预算上限 |
| `--actor_lr` | actor 学习率 |
| `--total_training_steps` | 训练总步数 |
| `--kl_topk` | full KL 的 top-k |
| `--kl_renorm_topk` | top-k 后是否重归一化 |
| `--n_gpus_per_node` | 单机 GPU 数 |
| `--nnodes` | 节点数 |

最常见的调参方向：

1. `train_batch_size`
2. `max_prompt_length`
3. `max_response_length`
4. `processor_max_image_tokens`
5. `actor_lr`
6. `kl_topk`

---

## 8. 当前实验的真实含义

现在这套实现对应的实验含义可以概括为：

- student 在 full image 上生成
- reference 在 crop image 条件下打分
- actor 学的是“如何在只看原图时，逼近 crop 条件下的 privileged 分布”

如果只记一句话，就记这个。
