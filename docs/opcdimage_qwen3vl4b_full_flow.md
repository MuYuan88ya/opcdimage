# OPCDImage 全流程

## 1. 数据准备

执行：

```bash
bash examples/on_policy_distillation_trainer/prepare_opcdimage_data.sh
```

它会读取 `ZwZ-RL-VQA-mini/train_crop_clean.csv`，输出：

- `data/opcdimage_qwen3vl4b/train.parquet`
- `data/opcdimage_qwen3vl4b/val.parquet`
- `data/opcdimage_qwen3vl4b/summary.json`

划分规则继续按原图分组，避免同图泄漏。

## 2. Dataset 到 trainer 的接口

训练时使用 `opcdimage_recipe/paired_vqa_dataset.py` 中的 `OPCDImagePairedVQADataset`。

dataset 会同时产出：

- `raw_prompt`

其中：

- `raw_prompt` 绑定 full image
- `extra_info.crop_image` 提供 privileged crop 信息
- trainer 再在 consolidate 路径里从 `raw_prompt` 派生 crop prompt

## 3. Trainer-native consolidate

主入口是：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

关键配置：

- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_type=full`
- `actor_rollout_ref.actor.kl_topk=256`

训练步内部：

1. 从 plain batch 生成 student response
2. 从 paired batch 中取 `raw_prompt` 和 `extra_info.crop_image`
3. trainer 内部构造 `gen_batch_with_crop`
4. 拼出 `batch_with_crop = crop prompt + student response`
5. 用 ref model 计算 `exp_log_probs`
6. actor 按 `opcd` consolidate 路径更新

## 4. 评估

离线评估脚本：

```bash
python opcdimage_recipe/evaluate_predictions.py \
  --dataset-file data/opcdimage_qwen3vl4b/val.parquet \
  --main-predictions <main_predictions> \
  --baseline-predictions <full_baseline_predictions> \
  --upper-bound-predictions <crop_upper_predictions>
```

当前默认输出：

- overall accuracy
- by megapixels
- by bbox area ratio
- gap closure

## 5. 当前实现重点

这一版的核心不是“再包装一层 distillation plugin”，而是：

- 新仓库
- trainer-native consolidate
- privileged crop batch
- ref model 直接产生 `exp_log_probs`

也就是说，这一版的主线已经从之前的 agent-loop Vision-OPCD，切换成了更接近 `opcd` 正式实现路径的版本。
