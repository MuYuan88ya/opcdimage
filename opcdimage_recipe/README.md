# OPCDImage Recipe

`opcdimage_recipe` 是当前 Vision-OPCDImage 实验的 recipe-local 实现。

这套实现保持的训练语义是：

- student 只看 `original_images` rollout
- privileged 分支不重新采样，而是复用同一条 student response
- privileged 分支只把 prompt 中的图像替换为 `crop_images`
- actor 真正反传的是 response-token 级 reverse KL，而不是 PPO reward loss

## 目录结构

当前按“训练 / 数据与打分 / 工具”三层组织。

### 1. 训练实现

- `opcdimage_recipe/training/main_ppo.py`
  - recipe-local 入口与 `OPCDImageTaskRunner`
  - 负责安装 recipe actor，并实例化 `OPCDImageRayTrainer`
- `opcdimage_recipe/training/trainer.py`
  - consolidate 主训练链路
  - 完成原图 rollout、crop prompt 重建、response 复用、`exp_log_probs` 计算、reverse-KL 更新
- `opcdimage_recipe/training/actor.py`
  - top-k support 生成 / gather
  - `stage_merge=True` 时的 reverse-KL actor update
- `opcdimage_recipe/training/utils.py`
  - recipe 训练辅助函数
  - 包括 `compose_prompt_response_tensors(...)`、reward manager 装载、recipe 选项读取

### 2. 数据与打分

- `opcdimage_recipe/paired_vqa_dataset.py`
  - `OPCDImagePairedVQADataset`
  - 约定 student prompt 使用 `original_images`
  - 提供 trainer 动态重建 crop prompt 所需的 `re_tokenize(...)`
- `opcdimage_recipe/core.py`
  - prompt 归一化
  - 答案提取
  - `build_crop_messages_from_raw_prompt(...)`
- `opcdimage_recipe/reward_fn.py`
  - 当前多选题实验使用的 `compute_score(...)`

### 3. 数据工具

- `opcdimage_recipe/data_tools.py`
  - prepared parquet 校验与数据侧检查
- `opcdimage_recipe/hf_data_tools.py`
  - HF 数据集下载 / 导出
- `opcdimage_recipe/upload_hf_dataset.py`
  - 上传数据集到 Hugging Face
- `opcdimage_recipe/evaluate_predictions.py`
  - 离线预测结果评估

### 4. 顶层兼容文件

下面三个顶层文件现在是兼容壳，保留旧导入路径和旧启动方式：

- `opcdimage_recipe/main_ppo.py`
- `opcdimage_recipe/ray_trainer.py`
- `opcdimage_recipe/dp_actor.py`

实际实现已经收敛到 `opcdimage_recipe/training/` 下。

## 训练入口

主训练脚本：

- `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`

它负责：

- 自动准备 / 校验 prepared dataset
- 组装 Hydra 参数
- 启动 `python -m opcdimage_recipe.main_ppo`

## 当前实现约束

以下约束是当前 recipe 正常运行必须满足的：

- privileged reference 通过 `actor_rollout_ref.ref.model.path` 指定
  - 不再使用错误的 `actor_rollout_ref.model.ref_model_path`
- 默认 `actor_rollout_ref.model.use_fused_kernels=False`
  - 当前 recipe actor 的 full top-k KL 路径需要 `return_all_logits=True`
  - 这条路径与 fused kernels 组合仍不兼容
- `trainer.balance_batch` 现在会在 actor 更新前真正生效
- reward manager 不再硬编码为 `NaiveRewardManager`
  - 会遵守 `reward.reward_manager.*`
  - 但当前实验仍要求 `reward.custom_reward_function.path` 指向本地打分函数

## 推荐阅读顺序

如果你要快速理解训练代码，建议按这个顺序看：

1. `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`
2. `opcdimage_recipe/training/main_ppo.py`
3. `opcdimage_recipe/training/trainer.py`
4. `opcdimage_recipe/training/actor.py`
5. `opcdimage_recipe/paired_vqa_dataset.py`
6. `opcdimage_recipe/core.py`
7. `opcdimage_recipe/reward_fn.py`

## 相关文档

- 代码链路说明：`docs/opcdimage_recipe_code_flow.md`
- 研究记录：`agents.md`
