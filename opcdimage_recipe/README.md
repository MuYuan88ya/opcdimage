# OPCDImage Recipe

`opcdimage_recipe` 是当前 Vision-OPCDImage 实验的 recipe-local 实现。

当前实验保持的训练语义是：

- student 只看 `original_images` rollout
- privileged 分支不重新采样，而是复用同一条 student response
- privileged 分支只把 prompt 中的图像替换为 `crop_images`
- actor 真正反传的是 response-token 级 reverse KL，而不是 PPO reward loss

## 单层结构

`opcdimage_recipe/` 现在只保留单层文件，不再使用 `training/` 子目录。

### 训练主线

- `opcdimage_recipe/main_ppo.py`
  - recipe-local 入口与 `OPCDImageTaskRunner`
  - 安装 recipe actor，并实例化 `OPCDImageRayTrainer`
- `opcdimage_recipe/ray_trainer.py`
  - consolidate 主训练链路
  - 包含 crop prompt 重建、response 复用、`exp_log_probs` 计算、reward manager 装载
- `opcdimage_recipe/dp_actor.py`
  - top-k support 生成 / gather
  - `stage_merge=True` 时的 reverse-KL actor update

### 训练期数据加载

- `opcdimage_recipe/paired_vqa_dataset.py`
  - `OPCDImagePairedVQADataset`
  - 约定 student prompt 使用 `original_images`
  - 只负责训练期样本读取与 `re_tokenize(...)`
- `opcdimage_recipe/core.py`
  - prompt 归一化
  - 答案提取
  - `build_crop_messages_from_raw_prompt(...)`
- `opcdimage_recipe/reward_fn.py`
  - 当前多选题实验使用的 `compute_score(...)`

### 离线数据工具

- `opcdimage_recipe/dataset_tools.py`
  - `prepare`
  - `validate`
  - `export`
  - `download`
  - `upload`
- `opcdimage_recipe/evaluate_predictions.py`
  - 离线预测结果评估

## 解耦原则

当前目录明确区分两类代码：

- 训练代码：
  - 只负责 dataset loading、rollout、privileged scoring、reverse-KL 更新
- 数据工具：
  - 只负责 prepared parquet 生成、校验、HF 导出/下载/上传

训练主线不应该依赖数据准备脚本模块。

## 训练入口

主训练脚本：

- `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`

它负责：

- 自动下载 / 校验 prepared dataset
- 组装 Hydra 参数
- 启动 `python -m opcdimage_recipe.main_ppo`

## 当前实现约束

以下约束是当前 recipe 正常运行必须满足的：

- privileged reference 通过 `actor_rollout_ref.ref.model.path` 指定
- 默认 `actor_rollout_ref.model.use_fused_kernels=False`
  - 当前 recipe actor 的 full top-k KL 路径需要 `return_all_logits=True`
  - 这条路径与 fused kernels 组合仍不兼容
- `trainer.balance_batch` 会在 actor 更新前生效
- reward manager 应遵守 `reward.reward_manager.*`
- custom reward function 通过 `reward.custom_reward_function.*` 接入

## 推荐阅读顺序

如果你要快速理解当前实现，建议按这个顺序看：

1. `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`
2. `opcdimage_recipe/main_ppo.py`
3. `opcdimage_recipe/ray_trainer.py`
4. `opcdimage_recipe/dp_actor.py`
5. `opcdimage_recipe/paired_vqa_dataset.py`
6. `opcdimage_recipe/core.py`
7. `opcdimage_recipe/reward_fn.py`
8. `opcdimage_recipe/dataset_tools.py`

## 相关文档

- 代码链路说明：`docs/opcdimage_recipe_code_flow.md`
- 研究记录与协作约束：`agents.md`
