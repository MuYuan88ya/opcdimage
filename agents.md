# OPCDImage Agents Notes

## Experiment Goal
- `opcdimage` 把 OPcD 的“特权上下文蒸馏”从文本经验迁移到图像条件。
- student / actor 只在 `original_images` 条件下 rollout。
- privileged reference 不重新采样，而是拿同一条 student response，在 `crop_images` 条件下重新打分。
- 训练目标是让 student 在原图上的 token 分布靠近 privileged crop 条件下的 token 分布。

## True Loss Semantics
- 当前实验不是标准 PPO reward optimization 再额外加 KL。
- actor 真正反传的是 response-token 级别的 reverse KL。
- reward function 主要提供训练监控和验证精度，不决定 actor 的主梯度方向。
- 当 `kl_loss_type=full` 且 `kl_topk>0` 时：
  - student 先在原图 prompt 上计算 response token 的 top-k support。
  - privileged ref 在 crop prompt 上只对这套 support 打分。
  - loss 使用 `KL(student || privileged)`，不是 upstream distillation 的 `KL(privileged || student)`。

## Current Recipe Contract
- 数据集类是 `OPCDImagePairedVQADataset`。
- 每条样本至少提供：
  - `raw_prompt`
  - `original_images`
  - `crop_images`
  - `extra_info.crop_image`
- recipe 训练约定固定为：
  - rollout prompt 不改写，仍然用原图。
  - privileged prompt 仅在 trainer 内部由 `build_crop_messages_from_raw_prompt(...)` 重建。
  - 同一条 rollout response 同时服务 student loss 和 privileged ref scoring。

## Why The Framework Was Restored
- `verl` core 之前混入了很多只属于 OPcDImage 实验的分支：
  - `stage=consolidate`
  - `privileged_mode`
  - `on_policy_merge`
  - textgame / off-policy / textual experience 兼容路径
- 这些逻辑不应该继续留在通用框架里。
- 当前重构把实验实现迁回 `opcdimage_recipe/`，并把以下文件恢复到 `third_party/verl_upstream`：
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/workers/actor/dp_actor.py`
  - `verl/trainer/config/ppo_trainer.yaml`
  - `_generated_ppo_*.yaml`

## Relation To Local `opcd`
- 本地 `C:/Users/LU/Desktop/rl4image/opcd` 仍然是文本 OPcD 的主要参考。
- 一致之处：
  - 同轨迹 response reuse
  - privileged branch 只负责打分
  - reverse KL 主导 actor 更新
- 差异只在特权上下文的载体：
  - `opcd` 是文本 experience / privileged prompt
  - `opcdimage` 是 crop image prompt

## Why Upstream Distillation Was Not Used As The Main Path
- upstream distillation 默认是 teacher-weighted forward KL。
- 它和当前实验需要保留的 reverse KL 目标不一致。
- upstream distillation 现成接口也不直接覆盖这里依赖的 `kl_topk_indices` support gather 方式。
- 所以当前最稳妥的实现是：
  - 保持 upstream `verl` 原样
  - 在 `opcdimage_recipe` 中实现 recipe-local trainer + actor

## New Entry Points
- 训练入口改为 `python -m opcdimage_recipe.main_ppo`
- shell 包装脚本仍是：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`
- recipe-specific 配置改为：
  - `++opcdimage.privileged_mode=crop`
  - `++opcdimage.on_policy_merge=True`

## Scope Deliberately Kept Out
- 当前 recipe 不再支持这些旧分支：
  - textgame
  - textual experience injection
  - generate/load off-policy batches
  - seqkd branch
  - exp_learner side paths
- 如果以后需要恢复，应该在 recipe 层单独扩展，而不是再次改写 `verl` core。
