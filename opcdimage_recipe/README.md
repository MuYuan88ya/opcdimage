# OPCDImage Recipe

这是 `opcdimage` 新仓库里的最小 Vision-OPCD recipe。

核心特点：

- 数据侧尽量对齐原版 `ZwZ-RL-VQA-mini` 格式
- crop 不再做成第二套数据集输入，而是在 trainer 里动态派生 privileged crop prompt
- trainer 侧走 `stage=consolidate`
- student rollout 看 full image
- trainer 在 step 内部从 `raw_prompt + crop_images` 派生 privileged crop batch
- privileged batch / ref model 看 crop image
- actor 直接消费 `exp_log_probs`

相关入口：

- 数据准备与校验：`opcdimage_recipe/data_tools.py`
- HF 导出与自动下载：`opcdimage_recipe/hf_data_tools.py`
- HF 上传：`opcdimage_recipe/upload_hf_dataset.py`
- 自定义 dataset：`opcdimage_recipe/paired_vqa_dataset.py`
- reward：`opcdimage_recipe/reward_fn.py`
- 离线评估：`opcdimage_recipe/evaluate_predictions.py`
- 训练脚本：`examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`
