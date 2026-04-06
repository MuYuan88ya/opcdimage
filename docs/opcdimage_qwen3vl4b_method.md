# OPCDImage 方法说明

## 目标

`opcdimage` 的目标不是继续沿用 recipe / agent-loop teacher sidecar，而是在新的 `verl` fork 中按 `opcd` 的 trainer-native 路径实现多模态版本的 OPCD。

首版固定问题设定：

- student 看 full image
- privileged batch / ref model 看 crop image
- student rollout 轨迹保持主导
- ref model 只在 student 轨迹上计算 `exp_log_probs`
- actor 直接按 `opcd sys_consolidate` 风格做 KL-style consolidate 更新

## 与文本 OPCD 的对应关系

文本版 `opcd`：

- plain prompt -> student rollout 条件
- system prompt / experience -> privileged 条件
- `batch_with_exp` -> privileged batch
- `ref_policy_wg.compute_ref_log_prob(batch_with_exp)` -> teacher/reference 打分

`opcdimage`：

- full-image prompt -> student rollout 条件
- crop-image prompt -> privileged 条件
- `batch_with_crop` -> privileged batch
- `ref_policy_wg.compute_ref_log_prob(batch_with_crop)` -> teacher/reference 打分

因此这里的 crop，不是推理时额外给 student 的输入，而是训练时 privileged context 的视觉等价物。

## 数据格式

prepared parquet 的单条样本固定包含：

- `prompt`
- `images`
- `ground_truth`
- `answer`
- `reward_model`
- `extra_info`

其中：

- `images` 是 full image
- `extra_info.crop_image` 是 crop image
- 数据格式尽量与原版 RLHFDataset 对齐，crop 不再被做成第二套数据集字段
- 真正的 privileged 差异由 trainer 在 step 内部派生，而不是由 dataset 直接产两套训练输入

## 训练流程

一次训练迭代的主流程如下：

1. dataloader 读出 paired batch
2. trainer 从 batch 中拆出 plain `gen_batch`
3. student 只对 `raw_prompt + full image` rollout
4. trainer 从同一批样本派生 `gen_batch_with_crop`
5. `gen_batch_with_crop` 使用 `raw_prompt + extra_info.crop_image`
6. trainer 用 `gen_batch_with_crop` 的 prompt 和 student response 重建 `batch_with_crop`
7. ref model 对 `batch_with_crop` 计算 `exp_log_probs`
8. actor 直接消费 `exp_log_probs`
9. actor 用 `kl_penalty(logprob=student, ref_logprob=exp, kl_penalty="full")` 更新

这条路径和 `opcd` 的关键一致点是：

- privileged 条件在 trainer step 内被显式构造
- 不是 teacher 重新采样答案
- 不是 agent loop 里单独请求 teacher logprob

## 当前默认配置

默认实验脚本固定为：

- model: `Qwen/Qwen3-VL-4B-Instruct`
- ref model: 默认与 student 初始 checkpoint 相同
- rollout backend: `vllm`
- actor backend: `FSDP`
- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_type=full`
- `actor_rollout_ref.actor.kl_topk=256`
- `actor_rollout_ref.actor.kl_renorm_topk=False`

## 当前边界

这一版已经按 `opcd` 的 trainer-native 路径落到新仓库中，但仍然只实现最小主线：

- 只支持 `Qwen3-VL-4B-Instruct + FSDP + vLLM`
- 只支持 single-image student / single-image crop privileged batch
- 默认不混 RLVR reward 作为主实验结论
- 不做 EMA teacher，不做每 K step teacher 同步

## 验证建议

在环境装好后，优先验证：

1. `dataset_tools.py prepare` 能生成 train/val parquet
2. 单样本跑通 `rollout -> batch_with_crop -> exp_log_probs -> update_actor`
3. `kl_topk_indices` 和 `exp_log_probs` 在 consolidate 主线上能稳定产出
4. 离线评估脚本能对 main / full baseline / crop upper bound 输出统一指标
