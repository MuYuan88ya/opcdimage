# OPCDImage AGENTS Notes

这份文件是 `opcdimage` 项目的后续会话说明书。

目标有两个：

1. 让后续对话能快速恢复对实验动机、实现方案和真实 loss 语义的理解。
2. 明确后续改代码时的约束，避免再次把实验逻辑写进 `verl` 框架层，或者把当前 reverse-KL 实验偷偷改成别的目标。

---

## 1. Project Goal

`opcdimage` 是把 OPcD 的“特权上下文蒸馏”思想从文本场景迁移到图像条件的实验。

当前实验想解决的问题是：

- 真实部署时，student / actor 只能看到完整原图 `original_images`。
- 但训练时，我们额外拥有一个更聚焦的训练特权视角，也就是目标区域 crop 图 `crop_images`。
- 希望 student 最终在“只看原图”的条件下，也能更接近“看过 crop 图时”的判别与生成能力。

所以这里的 crop image 不是测试时输入，也不是第二套数据任务，而是训练期的 privileged context。

---

## 2. Why This Experiment Exists

这个实验的直觉是：

- 原图通常包含无关背景、多个物体或视觉干扰。
- crop 图提供了更干净、更低歧义的特权条件。
- 如果直接让 student 用 sparse reward 自己摸索，学习信号会比较弱。
- 如果让 privileged 分支在 crop 条件下对同一条 response 给出 token 级分布约束，就能提供更密集、更稳定的训练信号。

这和本地参考实现 `C:/Users/LU/Desktop/rl4image/opcd` 的核心思想一致：

- privileged 分支负责“更优上下文条件下的打分”
- student 分支负责真实 rollout
- 主更新目标由 reverse KL 驱动，而不是把 privileged 分支当成另一个 sampler

---

## 3. Canonical Experiment Semantics

当前实验的标准语义必须固定为下面这条链路：

1. student / actor 只在 `original_images` 条件下 rollout。
2. privileged reference 不重新采样，而是复用同一条 student response。
3. privileged 分支只替换 prompt 中的图像为 `crop_images`，文本内容不改。
4. privileged ref 在 crop 条件下对同一条 response 重新打分。
5. actor 用 response-token 级 reverse KL，把 student(original image) 拉向 privileged(crop image)。

一句话概括：

```text
student 在原图上生成；
privileged 在 crop 图上对同一条 response 打分；
actor 最终优化 KL(student || privileged)。
```

---

## 4. True Loss Semantics

### 4.1 当前不是 PPO reward optimization + KL

当前实验不是：

```text
reward -> advantage -> PPO actor loss
```

再额外混一个 KL。

当前 actor 真正反传的是 reverse KL。

reward function 主要用于：

- 训练时记录 `curr_acc`
- 验证时做准确率评估

reward 不决定 actor 主梯度方向。

### 4.2 当前 reverse KL 的真实含义

当 `kl_loss_type=full` 且 `kl_topk>0` 时：

- student 先在原图 prompt 上计算 response token 的 top-k support
- privileged ref 在 crop prompt 上只对这套 support 打分
- loss 使用：

```text
KL(student || privileged)
```

不是 upstream distillation 常见的：

```text
KL(privileged || student)
```

也就是说，这里必须保留 reverse KL 方向，不能在后续改动里悄悄变成 forward KL。

---

## 5. Canonical Training Algorithm

当前实验的标准训练流程如下。

### 5.1 Dataset contract

数据集类是 `OPCDImagePairedVQADataset`。

每条样本至少要提供：

- `raw_prompt`
- `original_images`
- `crop_images`
- `extra_info.crop_image`
- `reward_model.ground_truth`

其中：

- rollout prompt 使用 `original_images`
- crop prompt 不在 dataset 里预先做第二套 tokenization
- crop prompt 由 trainer 内部动态重建

### 5.2 Per-step algorithm

每一步训练的标准语义是：

1. 用原图 prompt 生成 student response。
2. 用 `raw_prompt + crop_image` 动态重建 privileged prompt。
3. 把同一条 student response 同时拼接到：
   - student batch: `original prompt + same response`
   - privileged batch: `crop prompt + same response`
4. 如果需要 full top-k KL：
   - student 先生成 `kl_topk_indices`
   - privileged ref 只在这套 support 上计算 `exp_log_probs`
5. actor 以 `exp_log_probs` 为目标，执行 reverse-KL update。

### 5.3 Pseudocode

```python
batch = next(train_dataloader)

# 1. rollout on original image only
responses = actor.generate(original_prompt)

# 2. rebuild crop prompt inside trainer
crop_prompt = rebuild_from_raw_prompt(raw_prompt, crop_image)

# 3. reuse the same response in both branches
student_batch = compose(original_prompt, responses)
privileged_batch = compose(crop_prompt, responses)

# 4. optional top-k support for full KL
kl_topk_indices = actor.compute_support(student_batch)
exp_log_probs = ref.compute_log_prob(privileged_batch, kl_topk_indices)

# 5. reverse-KL actor update
loss = KL(student_log_prob || privileged_log_prob)
update_actor(loss)
```

---

## 6. Why Same-Response Reuse Matters

同轨迹 response reuse 是这个实验最重要的设计点之一，不能随便改。

原因是：

- 我们想比较的是“同一条行为轨迹”在不同视觉条件下的 token 分布差异。
- 如果 privileged 分支重新采样，student 和 privileged 的差异就混入了 rollout noise。
- 复用同一条 response，才能把差异主要归因于“原图 vs crop 图”的上下文条件不同。

所以后续实现里，不应该让 privileged 分支单独 rollout。

---

## 7. Why Crop Prompt Is Rebuilt In Trainer

当前实现把 crop prompt 的重建放在 trainer，而不是 dataset 里直接维护两套 prompt，原因是：

- 可以保持 dataset contract 简单
- 避免数据侧预先维护 duplicate prompt/tokenization
- 明确实验语义：student prompt 永远不变，privileged prompt 只是 trainer 内部的训练期派生视图

所以后续若没有明确收益，不要把 crop prompt 再下沉成另一套 dataset 主字段。

---

## 8. Current Recipe Contract

当前 recipe 的约定必须固定为：

- rollout prompt 不改写，仍然使用原图
- privileged prompt 仅在 trainer 内部由 `build_crop_messages_from_raw_prompt(...)` 重建
- privileged 分支只负责打分，不负责单独采样
- 同一条 rollout response 同时服务 student loss 和 privileged ref scoring
- recipe-specific 选项通过：
  - `++opcdimage.privileged_mode=crop`
  - `++opcdimage.on_policy_merge=True`
  进入配置

这些语义应该继续保留在 recipe 层，而不是塞回 `verl` 通用 config。

---

## 9. Why The Framework Was Restored

之前 `verl` core 混入了很多只属于 OPcDImage 实验的逻辑，例如：

- `stage=consolidate`
- `privileged_mode`
- `on_policy_merge`
- textgame / off-policy / textual experience 等旁支兼容路径

这些逻辑不应该继续留在通用框架里。

因此当前重构把实验实现迁回 `opcdimage_recipe/`，并把以下文件恢复到 `third_party/verl_upstream`：

- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/actor/dp_actor.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `_generated_ppo_*.yaml`

后续如果需要扩展实验，也应该优先扩展 recipe，而不是再次修改框架主线。

---

## 10. Relation To Local `opcd`

本地 `C:/Users/LU/Desktop/rl4image/opcd` 是文本 OPcD 的主要参考实现。

### 一致之处

- 同轨迹 response reuse
- privileged branch 只负责打分
- reverse KL 主导 actor 更新

### 差异之处

- `opcd` 的 privileged context 主要是文本 experience / privileged prompt
- `opcdimage` 的 privileged context 是 crop image prompt

所以 `opcdimage` 是“视觉条件版 OPcD”，而不是一套完全不同的训练目标。

---

## 11. Why Upstream Distillation Was Not Used As The Main Path

没有直接采用 upstream distillation 作为主实现，原因有两个。

### 11.1 目标方向不一致

upstream distillation 默认更接近 teacher-weighted forward KL。

而当前实验必须保留的是 reverse KL：

```text
KL(student || privileged)
```

### 11.2 support gather 语义不同

当前实验依赖的是：

1. student 先给出 response token 的 support
2. privileged 只在这套 support 上打分

这套 `kl_topk_indices` 驱动的 support gather 机制，是当前实验的核心部分之一。

因此最稳妥的方案是：

- 保持 upstream `verl` 原样
- 在 `opcdimage_recipe` 中实现 recipe-local trainer + actor

---

## 12. Current Code Map

当前推荐把下面这些文件视为实验主线：

- 训练脚本：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`
- recipe 入口：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/main_ppo.py`
- recipe trainer：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/ray_trainer.py`
- recipe actor：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/dp_actor.py`
- dataset：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/paired_vqa_dataset.py`
- crop prompt 重建：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/core.py`
- reward：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/reward_fn.py`
- 数据工具：
  - `C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/dataset_tools.py`

---

## 13. Working Constraints For Future Changes

后续改代码时，优先遵守下面这些约束。

### 13.1 Prefer recipe changes over framework changes

默认只改这些区域：

- `opcdimage_recipe/**`
- `examples/on_policy_distillation_trainer/opcdimage_consolidate.sh`
- `docs/**`
- `tests/**`

默认不要改：

- `verl/**`
- `third_party/verl_upstream/**`

除非确认存在“无法只在 recipe 层解决”的框架 bug。

### 13.2 If framework changes are unavoidable

如果确实必须改 `verl` 框架层，需要满足：

- 先明确说明为什么 recipe 层无法解决
- 改动必须最小化
- 不得引入只属于 OPcDImage 的实验分支语义
- 改完后要重新检查是否偏离 upstream 公共行为

### 13.3 All changes must preserve the experiment math

任何改动都必须符合当前实验的真实计算方法，不允许在不说明的情况下改变以下内容：

- student 只看 `original_images`
- privileged 只看 crop prompt
- privileged 不单独 rollout
- response 必须同轨迹复用
- actor 主更新目标必须仍然是 reverse KL
- 不得把当前目标偷换成 PPO reward 主导
- 不得把当前目标偷换成 forward KL / teacher-weighted distillation

---

## 14. Invariants That Should Stay True

后续实现里，下面这些不变量应该继续成立：

- `actor_rollout_ref.ref.model.path` 才是 privileged ref 模型路径
- full top-k KL 路径下，`return_all_logits=True` 与 fused kernels 当前不兼容
- `trainer.balance_batch` 应在 actor 更新前生效
- reward manager 应尊重 `reward.reward_manager.*`
- custom reward function 应继续通过 `reward.custom_reward_function.*` 接入

---

## 15. Scope Deliberately Kept Out

当前 recipe 刻意不再支持这些旧分支：

- textgame
- textual experience injection
- generate/load off-policy batches
- seqkd branch
- exp_learner side paths

如果以后确实要恢复，也应该：

- 先确认实验目标是否真的需要
- 再在 recipe 层独立扩展
- 不要再次改写 `verl` core

---

## 16. Suggested First Check In Future Sessions

后续继续开发或排查问题时，建议优先检查：

1. 当前改动是否只发生在 recipe 层。
2. 是否仍然保持“original rollout + crop scoring + same response reuse”。
3. actor 更新是否仍然是 reverse KL。
4. 是否误把 privileged 分支改成单独生成。
5. 是否误把 reward 重新接回 PPO 主梯度。

如果以上任何一条被破坏，应该先停下来重新对齐实验语义，再继续实现。
