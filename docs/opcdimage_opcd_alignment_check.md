# opcdimage 与 opcd 训练逻辑对齐检查

## 1. 结论

`opcdimage` 当前版本已经 **对齐了 opcd 的主训练语义**，具体是：

1. 训练入口仍然是 `trainer-native consolidate`，不是 agent-loop teacher sidecar。
2. student 只在 plain batch 上 rollout。
3. privileged batch 不是单独的数据流，而是在同一个 trainer step 里，从当前 batch 派生出来。
4. privileged batch 的 response 不是 teacher 生成的，而是直接复用 student rollout 出来的 response。
5. ref model 对 privileged batch 计算 `exp_log_probs`。
6. actor update 直接消费 `exp_log_probs`，按 `opcd` 风格做 KL-style consolidate。

所以如果问题是：

> `opcdimage` 的训练主循环有没有对齐 `opcd`？

答案是：

**主路径已经对齐。**

但如果问题是：

> `opcdimage` 是否已经把 `opcd` 的所有 trainer 分支、验证分支、文本经验分支都彻底视觉化改造完？

答案是：

**还没有。当前只保证主训练路径对齐。**

这也是当前版本的正确定位：

- 已经是 `opcd-style Vision-OPCD`
- 但还不是“所有辅助路径都完全视觉化”的最终版

---

## 2. 对齐了哪些核心步骤

### 2.1 trainer 落点对齐

`opcd` 的关键不是“有 teacher”，而是：

- 在 trainer 里做 `stage=consolidate`
- 在 step 内构造 privileged batch
- 用 ref model 计算 privileged reference logprob
- actor 直接吃这份 privileged logprob

当前 `opcdimage` 也是这一路径。

对应文件：

- `opcd`: [ray_trainer.py](C:\Users\LU\Desktop\rl4image\opcd\verl\verl\trainer\ppo\ray_trainer.py)
- `opcdimage`: [ray_trainer.py](C:\Users\LU\Desktop\rl4image\opcdimage\verl\trainer\ppo\ray_trainer.py)

当前训练脚本也明确走这条路径：

- [opcdimage_consolidate.sh](C:\Users\LU\Desktop\rl4image\opcdimage\examples\on_policy_distillation_trainer\opcdimage_consolidate.sh)

核心配置是：

- `trainer.stage=consolidate`
- `trainer.privileged_mode=crop`
- `trainer.on_policy_merge=True`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_type=full`
- `actor_rollout_ref.actor.kl_topk=256`

这说明当前不是 distillation plugin 主线，而是 trainer-native consolidate 主线。

### 2.2 privileged batch 的构造方式对齐

`opcd` 的做法不是维护两套 dataloader，而是在一个 step 里从当前 batch 派生出：

- `gen_batch_with_exp`
- `batch_with_exp`

当前 `opcdimage` 延续了这条逻辑，只是把 textual `experience/system prompt` 换成了 visual `crop prompt`。

在 `opcdimage` 里，这一步的对应实现是：

- 先从普通 `gen_batch` / `batch` 复制出 `gen_batch_with_exp` / `batch_with_exp`
- 当 `trainer.privileged_mode=crop` 时，不再往 `raw_prompt` 里插入 textual experience
- 而是直接取 dataset 里的 `raw_prompt_with_crop`

对应代码：

- [ray_trainer.py](C:\Users\LU\Desktop\rl4image\opcdimage\verl\trainer\ppo\ray_trainer.py)

这一点和 `opcd` 的结构是对齐的，只是 privileged context 的载体不同：

- `opcd`：system prompt / experience
- `opcdimage`：crop image

### 2.3 student rollout 与 teacher/ref scoring 的关系对齐

`opcd` 的关键语义是：

- student 先 rollout
- teacher/ref 不自己采样
- privileged batch 复用 student response

当前 `opcdimage` 也保持了这个语义。

trainer 先在 plain full-image batch 上 rollout，然后再在 `on_policy_merge` 分支中：

1. 取 `gen_batch_with_exp` 里的 privileged prompt
2. 取 `gen_batch_output` 里的 student response
3. 重建 `batch_with_exp`
4. 再调用 `ref_policy_wg.compute_ref_log_prob(batch_with_exp)`

这和 `opcd` 的主训练语义是一致的。

### 2.4 actor 的 KL 更新路径对齐

当前 `opcdimage` 没有继续沿用之前 `selfrl_image` 那条：

- `teacher_logprobs`
- `forward_kl_topk`
- distillation plugin

而是直接采用 `opcd` 风格 actor KL 路径。

对应实现：

- [dp_actor.py](C:\Users\LU\Desktop\rl4image\opcdimage\verl\workers\actor\dp_actor.py)

这条路径里：

1. actor 拿到 student 的 `log_prob`
2. trainer 传进来的 privileged `exp_log_probs`
3. 在 actor 里直接执行 `kl_penalty(logprob=student, ref_logprob=exp, kl_penalty="full")`

这就是 `opcd` consolidate 的核心训练方式。

### 2.5 top-k support merge 对齐

`opcd` 默认不是简单 sampled-token KL，而是：

- `full`
- `topk`
- 支持 `on_policy_merge`

当前 `opcdimage` 也保留了这条路径：

1. trainer 先让 actor 计算 top-k
2. 若开启 merge，再让 ref 补 support
3. 合并后写回 `batch_with_exp.batch["kl_topk_indices"]`
4. ref model 在这份 support 上计算 `exp_log_probs`
5. actor 再直接消费

因此在 loss 形态上，当前实现和 `opcd` 主线是对齐的，不再是之前 recipe 版本那种插件式 top-k distillation。

---

## 3. 当前 Vision-OPCD 数据接口是如何映射到 opcd 路径的

### 3.1 dataset 行接口

当前数据行约定为：

- `prompt`
- `prompt_with_crop`
- `images`
- `crop_images`
- `ground_truth`
- `extra_info`

对应准备脚本：

- [data_tools.py](C:\Users\LU\Desktop\rl4image\opcdimage\opcdimage_recipe\data_tools.py)

### 3.2 dataset 在 trainer 中提供两套 prompt

dataset 不再像之前 recipe 方案那样把 teacher prompt 只挂进 agent loop，而是显式提供：

- `raw_prompt`
- `raw_prompt_with_crop`

对应实现：

- [paired_vqa_dataset.py](C:\Users\LU\Desktop\rl4image\opcdimage\opcdimage_recipe\paired_vqa_dataset.py)

这使得 trainer 可以直接像 `opcd` 处理 `raw_prompt` / `batch_with_exp` 一样处理视觉 privileged prompt。

### 3.3 为什么 `prompt` 和 `prompt_with_crop` 现在文本相同

这是当前版本的一个刻意设计，不是 bug。

目前：

- `prompt` 和 `prompt_with_crop` 的文本内容相同
- 二者真正的差异在视觉输入：
  - `images` 是 full image
  - `crop_images` 是 crop image

也就是说，当前 privileged information 完全来自视觉上下文，而不是额外文本提示。

这正对应了本项目的研究目标：

- 把 crop 看作视觉 privileged context
- 而不是再额外给 teacher 文本特权

---

## 4. 还没有完全对齐的部分

这里需要明确写清，避免误以为已经 100% 完成了所有旁路。

### 4.1 验证路径还没有全部视觉化改造

训练主路径已经按 `crop` 改造，但 `ray_trainer.py` 里仍然保留了大量来自 `opcd` 的文本 `experience` / `train_system_prompt` 分支。

目前脚本通过下面的方式绕开这些未完全改造的旁路：

- `trainer.val_before_train=False`
- `trainer.test_freq` 设得很大
- `trainer.experience_path=''`

这意味着：

- 主训练路径可用
- 但不是所有验证/旁路分支都已经换成视觉 privileged 语义

### 4.2 现在仍然保留了很多 `opcd` 文本经验代码

这不是逻辑错误，但会带来两个结果：

1. 代码里还有一些与 `experience` 相关的分支不会在当前主实验触发
2. 如果后续开启这些分支，需要继续补视觉适配

### 4.3 端到端训练还没有在完整环境下实跑

当前已经完成：

- 代码静态编译
- 数据准备
- 数据校验
- 评估脚本跑通

但还没有完成：

- 真正的 `FSDP + vLLM + consolidate` 端到端训练验证

所以当前能下的结论是：

- 代码结构已经按 `opcd` 定死
- 主训练语义也已经对齐
- 但仍需完整环境下做一次真实 smoke run

---

## 5. 当前主训练流程

下面是 `opcdimage` 当前版本的真实训练流程。

### Step 1. dataset 读取 paired sample

每条样本同时提供：

- full image prompt
- crop image prompt
- ground truth

### Step 2. trainer 构造 plain batch

student rollout 使用：

- `raw_prompt`
- `images`

### Step 3. trainer 在同一步里构造 privileged batch

当 `trainer.privileged_mode=crop` 时，trainer 从当前 batch 派生：

- `gen_batch_with_exp`
- `batch_with_exp`

但这里的 “with_exp” 在视觉版里实际表示：

- `with_crop`

它使用的是：

- `raw_prompt_with_crop`
- `crop_images`

### Step 4. student 在 plain batch 上 rollout

student 生成 response。

### Step 5. trainer 用 privileged prompt + student response 重建序列

这一步对齐 `opcd` 的 `batch_with_exp` 重建方式：

- prompt 来自 privileged batch
- response 来自 student rollout

### Step 6. ref model 计算 privileged logprob

trainer 调用：

- `ref_policy_wg.compute_ref_log_prob(batch_with_exp)`

然后把：

- `ref_log_prob`

改名并写入：

- `exp_log_probs`

### Step 7. actor 直接消费 `exp_log_probs`

actor 计算 student 当前 logprob，再与 `exp_log_probs` 做 `full/topk` KL-style consolidate 更新。

这一步不经过 agent-loop teacher，也不经过 recipe distillation plugin。

---

## 6. 和旧版 selfrl_image Vision-OPCD 的根本区别

旧版 `selfrl_image` 方案是：

- rollout 仍用官方 trainer
- teacher 作为 agent-loop sidecar
- teacher 通过 vLLM `prompt_logprobs` 给 student 轨迹打分

当前 `opcdimage` 方案则是：

- trainer-native consolidate
- privileged batch 在 trainer 内构造
- ref model 在 privileged batch 上算 `exp_log_probs`
- actor 直接按 `opcd` KL 路径更新

所以 `opcdimage` 不是旧版的重命名，而是方法落点真正改到了 `opcd` 那条主干上。

---

## 7. 当前我对实现状态的判断

### 可以确认对齐的部分

- trainer-native consolidate 主线
- step 内派生 privileged batch
- plain rollout + privileged ref scoring
- `full/topk` actor KL consolidate
- `on_policy_merge=True` 的主语义

### 还需要后续补强的部分

- 验证分支彻底视觉化
- 端到端真实训练 smoke run
- 如有需要，再补 RLVR reward 与 ref/base anti-drift 的组合实验

---

## 8. 最终判断

如果以“是否沿着 `opcd` 的核心训练路径实现 Vision-OPCD”为标准，当前 `opcdimage` 的答案是：

**是。**

如果以“是否已经把 `opcd` 的所有文本经验分支、验证分支、辅助路径全部无死角改造成视觉版本”为标准，当前答案是：

**还没有完全做完。**

因此最准确的描述是：

**`opcdimage` 当前已经是 trainer-native、opcd-style 的 Vision-OPCD 主线实现；主训练逻辑已对齐，旁路分支仍待继续清理和验证。**
