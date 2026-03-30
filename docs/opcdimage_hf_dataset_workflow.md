# opcdimage Hugging Face 数据集工作流

## 1. 当前原则

现在的数据流刻意做了简化，只保留一条默认路径：

1. 数据统一来自 Hugging Face dataset repo：
   - `muyuho/opcdmini`
2. 本地数据准备只做三件事：
   - 下载 `prepared/*`
   - 下载并解压图片压缩包
   - 跑一遍数据校验

不再把“本地原始 CSV prepare”当成默认训练入口的一部分，也不再保留多套兜底逻辑。

---

## 2. HF repo 里有什么

当前 `muyuho/opcdmini` 里包含：

- `prepared/train.parquet`
- `prepared/val.parquet`
- `prepared/train.jsonl`
- `prepared/val.jsonl`
- `original_images.tar.gz`
- `crop_images.tar.gz`
- `summary.json`
- `README.md`

其中：

- `prepared/*` 是训练直接读取的样本表
- 两个 `tar.gz` 是图片内容

---

## 3. parquet 里的路径现在怎么存

现在 parquet 里保留的是相对路径，不会在下载后改写成绝对路径。

例如：

- `images/original_images/sa_9233167.jpg`
- `images/crop/sa_9233167__1275_157_2167_449__3845722d9ef8.png`

对应字段包括：

- `original_images`
- `crop_images`
- `extra_info.original_image`
- `extra_info.crop_image`

这样做的好处是：

- HF repo 内容保持可移植
- parquet 在不同机器上不需要重新生成
- 数据目录整体拷走后仍然成立

运行时再由 dataset 按 `train.parquet` 所在目录解析这些相对路径。

---

## 4. 运行时如何解析这些相对路径

训练时使用的是：

- [paired_vqa_dataset.py](C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/paired_vqa_dataset.py)

逻辑是：

1. 读取 `prepared/train.parquet` / `prepared/val.parquet`
2. 取 `parquet` 文件所在目录作为 `dataset_root`
3. 如果 parquet 位于 `prepared/` 下，就把上一级数据集目录作为真正的 `dataset_root`
4. 把样本里的相对图片路径解析成：
   - `${dataset_root}/images/original_images/...`
   - `${dataset_root}/images/crop/...`
5. 再把解析后的绝对路径送给 processor / trainer

所以：

- repo 内 manifest 保持相对路径
- 真正读图时仍然拿到本机可访问的绝对路径

---

## 5. 下载脚本现在做什么

核心入口是：

- [hf_data_tools.py](C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/hf_data_tools.py) 的 `download`

它现在只做这些事：

1. 从 `muyuho/opcdmini` 下载：
   - `prepared/*`
   - `summary.json`
   - `original_images.tar.gz`
   - `crop_images.tar.gz`
   - 如果 repo 里已经展开了 `images/*`，也允许直接拉
2. 如果本地还没有 `images/original_images` 和 `images/crop`，就自动解压两个压缩包
3. 保留下载得到的 repo 目录结构：
   - `${output_dir}/prepared/train.parquet`
   - `${output_dir}/prepared/val.parquet`
   - `${output_dir}/images/...`
4. 写一个 `.hf_dataset_source.json` 标记文件

它不会再重写 parquet 里的图片路径。

---

## 6. 训练脚本现在怎么准备数据

默认入口是：

- [opcdimage_consolidate.sh](C:/Users/LU/Desktop/rl4image/opcdimage/examples/on_policy_distillation_trainer/opcdimage_consolidate.sh)

数据准备逻辑已经化简成：

1. 如果 `${DATA_DIR}/prepared/train.parquet` 和 `${DATA_DIR}/prepared/val.parquet` 以及数据来源标记都已经存在，直接进入训练
2. 否则执行：

```bash
python3 opcdimage_recipe/hf_data_tools.py download \
  --output-dir data/opcdimage_qwen3vl4b \
  --repo-id muyuho/opcdmini
```

3. 下载并解压后，再执行：

```bash
python3 opcdimage_recipe/data_tools.py validate \
  --train-file data/opcdimage_qwen3vl4b/prepared/train.parquet \
  --val-file data/opcdimage_qwen3vl4b/prepared/val.parquet
```

也就是说，当前默认训练链路就是：

- `download -> extract -> validate -> train`

---

## 7. 单独准备数据的命令

如果你想先把数据准备好，再单独跑训练，可以直接执行：

```bash
bash examples/on_policy_distillation_trainer/prepare_opcdimage_data.sh
```

这个脚本现在也是极简版本，本质只做：

```bash
python3 opcdimage_recipe/hf_data_tools.py download ...
python3 opcdimage_recipe/data_tools.py validate ...
```

---

## 8. 导出和上传怎么理解

如果你后面要重新发布数据集，仍然可以用：

- [hf_data_tools.py](C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/hf_data_tools.py) `export`
- [upload_hf_dataset.py](C:/Users/LU/Desktop/rl4image/opcdimage/opcdimage_recipe/upload_hf_dataset.py)

这里的 `export` 会：

- 复制训练实际需要的 full / crop 图片
- 在导出后的 parquet 里写相对路径

也就是说，HF repo 上存的是“相对路径 manifest + 图片文件/压缩包”，而不是机器相关的绝对路径。

---

## 9. 推荐使用方式

Linux 机器上推荐直接这样用：

```bash
git clone --recurse-submodules https://github.com/MuYuan88ya/opcdimage.git
cd opcdimage
bash examples/on_policy_distillation_trainer/prepare_opcdimage_data.sh
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

如果你不想单独准备数据，也可以直接：

```bash
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

因为训练脚本会自动触发前面的下载和校验。

---

## 10. 当前设计的关键结论

现在这套实现可以记成一句话：

- HF 上的 parquet 只保存相对路径
- 本地准备数据只负责下载和解压
- 运行时由 dataset 把相对路径解析成本地绝对路径

这样数据集本身更干净，训练入口也更容易读和维护。
