# opcdimage Hugging Face 数据集工作流

## 1. 目标

为了让 `opcdimage` 迁移到 Linux 机器后可以直接跑实验，当前数据流改成了两级：

1. 本地如果已经有原始 `ZwZ-RL-VQA-mini`，可以直接本地生成 prepared dataset。
2. 如果本地没有原始数据，则自动从 Hugging Face dataset 下载最小可训练子集。

默认使用的 Hugging Face dataset repo id 为：

- `muyuho/opcdmini`

也可以通过环境变量覆盖：

- `OPCDIMAGE_HF_DATASET_REPO_ID`
- `OPCDIMAGE_PROXY`

---

## 2. 当前新增的脚本

### 2.1 导出 HF-ready 子集

- [hf_data_tools.py](C:\Users\LU\Desktop\rl4image\opcdimage\opcdimage_recipe\hf_data_tools.py) `export`

作用：

- 从本地 prepared dataset 中导出 HF-ready 目录
- 复制训练真正需要的 full / crop 图像子集
- 将 parquet 中的本地绝对路径改成相对路径

导出后的目录结构是：

- `README.md`
- `summary.json`
- `prepared/train.parquet`
- `prepared/val.parquet`
- `prepared/train.jsonl`
- `prepared/val.jsonl`
- `images/original_images/*`
- `images/crop/*`

### 2.2 上传到 HF dataset

- [upload_hf_dataset.py](C:\Users\LU\Desktop\rl4image\opcdimage\opcdimage_recipe\upload_hf_dataset.py)

作用：

- 创建 dataset repo
- 上传本地导出的 HF-ready 目录
- 支持两种模式：
  - `commit-batches`：按批次提交，适合小规模补传
  - `large-folder`：调用 `huggingface_hub` 的大目录续传逻辑，适合 Linux 机器长期断点续传

注意：

- 这一步需要本机已经登录 Hugging Face，或者设置有效 token
- 如果命中 Hugging Face 的每小时 commit 限流，优先改用 `--mode large-folder`

### 2.3 自动下载并本地重写

- [hf_data_tools.py](C:\Users\LU\Desktop\rl4image\opcdimage\opcdimage_recipe\hf_data_tools.py) `download`

作用：

1. 当本地 `train.parquet` / `val.parquet` 不存在时，从 HF dataset 下载：
   - `prepared/*`
   - `images/**/*`
   - `summary.json`
2. 将下载后的相对路径 manifest 重写成本地绝对路径 manifest
3. 自动解压 `original_images.tar.gz` 和 `crop_images.tar.gz`
3. 在目标目录下生成：
   - `train.parquet`
   - `val.parquet`
   - `train.jsonl`
   - `val.jsonl`
   - `.hf_dataset_source.json`

这样训练脚本和 dataset 类就不需要关心 Hub 上的相对路径格式。

如果设置了 `OPCDIMAGE_PROXY`，脚本会自动同步到：

- `HTTP_PROXY`
- `HTTPS_PROXY`
- `ALL_PROXY`

---

## 3. 训练脚本如何工作

### 3.1 数据准备入口

- [prepare_opcdimage_data.sh](C:\Users\LU\Desktop\rl4image\opcdimage\examples\on_policy_distillation_trainer\prepare_opcdimage_data.sh)

逻辑是：

1. 如果 `data/opcdimage_qwen3vl4b/train.parquet` 和 `val.parquet` 已存在，直接返回
2. 如果本地存在：
   - `../ZwZ-RL-VQA-mini/train_crop_clean.csv`
   - `../ZwZ-RL-VQA-mini/`
   
   就本地生成 prepared dataset
3. 否则自动调用 `hf_data_tools.py download` 从 Hub 下载

这里默认直接使用当前训练环境里的 `python3`，不要求额外安装 `uv`。

如果需要走代理，可以先设置：

```bash
export OPCDIMAGE_PROXY="http://127.0.0.1:7897"
```

### 3.2 训练入口

- [opcdimage_consolidate.sh](C:\Users\LU\Desktop\rl4image\opcdimage\examples\on_policy_distillation_trainer\opcdimage_consolidate.sh)

逻辑是：

1. 先检查 `DATA_DIR/train.parquet` 和 `DATA_DIR/val.parquet`
2. 如果缺失，则自动调用 `hf_data_tools.py download`
3. 数据准备完后，再进入 `trainer.stage=consolidate` 的主训练流程

这意味着：

- Linux 机器上如果没有原始数据，也可以直接跑训练脚本
- 只要能访问公开 HF dataset，就能自动把所需子集拉下来

---

## 4. 当前本地已导出的 HF-ready 目录

当前推荐在本地维护一份待上传目录，例如：

- `C:\Users\LU\Desktop\rl4image\opcdimage\hf_dataset\opcdmini_publish`

目录里已经包含：

- prepared parquet/jsonl
- 训练需要的 full/crop 图像子集
- README
- summary

其中：

- prepared parquet 里的 `images`
- `extra_info.original_image` / `extra_info.crop_image`

都已经被改写成 Hub 友好的相对路径，例如：

- `images/original_images/sa_9233167.jpg`
- `images/crop/sa_9233167__1275_157_2167_449__3845722d9ef8.png`

---

## 5. Linux 机器上的推荐用法

### 方案 A：只跑训练

直接执行：

```bash
export OPCDIMAGE_PROXY="http://127.0.0.1:7897"
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

如果本地没有 prepared dataset，它会自动尝试从 Hugging Face 下载。

### 方案 B：先单独拉数据

```bash
bash examples/on_policy_distillation_trainer/prepare_opcdimage_data.sh
```

然后再启动训练。

### 方案 C：切换到别的 dataset repo

```bash
export OPCDIMAGE_HF_DATASET_REPO_ID="your-name/your-opcdmini-repo"
bash examples/on_policy_distillation_trainer/opcdimage_consolidate.sh
```

### 方案 D：在 Linux 机器续传 HF 数据集

推荐优先使用大目录模式：

```bash
export HF_TOKEN="your-token"
export OPCDIMAGE_PROXY="http://127.0.0.1:7897"
uv run --no-project --with huggingface_hub python opcdimage_recipe/upload_hf_dataset.py \
  --local-dir hf_dataset/opcdmini_publish/archives \
  --repo-id muyuho/opcdmini \
  --exist-ok \
  --skip-existing \
  --mode large-folder \
  --num-workers 8
```

如果只想补传某一部分，也可以加 pattern，例如只传 crop：

```bash
uv run --no-project --with huggingface_hub python opcdimage_recipe/upload_hf_dataset.py \
  --local-dir hf_dataset/opcdmini_publish/archives \
  --repo-id muyuho/opcdmini \
  --exist-ok \
  --skip-existing \
  --mode large-folder \
  --allow-pattern "crop_images.tar.gz"
```

也可以直接使用封装脚本：

- [upload_opcdimage_hf_dataset.sh](C:\Users\LU\Desktop\rl4image\opcdimage\examples\on_policy_distillation_trainer\upload_opcdimage_hf_dataset.sh)

例如只续传剩余 crop 图：

```bash
export HF_TOKEN="your-token"
export OPCDIMAGE_PROXY="http://127.0.0.1:7897"
export OPCDIMAGE_HF_ONLY_CROP=true
bash examples/on_policy_distillation_trainer/upload_opcdimage_hf_dataset.sh
```

---

## 6. 当前状态

当前已经完成：

- HF-ready 子集导出
- 自动下载脚本
- 训练入口自动检测与下载
- 面向 Linux 的无原始数据运行路径

当前正在进行：

- 目标 repo：`muyuho/opcdmini`
- prepared 文件与部分图像已经上传
- 仍有剩余图像需要续传

上传过程中目前遇到的主要约束是：

- Hugging Face dataset repo 的每小时 commit 数限制

因此后续续传的推荐方式是：

- 减少小批次 commit
- 在 Linux 机器上优先使用 `large-folder` 模式续传
