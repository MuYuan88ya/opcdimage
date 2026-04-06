from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from opcdimage_recipe.core import normalize_problem

HF_DATASET_REPO_ID = os.environ.get("OPCDIMAGE_HF_DATASET_REPO_ID", "muyuho/opcdmini")
HF_PREPARED_SUBDIR = "prepared"
HF_ARCHIVE_FILENAMES = ["original_images.tar.gz", "crop_images.tar.gz"]


def _configure_proxy_from_env() -> None:
    proxy = os.environ.get("OPCDIMAGE_PROXY")
    if not proxy:
        return
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)
    os.environ.setdefault("ALL_PROXY", proxy)


@dataclass(frozen=True)
class PreparedSample:
    problem: str
    original_images: list[str]
    crop_images: list[str]
    bbox: list[float]
    answer: str
    ability: str
    data_source: str
    reward_model: dict[str, str]
    extra_info: dict[str, Any]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path}")


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        raise ValueError("Expected a non-empty list value, but got None.")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Expected a non-empty list string.")
        parsed = ast.literal_eval(stripped)
        if not isinstance(parsed, list):
            raise TypeError(f"Expected list literal, but got: {type(parsed)}")
        return parsed
    raise TypeError(f"Unsupported list field type: {type(value)}")


def parse_bbox(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if value is None:
        raise ValueError("bbox is required.")
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if not numbers:
            raise ValueError(f"Failed to parse bbox from: {value}")
        return [float(x) for x in numbers]
    raise TypeError(f"Unsupported bbox field type: {type(value)}")


def ensure_single_item_list(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, list):
        items = value
    elif hasattr(value, "tolist"):
        items = value.tolist()
    else:
        raise TypeError(f"Unsupported {field_name} container type: {type(value)}")

    if len(items) != 1:
        raise ValueError(f"each sample must contain exactly one item in '{field_name}'.")
    return items


def resolve_dataset_path(dataset_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    abs_path = path if path.is_absolute() else (dataset_root / path)
    abs_path = abs_path.resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {abs_path}")
    return abs_path


def relativize_dataset_path(dataset_root: Path, path_str: str) -> str:
    abs_path = resolve_dataset_path(dataset_root, path_str)
    try:
        rel_path = abs_path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path {abs_path} is not under dataset root {dataset_root}") from exc
    return rel_path.as_posix()


def infer_dataset_root_from_manifest(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    parent = manifest_path.parent
    if parent.name == "prepared":
        candidate = parent.parent
        if (candidate / "images").exists():
            return candidate
    return parent


def load_image_size(image_path: str, image_size_cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    cached = image_size_cache.get(image_path)
    if cached is not None:
        return cached

    with Image.open(image_path) as image:
        size = image.size
    image_size_cache[image_path] = size
    return size


def compute_bbox_area_ratio(bbox: list[float], image_width: int, image_height: int) -> float:
    if len(bbox) < 4:
        raise ValueError(f"bbox must contain at least four values, got {bbox}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image size must be positive, got {(image_width, image_height)}")

    x1, y1, x2, y2 = bbox[:4]
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    bbox_area = bbox_width * bbox_height
    image_area = float(image_width * image_height)
    return bbox_area / image_area


def assign_split(group_key: str, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if score < val_ratio else "train"


def build_sample(
    row: pd.Series,
    dataset_root: Path,
    val_ratio: float,
    seed: int,
    image_size_cache: dict[str, tuple[int, int]],
) -> tuple[str, PreparedSample]:
    original_images = parse_list(row["original_images"])
    crop_images = parse_list(row["crop_images"])
    if len(original_images) != 1:
        raise ValueError(f"Expected one full image per sample, got {len(original_images)}")
    if len(crop_images) != 1:
        raise ValueError(f"Expected one crop image per sample, got {len(crop_images)}")

    original_image = resolve_dataset_path(dataset_root, original_images[0])
    crop_image = resolve_dataset_path(dataset_root, crop_images[0])
    original_image_rel = relativize_dataset_path(dataset_root, original_images[0])
    crop_image_rel = relativize_dataset_path(dataset_root, crop_images[0])
    bbox = parse_bbox(row["bbox"])
    normalized_problem = normalize_problem(str(row["problem"]))
    answer = str(row["answer"]).strip()
    split = assign_split(original_image_rel, val_ratio=val_ratio, seed=seed)
    original_width, original_height = load_image_size(str(original_image), image_size_cache=image_size_cache)
    crop_width, crop_height = load_image_size(str(crop_image), image_size_cache=image_size_cache)
    bbox_area_ratio = compute_bbox_area_ratio(bbox, image_width=original_width, image_height=original_height)

    sample = PreparedSample(
        problem=normalized_problem,
        original_images=[original_image_rel],
        crop_images=[crop_image_rel],
        bbox=bbox,
        answer=answer,
        ability=str(row.get("ability", "")),
        data_source="opcdimage_vqa",
        reward_model={"ground_truth": answer},
        extra_info={
            "split": split,
            "question": normalized_problem.replace("<image>", "", 1).strip(),
            "raw_problem": str(row["problem"]),
            "original_image": original_image_rel,
            "crop_image": crop_image_rel,
            "original_width": original_width,
            "original_height": original_height,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "original_megapixels": (original_width * original_height) / 1_000_000.0,
            "crop_area_ratio": (crop_width * crop_height) / float(original_width * original_height),
            "bbox_area_ratio": bbox_area_ratio,
            "source_dataset": row.get("data_source", ""),
        },
    )
    return split, sample


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_prepare(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_table(input_path)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    image_size_cache: dict[str, tuple[int, int]] = {}

    for row_id, row in frame.iterrows():
        split, sample = build_sample(
            row,
            dataset_root=dataset_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
            image_size_cache=image_size_cache,
        )
        record = asdict(sample)
        record["sample_id"] = row_id
        if split == "train":
            train_records.append(record)
        else:
            val_records.append(record)

    if not train_records or not val_records:
        raise ValueError("Expected both train and val splits to be non-empty.")

    train_frame = pd.DataFrame(train_records)
    val_frame = pd.DataFrame(val_records)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    train_frame.to_parquet(train_path, index=False)
    val_frame.to_parquet(val_path, index=False)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)

    all_records = train_records + val_records
    all_extra_info = [record["extra_info"] for record in all_records]
    summary = {
        "input": str(input_path),
        "dataset_root": str(dataset_root),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "unique_images_cached": len(image_size_cache),
        "mean_original_megapixels": sum(info["original_megapixels"] for info in all_extra_info) / len(all_extra_info),
        "mean_crop_area_ratio": sum(info["crop_area_ratio"] for info in all_extra_info) / len(all_extra_info),
        "mean_bbox_area_ratio": sum(info["bbox_area_ratio"] for info in all_extra_info) / len(all_extra_info),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_frame(frame: pd.DataFrame, split_name: str, dataset_root: Path) -> dict[str, int]:
    required_columns = {
        "problem",
        "original_images",
        "crop_images",
        "bbox",
        "data_source",
        "ability",
        "answer",
        "reward_model",
        "extra_info",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise KeyError(f"{split_name}: missing columns {sorted(missing)}")

    prompt_contains_bbox_instruction = 0
    image_ids = set()

    for row in frame.to_dict("records"):
        problem = row["problem"]
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"{split_name}: problem must be a non-empty string.")
        images = ensure_single_item_list(row["original_images"], "original_images")
        crop_images = ensure_single_item_list(row["crop_images"], "crop_images")
        reward_model = row["reward_model"]
        extra_info = row["extra_info"]

        resolve_dataset_path(dataset_root, images[0])
        resolve_dataset_path(dataset_root, crop_images[0])
        if reward_model.get("ground_truth") != row["answer"]:
            raise ValueError(f"{split_name}: reward_model.ground_truth mismatch.")

        if "red bounding box" in problem.lower():
            prompt_contains_bbox_instruction += 1

        image_key = extra_info["original_image"]
        image_ids.add(image_key)

        if extra_info["crop_image"] != crop_images[0]:
            raise ValueError(f"{split_name}: extra_info.crop_image mismatch.")
        if extra_info["original_image"] != images[0]:
            raise ValueError(f"{split_name}: extra_info.original_image mismatch.")
        if extra_info["original_width"] <= 0 or extra_info["original_height"] <= 0:
            raise ValueError(f"{split_name}: invalid original image size in extra_info.")
        if extra_info["crop_width"] <= 0 or extra_info["crop_height"] <= 0:
            raise ValueError(f"{split_name}: invalid crop image size in extra_info.")
        if not (0.0 < extra_info["crop_area_ratio"] <= 1.0):
            raise ValueError(f"{split_name}: crop_area_ratio out of range: {extra_info['crop_area_ratio']}")
        if not (0.0 <= extra_info["bbox_area_ratio"] <= 1.0):
            raise ValueError(f"{split_name}: bbox_area_ratio out of range: {extra_info['bbox_area_ratio']}")

    if prompt_contains_bbox_instruction > 0:
        raise ValueError(f"{split_name}: found {prompt_contains_bbox_instruction} prompts still containing bbox text.")

    return {"rows": len(frame), "unique_original_images": len(image_ids)}


def run_validate(args: argparse.Namespace) -> None:
    train_file = args.train_file.resolve()
    val_file = args.val_file.resolve()
    train_frame = pd.read_parquet(train_file)
    val_frame = pd.read_parquet(val_file)

    train_stats = validate_frame(train_frame, "train", dataset_root=infer_dataset_root_from_manifest(train_file))
    val_stats = validate_frame(val_frame, "val", dataset_root=infer_dataset_root_from_manifest(val_file))

    train_images = {row["extra_info"]["original_image"] for row in train_frame.to_dict("records")}
    val_images = {row["extra_info"]["original_image"] for row in val_frame.to_dict("records")}
    overlap = sorted(train_images & val_images)
    if overlap:
        raise ValueError(f"train/val leakage detected on original images: {overlap[:5]}")

    print({"train": train_stats, "val": val_stats, "cross_split_original_image_overlap": 0})


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _relativize_export_path(path_str: str, source_root: Path) -> str:
    path = Path(path_str).resolve()
    try:
        rel = path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Path {path} is not under source root {source_root}") from exc
    return str(Path("images") / rel).replace("\\", "/")


def _rewrite_export_extra_info(extra_info: dict[str, Any], source_root: Path) -> dict[str, Any]:
    rewritten = dict(extra_info)
    for key in ["original_image", "crop_image"]:
        if key in rewritten and rewritten[key]:
            rewritten[key] = _relativize_export_path(str(rewritten[key]), source_root=source_root)
    return rewritten


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _export_split(split: str, source_dir: Path, output_dir: Path, source_root: Path) -> None:
    src = source_dir / f"{split}.parquet"
    df = pd.read_parquet(src)
    copied: set[str] = set()

    def rewrite_list(paths: list[str]) -> list[str]:
        rewritten = []
        for item in paths:
            rel = _relativize_export_path(str(item), source_root=source_root)
            rewritten.append(rel)
            if rel not in copied:
                _copy_if_needed(Path(item), output_dir / rel)
                copied.add(rel)
        return rewritten

    df["original_images"] = df["original_images"].map(rewrite_list)
    df["crop_images"] = df["crop_images"].map(rewrite_list)
    df["extra_info"] = df["extra_info"].map(lambda info: _rewrite_export_extra_info(info, source_root=source_root))

    prepared_dir = output_dir / HF_PREPARED_SUBDIR
    prepared_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = prepared_dir / f"{split}.parquet"
    jsonl_path = prepared_dir / f"{split}.jsonl"
    df.to_parquet(parquet_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def _write_readme(output_dir: Path, repo_id: str) -> None:
    readme = f"""# {repo_id}

HF-ready prepared subset for `opcdimage`.

Contents:

- `prepared/train.parquet`
- `prepared/val.parquet`
- `original_images.tar.gz`
- `crop_images.tar.gz`
- `summary.json`

This dataset is the minimal public subset required by the current `opcdimage` training recipe.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def run_export(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        _export_split(split=split, source_dir=source_dir, output_dir=output_dir, source_root=source_root)

    summary_src = source_dir / "summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, output_dir / "summary.json")

    _write_readme(output_dir, repo_id=args.repo_id)
    print(json.dumps({"output_dir": str(output_dir), "repo_id": args.repo_id}, ensure_ascii=False))


def _resolve_repo_path(repo_id: str, output_dir: Path, *, allow_patterns: list[str], force_download: bool) -> Path:
    local_repo_path = repo_id.removeprefix("file://")
    if Path(local_repo_path).exists():
        local_repo_path = Path(local_repo_path).resolve()
        if force_download and output_dir.exists():
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        if local_repo_path != output_dir:
            shutil.copytree(local_repo_path, output_dir, dirs_exist_ok=True)
        return output_dir.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            force_download=force_download,
            allow_patterns=allow_patterns,
        )
    ).resolve()


def _extract_archives_if_needed(snapshot_dir: Path, output_dir: Path) -> None:
    image_root = output_dir / "images"
    if (image_root / "original_images").exists() and (image_root / "crop").exists():
        return

    for archive_name in HF_ARCHIVE_FILENAMES:
        archive_path = snapshot_dir / archive_name
        if not archive_path.exists():
            alt_archive_path = snapshot_dir / "archives" / archive_name
            if alt_archive_path.exists():
                archive_path = alt_archive_path
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing expected archive: {archive_name}")

        with tarfile.open(archive_path, "r:gz") as handle:
            try:
                handle.extractall(output_dir, filter="data")
            except TypeError:
                handle.extractall(output_dir)


def _cleanup_downloaded_archives(output_dir: Path) -> None:
    archive_dirs = [output_dir, output_dir / "archives"]
    for archive_dir in archive_dirs:
        for archive_name in HF_ARCHIVE_FILENAMES:
            archive_path = archive_dir / archive_name
            if archive_path.exists():
                archive_path.unlink()


def _ensure_prepared_dir(snapshot_dir: Path) -> Path:
    prepared_dir = snapshot_dir / HF_PREPARED_SUBDIR
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Missing prepared directory: {prepared_dir}")
    for split in ["train", "val"]:
        parquet_path = prepared_dir / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing prepared split: {parquet_path}")
    return prepared_dir


def _is_ready_download_dir(output_dir: Path, repo_id: str) -> bool:
    marker_path = output_dir / ".hf_dataset_source.json"
    prepared_dir = output_dir / HF_PREPARED_SUBDIR
    image_root = output_dir / "images"
    if not marker_path.exists():
        return False
    if not prepared_dir.exists():
        return False
    if not (prepared_dir / "train.parquet").exists() or not (prepared_dir / "val.parquet").exists():
        return False
    if not (image_root / "original_images").exists() or not (image_root / "crop").exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return marker.get("repo_id") == repo_id


def _cleanup_legacy_root_outputs(output_dir: Path) -> None:
    for filename in ["train.parquet", "val.parquet", "train.jsonl", "val.jsonl"]:
        legacy_path = output_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()


def ensure_local_hf_dataset(
    output_dir: str | Path,
    repo_id: str = HF_DATASET_REPO_ID,
    force_download: bool = False,
) -> Path:
    _configure_proxy_from_env()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = output_dir / ".hf_dataset_source.json"
    if _is_ready_download_dir(output_dir=output_dir, repo_id=repo_id) and not force_download:
        _cleanup_downloaded_archives(output_dir)
        _cleanup_legacy_root_outputs(output_dir)
        return output_dir

    snapshot_dir = _resolve_repo_path(
        repo_id,
        output_dir=output_dir,
        allow_patterns=[
            "README.md",
            "summary.json",
            "prepared/*",
            "images/*",
            "images/**/*",
            *HF_ARCHIVE_FILENAMES,
            "archives/*",
        ],
        force_download=force_download,
    )

    _ensure_prepared_dir(snapshot_dir=snapshot_dir)
    _extract_archives_if_needed(snapshot_dir=snapshot_dir, output_dir=output_dir)

    marker_path.write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "snapshot_dir": str(snapshot_dir),
                "prepared_subdir": HF_PREPARED_SUBDIR,
                "relative_paths": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _cleanup_downloaded_archives(output_dir)
    _cleanup_legacy_root_outputs(output_dir)
    return output_dir


def run_download(args: argparse.Namespace) -> None:
    dataset_dir = ensure_local_hf_dataset(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        force_download=args.force_download,
    )
    print(json.dumps({"dataset_dir": str(dataset_dir), "repo_id": args.repo_id}, ensure_ascii=False))


def _iter_upload_files(local_dir: Path, include_readme: bool) -> list[Path]:
    files: list[Path] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "README.md" and not include_readme:
            continue
        files.append(path)

    def sort_key(path: Path) -> tuple[int, str]:
        rel = path.relative_to(local_dir).as_posix()
        if rel.startswith("prepared/"):
            return (0, rel)
        if rel == "summary.json":
            return (1, rel)
        if rel.startswith("images/original_images/"):
            return (2, rel)
        if rel.startswith("images/crop/"):
            return (3, rel)
        return (4, rel)

    return sorted(files, key=sort_key)


def _matches_any_pattern(path: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(path, pattern) for pattern in patterns)


def _filter_upload_files(
    files: list[Path],
    local_dir: Path,
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> list[Path]:
    if not allow_patterns and not ignore_patterns:
        return files

    filtered: list[Path] = []
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        if allow_patterns and not _matches_any_pattern(rel, allow_patterns):
            continue
        if ignore_patterns and _matches_any_pattern(rel, ignore_patterns):
            continue
        filtered.append(path)
    return filtered


def _parse_patterns(values: list[str] | None) -> list[str]:
    if not values:
        return []
    patterns: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                patterns.append(item)
    return patterns


def _upload_batch(
    api,
    repo_id: str,
    repo_type: str,
    files: list[Path],
    local_dir: Path,
    batch_index: int,
    total_batches: int,
) -> None:
    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(
            path_in_repo=path.relative_to(local_dir).as_posix(),
            path_or_fileobj=str(path),
        )
        for path in files
    ]
    result = api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=f"Upload opcdimage dataset batch {batch_index}/{total_batches}",
    )
    print(f"batch {batch_index}/{total_batches}: {result.commit_url}")


def run_upload(args: argparse.Namespace) -> None:
    _configure_proxy_from_env()

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    local_dir = args.local_dir.resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Local dataset folder not found: {local_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.start_batch <= 0:
        raise ValueError("start_batch must be >= 1")

    api = HfApi(token=None)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=args.exist_ok,
    )

    allow_patterns = _parse_patterns(args.allow_pattern)
    ignore_patterns = _parse_patterns(args.ignore_pattern)

    files = _iter_upload_files(local_dir=local_dir, include_readme=args.include_readme)
    files = _filter_upload_files(
        files=files,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    if args.skip_existing:
        repo_info = api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
        existing = {s.rfilename for s in repo_info.siblings}
        files = [path for path in files if path.relative_to(local_dir).as_posix() not in existing]

    print(f"repo_id={args.repo_id}")
    print(f"mode={args.mode}")
    print(f"total_files={len(files)}")
    if allow_patterns:
        print(f"allow_patterns={allow_patterns}")
    if ignore_patterns:
        print(f"ignore_patterns={ignore_patterns}")

    if args.mode == "large-folder":
        kwargs = {
            "repo_id": args.repo_id,
            "folder_path": local_dir,
            "repo_type": args.repo_type,
            "private": args.private,
            "allow_patterns": allow_patterns or None,
            "ignore_patterns": ignore_patterns or None,
            "num_workers": args.num_workers,
            "print_report": True,
        }
        try:
            api.upload_large_folder(**kwargs)
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                raise RuntimeError(
                    "Hit Hugging Face commit rate limit while running upload_large_folder. "
                    "Wait for the limit window to reset, then rerun the same command."
                ) from exc
            raise
        return

    total_batches = (len(files) + args.batch_size - 1) // args.batch_size
    end_batch = total_batches if args.end_batch <= 0 else min(args.end_batch, total_batches)
    if args.start_batch > total_batches and total_batches > 0:
        raise ValueError(f"start_batch={args.start_batch} exceeds total_batches={total_batches}")

    print(f"batch_size={args.batch_size}")
    print(f"total_batches={total_batches}")
    print(f"uploading_batches={args.start_batch}..{end_batch}")

    for batch_index in range(args.start_batch, end_batch + 1):
        start = (batch_index - 1) * args.batch_size
        end = min(start + args.batch_size, len(files))
        batch_files = files[start:end]
        while True:
            try:
                _upload_batch(
                    api=api,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    files=batch_files,
                    local_dir=local_dir,
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
                break
            except HfHubHTTPError as exc:
                if exc.response is None or exc.response.status_code != 429 or args.retry_seconds <= 0:
                    raise
                print(
                    f"batch {batch_index}/{total_batches} hit rate limit; "
                    f"sleeping {args.retry_seconds} seconds before retry"
                )
                time.sleep(args.retry_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset preparation and Hub tools for opcdimage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare train/val parquet from the source dataset.")
    prepare.add_argument("--input", type=Path, default=Path("..") / "ZwZ-RL-VQA-mini" / "train_crop_clean.csv")
    prepare.add_argument("--dataset-root", type=Path, default=Path("..") / "ZwZ-RL-VQA-mini")
    prepare.add_argument("--output-dir", type=Path, default=Path("data") / "opcdimage_qwen3vl4b")
    prepare.add_argument("--val-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.set_defaults(func=run_prepare)

    validate = subparsers.add_parser("validate", help="Validate prepared train/val parquet.")
    validate.add_argument(
        "--train-file", type=Path, default=Path("data") / "opcdimage_qwen3vl4b" / "prepared" / "train.parquet"
    )
    validate.add_argument(
        "--val-file", type=Path, default=Path("data") / "opcdimage_qwen3vl4b" / "prepared" / "val.parquet"
    )
    validate.set_defaults(func=run_validate)

    export_parser = subparsers.add_parser("export", help="Export local prepared data into a HF-ready folder.")
    export_parser.add_argument("--source-dir", type=Path, required=True)
    export_parser.add_argument("--source-root", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)
    export_parser.set_defaults(func=run_export)

    download_parser = subparsers.add_parser("download", help="Download the HF dataset and unpack it locally.")
    download_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)
    download_parser.add_argument("--force-download", action="store_true")
    download_parser.set_defaults(func=run_download)

    upload_parser = subparsers.add_parser("upload", help="Upload the HF-ready dataset folder to the Hub.")
    upload_parser.add_argument("--local-dir", type=Path, required=True)
    upload_parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)
    upload_parser.add_argument("--repo-type", type=str, default="dataset")
    upload_parser.add_argument("--private", action="store_true")
    upload_parser.add_argument("--exist-ok", action="store_true")
    upload_parser.add_argument("--batch-size", type=int, default=128)
    upload_parser.add_argument("--start-batch", type=int, default=1)
    upload_parser.add_argument("--end-batch", type=int, default=0, help="0 means upload through the last batch.")
    upload_parser.add_argument("--include-readme", action="store_true")
    upload_parser.add_argument("--skip-existing", action="store_true")
    upload_parser.add_argument("--mode", choices=["commit-batches", "large-folder"], default="commit-batches")
    upload_parser.add_argument("--allow-pattern", action="append", default=[])
    upload_parser.add_argument("--ignore-pattern", action="append", default=[])
    upload_parser.add_argument("--num-workers", type=int, default=8)
    upload_parser.add_argument("--retry-seconds", type=int, default=0)
    upload_parser.set_defaults(func=run_upload)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
