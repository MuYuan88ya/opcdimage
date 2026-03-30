from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from opcdimage_recipe.core import normalize_problem


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


def ensure_single_item_list(value, field_name: str):
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


def validate_frame(frame: pd.DataFrame, split_name: str, dataset_root: Path) -> dict:
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

    return {
        "rows": len(frame),
        "unique_original_images": len(image_ids),
    }


def run_validate(args: argparse.Namespace) -> None:
    train_file = args.train_file.resolve()
    val_file = args.val_file.resolve()
    train_frame = pd.read_parquet(train_file)
    val_frame = pd.read_parquet(val_file)

    train_stats = validate_frame(train_frame, "train", dataset_root=train_file.parent)
    val_stats = validate_frame(val_frame, "val", dataset_root=val_file.parent)

    train_images = {row["extra_info"]["original_image"] for row in train_frame.to_dict("records")}
    val_images = {row["extra_info"]["original_image"] for row in val_frame.to_dict("records")}
    overlap = sorted(train_images & val_images)
    if overlap:
        raise ValueError(f"train/val leakage detected on original images: {overlap[:5]}")

    print(
        {
            "train": train_stats,
            "val": val_stats,
            "cross_split_original_image_overlap": 0,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or validate opcdimage data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare train/val parquet from ZwZ-RL-VQA-mini.")
    prepare.add_argument("--input", type=Path, default=Path("..") / "ZwZ-RL-VQA-mini" / "train_crop_clean.csv")
    prepare.add_argument("--dataset-root", type=Path, default=Path("..") / "ZwZ-RL-VQA-mini")
    prepare.add_argument("--output-dir", type=Path, default=Path("data") / "opcdimage_qwen3vl4b")
    prepare.add_argument("--val-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.set_defaults(func=run_prepare)

    validate = subparsers.add_parser("validate", help="Validate prepared train/val parquet.")
    validate.add_argument("--train-file", type=Path, default=Path("data") / "opcdimage_qwen3vl4b" / "train.parquet")
    validate.add_argument("--val-file", type=Path, default=Path("data") / "opcdimage_qwen3vl4b" / "val.parquet")
    validate.set_defaults(func=run_validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
