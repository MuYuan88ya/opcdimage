from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download

HF_DATASET_REPO_ID = os.environ.get("OPCDIMAGE_HF_DATASET_REPO_ID", "muyuho/opcdmini")
HF_IMAGE_DATASET_REPO_ID = None
HF_PREPARED_SUBDIR = "prepared"
HF_ARCHIVE_FILENAMES = {
    "original": "original_images.tar.gz",
    "crop": "crop_images.tar.gz",
}


def _configure_proxy_from_env() -> None:
    proxy = os.environ.get("OPCDIMAGE_PROXY")
    if not proxy:
        return
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)
    os.environ.setdefault("ALL_PROXY", proxy)


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _relativize_path(path_str: str, source_root: Path) -> str:
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
            rewritten[key] = _relativize_path(str(rewritten[key]), source_root=source_root)
    return rewritten


def _rewrite_download_paths(value: Any, dataset_root: Path) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return [_rewrite_download_paths(item, dataset_root) for item in value]
    if isinstance(value, dict):
        rewritten = {}
        for key, item in value.items():
            if key in {"original_image", "crop_image"} and isinstance(item, str) and not Path(item).is_absolute():
                rewritten[key] = str((dataset_root / item).resolve())
            else:
                rewritten[key] = _rewrite_download_paths(item, dataset_root)
        return rewritten
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.parts and candidate.parts[0] == "images":
            return str((dataset_root / candidate).resolve())
    return value


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
            rel = _relativize_path(str(item), source_root=source_root)
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
- `images/original_images/*`
- `images/crop/*`
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
        return Path(local_repo_path).resolve()

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            force_download=force_download,
            allow_patterns=allow_patterns,
        )
    ).resolve()


def _extract_image_archives(snapshot_dir: Path, output_dir: Path) -> dict[str, str] | None:
    archive_paths: dict[str, Path] = {}
    for key, filename in HF_ARCHIVE_FILENAMES.items():
        for candidate in [snapshot_dir / filename, snapshot_dir / "archives" / filename]:
            if candidate.exists():
                archive_paths[key] = candidate
                break

    if len(archive_paths) != len(HF_ARCHIVE_FILENAMES):
        return None

    for archive_path in archive_paths.values():
        with tarfile.open(archive_path, "r:gz") as handle:
            try:
                handle.extractall(output_dir, filter="data")
            except TypeError:
                handle.extractall(output_dir)

    return {key: str(path.resolve()) for key, path in archive_paths.items()}


def _ensure_image_tree(
    prepared_snapshot_dir: Path,
    prepared_repo_id: str,
    output_dir: Path,
    *,
    image_repo_id: str | None,
    force_download: bool,
) -> dict[str, Any]:
    image_root = output_dir / "images"
    if (image_root / "original_images").exists() and (image_root / "crop").exists() and not force_download:
        return {"image_source": str(output_dir), "image_repo_id": image_repo_id, "archives": {}}

    if (prepared_snapshot_dir / "images").exists():
        return {
            "image_source": str(prepared_snapshot_dir.resolve()),
            "image_repo_id": image_repo_id,
            "archives": {},
        }

    archive_snapshot_dir = prepared_snapshot_dir
    if image_repo_id and image_repo_id != prepared_repo_id:
        archive_snapshot_dir = _resolve_repo_path(
            image_repo_id,
            output_dir=output_dir,
            allow_patterns=[
                HF_ARCHIVE_FILENAMES["original"],
                HF_ARCHIVE_FILENAMES["crop"],
                "archives/*",
                "images/*",
                "images/**/*",
            ],
            force_download=force_download,
        )

    if (archive_snapshot_dir / "images").exists():
        return {
            "image_source": str(archive_snapshot_dir.resolve()),
            "image_repo_id": image_repo_id,
            "archives": {},
        }

    archive_info = _extract_image_archives(snapshot_dir=archive_snapshot_dir, output_dir=output_dir)
    if archive_info is not None:
        return {
            "image_source": str(output_dir),
            "image_repo_id": image_repo_id,
            "archives": archive_info,
        }

    raise FileNotFoundError(
        "Could not find usable image files for the HF dataset. "
        f"prepared_snapshot_dir={prepared_snapshot_dir}, image_repo_id={image_repo_id}"
    )


def _materialize_split(snapshot_dir: Path, output_dir: Path, split: str, dataset_root: Path) -> None:
    prepared_path = snapshot_dir / HF_PREPARED_SUBDIR / f"{split}.parquet"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Missing prepared split: {prepared_path}")
    df = pd.read_parquet(prepared_path)
    for column in ["original_images", "crop_images", "extra_info"]:
        if column in df.columns:
            df[column] = df[column].map(lambda x: _rewrite_download_paths(x, dataset_root))
    output_path = output_dir / f"{split}.parquet"
    df.to_parquet(output_path, index=False)
    jsonl_path = output_dir / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def ensure_local_hf_dataset(
    output_dir: str | Path,
    repo_id: str = HF_DATASET_REPO_ID,
    image_repo_id: str | None = HF_IMAGE_DATASET_REPO_ID,
    force_download: bool = False,
) -> Path:
    _configure_proxy_from_env()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_repo_id:
        image_repo_id = repo_id

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    marker_path = output_dir / ".hf_dataset_source.json"
    if train_path.exists() and val_path.exists() and marker_path.exists() and not force_download:
        return output_dir

    snapshot_dir = _resolve_repo_path(
        repo_id,
        output_dir=output_dir,
        allow_patterns=[
            "README.md",
            "summary.json",
            "prepared/*",
            HF_ARCHIVE_FILENAMES["original"],
            HF_ARCHIVE_FILENAMES["crop"],
            "archives/*",
            "images/*",
            "images/**/*",
        ],
        force_download=force_download,
    )

    image_info = _ensure_image_tree(
        prepared_snapshot_dir=snapshot_dir,
        prepared_repo_id=repo_id,
        output_dir=output_dir,
        image_repo_id=image_repo_id,
        force_download=force_download,
    )
    dataset_root = Path(image_info["image_source"]).resolve()

    for split in ["train", "val"]:
        _materialize_split(snapshot_dir=snapshot_dir, output_dir=output_dir, split=split, dataset_root=dataset_root)

    summary_src = snapshot_dir / "summary.json"
    if summary_src.exists():
        summary_dst = output_dir / "summary.json"
        if summary_src.resolve() != summary_dst.resolve():
            shutil.copy2(summary_src, summary_dst)

    marker_path.write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "image_repo_id": image_repo_id,
                "snapshot_dir": str(snapshot_dir),
                "image_source": image_info["image_source"],
                "prepared_subdir": HF_PREPARED_SUBDIR,
                "archives": image_info["archives"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def run_download(args: argparse.Namespace) -> None:
    dataset_dir = ensure_local_hf_dataset(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        force_download=args.force_download,
    )
    print(
        json.dumps(
            {"dataset_dir": str(dataset_dir), "repo_id": args.repo_id},
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HF export/download tools for opcdimage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export local prepared data into a HF-ready folder.")
    export_parser.add_argument("--source-dir", type=Path, required=True)
    export_parser.add_argument("--source-root", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)

    download_parser = subparsers.add_parser("download", help="Download the HF-ready dataset and rewrite paths.")
    download_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)
    download_parser.add_argument("--force-download", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "export":
        run_export(args)
        return
    if args.command == "download":
        run_download(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
