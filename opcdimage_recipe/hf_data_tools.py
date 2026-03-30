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
HF_PREPARED_SUBDIR = "prepared"
HF_ARCHIVE_FILENAMES = [
    "original_images.tar.gz",
    "crop_images.tar.gz",
]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HF export/download tools for opcdimage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export local prepared data into a HF-ready folder.")
    export_parser.add_argument("--source-dir", type=Path, required=True)
    export_parser.add_argument("--source-root", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)

    download_parser = subparsers.add_parser("download", help="Download the HF dataset and unpack it locally.")
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
