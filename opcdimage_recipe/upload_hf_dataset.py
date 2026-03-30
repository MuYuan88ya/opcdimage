from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

HF_DATASET_REPO_ID = os.environ.get("OPCDIMAGE_HF_DATASET_REPO_ID", "muyuho/opcdmini")


def _configure_proxy_from_env() -> None:
    proxy = os.environ.get("OPCDIMAGE_PROXY")
    if not proxy:
        return
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)
    os.environ.setdefault("ALL_PROXY", proxy)


def _iter_files(local_dir: Path, include_readme: bool) -> list[Path]:
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


def _filter_files(
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


def _upload_batch(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    files: list[Path],
    local_dir: Path,
    batch_index: int,
    total_batches: int,
) -> None:
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


def main() -> None:
    _configure_proxy_from_env()

    parser = argparse.ArgumentParser(description="Upload the HF-ready opcdimage dataset folder to the Hub in batches.")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=HF_DATASET_REPO_ID)
    parser.add_argument("--repo-type", type=str, default="dataset")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--end-batch", type=int, default=0, help="0 means upload through the last batch.")
    parser.add_argument("--include-readme", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mode", choices=["commit-batches", "large-folder"], default="commit-batches")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--retry-seconds", type=int, default=0)
    args = parser.parse_args()

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

    files = _iter_files(local_dir=local_dir, include_readme=args.include_readme)
    files = _filter_files(files=files, local_dir=local_dir, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
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


if __name__ == "__main__":
    main()
