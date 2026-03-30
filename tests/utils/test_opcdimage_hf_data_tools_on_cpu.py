from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd

from opcdimage_recipe.hf_data_tools import ensure_local_hf_dataset


def _write_archive(archive_path: Path, source_root: Path, members: list[Path]) -> None:
    with tarfile.open(archive_path, "w:gz") as handle:
        for member in members:
            handle.add(member, arcname=member.relative_to(source_root).as_posix())


def test_ensure_local_hf_dataset_keeps_relative_paths_and_extracts_archives(tmp_path):
    dataset_repo = tmp_path / "dataset_repo"
    prepared_dir = dataset_repo / "prepared"
    prepared_dir.mkdir(parents=True)

    staging_root = tmp_path / "staging"
    original_image = staging_root / "images" / "original_images" / "sample.jpg"
    crop_image = staging_root / "images" / "crop" / "sample_crop.png"
    original_image.parent.mkdir(parents=True, exist_ok=True)
    crop_image.parent.mkdir(parents=True, exist_ok=True)
    original_image.write_bytes(b"original-image")
    crop_image.write_bytes(b"crop-image")

    row = {
        "problem": "Which option is correct?",
        "original_images": ["images/original_images/sample.jpg"],
        "crop_images": ["images/crop/sample_crop.png"],
        "extra_info": {
            "original_image": "images/original_images/sample.jpg",
            "crop_image": "images/crop/sample_crop.png",
        },
    }
    pd.DataFrame([row]).to_parquet(prepared_dir / "train.parquet", index=False)
    pd.DataFrame([row]).to_parquet(prepared_dir / "val.parquet", index=False)
    (dataset_repo / "summary.json").write_text(json.dumps({"rows": 2}), encoding="utf-8")
    _write_archive(dataset_repo / "original_images.tar.gz", staging_root, [original_image])
    _write_archive(dataset_repo / "crop_images.tar.gz", staging_root, [crop_image])

    output_dir = tmp_path / "downloaded"
    ensure_local_hf_dataset(output_dir=output_dir, repo_id=f"file://{dataset_repo}")

    train_df = pd.read_parquet(output_dir / "train.parquet")
    assert train_df.loc[0, "original_images"][0] == "images/original_images/sample.jpg"
    assert train_df.loc[0, "crop_images"][0] == "images/crop/sample_crop.png"
    assert (output_dir / "images" / "original_images" / "sample.jpg").exists()
    assert (output_dir / "images" / "crop" / "sample_crop.png").exists()

    extra_info = train_df.loc[0, "extra_info"]
    assert extra_info["original_image"] == "images/original_images/sample.jpg"
    assert extra_info["crop_image"] == "images/crop/sample_crop.png"

    marker = json.loads((output_dir / ".hf_dataset_source.json").read_text(encoding="utf-8"))
    assert marker["repo_id"] == f"file://{dataset_repo}"
    assert marker["relative_paths"] is True
