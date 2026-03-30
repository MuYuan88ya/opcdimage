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


def test_ensure_local_hf_dataset_supports_split_image_archive_repo(tmp_path):
    prepared_repo = tmp_path / "prepared_repo"
    prepared_dir = prepared_repo / "prepared"
    prepared_dir.mkdir(parents=True)

    image_repo = tmp_path / "image_repo"
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
    (prepared_repo / "summary.json").write_text(json.dumps({"rows": 2}), encoding="utf-8")

    image_repo.mkdir(parents=True)
    _write_archive(image_repo / "original_images.tar.gz", staging_root, [original_image])
    _write_archive(image_repo / "crop_images.tar.gz", staging_root, [crop_image])

    output_dir = tmp_path / "downloaded"
    ensure_local_hf_dataset(
        output_dir=output_dir,
        repo_id=f"file://{prepared_repo}",
        image_repo_id=f"file://{image_repo}",
    )

    train_df = pd.read_parquet(output_dir / "train.parquet")
    assert Path(train_df.loc[0, "original_images"][0]).exists()
    assert Path(train_df.loc[0, "crop_images"][0]).exists()
    assert str(output_dir / "images" / "original_images" / "sample.jpg") == train_df.loc[0, "original_images"][0]
    assert str(output_dir / "images" / "crop" / "sample_crop.png") == train_df.loc[0, "crop_images"][0]

    extra_info = train_df.loc[0, "extra_info"]
    assert extra_info["original_image"] == str(output_dir / "images" / "original_images" / "sample.jpg")
    assert extra_info["crop_image"] == str(output_dir / "images" / "crop" / "sample_crop.png")

    marker = json.loads((output_dir / ".hf_dataset_source.json").read_text(encoding="utf-8"))
    assert marker["repo_id"] == f"file://{prepared_repo}"
    assert marker["image_repo_id"] == f"file://{image_repo}"
    assert marker["archives"]["original"].endswith("original_images.tar.gz")
    assert marker["archives"]["crop"].endswith("crop_images.tar.gz")
