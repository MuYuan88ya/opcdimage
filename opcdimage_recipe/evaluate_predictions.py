import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from opcdimage_recipe.core import extract_choice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OPCDImage predictions on prepared parquet data.")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("data") / "opcdimage_qwen3vl4b" / "val.parquet",
        help="Prepared dataset parquet used as the evaluation reference.",
    )
    parser.add_argument("--main-predictions", type=Path, required=True, help="Main method predictions.")
    parser.add_argument("--baseline-predictions", type=Path, help="Optional full-image baseline predictions.")
    parser.add_argument("--upper-bound-predictions", type=Path, help="Optional crop upper-bound predictions.")
    parser.add_argument("--id-key", type=str, default="sample_id", help="Sample id column shared by all files.")
    parser.add_argument(
        "--prediction-key",
        type=str,
        default="solution_str",
        help="Prediction text column. If absent but `pred` exists, `pred` will be used.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to save the evaluation summary as JSON.")
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file format: {path}")


def standardize_predictions(frame: pd.DataFrame, *, id_key: str, prediction_key: str, prefix: str) -> pd.DataFrame:
    if id_key not in frame.columns:
        raise KeyError(f"{prefix}: missing id column '{id_key}'.")
    if frame[id_key].duplicated().any():
        duplicate_ids = frame.loc[frame[id_key].duplicated(), id_key].astype(str).tolist()
        raise ValueError(f"{prefix}: duplicated prediction ids detected, e.g. {duplicate_ids[:5]}")

    if "pred" in frame.columns:
        extracted = frame["pred"].astype(str).str.upper()
    elif prediction_key in frame.columns:
        extracted = frame[prediction_key].map(extract_choice)
    else:
        raise KeyError(f"{prefix}: missing '{prediction_key}' and 'pred' columns.")

    result = frame[[id_key]].copy()
    result[f"{prefix}_pred"] = extracted.fillna("[INVALID]")
    return result


def build_eval_frame(dataset_frame: pd.DataFrame, *, id_key: str) -> pd.DataFrame:
    if id_key not in dataset_frame.columns:
        raise KeyError(f"dataset is missing id column '{id_key}'.")

    eval_frame = dataset_frame[[id_key, "ground_truth", "extra_info"]].copy()
    eval_frame["gold"] = eval_frame["ground_truth"].astype(str).str.upper()
    eval_frame["original_megapixels"] = eval_frame["extra_info"].map(lambda info: float(info["original_megapixels"]))
    eval_frame["bbox_area_ratio"] = eval_frame["extra_info"].map(lambda info: float(info["bbox_area_ratio"]))
    return eval_frame


def compute_accuracy_table(frame: pd.DataFrame, prediction_column: str, metric_name: str) -> dict[str, Any]:
    correct = frame[prediction_column] == frame["gold"]
    return {
        "metric": metric_name,
        "count": int(len(frame)),
        "accuracy": float(correct.mean()) if len(frame) else 0.0,
    }


def summarize_by_bins(frame: pd.DataFrame, value_column: str, prediction_column: str, bins, labels) -> list[dict[str, Any]]:
    binned = frame.copy()
    binned["bin"] = pd.cut(binned[value_column], bins=bins, labels=labels, include_lowest=True, right=True)
    rows: list[dict[str, Any]] = []
    for bin_label, group in binned.groupby("bin", dropna=False, observed=False):
        if len(group) == 0:
            continue
        rows.append(
            {
                "bin": str(bin_label),
                "count": int(len(group)),
                "accuracy": float((group[prediction_column] == group["gold"]).mean()),
            }
        )
    return rows


def compute_gap_closure(frame: pd.DataFrame) -> dict[str, float] | None:
    required_columns = {"baseline_pred", "main_pred", "upper_pred"}
    if not required_columns.issubset(frame.columns):
        return None

    baseline_acc = float((frame["baseline_pred"] == frame["gold"]).mean())
    main_acc = float((frame["main_pred"] == frame["gold"]).mean())
    upper_acc = float((frame["upper_pred"] == frame["gold"]).mean())
    gap = upper_acc - baseline_acc
    if gap <= 0:
        return {
            "baseline_accuracy": baseline_acc,
            "main_accuracy": main_acc,
            "upper_accuracy": upper_acc,
            "gap_closure": 0.0,
        }

    return {
        "baseline_accuracy": baseline_acc,
        "main_accuracy": main_acc,
        "upper_accuracy": upper_acc,
        "gap_closure": (main_acc - baseline_acc) / gap,
    }


def merge_predictions(eval_frame: pd.DataFrame, preds: pd.DataFrame, *, id_key: str, prefix: str) -> pd.DataFrame:
    merged = eval_frame.merge(preds, on=id_key, how="inner")
    if len(merged) != len(eval_frame):
        raise ValueError(
            f"{prefix} predictions do not cover the full evaluation set: "
            f"{len(merged)} merged vs {len(eval_frame)} rows."
        )
    return merged


def main() -> None:
    args = parse_args()
    dataset_frame = build_eval_frame(load_table(args.dataset_file.resolve()), id_key=args.id_key)

    main_preds = standardize_predictions(
        load_table(args.main_predictions.resolve()),
        id_key=args.id_key,
        prediction_key=args.prediction_key,
        prefix="main",
    )
    eval_frame = merge_predictions(dataset_frame, main_preds, id_key=args.id_key, prefix="main")

    if args.baseline_predictions is not None:
        baseline_preds = standardize_predictions(
            load_table(args.baseline_predictions.resolve()),
            id_key=args.id_key,
            prediction_key=args.prediction_key,
            prefix="baseline",
        )
        eval_frame = merge_predictions(eval_frame, baseline_preds, id_key=args.id_key, prefix="baseline")

    if args.upper_bound_predictions is not None:
        upper_preds = standardize_predictions(
            load_table(args.upper_bound_predictions.resolve()),
            id_key=args.id_key,
            prediction_key=args.prediction_key,
            prefix="upper",
        )
        eval_frame = merge_predictions(eval_frame, upper_preds, id_key=args.id_key, prefix="upper")

    summary: dict[str, Any] = {
        "overall": compute_accuracy_table(eval_frame, "main_pred", "main_accuracy"),
        "by_megapixels": summarize_by_bins(
            eval_frame,
            value_column="original_megapixels",
            prediction_column="main_pred",
            bins=[0.0, 1.0, 2.0, 4.0, 8.0, float("inf")],
            labels=["<=1MP", "1-2MP", "2-4MP", "4-8MP", ">8MP"],
        ),
        "by_bbox_area_ratio": summarize_by_bins(
            eval_frame,
            value_column="bbox_area_ratio",
            prediction_column="main_pred",
            bins=[0.0, 0.01, 0.03, 0.1, 1.0],
            labels=["<=1%", "1%-3%", "3%-10%", ">10%"],
        ),
    }

    if "baseline_pred" in eval_frame.columns:
        summary["baseline"] = compute_accuracy_table(eval_frame, "baseline_pred", "baseline_accuracy")
    if "upper_pred" in eval_frame.columns:
        summary["upper_bound"] = compute_accuracy_table(eval_frame, "upper_pred", "upper_bound_accuracy")

    gap_closure = compute_gap_closure(eval_frame)
    if gap_closure is not None:
        summary["gap_closure"] = gap_closure

    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_json)
    if args.output is not None:
        args.output.write_text(summary_json, encoding="utf-8")


if __name__ == "__main__":
    main()
