from typing import Any

from opcdimage_recipe.core import extract_choice


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    del data_source
    del extra_info

    pred = extract_choice(solution_str)
    gold = str(ground_truth).strip().upper()
    correct = pred == gold
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "pred": pred or "[INVALID]",
    }
