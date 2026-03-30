from copy import deepcopy
import re
from typing import Any


BOUNDING_BOX_PATTERNS = (
    "only focus on the objects inside the red bounding box in the image to answer this question.",
    "only focus on the object inside the red bounding box in the image to answer this question.",
    "only focus on the region inside the red bounding box in the image to answer this question.",
)

LETTER_PATTERN = re.compile(r"\b([A-Z])\b")
BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-Z])\s*\}")
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)


def normalize_problem(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string.")

    text = problem.replace("\r\n", "\n").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip()

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    kept_blocks: list[str] = []
    for block in blocks:
        normalized_block = " ".join(block.lower().split())
        if any(pattern in normalized_block for pattern in BOUNDING_BOX_PATTERNS):
            continue
        if "red bounding box" in normalized_block and ("focus" in normalized_block or "inside" in normalized_block):
            continue
        kept_blocks.append(block)

    if not kept_blocks:
        raise ValueError(f"Prompt normalization removed all content: {problem}")

    return "<image>" + "\n\n".join(kept_blocks).strip()


def extract_choice(solution_str: str) -> str | None:
    if not isinstance(solution_str, str):
        return None

    upper = solution_str.upper()
    for pattern in (ANSWER_TAG_PATTERN, BOXED_PATTERN):
        matches = pattern.findall(upper)
        if matches:
            return matches[-1]

    letter_matches = LETTER_PATTERN.findall(upper)
    if letter_matches:
        return letter_matches[-1]
    return None


def build_crop_messages_from_raw_prompt(raw_prompt: Any, crop_image: str) -> list[dict[str, Any]]:
    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, (str, bytes, bytearray)):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, list) or not raw_prompt:
        raise ValueError("raw_prompt must be a non-empty list of chat messages.")
    if not isinstance(crop_image, str) or not crop_image:
        raise ValueError("crop_image must be a non-empty string path.")

    messages = deepcopy(raw_prompt)
    image_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_count += 1
                item["image"] = crop_image
                if "path" in item:
                    item["path"] = crop_image
                item.pop("bytes", None)

    if image_count != 1:
        raise ValueError(f"Expected exactly one image in raw_prompt, got {image_count}.")
    return messages


def ensure_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")
    return value
