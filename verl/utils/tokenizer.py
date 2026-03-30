# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""

import os
import types
import warnings

__all__ = ["hf_tokenizer", "hf_processor", "normalize_token_ids"]


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def _read_int_env(var_name: str):
    value = os.getenv(var_name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"{var_name} must be an integer, got: {value!r}") from e


def _normalize_hw(value):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if value is None:
        raise ValueError("patch_size is required to derive max_pixels from max_image_tokens")
    return int(value), int(value)


def _max_pixels_from_image_tokens(image_processor, max_image_tokens: int) -> int | None:
    patch_h, patch_w = _normalize_hw(getattr(image_processor, "patch_size", None))
    merge_size = int(getattr(image_processor, "merge_size", 1))
    pixels_per_visual_token = patch_h * patch_w * merge_size * merge_size
    return int(max_image_tokens) * pixels_per_visual_token


def _apply_image_processor_limits(processor, *, max_pixels: int | None, min_pixels: int | None):
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return

    size = getattr(image_processor, "size", None)
    if not isinstance(size, dict):
        size = {}

    if min_pixels is not None:
        size["shortest_edge"] = int(min_pixels)
        setattr(image_processor, "min_pixels", int(min_pixels))

    if max_pixels is not None:
        size["longest_edge"] = int(max_pixels)
        setattr(image_processor, "max_pixels", int(max_pixels))

    if size:
        image_processor.size = size


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    explicit_min_pixels = kwargs.pop("min_pixels", None)
    explicit_max_pixels = kwargs.pop("max_pixels", None)
    explicit_max_image_tokens = kwargs.pop("max_image_tokens", None)

    env_min_pixels = _read_int_env("VERL_PROCESSOR_MIN_PIXELS")
    env_max_pixels = _read_int_env("VERL_PROCESSOR_MAX_PIXELS")
    env_max_image_tokens = _read_int_env("VERL_PROCESSOR_MAX_IMAGE_TOKENS")

    min_pixels = explicit_min_pixels if explicit_min_pixels is not None else env_min_pixels
    max_pixels = explicit_max_pixels if explicit_max_pixels is not None else env_max_pixels
    max_image_tokens = (
        explicit_max_image_tokens if explicit_max_image_tokens is not None else env_max_image_tokens
    )

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        if isinstance(processor, PreTrainedTokenizerBase):
            return None

        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor
        processor.config = config
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)

        if processor is not None and getattr(processor, "image_processor", None) is not None:
            resolved_max_pixels = max_pixels
            if resolved_max_pixels is None and max_image_tokens is not None:
                resolved_max_pixels = _max_pixels_from_image_tokens(processor.image_processor, max_image_tokens)
                warnings.warn(
                    (
                        f"Apply processor image cap: max_image_tokens={max_image_tokens} -> "
                        f"max_pixels={resolved_max_pixels}"
                    ),
                    stacklevel=1,
                )
            _apply_image_processor_limits(
                processor,
                max_pixels=resolved_max_pixels,
                min_pixels=min_pixels,
            )
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
