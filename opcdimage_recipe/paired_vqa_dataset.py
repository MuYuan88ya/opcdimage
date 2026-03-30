import copy
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


class OPCDImagePairedVQADataset(RLHFDataset):
    """Keep dataset format close to upstream RLHFDataset; crop stays in trainer logic."""

    prompt_key = "problem"
    image_key = "original_images"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_data_file = Path(self.data_files[0]).resolve()
        self.dataset_root = first_data_file.parent

    def _resolve_dataset_path(self, value: Any) -> Any:
        if isinstance(value, str):
            path = Path(value)
            if path.is_absolute():
                return str(path)
            return str((self.dataset_root / path).resolve())
        if isinstance(value, list):
            return [self._resolve_dataset_path(item) for item in value]
        if isinstance(value, dict):
            resolved = dict(value)
            for key in ("original_image", "crop_image"):
                if key in resolved:
                    resolved[key] = self._resolve_dataset_path(resolved[key])
            return resolved
        return value

    @staticmethod
    def _normalize_image_payload(image: Any) -> dict[str, Any]:
        if isinstance(image, str):
            return {"type": "image", "image": image}
        if isinstance(image, Image.Image):
            return {"type": "image", "image": image.convert("RGB")}
        if isinstance(image, dict):
            image_dict = dict(image)
            if "bytes" in image_dict and image_dict["bytes"] is not None:
                image_dict["image"] = Image.open(BytesIO(image_dict["bytes"])).convert("RGB")
            elif "path" in image_dict and "image" not in image_dict:
                image_dict["image"] = image_dict["path"]
            return {"type": "image", **image_dict}
        raise TypeError(f"Unsupported image payload type: {type(image)}")

    def _build_messages(self, example: dict):
        prompt = example[self.prompt_key]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("problem must be a non-empty string.")
        messages = [{"role": "user", "content": prompt}]
        images = list(example.get(self.image_key) or [])
        image_offset = 0

        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = content.split("<image>")
            for segment_id, segment in enumerate(segments):
                if segment_id > 0:
                    if image_offset >= len(images):
                        raise ValueError(
                            f"Prompt expects more images than provided for key '{self.image_key}': "
                            f"{image_offset} >= {len(images)}"
                        )
                    content_list.append(self._normalize_image_payload(images[image_offset]))
                    image_offset += 1
                if segment:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        if image_offset != len(images):
            raise ValueError(
                f"Unused images detected for key '{self.image_key}': consumed {image_offset}, provided {len(images)}"
            )
        return messages

    def re_tokenize(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        apply_kwargs = dict(**self.apply_chat_template_kwargs)
        if self.tool_schemas is not None:
            apply_kwargs["tools"] = self.tool_schemas

        raw_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )

        images = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item.get("image", item.get("path")))

        model_inputs = dict(self.processor(text=[raw_prompt], images=images, return_tensors="pt"))
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        is_qwen2vl = (
            hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        )
        if is_qwen2vl:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask.squeeze(0),
            )
        else:
            position_ids = compute_position_id_with_mask(attention_mask)[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        multi_modal_inputs = dict(model_inputs)
        multi_modal_inputs.pop("second_per_grid_ts", None)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }

    def __getitem__(self, item):
        row_dict: dict = copy.deepcopy(self.dataframe[item])
        row_dict["original_images"] = self._resolve_dataset_path(row_dict.get("original_images") or [])
        row_dict["crop_images"] = self._resolve_dataset_path(row_dict.get("crop_images") or [])
        if "extra_info" in row_dict and row_dict["extra_info"] is not None:
            row_dict["extra_info"] = self._resolve_dataset_path(row_dict["extra_info"])
        row_dict["raw_prompt"] = self._build_messages(row_dict)
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        crop_images = row_dict.get("crop_images") or []
        if len(crop_images) != 1:
            raise KeyError("Missing required crop_images field for opcdimage privileged crop training.")
        if not row_dict["extra_info"].get("crop_image"):
            raise KeyError("Missing required extra_info.crop_image for opcdimage privileged crop training.")

        index = row_dict.get("extra_info", {}).get("index", item)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})

        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
