import copy
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset


class OPCDImagePairedVQADataset(RLHFDataset):
    """Keep dataset format close to upstream RLHFDataset; crop stays in trainer logic."""

    prompt_key = "problem"
    image_key = "original_images"

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

        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            raise ValueError("processor must return position_ids for opcdimage vision training.")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids[0],
        }

    def __getitem__(self, item):
        row_dict: dict = copy.deepcopy(self.dataframe[item])
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
