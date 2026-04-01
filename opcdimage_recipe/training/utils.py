from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf

from verl.utils.import_utils import load_extern_object
from verl.utils.torch_functional import get_response_mask
from verl.workers.reward_manager import get_reward_manager_cls


def get_opcdimage_options(config) -> dict[str, object]:
    privileged_mode = OmegaConf.select(config, "opcdimage.privileged_mode") or "crop"
    on_policy_merge = OmegaConf.select(config, "opcdimage.on_policy_merge")
    if on_policy_merge is None:
        on_policy_merge = True
    return {"privileged_mode": privileged_mode, "on_policy_merge": bool(on_policy_merge)}


def load_opcdimage_reward_manager(config, tokenizer, num_examine: int = 0):
    reward_manager_source = OmegaConf.select(config, "reward.reward_manager.source") or "register"
    reward_manager_name = OmegaConf.select(config, "reward.reward_manager.name") or "naive"

    if reward_manager_source == "register":
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    elif reward_manager_source == "importlib":
        module_path = OmegaConf.select(config, "reward.reward_manager.module.path")
        module_name = OmegaConf.select(config, "reward.reward_manager.module.name") or reward_manager_name
        if not module_path:
            raise ValueError("reward.reward_manager.module.path must be set when source=importlib.")
        reward_manager_cls = load_extern_object(module_path, module_name)
    else:
        raise ValueError(f"Unsupported reward manager source: {reward_manager_source}")

    reward_fn_path = OmegaConf.select(config, "reward.custom_reward_function.path")
    reward_fn_name = OmegaConf.select(config, "reward.custom_reward_function.name") or "compute_score"
    if not reward_fn_path:
        raise ValueError("opcdimage recipe requires reward.custom_reward_function.path to be set.")
    compute_score = load_extern_object(reward_fn_path, reward_fn_name)

    reward_kwargs_cfg = OmegaConf.select(config, "reward.reward_kwargs")
    reward_kwargs = (
        dict(OmegaConf.to_container(reward_kwargs_cfg, resolve=True)) if reward_kwargs_cfg is not None else {}
    )

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compose_prompt_response_tensors(
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    responses: torch.Tensor,
    eos_token_id: int,
) -> dict[str, torch.Tensor]:
    response_attention_mask = get_response_mask(
        response_id=responses,
        eos_token=eos_token_id,
        dtype=prompt_attention_mask.dtype,
    )
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    sequence = torch.cat([prompt_input_ids, responses], dim=-1)

    response_length = responses.size(1)
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(prompt_input_ids.size(0), -1)
    if prompt_position_ids.dim() == 3:
        delta_position_id = delta_position_id.view(prompt_input_ids.size(0), 1, -1).expand(
            prompt_input_ids.size(0), 3, -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

    return {
        "prompts": prompt_input_ids,
        "responses": responses,
        "input_ids": sequence,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def to_python_list(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    return value
